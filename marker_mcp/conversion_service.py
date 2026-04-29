"""Async-safe document conversion service wrapping the Marker library.

Models are initialised at module load time so that subsequent tool calls are fast.
All blocking conversion calls are offloaded to a thread via asyncio.to_thread().

LLM service selection is driven entirely by environment variables so that the
MCP tool signatures stay simple (just use_llm=True) while the backend is
fully configurable. See README.md § LLM-Enhanced Conversion for details.
"""

import asyncio
import base64
import contextlib
import gc
import mimetypes
import os
import re
import tempfile
import warnings
from difflib import SequenceMatcher
from inspect import Parameter, signature
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

_AUTO_DEVICE_SENTINEL = "__auto__"
_MODEL_DTYPE_CHOICES = {"float16", "float32", "bfloat16"}
_GPU_MEMORY_PROFILE_CHOICES = {"low-vram"}
_DEFAULT_PAGE_TILE_OVERLAP_PX = 96
_MIN_PAGE_TILE_OVERLAP_PX = 24
_PAGE_TILE_RENDER_DPI = 192
_DEFAULT_GPU_OOM_TILE_HEIGHT_PX = 1600
_DEFAULT_PDFTEXT_WORKERS = 8
_PAGE_SEPARATOR_DASH_COUNT = 48
_LOW_VRAM_GPU_CONFIG = {
    "layout_batch_size": 2,
    "detection_batch_size": 1,
    "recognition_batch_size": 16,
    "ocr_error_batch_size": 4,
    "table_rec_batch_size": 4,
    "equation_batch_size": 4,
}
_GPU_MEMORY_PROFILE_CONFIGS = {
    "low-vram": _LOW_VRAM_GPU_CONFIG,
}
_OCR_DEVICE_ALIAS_MAP = {
    "auto": _AUTO_DEVICE_SENTINEL,
    "cpu": "cpu",
    "cuda": "cuda",
    "nvidia": "cuda",
    "rocm": "cuda",
    "amd": "cuda",
    "mps": "mps",
}


def _resolve_ocr_device_override(raw_value: str | None) -> str | None:
    """Map the MCP OCR device selector to Marker's TORCH_DEVICE values.

    Marker expects torch-style device strings (`cpu`, `cuda`, `mps`). For ROCm
    builds, PyTorch still uses the `cuda` device string, so the AMD/ROCm aliases
    deliberately map to `cuda`.
    """
    if raw_value is None:
        return None

    normalized = raw_value.strip().lower()
    if not normalized:
        return None

    if normalized not in _OCR_DEVICE_ALIAS_MAP:
        supported = ", ".join(sorted(_OCR_DEVICE_ALIAS_MAP))
        raise ValueError(
            f"Unsupported MARKER_MCP_OCR_DEVICE={raw_value!r}. "
            f"Expected one of: {supported}."
        )

    return _OCR_DEVICE_ALIAS_MAP[normalized]


def _apply_ocr_device_override_from_env() -> None:
    """Translate MARKER_MCP_OCR_DEVICE into Marker's TORCH_DEVICE env var."""
    override = _resolve_ocr_device_override(os.environ.get("MARKER_MCP_OCR_DEVICE"))
    if override is None:
        return

    if override == _AUTO_DEVICE_SENTINEL:
        os.environ.pop("TORCH_DEVICE", None)
        return

    os.environ["TORCH_DEVICE"] = override


def _resolve_model_dtype_override(raw_value: str | None) -> str | None:
    """Validate the optional model dtype override for Marker model loading."""
    if raw_value is None:
        return None

    normalized = raw_value.strip().lower()
    if not normalized:
        return None

    if normalized not in _MODEL_DTYPE_CHOICES:
        supported = ", ".join(sorted(_MODEL_DTYPE_CHOICES))
        raise ValueError(
            f"Unsupported MARKER_MCP_MODEL_DTYPE={raw_value!r}. "
            f"Expected one of: {supported}."
        )

    return normalized


def _resolve_gpu_memory_profile(raw_value: str | None) -> str | None:
    """Validate the optional GPU-memory profile used to tune Marker batch sizes."""
    if raw_value is None:
        return None

    normalized = raw_value.strip().lower()
    if not normalized:
        return None

    if normalized not in _GPU_MEMORY_PROFILE_CHOICES:
        supported = ", ".join(sorted(_GPU_MEMORY_PROFILE_CHOICES))
        raise ValueError(
            f"Unsupported gpu_memory_profile={raw_value!r}. "
            f"Expected one of: {supported}."
        )

    return normalized


_apply_ocr_device_override_from_env()

from marker.config.parser import ConfigParser
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

try:
    from PIL import Image as PILImage
except Exception:  # pragma: no cover - Pillow is an indirect Marker dependency.
    PILImage = None

# ---------------------------------------------------------------------------
# LLM service configuration — resolved from environment variables.
#
# MARKER_LLM_SERVICE   import path of the service class (overrides all others)
#
# Ollama (local):
#   OLLAMA_BASE_URL    base URL            (default: http://localhost:11434)
#   OLLAMA_MODEL       vision model name   (recommended: gemma4:31b-cloud)
#
# OpenAI / compatible:
#   OPENAI_API_KEY     API key
#   OPENAI_BASE_URL    base URL            (default: https://api.openai.com/v1)
#   OPENAI_MODEL       model name          (default: gpt-4o-mini)
#
# Gemini (default when GOOGLE_API_KEY is set):
#   GOOGLE_API_KEY     Gemini API key
#
# Claude:
#   CLAUDE_API_KEY     API key
#   CLAUDE_MODEL       model name          (default: claude-3-5-sonnet-20241022)
# ---------------------------------------------------------------------------

def _llm_options_from_env() -> dict:
    """Build the LLM-related options dict from environment variables."""
    opts: dict = {}

    def _int_env(name: str) -> int | None:
        raw = os.environ.get(name)
        if raw is None:
            return None
        raw = raw.strip()
        if not raw:
            return None
        return int(raw)

    explicit_service = os.environ.get("MARKER_LLM_SERVICE")
    if explicit_service:
        opts["llm_service"] = explicit_service

    # Ollama — detected by OLLAMA_BASE_URL or OLLAMA_MODEL being set
    ollama_url = os.environ.get("OLLAMA_BASE_URL")
    ollama_model = os.environ.get("OLLAMA_MODEL")
    if (ollama_url or ollama_model) and not explicit_service:
        opts["llm_service"] = "marker_mcp.ollama_service.OllamaService"
    if ollama_url:
        opts["ollama_base_url"] = ollama_url
    if ollama_model:
        opts["ollama_model"] = ollama_model
    ollama_batch_size = _int_env("OLLAMA_BATCH_SIZE")
    ollama_batch_wait_ms = _int_env("OLLAMA_BATCH_WAIT_MS")
    if ollama_batch_size is not None:
        opts["ollama_batch_size"] = ollama_batch_size
    if ollama_batch_wait_ms is not None:
        opts["ollama_batch_wait_ms"] = ollama_batch_wait_ms

    # OpenAI-compatible — detected by OPENAI_API_KEY being set
    openai_key = os.environ.get("OPENAI_API_KEY")
    openai_url = os.environ.get("OPENAI_BASE_URL")
    openai_model = os.environ.get("OPENAI_MODEL")
    if openai_key and not explicit_service and "llm_service" not in opts:
        opts["llm_service"] = "marker.services.openai.OpenAIService"
    if openai_key:
        opts["openai_api_key"] = openai_key
    if openai_url:
        opts["openai_base_url"] = openai_url
    if openai_model:
        opts["openai_model"] = openai_model

    # Claude
    claude_key = os.environ.get("CLAUDE_API_KEY")
    claude_model = os.environ.get("CLAUDE_MODEL")
    if claude_key and not explicit_service and "llm_service" not in opts:
        opts["llm_service"] = "marker.services.claude.ClaudeService"
    if claude_key:
        opts["claude_api_key"] = claude_key
    if claude_model:
        opts["claude_model_name"] = claude_model

    # Gemini is the marker default — just needs GOOGLE_API_KEY in env, no extra opts needed.

    return opts


def _resolve_model_dtype_argument() -> Any | None:
    """Return the dtype object (or string fallback) expected by Marker."""
    dtype_name = _resolve_model_dtype_override(os.environ.get("MARKER_MCP_MODEL_DTYPE"))
    if dtype_name is None:
        return None

    try:
        import torch
    except Exception:
        return dtype_name

    return getattr(torch, dtype_name, dtype_name)


def _create_models(device: str | None = None) -> dict:
    """Create the Marker model dictionary with any supported runtime overrides."""
    model_dtype = _resolve_model_dtype_argument()
    supported_kwargs: list[str] = []
    create_models_signature = None
    try:
        create_models_signature = signature(create_model_dict)
    except (TypeError, ValueError):
        create_models_signature = None

    base_kwargs: dict[str, Any] = {}
    if create_models_signature is not None:
        parameters = create_models_signature.parameters
        supports_kwargs = any(
            param.kind is Parameter.VAR_KEYWORD for param in parameters.values()
        )
        if device is not None and ("device" in parameters or supports_kwargs):
            base_kwargs["device"] = device
        if model_dtype is not None and ("model_dtype" in parameters or supports_kwargs):
            supported_kwargs.append("model_dtype")
        if model_dtype is not None and ("dtype" in parameters or supports_kwargs):
            supported_kwargs.append("dtype")
    else:
        if device is not None:
            base_kwargs["device"] = device
        if model_dtype is not None:
            supported_kwargs = ["model_dtype", "dtype"]

    if model_dtype is None:
        return create_model_dict(**base_kwargs)

    if not supported_kwargs:
        supported_kwargs = ["model_dtype", "dtype"]

    errors: list[str] = []
    for kwarg in dict.fromkeys(supported_kwargs):
        try:
            return create_model_dict(**{**base_kwargs, kwarg: model_dtype})
        except TypeError as exc:
            errors.append(f"{kwarg}: {exc}")

    joined_errors = "; ".join(errors) if errors else "no supported dtype keyword found"
    raise RuntimeError(
        "Configured MARKER_MCP_MODEL_DTYPE, but this Marker build does not accept a "
        f"dtype override ({joined_errors})."
    )


# ---------------------------------------------------------------------------
# Module-level model initialisation — fail fast with a clear error message.
# ---------------------------------------------------------------------------
_MODELS: Optional[dict] = None

try:
    _MODELS = _create_models()
    print("✅ Marker models loaded successfully.")
except Exception as _exc:
    print(f"❌ Failed to load Marker models: {_exc}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_converter(options: dict) -> PdfConverter:
    """Build a configured PdfConverter from an options dict."""
    config_parser = ConfigParser(options)
    config_dict = config_parser.generate_config_dict()
    gpu_memory_profile = _resolve_gpu_memory_profile(options.get("gpu_memory_profile"))
    config_dict.pop("gpu_memory_profile", None)
    if gpu_memory_profile is not None:
        for key, value in _GPU_MEMORY_PROFILE_CONFIGS[gpu_memory_profile].items():
            config_dict.setdefault(key, value)
    config_dict.setdefault("pdftext_workers", _DEFAULT_PDFTEXT_WORKERS)
    return PdfConverter(
        config=config_dict,
        artifact_dict=_MODELS,
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer(),
        llm_service=config_parser.get_llm_service(),
    )


def _ensure_models_available() -> None:
    """Raise a consistent error when Marker model initialisation failed."""
    if _MODELS is None:
        raise RuntimeError(
            "Marker models are not available. Check startup logs for initialisation errors."
        )


def _release_torch_memory() -> None:
    """Best-effort release of Python and CUDA allocations between retries."""
    gc.collect()
    try:
        import torch
    except Exception:
        return

    if hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def _close_image(image: Any) -> None:
    """Best-effort close for Pillow images created during tiling/rendering."""
    close = getattr(image, "close", None)
    if callable(close):
        close()


def _reload_models_for_device(device: str | None) -> None:
    """Recreate the Marker model bundle for a specific runtime device."""
    global _MODELS

    with _temporary_torch_device(device):
        _MODELS = None
        _release_torch_memory()
        _MODELS = _create_models(device=device)


def _looks_like_assets(payload: Any) -> bool:
    """Best-effort detection for structured artifact/path payloads."""
    if not isinstance(payload, dict) or not payload:
        return False

    asset_keys = {"artifacts", "assets", "attachments", "files", "images", "paths"}
    if asset_keys.intersection(payload):
        return True

    return any(
        isinstance(value, (dict, list, tuple)) and "path" in str(value).lower()
        for value in payload.values()
    )


def _is_pil_image(value: Any) -> bool:
    """Return True when the value is a Pillow image instance."""
    return PILImage is not None and isinstance(value, PILImage.Image)


def _image_format_for_filename(filename: str, image: Any) -> str:
    """Infer the most appropriate Pillow save format for an exported image."""
    format_by_suffix = {
        ".bmp": "BMP",
        ".gif": "GIF",
        ".jpeg": "JPEG",
        ".jpg": "JPEG",
        ".png": "PNG",
        ".tif": "TIFF",
        ".tiff": "TIFF",
        ".webp": "WEBP",
    }
    suffix = Path(filename).suffix.lower()
    if suffix in format_by_suffix:
        return format_by_suffix[suffix]

    image_format = getattr(image, "format", None)
    if isinstance(image_format, str) and image_format:
        return image_format.upper()

    return "PNG"


def _serialize_figure_asset(filename: str, image: Any) -> dict[str, str]:
    """Encode a Marker figure image as a JSON-serializable asset entry."""
    image_format = _image_format_for_filename(filename, image)
    image_to_save = image
    if image_format == "JPEG" and getattr(image, "mode", None) not in {"L", "RGB"}:
        image_to_save = image.convert("RGB")

    buffer = BytesIO()
    image_to_save.save(buffer, format=image_format)

    media_type = mimetypes.guess_type(filename)[0] or f"image/{image_format.lower()}"
    return {
        "filename": filename,
        "content_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
        "media_type": media_type,
    }


def _extract_figure_assets(payload: Any) -> tuple[Any, list[dict[str, str]]]:
    """Split top-level Pillow figure mappings from a rendered payload dict."""
    if not isinstance(payload, dict):
        return payload, []

    metadata: dict[str, Any] = {}
    image_assets: list[dict[str, str]] = []

    for key, value in payload.items():
        if isinstance(key, str) and _is_pil_image(value):
            image_assets.append(_serialize_figure_asset(key, value))
            continue
        metadata[key] = value

    return metadata, image_assets


def _json_safe_value(value: Any) -> Any:
    """Recursively normalize common structured-result values for JSON output."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, list):
        return [_json_safe_value(item) for item in value]
    return value


def _coerce_rendered_result(
    text: Any,
    extra_one: Any = None,
    extra_two: Any = None,
    warnings_list: list[str] | None = None,
) -> dict:
    """Convert Marker's rendered output into a stable MCP-facing result payload."""
    metadata: dict[str, Any] = {}
    assets: dict[str, Any] = {}

    for extra in (extra_one, extra_two):
        if extra in (None, "", {}, []):
            continue
        extra, figure_assets = _extract_figure_assets(extra)
        if figure_assets:
            assets.setdefault("images", []).extend(figure_assets)
        if _looks_like_assets(extra):
            if isinstance(extra, dict):
                assets.update(_json_safe_value(extra))
            continue
        if isinstance(extra, dict):
            metadata.update(_json_safe_value(extra))

    return {
        "text": str(text),
        "metadata": _json_safe_value(metadata),
        "warnings": list(warnings_list or []),
        "assets": _json_safe_value(assets),
    }


def _convert_sync(filepath: str, options: dict) -> str:
    """Synchronous conversion — always called via asyncio.to_thread()."""
    _ensure_models_available()
    converter = _build_converter(options)
    rendered = converter(filepath)
    text, _, _ = text_from_rendered(rendered)
    return str(text)


def _convert_sync_result(filepath: str, options: dict) -> dict:
    """Synchronous structured conversion — always called via asyncio.to_thread()."""
    _ensure_models_available()
    converter = _build_converter(options)
    rendered = converter(filepath)
    rendered_output = text_from_rendered(rendered)

    if isinstance(rendered_output, tuple):
        text = rendered_output[0]
        extra_one = rendered_output[1] if len(rendered_output) > 1 else None
        extra_two = rendered_output[2] if len(rendered_output) > 2 else None
    else:
        text = rendered_output
        extra_one = None
        extra_two = None

    return _coerce_rendered_result(text, extra_one, extra_two)


def _is_cuda_oom_error(exc: Exception) -> bool:
    """Return True when an exception looks like a CUDA/ROCm out-of-memory failure."""
    message = str(exc).lower()
    name = type(exc).__name__.lower()
    return (
        "cuda out of memory" in message
        or "hip out of memory" in message
        or "outofmemory" in name
    )


def _is_llm_failure(exc: Exception) -> bool:
    """Return True when an exception looks attributable to the optional LLM path."""
    message = str(exc).lower()
    llm_markers = (
        "llm",
        "ollama",
        "openai",
        "anthropic",
        "claude",
        "gemini",
        "google",
    )
    return any(marker in message for marker in llm_markers)


def _without_llm_options(options: dict) -> dict:
    """Strip LLM-specific flags for a controlled non-LLM retry."""
    filtered = dict(options)
    filtered.pop("use_llm", None)

    for key in list(filtered):
        if key == "llm_service" or key.startswith(("ollama_", "openai_", "claude_", "gemini_")):
            filtered.pop(key, None)

    return filtered


@contextlib.contextmanager
def _temporary_torch_device(device: str | None):
    """Temporarily override TORCH_DEVICE for a single conversion attempt."""
    original = os.environ.get("TORCH_DEVICE")
    had_original = "TORCH_DEVICE" in os.environ

    if device is None:
        os.environ.pop("TORCH_DEVICE", None)
    else:
        os.environ["TORCH_DEVICE"] = device

    try:
        yield
    finally:
        if had_original:
            os.environ["TORCH_DEVICE"] = original if original is not None else ""
            if original is None:
                os.environ.pop("TORCH_DEVICE", None)
        else:
            os.environ.pop("TORCH_DEVICE", None)


async def _run_conversion_with_fallbacks(
    filepath: str,
    options: dict,
    sync_converter,
    warnings_list: list[str] | None = None,
    allow_cpu_fallback: bool = True,
):
    """Run a conversion attempt with explicit CUDA-OOM and optional-LLM fallbacks."""
    opts = dict(options)
    cpu_fallback_enabled = False
    cpu_fallback_used = False
    llm_fallback_used = False
    original_torch_device = os.environ.get("TORCH_DEVICE")
    had_original_torch_device = "TORCH_DEVICE" in os.environ

    try:
        while True:
            context_manager = (
                _temporary_torch_device("cpu") if cpu_fallback_enabled else contextlib.nullcontext()
            )
            with context_manager:
                try:
                    return await asyncio.to_thread(sync_converter, filepath, opts)
                except Exception as exc:
                    if allow_cpu_fallback and not cpu_fallback_enabled and _is_cuda_oom_error(exc):
                        cpu_fallback_enabled = True
                        cpu_fallback_used = True
                        warning_message = (
                            "CUDA out-of-memory during Marker conversion; retrying this step on CPU."
                        )
                        if warnings_list is not None:
                            warnings_list.append(warning_message)
                        warnings.warn(warning_message, UserWarning, stacklevel=2)
                        _reload_models_for_device("cpu")
                        continue

                    if opts.get("use_llm") and not llm_fallback_used and _is_llm_failure(exc):
                        llm_fallback_used = True
                        warning_message = (
                            "LLM-assisted Marker conversion failed; retrying without LLM enhancements."
                        )
                        if warnings_list is not None:
                            warnings_list.append(warning_message)
                        warnings.warn(warning_message, UserWarning, stacklevel=2)
                        opts = _without_llm_options(opts)
                        continue

                    raise
                finally:
                    _release_torch_memory()
    finally:
        if cpu_fallback_used and (had_original_torch_device or original_torch_device is None):
            _reload_models_for_device(original_torch_device)


def _is_single_page_pdf_step(filepath: str, options: dict) -> bool:
    """Return True when the requested conversion step targets exactly one PDF page."""
    if Path(filepath).suffix.lower() != ".pdf":
        return False
    page_range = options.get("page_range")
    if not page_range:
        return False
    return len(_parse_page_range(page_range)) == 1


def _gpu_oom_tile_height_attempts(initial_height: int = _DEFAULT_GPU_OOM_TILE_HEIGHT_PX) -> list[int]:
    """Return descending GPU tile heights to try before falling back to CPU."""
    attempts: list[int] = []
    height = initial_height
    while height >= 400:
        attempts.append(height)
        if height == 400:
            break
        height = max(400, height // 2)
    return attempts


def _gpu_memory_profile_retry_attempts(options: dict) -> list[str]:
    """Return GPU-memory profiles to try before page tiling or CPU fallback."""
    current_profile = _resolve_gpu_memory_profile(options.get("gpu_memory_profile"))
    return [
        profile
        for profile in ("low-vram",)
        if profile != current_profile
    ]


async def _run_conversion_with_gpu_memory_profile_recovery(
    filepath: str,
    options: dict,
    sync_converter,
    warnings_list: list[str] | None = None,
    adaptive_options: dict | None = None,
):
    """Retry a failed GPU step with smaller Marker batch sizes before page tiling."""
    for profile in _gpu_memory_profile_retry_attempts(options):
        warning_message = (
            "CUDA out-of-memory during single-page conversion; retrying with GPU memory "
            f"profile '{profile}' before page tiling."
        )
        if warnings_list is not None:
            warnings_list.append(warning_message)
        warnings.warn(warning_message, UserWarning, stacklevel=2)

        profiled_options = {**options, "gpu_memory_profile": profile}
        try:
            result = await _run_conversion_with_fallbacks(
                filepath,
                profiled_options,
                sync_converter,
                warnings_list=warnings_list,
                allow_cpu_fallback=False,
            )
        except Exception as exc:
            if not _is_cuda_oom_error(exc):
                raise
            continue

        if adaptive_options is not None:
            adaptive_options["gpu_memory_profile"] = profile
        return result

    return None


async def _run_conversion_with_gpu_page_tiling_recovery(
    filepath: str,
    options: dict,
    sync_converter,
    warnings_list: list[str] | None = None,
    adaptive_options: dict | None = None,
):
    """Prefer GPU page tiling over CPU fallback for single-page PDF OOMs."""
    if options.get("max_page_height_px") is not None or not _is_single_page_pdf_step(filepath, options):
        return await _run_conversion_with_fallbacks(
            filepath,
            options,
            sync_converter,
            warnings_list=warnings_list,
        )

    try:
        return await _run_conversion_with_fallbacks(
            filepath,
            options,
            sync_converter,
            warnings_list=warnings_list,
            allow_cpu_fallback=False,
        )
    except Exception as exc:
        if not _is_cuda_oom_error(exc):
            raise

    profiled_result = await _run_conversion_with_gpu_memory_profile_recovery(
        filepath,
        options,
        sync_converter,
        warnings_list=warnings_list,
        adaptive_options=adaptive_options,
    )
    if profiled_result is not None:
        return profiled_result

    for tile_height in _gpu_oom_tile_height_attempts():
        warning_message = (
            "CUDA out-of-memory during single-page conversion; retrying with GPU page tiling "
            f"({tile_height}px) before CPU fallback."
        )
        if warnings_list is not None:
            warnings_list.append(warning_message)
        warnings.warn(warning_message, UserWarning, stacklevel=2)

        tiled_options = {**options, "max_page_height_px": tile_height}
        try:
            tiled_result = await _convert_file_with_page_tiling(
                filepath,
                tiled_options,
                _convert_sync_result,
                warnings_list=warnings_list,
                allow_cpu_fallback=False,
            )
            if sync_converter is _convert_sync:
                return str(tiled_result["text"])
            return tiled_result
        except Exception as tiled_exc:
            if not _is_cuda_oom_error(tiled_exc):
                raise

    fallback_warning = (
        "GPU page tiling also ran out of memory at all configured tile sizes; retrying the step on CPU."
    )
    if warnings_list is not None:
        warnings_list.append(fallback_warning)
    warnings.warn(fallback_warning, UserWarning, stacklevel=2)
    return await _run_conversion_with_fallbacks(
        filepath,
        options,
        sync_converter,
        warnings_list=warnings_list,
        allow_cpu_fallback=True,
    )


def _chunk_page_ranges(total_pages: int, max_pages_per_chunk: int) -> list[str]:
    """Split a page count into Marker page_range chunks."""
    if total_pages <= 0:
        return []
    if max_pages_per_chunk <= 0:
        raise ValueError("max_pages_per_chunk must be greater than zero.")

    ranges: list[str] = []
    for start in range(0, total_pages, max_pages_per_chunk):
        end = min(start + max_pages_per_chunk - 1, total_pages - 1)
        ranges.append(f"{start}-{end}")
    return ranges


def _requested_page_numbers(filepath: str, options: dict) -> list[int]:
    """Return the ordered page numbers requested for a PDF conversion."""
    page_range = options.get("page_range")
    if page_range:
        return _parse_page_range(page_range)
    return list(range(_get_pdf_page_count(filepath)))


def _parse_page_range(page_range: str) -> list[int]:
    """Expand a Marker-style page_range string into ordered page numbers."""
    pages: list[int] = []
    for raw_part in page_range.split(","):
        part = raw_part.strip()
        if not part:
            continue

        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text.strip())
            end = int(end_text.strip())
            if start < 0 or end < 0:
                raise ValueError("page_range values must be non-negative.")
            if end < start:
                raise ValueError("page_range end must be greater than or equal to start.")
            pages.extend(range(start, end + 1))
            continue

        page = int(part)
        if page < 0:
            raise ValueError("page_range values must be non-negative.")
        pages.append(page)

    return pages


def _page_numbers_to_page_range(pages: list[int]) -> str:
    """Compact an ordered page list back into Marker page_range syntax."""
    if not pages:
        return ""

    segments: list[str] = []
    start = previous = pages[0]

    for page in pages[1:]:
        if page == previous + 1:
            previous = page
            continue

        segments.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = page

    segments.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(segments)


def _chunk_requested_page_range(page_range: str, max_pages_per_chunk: int) -> list[str]:
    """Split an explicit page_range into smaller Marker page_range chunks."""
    if max_pages_per_chunk <= 0:
        raise ValueError("max_pages_per_chunk must be greater than zero.")

    pages = _parse_page_range(page_range)
    if not pages:
        return []

    return [
        _page_numbers_to_page_range(pages[start : start + max_pages_per_chunk])
        for start in range(0, len(pages), max_pages_per_chunk)
    ]


def _merge_chunk_texts(chunk_texts: list[str]) -> str:
    """Merge ordered chunk outputs into the same text shape as a single conversion."""
    return "\n\n".join(text for text in chunk_texts if text)


def _tile_vertical_bounds(
    height: int,
    max_tile_height: int,
    overlap_px: int = _DEFAULT_PAGE_TILE_OVERLAP_PX,
) -> list[tuple[int, int]]:
    """Split an image height into overlapping vertical strips."""
    if max_tile_height <= 0:
        raise ValueError("max_page_height_px must be greater than zero.")
    if overlap_px < 0:
        raise ValueError("page tile overlap must be non-negative.")
    if overlap_px >= max_tile_height:
        raise ValueError("page tile overlap must be smaller than max_page_height_px.")
    if height <= 0:
        return []
    if height <= max_tile_height:
        return [(0, height)]

    bounds: list[tuple[int, int]] = []
    top = 0
    while top < height:
        bottom = min(top + max_tile_height, height)
        bounds.append((top, bottom))
        if bottom >= height:
            break
        top = bottom - overlap_px
    return bounds


def _tile_page_image(
    image: Any,
    max_tile_height: int,
    overlap_px: int | None = None,
) -> list[Any]:
    """Split a page image into overlapping full-width vertical tiles."""
    if PILImage is None or not isinstance(image, PILImage.Image):
        raise TypeError("page tiling requires Pillow image inputs.")

    effective_overlap = (
        _page_tile_overlap_px(max_tile_height)
        if overlap_px is None
        else overlap_px
    )

    return [
        image.crop((0, top, image.width, bottom))
        for top, bottom in _tile_vertical_bounds(image.height, max_tile_height, effective_overlap)
    ]


def _normalized_nonempty_lines(text: str) -> list[str]:
    """Normalize non-empty lines for overlap detection."""
    normalized_lines: list[str] = []
    for line in text.splitlines():
        normalized = " ".join(line.split())
        if normalized:
            normalized_lines.append(normalized)
    return normalized_lines


def _trim_overlapping_line_prefix(previous_text: str, current_text: str, max_lines: int = 12) -> str:
    """Trim a duplicated leading line prefix caused by tile overlap."""
    previous_lines = _normalized_nonempty_lines(previous_text)
    current_lines = _normalized_nonempty_lines(current_text)
    if not previous_lines or not current_lines:
        return current_text

    overlap_size = 0
    max_overlap = min(len(previous_lines), len(current_lines), max_lines)
    for size in range(max_overlap, 0, -1):
        if previous_lines[-size:] == current_lines[:size]:
            overlap_size = size
            break

    if overlap_size == 0:
        return current_text

    lines = current_text.splitlines()
    consumed = 0
    trim_index = 0
    while trim_index < len(lines) and consumed < overlap_size:
        if " ".join(lines[trim_index].split()):
            consumed += 1
        trim_index += 1

    return "\n".join(lines[trim_index:]).lstrip("\n")


def _page_tile_overlap_px(max_tile_height: int) -> int:
    """Choose a smaller overlap for smaller tiles to reduce duplicate OCR regions."""
    if max_tile_height <= 0:
        raise ValueError("max_page_height_px must be greater than zero.")

    overlap_px = min(_DEFAULT_PAGE_TILE_OVERLAP_PX, max_tile_height // 6)
    overlap_px = max(_MIN_PAGE_TILE_OVERLAP_PX, overlap_px)
    return min(overlap_px, max_tile_height - 1)


def _page_separator(page_number: int) -> str:
    """Return the Marker-style page separator used by paginated markdown output."""
    return f"{{{page_number}}}{'-' * _PAGE_SEPARATOR_DASH_COUNT}"


def _strip_tile_page_separators(text: str) -> str:
    """Drop per-tile pagination markers before merging tiles back into full pages."""
    cleaned_lines = [
        line
        for line in str(text).splitlines()
        if not re.fullmatch(r"\{\d+\}-+", line.strip())
    ]
    return "\n".join(cleaned_lines).strip()


def _split_markdown_blocks(text: str) -> list[str]:
    """Split markdown into paragraph-like blocks separated by blank lines."""
    blocks: list[str] = []
    current: list[str] = []

    for line in text.splitlines():
        if line.strip():
            current.append(line.rstrip())
            continue
        if current:
            blocks.append("\n".join(current).strip())
            current = []

    if current:
        blocks.append("\n".join(current).strip())
    return blocks


def _normalize_tiled_block(text: str) -> str:
    """Normalize a markdown block for duplicate detection after tiling."""
    normalized = str(text)
    normalized = re.sub(r"!\[[^\]]*\]\([^)]+\)", "image", normalized)
    normalized = re.sub(r"<[^>]+>", "", normalized)
    normalized = normalized.replace("\\*", "*")
    normalized = re.sub(r"[#*_`|>-]+", " ", normalized)
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip().lower()


def _looks_like_repeated_phrase_gibberish(text: str) -> bool:
    """Detect obvious OCR loops where the same short phrase repeats excessively."""
    words = re.findall(r"\b[\w'-]+\b", text.lower())
    if len(words) < 12:
        return False

    for phrase_len in (2, 3, 4):
        if len(words) < phrase_len * 4:
            continue
        phrase = words[:phrase_len]
        repetitions = 0
        position = 0
        while position + phrase_len <= len(words) and words[position : position + phrase_len] == phrase:
            repetitions += 1
            position += phrase_len
        if repetitions >= 4 and position >= int(len(words) * 0.75):
            return True
    return False


def _tiled_block_quality_score(text: str) -> tuple[int, int]:
    """Prefer fuller, cleaner OCR blocks when overlaps produce near-duplicates."""
    normalized = _normalize_tiled_block(text)
    alpha_words = re.findall(r"[A-Za-z]+", text)
    vowel_words = sum(any(char in "aeiouAEIOU" for char in word) for word in alpha_words)
    penalties = len(re.findall(r"\\\*\d+", text))
    penalties += len(re.findall(r"\b\d{3,}\b", text))
    penalties += len(re.findall(r"([A-Za-z]{2,})(?: \1){3,}", text.lower()))
    return (vowel_words - (penalties * 3), len(normalized))


def _blocks_should_deduplicate(previous_block: str, current_block: str) -> bool:
    """Return True when two nearby tiled blocks are likely duplicate overlap content."""
    previous_normalized = _normalize_tiled_block(previous_block)
    current_normalized = _normalize_tiled_block(current_block)
    if not previous_normalized or not current_normalized:
        return False
    if previous_normalized == current_normalized:
        return True

    minimum_length = min(len(previous_normalized), len(current_normalized))
    if minimum_length >= 40 and (
        previous_normalized in current_normalized
        or current_normalized in previous_normalized
    ):
        return True

    if minimum_length < 24:
        return False

    return SequenceMatcher(None, previous_normalized, current_normalized).ratio() >= 0.88


def _cleanup_tiled_page_text(text: str) -> str:
    """Remove tile-only artifacts and replace weaker overlap fragments with better blocks."""
    blocks = _split_markdown_blocks(_strip_tile_page_separators(text))
    cleaned_blocks: list[str] = []
    seen_image_blocks: set[str] = set()

    for block in blocks:
        if not block:
            continue
        if _looks_like_repeated_phrase_gibberish(block):
            continue
        if block.startswith("!["):
            if block in seen_image_blocks:
                continue
            seen_image_blocks.add(block)

        deduplicated = False
        for index in range(len(cleaned_blocks) - 1, max(-1, len(cleaned_blocks) - 5), -1):
            previous_block = cleaned_blocks[index]
            if not _blocks_should_deduplicate(previous_block, block):
                continue

            if _tiled_block_quality_score(block) > _tiled_block_quality_score(previous_block):
                cleaned_blocks[index] = block
            deduplicated = True
            break

        if not deduplicated:
            cleaned_blocks.append(block)

    return "\n\n".join(cleaned_blocks).strip()


def _merge_tiled_texts(chunk_texts: list[str]) -> str:
    """Merge tile text in order while trimming duplicated overlap lines."""
    merged_text = ""
    for text in chunk_texts:
        cleaned_text = _strip_tile_page_separators(str(text))
        if not cleaned_text:
            continue
        if not merged_text:
            merged_text = cleaned_text
            continue

        trimmed = _trim_overlapping_line_prefix(merged_text, cleaned_text)
        if trimmed:
            merged_text = f"{merged_text.rstrip()}\n\n{trimmed.lstrip()}".rstrip()
    return _cleanup_tiled_page_text(merged_text)


def _should_chunk_conversion(filepath: str, options: dict) -> bool:
    """Return True when option-driven PDF chunking should be used."""
    max_pages_per_chunk = options.get("max_pages_per_chunk")
    return max_pages_per_chunk is not None and Path(filepath).suffix.lower() == ".pdf"


def _should_tile_pages(filepath: str, options: dict) -> bool:
    """Return True when page-image tiling should be used for a PDF."""
    return options.get("max_page_height_px") is not None and Path(filepath).suffix.lower() == ".pdf"


def _get_pdf_page_count(filepath: str) -> int:
    """Return the number of pages in a PDF."""
    from pypdf import PdfReader

    return len(PdfReader(filepath).pages)


def _options_without_chunking(options: dict) -> dict:
    """Strip chunk-only orchestration options before calling Marker."""
    filtered = dict(options)
    filtered.pop("max_pages_per_chunk", None)
    return filtered


def _options_without_page_tiling(options: dict) -> dict:
    """Strip page-tiling orchestration options before calling Marker."""
    filtered = dict(options)
    filtered.pop("max_page_height_px", None)
    return filtered


def _render_pdf_page_image(filepath: str, page_number: int, dpi: int = _PAGE_TILE_RENDER_DPI) -> Any:
    """Render a single PDF page to a Pillow image for tiled conversion."""
    import pypdfium2 as pdfium
    from marker.providers.pdf import PdfProvider

    doc = None
    try:
        doc = pdfium.PdfDocument(filepath)
        doc.init_forms()
        return PdfProvider._render_image(doc, page_number, dpi, flatten_page=True)
    finally:
        if doc is not None:
            doc.close()


def _merge_structured_results(chunk_results: list[dict], text_merger=_merge_chunk_texts) -> dict:
    """Merge structured conversion payloads using a caller-provided text merger."""
    metadata: dict[str, Any] = {}
    merged_assets: dict[str, Any] = {}
    merged_warnings: list[str] = []
    total_page_count = 0
    saw_page_count = False

    for chunk_result in chunk_results:
        chunk_metadata = dict(chunk_result.get("metadata") or {})
        chunk_page_count = chunk_metadata.pop("page_count", None)
        if isinstance(chunk_page_count, int):
            total_page_count += chunk_page_count
            saw_page_count = True

        metadata.update(chunk_metadata)
        merged_warnings.extend(chunk_result.get("warnings") or [])

        for key, value in (chunk_result.get("assets") or {}).items():
            if key not in merged_assets:
                if isinstance(value, list):
                    merged_assets[key] = list(value)
                elif isinstance(value, dict):
                    merged_assets[key] = dict(value)
                else:
                    merged_assets[key] = value
                continue

            existing = merged_assets[key]
            if isinstance(existing, list) and isinstance(value, list):
                existing.extend(value)
            elif isinstance(existing, dict) and isinstance(value, dict):
                existing.update(value)
            else:
                merged_assets[key] = value

    if saw_page_count:
        metadata["page_count"] = total_page_count

    return {
        "text": text_merger([str(chunk_result.get("text", "")) for chunk_result in chunk_results]),
        "metadata": metadata,
        "warnings": merged_warnings,
        "assets": merged_assets,
    }


async def _convert_file_in_chunks(
    filepath: str,
    options: dict,
    sync_converter,
    warnings_list: list[str] | None = None,
) -> list[Any]:
    """Run sequential chunked conversions for large PDFs."""
    base_options = _options_without_chunking(options)
    max_pages_per_chunk = options["max_pages_per_chunk"]
    requested_page_range = base_options.get("page_range")

    if requested_page_range:
        chunk_ranges = _chunk_requested_page_range(requested_page_range, max_pages_per_chunk)
        if len(chunk_ranges) <= 1:
            return [
                await _run_conversion_with_gpu_page_tiling_recovery(
                    filepath,
                    base_options,
                    sync_converter,
                    warnings_list=warnings_list,
                    adaptive_options=base_options,
                )
            ]

        chunk_results: list[Any] = []
        for page_range in chunk_ranges:
            chunk_options = {**base_options, "page_range": page_range}
            try:
                chunk_results.append(
                    await _run_conversion_with_gpu_page_tiling_recovery(
                        filepath,
                        chunk_options,
                        sync_converter,
                        warnings_list=warnings_list,
                        adaptive_options=base_options,
                    )
                )
            finally:
                _release_torch_memory()
        return chunk_results

    total_pages = _get_pdf_page_count(filepath)
    if total_pages <= max_pages_per_chunk:
        return [
                await _run_conversion_with_gpu_page_tiling_recovery(
                    filepath,
                    base_options,
                    sync_converter,
                    warnings_list=warnings_list,
                    adaptive_options=base_options,
                )
            ]

    chunk_results: list[Any] = []
    for page_range in _chunk_page_ranges(total_pages, max_pages_per_chunk):
        chunk_options = {**base_options, "page_range": page_range}
        try:
            chunk_results.append(
                await _run_conversion_with_gpu_page_tiling_recovery(
                    filepath,
                    chunk_options,
                    sync_converter,
                    warnings_list=warnings_list,
                    adaptive_options=base_options,
                )
            )
        finally:
            _release_torch_memory()
    return chunk_results


def _merge_chunk_results(chunk_results: list[dict]) -> dict:
    """Merge structured chunk payloads into a single structured result."""
    return _merge_structured_results(chunk_results, text_merger=_merge_chunk_texts)


async def _convert_file_with_page_tiling(
    filepath: str,
    options: dict,
    sync_converter,
    warnings_list: list[str] | None = None,
    allow_cpu_fallback: bool = True,
) -> dict:
    """Convert a PDF by rasterizing requested pages into smaller image tiles."""
    base_options = _options_without_page_tiling(_options_without_chunking(options))
    paginate_output = bool(base_options.pop("paginate_output", True))
    max_page_height_px = options["max_page_height_px"]
    requested_pages = _requested_page_numbers(filepath, options)
    base_options.pop("page_range", None)

    warning_message = (
        "Experimental page tiling enabled: PDF pages are rasterized into smaller image strips, "
        "which can reduce peak memory but may alter reading order and OCR quality."
    )
    if warnings_list is not None and warning_message not in warnings_list:
        warnings_list.append(warning_message)

    page_results: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="marker-page-tiles-") as temp_dir:
        temp_dir_path = Path(temp_dir)

        for page_number in requested_pages:
            page_image = None
            tile_images: list[Any] = []
            tile_results: list[dict] = []
            try:
                page_image = _render_pdf_page_image(filepath, page_number, dpi=_PAGE_TILE_RENDER_DPI)
                tile_images = _tile_page_image(page_image, max_page_height_px)

                for tile_index, tile_image in enumerate(tile_images):
                    tile_path = temp_dir_path / f"page_{page_number:04d}_tile_{tile_index:03d}.png"
                    try:
                        tile_image.save(tile_path, format="PNG")
                        tile_results.append(
                            await _run_conversion_with_fallbacks(
                                str(tile_path),
                                base_options,
                                sync_converter,
                                warnings_list=warnings_list,
                                allow_cpu_fallback=allow_cpu_fallback,
                            )
                        )
                    finally:
                        _close_image(tile_image)
                        tile_path.unlink(missing_ok=True)
                        _release_torch_memory()

                page_result = _merge_structured_results(tile_results, text_merger=_merge_tiled_texts)
                page_result.setdefault("metadata", {})
                page_result["metadata"]["page_count"] = 1
                page_text = _strip_tile_page_separators(str(page_result.get("text", "")))
                if page_text and paginate_output:
                    page_text = f"{_page_separator(page_number)}\n\n{page_text}".rstrip()
                page_result["text"] = page_text
                page_results.append(page_result)
            finally:
                if page_image is not None:
                    _close_image(page_image)
                _release_torch_memory()

    return _merge_chunk_results(page_results)


# ---------------------------------------------------------------------------
# Public async API
# ---------------------------------------------------------------------------

async def convert_file(filepath: str, options: Optional[dict] = None) -> str:
    """Convert a document at the given file path and return the converted text."""
    opts = {"output_format": "markdown", **(options or {})}
    if _should_tile_pages(filepath, opts):
        result = await _convert_file_with_page_tiling(
            filepath,
            opts,
            _convert_sync_result,
        )
        return str(result["text"])

    if _should_chunk_conversion(filepath, opts):
        chunk_results = await _convert_file_in_chunks(filepath, opts, _convert_sync)
        return _merge_chunk_texts([str(chunk_result) for chunk_result in chunk_results])

    return await _run_conversion_with_gpu_page_tiling_recovery(
        filepath,
        _options_without_page_tiling(_options_without_chunking(opts)),
        _convert_sync,
    )


async def convert_file_result(filepath: str, options: Optional[dict] = None) -> dict:
    """Convert a document at the given file path and return a structured result."""
    opts = {"output_format": "markdown", **(options or {})}
    fallback_warnings: list[str] = []
    if _should_tile_pages(filepath, opts):
        result = await _convert_file_with_page_tiling(
            filepath,
            opts,
            _convert_sync_result,
            warnings_list=fallback_warnings,
        )
    elif _should_chunk_conversion(filepath, opts):
        chunk_results = await _convert_file_in_chunks(
            filepath,
            opts,
            _convert_sync_result,
            warnings_list=fallback_warnings,
        )
        result = _merge_chunk_results(chunk_results)
    else:
        result = await _run_conversion_with_gpu_page_tiling_recovery(
            filepath,
            _options_without_page_tiling(_options_without_chunking(opts)),
            _convert_sync_result,
            warnings_list=fallback_warnings,
        )

    if fallback_warnings:
        result["warnings"] = [*fallback_warnings, *result.get("warnings", [])]

    return result


async def convert_bytes(content: bytes, filename: str, options: Optional[dict] = None) -> str:
    """Convert a document from raw bytes.

    Writes the bytes to a temporary file, converts it, then cleans up.
    """
    suffix = Path(filename).suffix or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    try:
        try:
            return await convert_file(tmp_path, options)
        except Exception as exc:
            if not _is_cuda_oom_error(exc):
                raise
            return await convert_file(tmp_path, options)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


async def convert_bytes_result(
    content: bytes,
    filename: str,
    options: Optional[dict] = None,
) -> dict:
    """Convert raw bytes and return a structured result payload."""
    suffix = Path(filename).suffix or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    try:
        return await convert_file_result(tmp_path, options)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


async def convert_bytes_batch(files: list[dict]) -> list[dict]:
    """Batch-convert multiple documents.

    Each entry must contain:
        - filename (str): original filename including extension
        - content_base64 (str): base64-encoded file bytes
        - options (dict, optional): per-file conversion options

    Returns a list of result dicts: {filename, success, content, error}.
    """
    results: list[dict] = []
    for file in files:
        filename = file.get("filename", "document.pdf")
        try:
            content = base64.b64decode(file.get("content_base64", ""))
            text = await convert_bytes(content, filename, file.get("options"))
            results.append({"filename": filename, "success": True, "content": text, "error": None})
        except Exception as exc:
            results.append({"filename": filename, "success": False, "content": None, "error": str(exc)})
    return results


def get_status() -> dict:
    """Return the converter readiness status (synchronous, safe to call from anywhere)."""
    return {
        "initialized": _MODELS is not None,
        "status": "ready" if _MODELS is not None else "failed",
        "message": (
            "Marker models loaded and ready."
            if _MODELS is not None
            else "Model initialisation failed — check startup logs."
        ),
    }
