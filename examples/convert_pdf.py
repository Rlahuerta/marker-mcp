import base64
import argparse
import asyncio
import os
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert a PDF document to Markdown with marker-mcp.")
    parser.add_argument("input_pdf", type=Path, help="Path to the input PDF document.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Path to the output Markdown file. Defaults to <input>.md",
    )
    parser.add_argument(
        "--page-range",
        default=None,
        help='Optional page range, for example "0-3" or "0,2,4-6".',
    )
    parser.add_argument(
        "--max-pages-per-chunk",
        type=int,
        default=4,
        help="Sequential PDF chunk size used to reduce peak memory pressure.",
    )
    parser.add_argument(
        "--max-page-height-px",
        type=int,
        default=None,
        help="Experimental rasterized page-strip height used to split large PDF pages vertically.",
    )
    parser.add_argument(
        "--gpu-memory-profile",
        choices=["low-vram"],
        default=None,
        help="Optional GPU memory tuning profile that lowers Marker batch sizes while staying on GPU.",
    )
    parser.add_argument(
        "--ocr-device",
        choices=["auto", "cpu", "cuda", "nvidia", "amd", "rocm", "mps"],
        default=None,
        help="Optional OCR runtime override. Use cpu when VRAM is limited.",
    )
    parser.add_argument(
        "--model-dtype",
        choices=["float32", "float16", "bfloat16"],
        default=None,
        help="Experimental Marker model dtype override.",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Enable Marker LLM enhancement using the environment-configured service.",
    )
    parser.add_argument(
        "--paginate-output",
        action="store_true",
        default=True,
        help="Insert page separators in the Markdown output.",
    )
    parser.add_argument(
        "--no-paginate-output",
        action="store_false",
        dest="paginate_output",
        help="Disable page separators in the Markdown output.",
    )
    return parser


def resolve_output_path(input_pdf: Path, output: Path | None) -> Path:
    return output if output is not None else input_pdf.with_suffix(".md")


def build_options(args: argparse.Namespace) -> dict:
    options: dict = {
        "output_format": "markdown",
        "paginate_output": args.paginate_output,
    }
    if args.page_range:
        options["page_range"] = args.page_range
    if args.max_pages_per_chunk is not None:
        options["max_pages_per_chunk"] = args.max_pages_per_chunk
    if args.max_page_height_px is not None:
        options["max_page_height_px"] = args.max_page_height_px
    if args.gpu_memory_profile is not None:
        options["gpu_memory_profile"] = args.gpu_memory_profile
    if args.use_llm:
        options["use_llm"] = True
    return options


def _resolve_asset_path(output_dir: Path, filename: str) -> Path:
    if not filename:
        raise ValueError("Asset filename cannot be empty.")

    base_dir = output_dir.resolve()
    asset_path = (base_dir / filename).resolve()
    if asset_path != base_dir and base_dir not in asset_path.parents:
        raise ValueError(f"Refusing to write asset outside output directory: {filename}")

    asset_path.parent.mkdir(parents=True, exist_ok=True)
    return asset_path


def save_assets(output_dir: Path, assets: dict) -> list[Path]:
    saved_paths: list[Path] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for group_assets in assets.values():
        if not isinstance(group_assets, list):
            continue
        for asset in group_assets:
            if not isinstance(asset, dict):
                continue

            filename = asset.get("filename")
            content_base64 = asset.get("content_base64")
            if not filename or not content_base64:
                continue

            asset_path = _resolve_asset_path(output_dir, str(filename))
            asset_path.write_bytes(base64.b64decode(content_base64))
            saved_paths.append(asset_path)

    return saved_paths


async def convert_pdf(args: argparse.Namespace) -> Path:
    if args.ocr_device is not None:
        os.environ["MARKER_MCP_OCR_DEVICE"] = args.ocr_device
    if args.model_dtype is not None:
        os.environ["MARKER_MCP_MODEL_DTYPE"] = args.model_dtype

    from marker_mcp.conversion_service import convert_file_result

    input_pdf = args.input_pdf.resolve()
    output_md = resolve_output_path(input_pdf, args.output)
    result = await convert_file_result(str(input_pdf), build_options(args))

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(result["text"], encoding="utf-8")
    print(f"Wrote {output_md}")

    saved_assets = save_assets(output_md.parent, result["assets"])
    if saved_assets:
        print("Saved assets:")
        for asset_path in saved_assets:
            print(f"  - {asset_path}")

    if result["warnings"]:
        print("Warnings:")
        for warning in result["warnings"]:
            print(f"  - {warning}")

    if result["assets"] and not saved_assets:
        print("Assets:")
        print(result["assets"])

    return output_md


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    asyncio.run(convert_pdf(args))


if __name__ == "__main__":
    #  python convert_pdf.py risks-13-00247-v2.pdf --page-range 0-5 --ocr-device "cuda" --model-dtype "float16" --max-pages-per-chunk 1 -o risks-13-00247-v2.md
    #  python convert_pdf.py risks-13-00247-v2.pdf --page-range 0-10 --ocr-device "cuda" --model-dtype "float16" --gpu-memory-profile low-vram --max-pages-per-chunk 1 -o risks-13-00247-v2.md
    #  python convert_pdf.py risks-13-00247-v2.pdf --page-range 0-10 --ocr-device "cuda" --max-page-height-px 300 -o risks-13-00247-v2.md

    main()
