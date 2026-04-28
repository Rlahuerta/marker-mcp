import argparse
import base64
import importlib.util
from pathlib import Path
from unittest.mock import AsyncMock, patch


def _load_convert_pdf_module():
    example_path = Path(__file__).resolve().parent.parent / "examples" / "convert_pdf.py"
    spec = importlib.util.spec_from_file_location("examples.convert_pdf", example_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_convert_pdf_example_defaults_output_to_input_stem():
    module = _load_convert_pdf_module()
    input_pdf = Path("/tmp/example-paper.pdf")

    assert module.resolve_output_path(input_pdf, None) == Path("/tmp/example-paper.md")


def test_convert_pdf_example_builds_chunking_options():
    module = _load_convert_pdf_module()
    args = argparse.Namespace(
        paginate_output=True,
        page_range="0-2",
        max_pages_per_chunk=4,
        max_page_height_px=1600,
        gpu_memory_profile="low-vram",
        use_llm=True,
    )

    assert module.build_options(args) == {
        "output_format": "markdown",
        "paginate_output": True,
        "page_range": "0-2",
        "max_pages_per_chunk": 4,
        "max_page_height_px": 1600,
        "gpu_memory_profile": "low-vram",
        "use_llm": True,
    }


async def test_convert_pdf_example_writes_output_file(tmp_path):
    module = _load_convert_pdf_module()
    input_pdf = tmp_path / "input.pdf"
    input_pdf.write_bytes(b"%PDF-1.4\n")
    output_md = tmp_path / "output.md"
    args = argparse.Namespace(
        input_pdf=input_pdf,
        output=output_md,
        page_range=None,
        max_pages_per_chunk=2,
        max_page_height_px=None,
        gpu_memory_profile=None,
        ocr_device="cpu",
        model_dtype="bfloat16",
        use_llm=False,
        paginate_output=True,
    )

    with patch(
        "marker_mcp.conversion_service.convert_file_result",
        new=AsyncMock(return_value={"text": "# Converted", "warnings": [], "assets": {}}),
    ) as mock_convert:
        written = await module.convert_pdf(args)

    assert written == output_md
    assert output_md.read_text(encoding="utf-8") == "# Converted"
    mock_convert.assert_awaited_once()


async def test_convert_pdf_example_writes_image_assets_next_to_markdown(tmp_path):
    module = _load_convert_pdf_module()
    input_pdf = tmp_path / "input.pdf"
    input_pdf.write_bytes(b"%PDF-1.4\n")
    output_md = tmp_path / "output.md"
    args = argparse.Namespace(
        input_pdf=input_pdf,
        output=output_md,
        page_range=None,
        max_pages_per_chunk=2,
        max_page_height_px=None,
        gpu_memory_profile=None,
        ocr_device=None,
        model_dtype=None,
        use_llm=False,
        paginate_output=True,
    )

    with patch(
        "marker_mcp.conversion_service.convert_file_result",
        new=AsyncMock(
            return_value={
                "text": "![](_page_0_Picture_2.jpeg)\n",
                "warnings": [],
                "assets": {
                    "images": [
                        {
                            "filename": "_page_0_Picture_2.jpeg",
                            "content_base64": base64.b64encode(b"image-bytes").decode("ascii"),
                            "media_type": "image/jpeg",
                        }
                    ]
                },
            }
        ),
    ):
        written = await module.convert_pdf(args)

    assert written == output_md
    assert output_md.read_text(encoding="utf-8") == "![](_page_0_Picture_2.jpeg)\n"
    assert (tmp_path / "_page_0_Picture_2.jpeg").read_bytes() == b"image-bytes"
