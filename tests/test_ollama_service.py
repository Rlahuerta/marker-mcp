import json
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

from pydantic import BaseModel

from marker_mcp.ollama_service import OllamaService


class BlockSchema(BaseModel):
    id: str
    html: str


class SectionHeaderSchema(BaseModel):
    analysis: str
    correction_type: str
    blocks: list[BlockSchema]


class TableSchema(BaseModel):
    comparison: str
    corrected_html: str
    analysis: str
    score: int


class FormSchema(BaseModel):
    comparison: str
    corrected_html: str


class EquationSchema(BaseModel):
    analysis: str
    corrected_equation: str


class MarkdownSchema(BaseModel):
    corrected_markdown: str


class ImageSchema(BaseModel):
    image_description: str


def test_ollama_service_extracts_fenced_json_from_cloud_response():
    service = OllamaService(
        {
            "ollama_base_url": "http://localhost:11434",
            "ollama_model": "gemma4:31b-cloud",
        }
    )
    response_text = (
        "Analysis: The first header is the main title and should be an h1.\n\n"
        "corrections_needed\n"
        "```json\n"
        "[\n"
        '  {"id": "/page/0/SectionHeader/2", "html": "<h1>ABSTRACT</h1>"}\n'
        "]\n"
        "```"
    )
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "model": "gemma4:31b",
        "response": response_text,
        "done": True,
    }
    mock_response.raise_for_status.return_value = None

    with patch("marker_mcp.ollama_service.requests.post", return_value=mock_response):
        payload = service("prompt", None, None, SectionHeaderSchema)

    assert payload == {
        "analysis": "The first header is the main title and should be an h1.",
        "correction_type": "corrections_needed",
        "blocks": [
            {"id": "/page/0/SectionHeader/2", "html": "<h1>ABSTRACT</h1>"},
        ],
    }


def test_ollama_service_tolerates_missing_token_counts():
    service = OllamaService(
        {
            "ollama_base_url": "http://localhost:11434",
            "ollama_model": "gemma4:31b-cloud",
        }
    )
    mock_block = MagicMock()
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "model": "gemma4:31b",
        "response": json.dumps(
            {
                "analysis": "No fixes needed.",
                "correction_type": "no_corrections",
                "blocks": [],
            }
        ),
        "done": True,
    }
    mock_response.raise_for_status.return_value = None

    with patch("marker_mcp.ollama_service.requests.post", return_value=mock_response):
        payload = service("prompt", None, mock_block, SectionHeaderSchema)

    assert payload["correction_type"] == "no_corrections"
    mock_block.update_metadata.assert_not_called()


def test_ollama_service_extracts_table_payload_from_fenced_html_response():
    service = OllamaService(
        {
            "ollama_base_url": "http://localhost:11434",
            "ollama_model": "gemma4:31b-cloud",
        }
    )
    response_text = (
        "comparison: The provided HTML interprets the text as a table, but the image is actually "
        "two columns of plain text.\n\n"
        "```html\n"
        "<table>\n"
        "  <tr><td>Left column</td><td>Right column</td></tr>\n"
        "</table>\n"
        "```\n\n"
        "analysis: I corrected the OCR mistakes and preserved the two-column layout.\n"
        "score: 5"
    )
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "model": "gemma4:31b",
        "response": response_text,
        "done": True,
    }
    mock_response.raise_for_status.return_value = None

    with patch("marker_mcp.ollama_service.requests.post", return_value=mock_response):
        payload = service("prompt", None, None, TableSchema)

    assert payload == {
        "comparison": (
            "The provided HTML interprets the text as a table, but the image is actually "
            "two columns of plain text."
        ),
        "corrected_html": "<table>\n  <tr><td>Left column</td><td>Right column</td></tr>\n</table>",
        "analysis": "I corrected the OCR mistakes and preserved the two-column layout.",
        "score": 5,
    }


def test_ollama_service_extracts_form_no_corrections_response():
    service = OllamaService(
        {
            "ollama_base_url": "http://localhost:11434",
            "ollama_model": "gemma4:31b-cloud",
        }
    )
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "model": "gemma4:31b",
        "response": "No corrections needed.",
        "done": True,
    }
    mock_response.raise_for_status.return_value = None

    with patch("marker_mcp.ollama_service.requests.post", return_value=mock_response):
        payload = service("prompt", None, None, FormSchema)

    assert payload == {
        "comparison": "",
        "corrected_html": "No corrections needed.",
    }


def test_ollama_service_extracts_equation_payload_from_fenced_html_response():
    service = OllamaService(
        {
            "ollama_base_url": "http://localhost:11434",
            "ollama_model": "gemma4:31b-cloud",
        }
    )
    response_text = (
        "analysis: The previous HTML was close, but the image includes the phrase "
        "\"for example\" above the equation.\n\n"
        "```html\n"
        "<p>for example</p>\n"
        "<math display=\"block\">2^{-4}</math>\n"
        "```"
    )
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "model": "gemma4:31b",
        "response": response_text,
        "done": True,
    }
    mock_response.raise_for_status.return_value = None

    with patch("marker_mcp.ollama_service.requests.post", return_value=mock_response):
        payload = service("prompt", None, None, EquationSchema)

    assert payload == {
        "analysis": (
            "The previous HTML was close, but the image includes the phrase "
            "\"for example\" above the equation."
        ),
        "corrected_equation": "<p>for example</p>\n<math display=\"block\">2^{-4}</math>",
    }


def test_ollama_service_extracts_single_markdown_field():
    service = OllamaService(
        {
            "ollama_base_url": "http://localhost:11434",
            "ollama_model": "gemma4:31b-cloud",
        }
    )
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "model": "gemma4:31b",
        "response": "```markdown\n## Header\n\n- item\n```",
        "done": True,
    }
    mock_response.raise_for_status.return_value = None

    with patch("marker_mcp.ollama_service.requests.post", return_value=mock_response):
        payload = service("prompt", None, None, MarkdownSchema)

    assert payload == {"corrected_markdown": "## Header\n\n- item"}


def test_ollama_service_extracts_single_text_field():
    service = OllamaService(
        {
            "ollama_base_url": "http://localhost:11434",
            "ollama_model": "gemma4:31b-cloud",
        }
    )
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "model": "gemma4:31b",
        "response": "A bar chart showing three values: 20, 15, and 10.",
        "done": True,
    }
    mock_response.raise_for_status.return_value = None

    with patch("marker_mcp.ollama_service.requests.post", return_value=mock_response):
        payload = service("prompt", None, None, ImageSchema)

    assert payload == {"image_description": "A bar chart showing three values: 20, 15, and 10."}


def test_ollama_service_batches_compatible_requests():
    service = OllamaService(
        {
            "ollama_base_url": "http://localhost:11434",
            "ollama_model": "gemma4:31b-cloud",
            "ollama_batch_size": 4,
            "ollama_batch_wait_ms": 10,
        }
    )
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "model": "gemma4:31b",
        "response": json.dumps(
            {
                "responses": [
                    {"corrected_markdown": "## First"},
                    {"corrected_markdown": "## Second"},
                ]
            }
        ),
        "done": True,
        "prompt_eval_count": 100,
        "eval_count": 20,
    }
    mock_response.raise_for_status.return_value = None

    with patch("marker_mcp.ollama_service.requests.post", return_value=mock_response) as mock_post:
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_one = executor.submit(service, "prompt one", None, None, MarkdownSchema)
            future_two = executor.submit(service, "prompt two", None, None, MarkdownSchema)

        first = future_one.result()
        second = future_two.result()

    assert first == {"corrected_markdown": "## First"}
    assert second == {"corrected_markdown": "## Second"}
    assert mock_post.call_count == 1
    batched_prompt = mock_post.call_args.kwargs["json"]["prompt"]
    assert "Task 1:" in batched_prompt
    assert "Task 2:" in batched_prompt


def test_ollama_service_falls_back_to_individual_requests_when_batch_parse_fails():
    service = OllamaService(
        {
            "ollama_base_url": "http://localhost:11434",
            "ollama_model": "gemma4:31b-cloud",
            "ollama_batch_size": 4,
            "ollama_batch_wait_ms": 10,
        }
    )
    batched_response = MagicMock()
    batched_response.json.return_value = {
        "model": "gemma4:31b",
        "response": "not valid batched json",
        "done": True,
    }
    batched_response.raise_for_status.return_value = None

    single_one = MagicMock()
    single_one.json.return_value = {
        "model": "gemma4:31b",
        "response": json.dumps({"corrected_markdown": "## First"}),
        "done": True,
    }
    single_one.raise_for_status.return_value = None

    single_two = MagicMock()
    single_two.json.return_value = {
        "model": "gemma4:31b",
        "response": json.dumps({"corrected_markdown": "## Second"}),
        "done": True,
    }
    single_two.raise_for_status.return_value = None

    with patch(
        "marker_mcp.ollama_service.requests.post",
        side_effect=[batched_response, single_one, single_two],
    ) as mock_post:
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_one = executor.submit(service, "prompt one", None, None, MarkdownSchema)
            future_two = executor.submit(service, "prompt two", None, None, MarkdownSchema)

        first = future_one.result()
        second = future_two.result()

    assert first == {"corrected_markdown": "## First"}
    assert second == {"corrected_markdown": "## Second"}
    assert mock_post.call_count == 3
