"""More tolerant Ollama service wrapper for Marker.

The upstream Marker Ollama service expects the model to return a bare JSON string in
`response`. Ollama Cloud models can instead return prose plus fenced JSON or HTML,
which makes the upstream `json.loads(response)` path fail even though the answer is
usable.
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Annotated, Any, List

import PIL
import requests
from marker.logger import get_logger
from marker.schema.blocks import Block
from marker.services import BaseService
from pydantic import BaseModel

logger = get_logger()


@dataclass
class _PendingRequest:
    prompt: str
    image: PIL.Image.Image | list[PIL.Image.Image] | None
    block: Block | None
    response_schema: type[BaseModel]
    timeout: int
    event: threading.Event = field(default_factory=threading.Event)
    result: dict[str, Any] = field(default_factory=dict)


class OllamaService(BaseService):
    timeout: Annotated[int, "The timeout to use for the service."] = 180
    ollama_base_url: Annotated[
        str, "The base url to use for ollama. No trailing slash."
    ] = "http://localhost:11434"
    ollama_model: Annotated[str, "The model name to use for ollama."] = (
        "gemma4:31b-cloud"
    )
    ollama_batch_size: Annotated[
        int, "Maximum number of compatible Ollama requests to combine into one call."
    ] = 3
    ollama_batch_wait_ms: Annotated[
        int, "How long to wait for additional compatible requests before dispatching a batch."
    ] = 50

    def __init__(self, config=None):
        super().__init__(config)
        self._batch_lock = threading.Lock()
        self._pending_batches: dict[tuple[str, str], list[_PendingRequest]] = {}

    def process_images(self, images):
        image_bytes = [self.img_to_base64(img) for img in images]
        return image_bytes

    @staticmethod
    def _build_format_schema(response_schema: type[BaseModel]) -> dict[str, Any]:
        schema = response_schema.model_json_schema()
        return {
            "type": "object",
            "properties": schema.get("properties", {}),
            "required": schema.get("required", []),
        }

    @staticmethod
    def _flatten_image_inputs(
        image: PIL.Image.Image | list[PIL.Image.Image] | None,
    ) -> list[PIL.Image.Image]:
        if image is None:
            return []
        if isinstance(image, list):
            return image
        return [image]

    def _is_batchable_schema(self, response_schema: type[BaseModel]) -> bool:
        properties = response_schema.model_json_schema().get("properties", {})
        if not properties:
            return False

        scalar_types = {"string", "integer", "number", "boolean"}
        return all(property_schema.get("type") in scalar_types for property_schema in properties.values())

    def _can_batch_request(
        self,
        image: PIL.Image.Image | list[PIL.Image.Image] | None,
        response_schema: type[BaseModel],
    ) -> bool:
        if self.ollama_batch_size <= 1:
            return False
        if not self._is_batchable_schema(response_schema):
            return False
        return len(self._flatten_image_inputs(image)) <= 1

    def _batch_key(self, response_schema: type[BaseModel]) -> tuple[str, str]:
        return (response_schema.__module__, response_schema.__qualname__)

    def _call_ollama_api(
        self,
        prompt: str,
        image: PIL.Image.Image | list[PIL.Image.Image] | None,
        format_schema: dict[str, Any],
        timeout: int,
    ) -> dict[str, Any]:
        url = f"{self.ollama_base_url.rstrip('/')}/api/generate"
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
            "format": format_schema,
            "images": self.format_image_for_llm(image),
        }
        response = requests.post(url, json=payload, headers=headers, timeout=timeout)
        response.raise_for_status()
        return response.json()

    def _update_block_metadata(
        self,
        blocks: list[Block | None],
        prompt_eval_count: int | None,
        eval_count: int | None,
    ) -> None:
        if not isinstance(prompt_eval_count, int) or not isinstance(eval_count, int):
            return

        valid_blocks = [block for block in blocks if block is not None]
        if not valid_blocks:
            return

        total_tokens = prompt_eval_count + eval_count
        tokens_per_block = max(1, total_tokens // len(valid_blocks))
        for block in valid_blocks:
            block.update_metadata(llm_request_count=1, llm_tokens_used=tokens_per_block)

    def _build_batch_prompt(
        self,
        requests_batch: list[_PendingRequest],
        item_schema: dict[str, Any],
    ) -> tuple[str, list[PIL.Image.Image]]:
        prompt_parts = [
            "You are handling multiple independent document-vision tasks.",
            "Solve each task independently using only its assigned image(s) and prompt.",
            'Return one JSON object: {"responses": [...]}',
            "The responses array must contain exactly one response object per task, in the same order.",
            "Each response object must follow this JSON schema:",
            "```json",
            json.dumps(item_schema, ensure_ascii=False),
            "```",
            "",
        ]

        flattened_images: list[PIL.Image.Image] = []
        image_cursor = 1
        for index, pending in enumerate(requests_batch, start=1):
            images = self._flatten_image_inputs(pending.image)
            if images:
                start_idx = image_cursor
                end_idx = image_cursor + len(images) - 1
                image_ref = (
                    f"Use image {start_idx}."
                    if start_idx == end_idx
                    else f"Use images {start_idx}-{end_idx}."
                )
                flattened_images.extend(images)
                image_cursor = end_idx + 1
            else:
                image_ref = "This task has no image."

            prompt_parts.extend(
                [
                    f"Task {index}: {image_ref}",
                    "Apply the following task prompt exactly as written:",
                    pending.prompt,
                    "",
                ]
            )

        return "\n".join(prompt_parts), flattened_images

    def _coerce_batch_response_payload(
        self,
        data: str,
        response_schema: type[BaseModel],
        expected_count: int,
    ) -> list[dict[str, Any]]:
        for candidate in self._extract_json_candidates(data):
            responses: list[Any] | None = None
            if isinstance(candidate, dict):
                maybe_responses = candidate.get("responses")
                if isinstance(maybe_responses, list):
                    responses = maybe_responses
            elif isinstance(candidate, list):
                responses = candidate

            if responses is None or len(responses) != expected_count:
                continue

            return [
                response_schema.model_validate(response).model_dump()
                for response in responses
            ]

        raise json.JSONDecodeError(
            "Could not extract batched JSON from Ollama response",
            data,
            0,
        )

    def _execute_single_request(
        self,
        prompt: str,
        image: PIL.Image.Image | list[PIL.Image.Image] | None,
        block: Block | None,
        response_schema: type[BaseModel],
        timeout: int,
    ) -> dict[str, Any]:
        response_data = self._call_ollama_api(
            prompt=prompt,
            image=image,
            format_schema=self._build_format_schema(response_schema),
            timeout=timeout,
        )
        self._update_block_metadata(
            [block],
            response_data.get("prompt_eval_count"),
            response_data.get("eval_count"),
        )
        return self._coerce_response_payload(str(response_data.get("response", "")), response_schema)

    def _execute_batched_requests(self, key: tuple[str, str], requests_batch: list[_PendingRequest]) -> None:
        if len(requests_batch) == 1:
            pending = requests_batch[0]
            try:
                pending.result = self._execute_single_request(
                    prompt=pending.prompt,
                    image=pending.image,
                    block=pending.block,
                    response_schema=pending.response_schema,
                    timeout=pending.timeout,
                )
            except Exception as exc:
                logger.warning(f"Ollama inference failed: {exc}")
                pending.result = {}
            finally:
                pending.event.set()
            return

        try:
            response_schema = requests_batch[0].response_schema
            item_schema = self._build_format_schema(response_schema)
            batch_prompt, flattened_images = self._build_batch_prompt(requests_batch, item_schema)
            batch_response = self._call_ollama_api(
                prompt=batch_prompt,
                image=flattened_images,
                format_schema={
                    "type": "object",
                    "properties": {
                        "responses": {
                            "type": "array",
                            "items": item_schema,
                        }
                    },
                    "required": ["responses"],
                },
                timeout=max(pending.timeout for pending in requests_batch) * len(requests_batch),
            )

            parsed_responses = self._coerce_batch_response_payload(
                str(batch_response.get("response", "")),
                response_schema,
                len(requests_batch),
            )
            self._update_block_metadata(
                [pending.block for pending in requests_batch],
                batch_response.get("prompt_eval_count"),
                batch_response.get("eval_count"),
            )
            for pending, parsed in zip(requests_batch, parsed_responses, strict=True):
                pending.result = parsed
        except Exception as exc:
            logger.info(
                "Batched Ollama request failed; retrying individual requests: %s",
                exc,
            )
            for pending in requests_batch:
                try:
                    pending.result = self._execute_single_request(
                        prompt=pending.prompt,
                        image=pending.image,
                        block=pending.block,
                        response_schema=pending.response_schema,
                        timeout=pending.timeout,
                    )
                except Exception as single_exc:
                    logger.warning(f"Ollama inference failed: {single_exc}")
                    pending.result = {}
        finally:
            for pending in requests_batch:
                pending.event.set()

    def _dispatch_batched_request(
        self,
        prompt: str,
        image: PIL.Image.Image | list[PIL.Image.Image] | None,
        block: Block | None,
        response_schema: type[BaseModel],
        timeout: int,
    ) -> dict[str, Any]:
        pending = _PendingRequest(
            prompt=prompt,
            image=image,
            block=block,
            response_schema=response_schema,
            timeout=timeout,
        )
        key = self._batch_key(response_schema)

        with self._batch_lock:
            batch = self._pending_batches.setdefault(key, [])
            batch.append(pending)
            is_leader = len(batch) == 1

        if is_leader:
            time.sleep(max(0, self.ollama_batch_wait_ms) / 1000)
            with self._batch_lock:
                requests_batch = self._pending_batches.pop(key, [])
            self._execute_batched_requests(key, requests_batch)
        else:
            pending.event.wait()

        return pending.result

    def _extract_json_candidates(self, text: str) -> list[Any]:
        candidates: list[Any] = []
        stripped = text.strip()
        if not stripped:
            return candidates

        for candidate in (
            stripped,
            self._extract_fenced_json(stripped),
            self._extract_balanced_json(stripped),
        ):
            if not candidate:
                continue
            try:
                candidates.append(json.loads(candidate))
            except Exception:
                continue
        return candidates

    @staticmethod
    def _extract_fenced_block(text: str, language: str) -> str | None:
        match = re.search(
            rf"```{re.escape(language)}\s*(.*?)\s*```",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if match:
            return match.group(1).strip()
        return None

    @staticmethod
    def _extract_fenced_json(text: str) -> str | None:
        return OllamaService._extract_fenced_block(text, "json")

    @staticmethod
    def _extract_balanced_json(text: str) -> str | None:
        start_positions = [idx for idx, char in enumerate(text) if char in "[{"]
        for start in start_positions:
            opening = text[start]
            closing = "]" if opening == "[" else "}"
            depth = 0
            in_string = False
            escape = False
            for idx in range(start, len(text)):
                char = text[idx]
                if escape:
                    escape = False
                    continue
                if char == "\\":
                    escape = True
                    continue
                if char == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if char == opening:
                    depth += 1
                elif char == closing:
                    depth -= 1
                    if depth == 0:
                        return text[start : idx + 1]
        return None

    @staticmethod
    def _extract_analysis(text: str) -> str:
        match = re.search(
            r"analysis\s*:\s*(.*?)(?:\n\s*(?:no_corrections|corrections_needed)\b|```json|\Z)",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if not match:
            return ""
        return match.group(1).strip()

    @staticmethod
    def _extract_correction_type(text: str) -> str:
        lowered = text.lower()
        if "no_corrections" in lowered:
            return "no_corrections"
        if "corrections_needed" in lowered:
            return "corrections_needed"
        return "corrections_needed"

    @staticmethod
    def _extract_labeled_value(text: str, label: str, stops: tuple[str, ...]) -> str:
        stop_pattern = "|".join(re.escape(stop) for stop in stops if stop)
        terminator = (
            rf"(?:\n\s*(?:{stop_pattern})\s*:|```html|\Z)"
            if stop_pattern
            else r"(?:```html|\Z)"
        )
        match = re.search(
            rf"{re.escape(label)}\s*:\s*(.*?){terminator}",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if not match:
            return ""
        return match.group(1).strip()

    def _extract_corrected_block(self, text: str, language: str = "html") -> str:
        corrected = self._extract_fenced_block(text, language)
        if corrected is not None:
            return corrected
        if "no corrections needed" in text.lower():
            return "No corrections needed."
        return ""

    def _extract_table_payload(self, text: str) -> dict[str, Any]:
        score_match = re.search(r"score\s*:\s*(\d+)", text, re.IGNORECASE)
        score = int(score_match.group(1)) if score_match else 5
        return {
            "comparison": self._extract_labeled_value(text, "comparison", ("analysis", "score")),
            "corrected_html": self._extract_corrected_block(text),
            "analysis": self._extract_labeled_value(text, "analysis", ("score",)),
            "score": score,
        }

    def _extract_comparison_html_payload(self, text: str) -> dict[str, Any]:
        return {
            "comparison": self._extract_labeled_value(text, "comparison", ()),
            "corrected_html": self._extract_corrected_block(text),
        }

    def _extract_analysis_html_payload(self, text: str, output_field: str) -> dict[str, Any]:
        return {
            "analysis": self._extract_labeled_value(text, "analysis", ()),
            output_field: self._extract_corrected_block(text),
        }

    def _extract_single_field_value(self, text: str, field_name: str) -> str:
        if field_name in {"markdown", "corrected_markdown"}:
            markdown = self._extract_fenced_block(text, "markdown")
            if markdown is None:
                markdown = self._extract_fenced_block(text, "md")
            if markdown is not None:
                return markdown
        if field_name in {"corrected_html", "corrected_equation"}:
            return self._extract_corrected_block(text)
        if "no corrections needed" in text.lower():
            return "No corrections needed."
        return text.strip()

    def _coerce_response_payload(self, data: str, response_schema: type[BaseModel]) -> dict[str, Any]:
        schema = response_schema.model_json_schema()
        properties = schema.get("properties", {})

        for candidate in self._extract_json_candidates(data):
            if isinstance(candidate, dict):
                try:
                    return response_schema.model_validate(candidate).model_dump()
                except Exception:
                    pass

            if isinstance(candidate, list) and "blocks" in properties:
                wrapped = {
                    "analysis": self._extract_analysis(data),
                    "correction_type": self._extract_correction_type(data),
                    "blocks": candidate,
                }
                try:
                    return response_schema.model_validate(wrapped).model_dump()
                except Exception:
                    pass

        # Final heuristic for schemas like SectionHeaderSchema where the model returns
        # prose plus `corrections_needed`/`no_corrections` but no strict object wrapper.
        if {"analysis", "correction_type", "blocks"}.issubset(properties):
            wrapped = {
                "analysis": self._extract_analysis(data),
                "correction_type": self._extract_correction_type(data),
                "blocks": [],
            }
            return response_schema.model_validate(wrapped).model_dump()

        if {"comparison", "corrected_html", "analysis", "score"}.issubset(properties):
            return response_schema.model_validate(self._extract_table_payload(data)).model_dump()

        if {"comparison", "corrected_html"}.issubset(properties):
            return response_schema.model_validate(
                self._extract_comparison_html_payload(data)
            ).model_dump()

        if {"analysis", "corrected_html"}.issubset(properties):
            return response_schema.model_validate(
                self._extract_analysis_html_payload(data, "corrected_html")
            ).model_dump()

        if {"analysis", "corrected_equation"}.issubset(properties):
            return response_schema.model_validate(
                self._extract_analysis_html_payload(data, "corrected_equation")
            ).model_dump()

        string_fields = [
            name for name, property_schema in properties.items() if property_schema.get("type") == "string"
        ]
        if len(properties) == 1 and len(string_fields) == 1:
            field_name = string_fields[0]
            return response_schema.model_validate(
                {field_name: self._extract_single_field_value(data, field_name)}
            ).model_dump()

        raise json.JSONDecodeError("Could not extract JSON from Ollama response", data, 0)

    def __call__(
        self,
        prompt: str,
        image: PIL.Image.Image | List[PIL.Image.Image] | None,
        block: Block | None,
        response_schema: type[BaseModel],
        max_retries: int | None = None,
        timeout: int | None = None,
    ):
        try:
            effective_timeout = timeout or self.timeout
            if self._can_batch_request(image, response_schema):
                return self._dispatch_batched_request(
                    prompt=prompt,
                    image=image,
                    block=block,
                    response_schema=response_schema,
                    timeout=effective_timeout,
                )

            return self._execute_single_request(
                prompt=prompt,
                image=image,
                block=block,
                response_schema=response_schema,
                timeout=effective_timeout,
            )
        except Exception as exc:
            logger.warning(f"Ollama inference failed: {exc}")

        return {}
