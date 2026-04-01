"""Sarvam provider runner with injectable transport layer."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from time import perf_counter
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from code_switch_failure_map.models.base import BaseModelRunner, ProviderResponse
from code_switch_failure_map.models.tokenizer_stats import TokenUsage, usage_from_provider
from code_switch_failure_map.schemas.sample import SampleRecord
from code_switch_failure_map.schemas.taxonomy import PromptLanguage

SarvamTransport = Callable[[str], ProviderResponse | tuple[str, TokenUsage | None] | str]
_SARVAM_CHAT_COMPLETIONS_URL = "https://api.sarvam.ai/v1/chat/completions"


class SarvamRunner(BaseModelRunner):
    """Production-clean Sarvam runner with an injection point for API client."""

    def __init__(
        self,
        *,
        prompt_language: PromptLanguage,
        model_name: str = "sarvam-30b",
        transport: SarvamTransport | None = None,
    ) -> None:
        super().__init__(model_name=model_name, prompt_language=prompt_language)
        self._transport = transport

    def invoke_model(self, *, prompt_text: str, sample: SampleRecord) -> ProviderResponse:
        if self._transport is None:
            return self._invoke_live(prompt_text=prompt_text)

        started = perf_counter()
        response = self._transport(prompt_text)
        latency_ms = (perf_counter() - started) * 1000

        if isinstance(response, ProviderResponse):
            return ProviderResponse(
                raw_text=response.raw_text,
                token_usage=response.token_usage,
                latency_ms=response.latency_ms if response.latency_ms is not None else latency_ms,
            )
        if isinstance(response, tuple):
            raw_text, usage = response
            return ProviderResponse(raw_text=raw_text, token_usage=usage, latency_ms=latency_ms)
        return ProviderResponse(raw_text=response, latency_ms=latency_ms)

    def _invoke_live(self, *, prompt_text: str) -> ProviderResponse:
        api_key = os.getenv("SARVAM_API_KEY")
        if not api_key:
            raise RuntimeError("Missing SARVAM_API_KEY; configure environment or inject Sarvam transport")

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt_text}],
            "temperature": 0,
        }
        request = Request(
            _SARVAM_CHAT_COMPLETIONS_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        started = perf_counter()
        try:
            with urlopen(request, timeout=120) as response:
                response_body = response.read().decode("utf-8")
        except HTTPError as exc:
            message = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Sarvam request failed with HTTP {exc.code}: {message}") from exc
        except URLError as exc:
            raise RuntimeError(f"Sarvam request failed: {exc.reason}") from exc

        latency_ms = (perf_counter() - started) * 1000
        decoded = json.loads(response_body)
        return ProviderResponse(
            raw_text=_extract_sarvam_text(decoded),
            token_usage=_extract_usage(decoded.get("usage")),
            latency_ms=latency_ms,
        )


def _extract_sarvam_text(decoded: dict[str, object]) -> str:
    choices = decoded.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("Sarvam response missing choices[0]")

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise RuntimeError("Sarvam response choices[0] is invalid")

    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise RuntimeError("Sarvam response missing message payload")

    text = _extract_text_candidate(message.get("content"))
    if text is not None:
        return text

    for fallback_key in ("output_text", "reasoning_content", "refusal"):
        text = _extract_text_candidate(message.get(fallback_key))
        if text is not None:
            return text

    text = _extract_text_candidate(first_choice.get("text"))
    if text is not None:
        return text

    top_level_keys = sorted(str(key) for key in decoded.keys())
    choice_keys = sorted(str(key) for key in first_choice.keys())
    message_keys = sorted(str(key) for key in message.keys())
    raise RuntimeError(
        "Sarvam response missing assistant text content. "
        f"top_level_keys={top_level_keys}; choice_keys={choice_keys}; message_keys={message_keys}"
    )


def _extract_text_candidate(value: object) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else None

    if isinstance(value, dict):
        for key in ("text", "content", "value"):
            nested = _extract_text_candidate(value.get(key))
            if nested is not None:
                return nested
        return None

    if isinstance(value, list):
        text_chunks: list[str] = []
        for item in value:
            nested = _extract_text_candidate(item)
            if nested is not None:
                text_chunks.append(nested)
        if text_chunks:
            return "".join(text_chunks)

    return None


def _extract_usage(usage_payload: object) -> TokenUsage | None:
    if not isinstance(usage_payload, dict):
        return None
    return usage_from_provider(
        input_tokens=_coerce_int(usage_payload.get("prompt_tokens")),
        output_tokens=_coerce_int(usage_payload.get("completion_tokens")),
        total_tokens=_coerce_int(usage_payload.get("total_tokens")),
    )


def _coerce_int(value: object) -> int | None:
    return value if isinstance(value, int) else None
