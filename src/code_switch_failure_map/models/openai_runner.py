"""OpenAI provider runner with injectable transport layer."""

from __future__ import annotations

import os
from collections.abc import Callable
from time import perf_counter

from code_switch_failure_map.models.base import BaseModelRunner, ProviderResponse
from code_switch_failure_map.models.tokenizer_stats import TokenUsage
from code_switch_failure_map.schemas.sample import SampleRecord
from code_switch_failure_map.schemas.taxonomy import PromptLanguage

OpenAITransport = Callable[[str], ProviderResponse | tuple[str, TokenUsage | None] | str]


class OpenAIRunner(BaseModelRunner):
    """Production-clean GPT runner with swappable transport/client dependency."""

    def __init__(
        self,
        *,
        prompt_language: PromptLanguage,
        model_name: str = "gpt-4o-mini",
        transport: OpenAITransport | None = None,
    ) -> None:
        super().__init__(model_name=model_name, prompt_language=prompt_language)
        self._transport = transport

    def invoke_model(self, *, prompt_text: str, sample: SampleRecord) -> ProviderResponse:
        if self._transport is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("Missing OPENAI_API_KEY; configure environment or inject OpenAI transport")
            raise RuntimeError("OpenAI live transport not wired in this repo. Inject transport callable for execution.")

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
