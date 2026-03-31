"""Minimal token usage helpers for model runner outputs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import ceil


@dataclass(frozen=True)
class TokenUsage:
    """Token accounting values for a single request/response cycle."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


TokenEstimator = Callable[[str], int]


def usage_from_provider(
    *,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    total_tokens: int | None = None,
) -> TokenUsage:
    """Create a usage object from provider-reported counts."""
    return TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens, total_tokens=total_tokens)


def estimate_tokens_basic(text: str) -> int:
    """Very small fallback estimator when provider counts are unavailable."""
    stripped = text.strip()
    if not stripped:
        return 0
    return ceil(len(stripped) / 4)


def merge_token_usage(
    *,
    prompt_text: str,
    raw_response: str,
    usage: TokenUsage | None,
    estimator: TokenEstimator | None = None,
) -> TokenUsage:
    """Merge provider counts with estimated fallbacks for missing values."""
    estimator_fn = estimator or estimate_tokens_basic
    usage = usage or TokenUsage()

    input_tokens = usage.input_tokens if usage.input_tokens is not None else estimator_fn(prompt_text)
    output_tokens = usage.output_tokens if usage.output_tokens is not None else estimator_fn(raw_response)
    total_tokens = usage.total_tokens
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )
