"""Helpers for stable failure bucket ordering and tabulation."""

from __future__ import annotations

from code_switch_failure_map.schemas.taxonomy import FailureCategory

FAILURE_BUCKET_ORDER = [
    FailureCategory.TEMPORAL_INVERSION,
    FailureCategory.INTENT_CONFUSION,
    FailureCategory.SENTIMENT_INVERSION,
    FailureCategory.ENTITY_DRIFT,
    FailureCategory.HALLUCINATION,
    FailureCategory.SCHEMA_FAILURE,
    FailureCategory.OMISSION,
    FailureCategory.OVER_EXTRACTION,
    FailureCategory.OTHER,
]


def sorted_failure_buckets(buckets: set[FailureCategory]) -> list[FailureCategory]:
    """Return failure buckets in a stable presentation order."""
    present = set(buckets)
    ordered = [bucket for bucket in FAILURE_BUCKET_ORDER if bucket in present]
    remaining = sorted(bucket for bucket in present if bucket not in set(ordered))
    return ordered + remaining


def bucket_names(buckets: set[FailureCategory]) -> list[str]:
    """Return ordered string values for serialization."""
    return [bucket.value for bucket in sorted_failure_buckets(buckets)]
