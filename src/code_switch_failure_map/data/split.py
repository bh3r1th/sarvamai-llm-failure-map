"""Deterministic utilities for assigning curated/golden data buckets."""

from __future__ import annotations

import hashlib

from code_switch_failure_map.schemas.sample import SampleRecord


BUCKET_CURATED = "curated"
BUCKET_GOLDEN = "golden"


def assign_bucket(sample_id: str, golden_ratio: float = 0.2) -> str:
    """Assign a stable bucket based on sample id hashing."""
    if not 0.0 < golden_ratio < 1.0:
        raise ValueError("golden_ratio must be between 0 and 1")

    digest = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()
    slot = int(digest[:8], 16) / 0xFFFFFFFF
    return BUCKET_GOLDEN if slot < golden_ratio else BUCKET_CURATED


def split_records(records: list[SampleRecord], golden_ratio: float = 0.2) -> tuple[list[SampleRecord], list[SampleRecord]]:
    """Split records into curated and golden sets deterministically."""
    curated: list[SampleRecord] = []
    golden: list[SampleRecord] = []
    for record in records:
        bucket = assign_bucket(record.sample_id, golden_ratio=golden_ratio)
        if bucket == BUCKET_GOLDEN:
            golden.append(record)
        else:
            curated.append(record)
    return curated, golden
