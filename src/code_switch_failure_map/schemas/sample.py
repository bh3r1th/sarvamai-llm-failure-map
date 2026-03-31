"""Schema definitions for annotated dataset samples."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from code_switch_failure_map.schemas.taxonomy import (
    EntityType,
    IntentLabel,
    PromptLanguage,
    SliceTag,
    SourceSplit,
)

SampleText = Annotated[str, StringConstraints(min_length=1)]


class EntityMention(BaseModel):
    """Structured representation of a single entity mention."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: EntityType
    text: Annotated[str, StringConstraints(min_length=1)]
    normalized_value: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    start_char: int | None = Field(default=None, ge=0)
    end_char: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_offsets(self) -> "EntityMention":
        has_start = self.start_char is not None
        has_end = self.end_char is not None

        if has_start != has_end:
            raise ValueError("start_char and end_char must both be provided when spans are used")

        if has_start and has_end and self.end_char <= self.start_char:
            raise ValueError("end_char must be greater than start_char when offsets are provided")
        return self


class MetadataFlags(BaseModel):
    """Boolean metadata for common error-prone input phenomena."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code_switching: bool = False
    transliteration_noise: bool = False
    ambiguity: bool = False


class SampleRecord(BaseModel):
    """One labeled sample used in prompting and evaluation."""

    model_config = ConfigDict(extra="forbid")

    sample_id: Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)]
    source_split: SourceSplit
    text: SampleText
    normalized_text: str | None = None
    gold_intent: IntentLabel
    gold_entities: list[EntityMention] = Field(default_factory=list)
    metadata_flags: MetadataFlags = Field(default_factory=MetadataFlags)
    slice_tags: set[SliceTag] = Field(default_factory=set)
    prompt_variant: Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)]
    prompt_language: PromptLanguage

    @model_validator(mode="after")
    def validate_slice_tags(self) -> "SampleRecord":
        required_for_true_flags = {
            SliceTag.CODE_SWITCHING: self.metadata_flags.code_switching,
            SliceTag.TRANSLITERATION_NOISE: self.metadata_flags.transliteration_noise,
            SliceTag.AMBIGUITY: self.metadata_flags.ambiguity,
        }
        for tag, is_enabled in required_for_true_flags.items():
            if is_enabled and tag not in self.slice_tags:
                raise ValueError(f"slice_tags must include '{tag.value}' when corresponding metadata flag is true")

        expected_prompt_tag = {
            PromptLanguage.ENGLISH: SliceTag.PROMPT_LANGUAGE_EN,
            PromptLanguage.HINGLISH: SliceTag.PROMPT_LANGUAGE_HINGLISH,
        }[self.prompt_language]

        if expected_prompt_tag not in self.slice_tags:
            raise ValueError(f"slice_tags must include '{expected_prompt_tag.value}'")

        incompatible_prompt_tag = {
            PromptLanguage.ENGLISH: SliceTag.PROMPT_LANGUAGE_HINGLISH,
            PromptLanguage.HINGLISH: SliceTag.PROMPT_LANGUAGE_EN,
        }[self.prompt_language]

        if incompatible_prompt_tag in self.slice_tags:
            raise ValueError(f"slice_tags cannot include '{incompatible_prompt_tag.value}' for this prompt_language")

        return self
