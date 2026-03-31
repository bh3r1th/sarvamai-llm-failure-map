"""Schema definitions for annotated dataset samples."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from code_switch_failure_map.schemas.taxonomy import PromptLanguage, SliceTag, SourceSplit

IntentLabel = Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)]
SampleText = Annotated[str, StringConstraints(min_length=1)]


class EntityMention(BaseModel):
    """Structured representation of a single entity mention."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    label: Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)]
    value: Annotated[str, StringConstraints(min_length=1)]
    start_char: int | None = Field(default=None, ge=0)
    end_char: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_offsets(self) -> "EntityMention":
        if self.start_char is not None and self.end_char is not None and self.end_char <= self.start_char:
            raise ValueError("end_char must be greater than start_char when both offsets are provided")
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
    gold_entities: list[EntityMention] | dict[str, str | list[str]]
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

        if SliceTag.PROMPT_LANGUAGE not in self.slice_tags:
            raise ValueError("slice_tags must include 'prompt_language'")

        return self
