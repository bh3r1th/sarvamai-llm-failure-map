"""Behavioral tests for aggregate evaluation summaries."""

from __future__ import annotations

from code_switch_failure_map.eval.aggregate import classify_boundary_failure, evaluate_prediction_set, summarize_aggregates
from code_switch_failure_map.schemas.prediction import ParsedEntity, ParsedPrediction, PredictionRecord
from code_switch_failure_map.schemas.sample import EntityMention, MetadataFlags, SampleRecord
from code_switch_failure_map.schemas.taxonomy import (
    EntityType,
    FailureCategory,
    IntentLabel,
    PromptLanguage,
    SliceTag,
    SourceSplit,
)


def _sample(
    *,
    sample_id: str,
    text: str,
    intent: IntentLabel,
    entities: list[EntityMention] | None = None,
    slice_tags: set[SliceTag] | None = None,
) -> SampleRecord:
    return SampleRecord(
        sample_id=sample_id,
        source_split=SourceSplit.GOLDEN,
        text=text,
        gold_intent=intent,
        gold_entities=entities or [],
        metadata_flags=MetadataFlags(),
        slice_tags=slice_tags or {SliceTag.PROMPT_LANGUAGE_HINGLISH},
        prompt_variant="baseline_v1",
        prompt_language=PromptLanguage.HINGLISH,
    )


def _prediction(
    *,
    sample_id: str,
    intent: str | None,
    entities: list[ParsedEntity] | None = None,
    parse_success: bool = True,
    schema_failure: bool = False,
) -> PredictionRecord:
    return PredictionRecord(
        sample_id=sample_id,
        model_name="demo-model",
        prompt_language=PromptLanguage.HINGLISH,
        prompt_text="prompt",
        raw_response='{"intent": null, "entities": []}',
        parsed_prediction=ParsedPrediction(intent=intent, entities=entities or []) if parse_success else None,
        parse_success=parse_success,
        schema_failure=schema_failure,
    )


def test_correct_intent_scoring_and_confusion_summary() -> None:
    gold = [
        _sample(sample_id="s1", text="rahul ko call karo", intent=IntentLabel.CALL_REQUEST),
        _sample(sample_id="s2", text="kal ka weather", intent=IntentLabel.INFORMATION_QUERY),
    ]
    predictions = [
        _prediction(sample_id="s1", intent="call_request"),
        _prediction(sample_id="s2", intent="message_send"),
    ]

    results, issues = evaluate_prediction_set(gold_records=gold, predictions=predictions)
    summary = summarize_aggregates(results)

    assert issues == []
    assert [result.intent_correct for result in results] == [True, False]
    assert summary["by_model"][0]["intent_accuracy"] == 0.5
    assert summary["intent_confusion"] == [
        {
            "model_name": "demo-model",
            "prompt_language": "hinglish",
            "gold_intent": "call_request",
            "predicted_intent": "call_request",
            "count": 1,
        },
        {
            "model_name": "demo-model",
            "prompt_language": "hinglish",
            "gold_intent": "information_query",
            "predicted_intent": "message_send",
            "count": 1,
        },
    ]


def test_aggregate_summary_correctness() -> None:
    gold = [
        _sample(
            sample_id="s1",
            text="kal Rahul ko call karo",
            intent=IntentLabel.CALL_REQUEST,
            entities=[EntityMention(type=EntityType.CONTACT, text="Rahul")],
            slice_tags={SliceTag.AMBIGUITY, SliceTag.PROMPT_LANGUAGE_HINGLISH},
        ),
        _sample(
            sample_id="s2",
            text="Aman ko yaad dilao",
            intent=IntentLabel.REMINDER_CREATE,
            entities=[EntityMention(type=EntityType.CONTACT, text="Aman")],
            slice_tags={SliceTag.CODE_SWITCHING, SliceTag.PROMPT_LANGUAGE_HINGLISH},
        ),
    ]
    predictions = [
        _prediction(sample_id="s1", intent="call_request", entities=[ParsedEntity(label="contact", value="rahul")]),
        _prediction(sample_id="s2", intent="reminder_create", entities=[ParsedEntity(label="person", value="aman")]),
    ]

    results, issues = evaluate_prediction_set(gold_records=gold, predictions=predictions)
    summary = summarize_aggregates(results)

    assert issues == []
    assert summary["by_model"][0]["samples"] == 2
    assert summary["by_model"][0]["intent_accuracy"] == 1.0
    assert summary["by_model"][0]["entity_f1"] == 0.5
    assert any(row["slice_tag"] == "ambiguity" and row["samples"] == 1 for row in summary["by_slice_tag"])
    assert any(row["slice_tag"] == "code_switching" and row["samples"] == 1 for row in summary["by_slice_tag"])
    assert any(row["failure_bucket"] == FailureCategory.HALLUCINATION.value and row["samples"] == 1 for row in summary["by_failure_bucket"])


def test_aggregate_summary_includes_language_family_rows_when_present() -> None:
    gold = [
        _sample(
            sample_id="s1",
            text="kal Rahul ko call karo",
            intent=IntentLabel.CALL_REQUEST,
            entities=[EntityMention(type=EntityType.CONTACT, text="Rahul")],
            slice_tags={SliceTag.HINGLISH, SliceTag.PROMPT_LANGUAGE_HINGLISH},
        ),
        _sample(
            sample_id="s2",
            text="amma ki meeting schedule chey",
            intent=IntentLabel.MEETING_SCHEDULE,
            entities=[EntityMention(type=EntityType.PERSON, text="amma")],
            slice_tags={SliceTag.TELUGU_ENGLISH, SliceTag.PROMPT_LANGUAGE_HINGLISH},
        ),
    ]
    predictions = [
        _prediction(sample_id="s1", intent="call_request", entities=[ParsedEntity(label="contact", value="rahul")]),
        _prediction(sample_id="s2", intent="meeting_schedule", entities=[ParsedEntity(label="person", value="amma")]),
    ]

    results, issues = evaluate_prediction_set(gold_records=gold, predictions=predictions)
    summary = summarize_aggregates(results)

    assert issues == []
    assert any(row["language_family"] == "hinglish" and row["samples"] == 1 for row in summary["by_language_family"])
    assert any(
        row["language_family"] == "telugu_english" and row["samples"] == 1
        for row in summary["by_language_family"]
    )


def test_aggregate_summary_derives_language_family_from_legacy_sample_ids() -> None:
    gold = [
        _sample(
            sample_id="cur_v2_hi_001",
            text="kal Rahul ko call karo",
            intent=IntentLabel.CALL_REQUEST,
            entities=[EntityMention(type=EntityType.CONTACT, text="Rahul")],
            slice_tags={SliceTag.CODE_SWITCHING, SliceTag.PROMPT_LANGUAGE_HINGLISH},
        ),
        _sample(
            sample_id="cur_v2_te_001",
            text="amma ki meeting schedule chey",
            intent=IntentLabel.MEETING_SCHEDULE,
            entities=[EntityMention(type=EntityType.PERSON, text="amma")],
            slice_tags={SliceTag.CODE_SWITCHING, SliceTag.PROMPT_LANGUAGE_HINGLISH},
        ),
    ]
    predictions = [
        _prediction(sample_id="cur_v2_hi_001", intent="call_request", entities=[ParsedEntity(label="contact", value="rahul")]),
        _prediction(sample_id="cur_v2_te_001", intent="meeting_schedule", entities=[ParsedEntity(label="person", value="amma")]),
    ]

    results, issues = evaluate_prediction_set(gold_records=gold, predictions=predictions)
    summary = summarize_aggregates(results)

    assert issues == []
    assert summary["by_language_family"] == [
        {
            "language_family": "hinglish",
            "samples": 1,
            "intent_accuracy": 1.0,
            "entity_precision": 1.0,
            "entity_recall": 1.0,
            "entity_f1": 1.0,
            "exact_match_rate": 1.0,
            "schema_failure_rate": 0.0,
            "schema_failures": 0,
            "exact_matches": 1,
        },
        {
            "language_family": "telugu_english",
            "samples": 1,
            "intent_accuracy": 1.0,
            "entity_precision": 1.0,
            "entity_recall": 1.0,
            "entity_f1": 1.0,
            "exact_match_rate": 1.0,
            "schema_failure_rate": 0.0,
            "schema_failures": 0,
            "exact_matches": 1,
        },
    ]
    assert any(
        row["language_family"] == "hinglish"
        for row in summary["boundary_failure_by_language_family"]
    )
    assert any(
        row["language_family"] == "telugu_english"
        for row in summary["boundary_failure_by_language_family"]
    )


def test_boundary_failure_category_assignment_is_deterministic() -> None:
    exact_match = _sample(
        sample_id="s1",
        text="rahul ko call karo",
        intent=IntentLabel.CALL_REQUEST,
        entities=[EntityMention(type=EntityType.CONTACT, text="Rahul")],
    )
    missing_entity = _sample(
        sample_id="s2",
        text="rahul ko kal call karo",
        intent=IntentLabel.CALL_REQUEST,
        entities=[
            EntityMention(type=EntityType.CONTACT, text="Rahul"),
            EntityMention(type=EntityType.DATE, text="kal"),
        ],
    )
    extra_entity = _sample(
        sample_id="s3",
        text="rahul ko call karo",
        intent=IntentLabel.CALL_REQUEST,
        entities=[EntityMention(type=EntityType.CONTACT, text="Rahul")],
    )
    drift_entity = _sample(
        sample_id="s4",
        text="rahul ko kal call karo",
        intent=IntentLabel.CALL_REQUEST,
        entities=[EntityMention(type=EntityType.CONTACT, text="Rahul")],
    )
    wrong_intent_partial_entity = _sample(
        sample_id="s5",
        text="rahul ko call karo",
        intent=IntentLabel.CALL_REQUEST,
        entities=[EntityMention(type=EntityType.CONTACT, text="Rahul")],
    )
    intent_correct_drift = _sample(
        sample_id="s6",
        text="rahul ko kal call karo",
        intent=IntentLabel.CALL_REQUEST,
        entities=[
            EntityMention(type=EntityType.CONTACT, text="Rahul"),
            EntityMention(type=EntityType.DATE, text="kal"),
        ],
    )
    wrong_intent_partial_with_tp = _sample(
        sample_id="s7",
        text="rahul ko kal call karo",
        intent=IntentLabel.CALL_REQUEST,
        entities=[
            EntityMention(type=EntityType.CONTACT, text="Rahul"),
            EntityMention(type=EntityType.DATE, text="kal"),
        ],
    )
    unparsed = _sample(
        sample_id="s8",
        text="rahul ko call karo",
        intent=IntentLabel.CALL_REQUEST,
        entities=[EntityMention(type=EntityType.CONTACT, text="Rahul")],
    )

    predictions = [
        _prediction(sample_id="s1", intent="call_request", entities=[ParsedEntity(label="contact", value="rahul")]),
        _prediction(sample_id="s2", intent="call_request", entities=[ParsedEntity(label="contact", value="rahul")]),
        _prediction(
            sample_id="s3",
            intent="call_request",
            entities=[
                ParsedEntity(label="contact", value="rahul"),
                ParsedEntity(label="date", value="kal"),
            ],
        ),
        _prediction(sample_id="s4", intent="call_request", entities=[ParsedEntity(label="date", value="kal")]),
        _prediction(sample_id="s5", intent="message_send", entities=[ParsedEntity(label="contact", value="rahul")]),
        _prediction(
            sample_id="s6",
            intent="call_request",
            entities=[
                ParsedEntity(label="contact", value="rahul"),
                ParsedEntity(label="date", value="aaj"),
            ],
        ),
        _prediction(
            sample_id="s7",
            intent="message_send",
            entities=[ParsedEntity(label="contact", value="rahul")],
        ),
        _prediction(sample_id="s8", intent=None, parse_success=False, schema_failure=True),
    ]

    results, issues = evaluate_prediction_set(
        gold_records=[
            exact_match,
            missing_entity,
            extra_entity,
            drift_entity,
            wrong_intent_partial_entity,
            intent_correct_drift,
            wrong_intent_partial_with_tp,
            unparsed,
        ],
        predictions=predictions,
    )

    assert issues == []
    assert [classify_boundary_failure(result) for result in results] == [
        "full_exact_match",
        "intent_correct_entity_missing",
        "intent_correct_entity_extra",
        "exact_intent_only",
        "exact_entities_only",
        "intent_correct_entity_drift",
        "intent_wrong_entity_partially_correct",
        "null_or_unparsed_prediction",
    ]


def test_boundary_failure_aggregates_preserve_existing_summary_behavior() -> None:
    gold = [
        _sample(sample_id="s1", text="rahul ko call karo", intent=IntentLabel.CALL_REQUEST),
        _sample(sample_id="s2", text="kal ka weather", intent=IntentLabel.INFORMATION_QUERY),
    ]
    predictions = [
        _prediction(sample_id="s1", intent="call_request"),
        _prediction(sample_id="s2", intent="message_send"),
    ]

    results, _ = evaluate_prediction_set(gold_records=gold, predictions=predictions)
    summary = summarize_aggregates(results)

    assert summary["by_model"][0]["samples"] == 2
    assert summary["by_model"][0]["intent_accuracy"] == 0.5
    assert any(
        row["boundary_failure_category"] == "full_exact_match" and row["samples"] == 1
        for row in summary["boundary_failure_summary"]
    )
    assert any(
        row["boundary_failure_category"] == "exact_entities_only" and row["samples"] == 1
        for row in summary["boundary_failure_summary"]
    )
