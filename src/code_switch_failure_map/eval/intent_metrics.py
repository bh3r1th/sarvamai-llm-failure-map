"""Intent scoring helpers for extraction evaluation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from code_switch_failure_map.schemas.evaluation import EvaluationResult
from code_switch_failure_map.schemas.prediction import PredictionRecord
from code_switch_failure_map.schemas.sample import SampleRecord


@dataclass(frozen=True)
class IntentScore:
    """Outcome of intent scoring for one sample."""

    gold_intent: str
    predicted_intent: str | None
    exact_match: bool


def score_intent(sample: SampleRecord, prediction: PredictionRecord) -> IntentScore:
    """Return exact-match intent scoring for one gold/prediction pair."""
    predicted_intent = prediction.parsed_prediction.intent if prediction.parsed_prediction is not None else None
    gold_intent = sample.gold_intent.value
    return IntentScore(
        gold_intent=gold_intent,
        predicted_intent=predicted_intent,
        exact_match=predicted_intent == gold_intent,
    )


def build_confusion_summary(results: list[EvaluationResult]) -> list[dict[str, object]]:
    """Summarize gold vs predicted intent counts by model and prompt language."""
    counter: Counter[tuple[str, str, str, str | None]] = Counter()
    for result in results:
        counter[(result.model_name, result.prompt_language.value, result.gold_intent or "", result.predicted_intent)] += 1

    rows: list[dict[str, object]] = []
    ordered_items = sorted(
        counter.items(),
        key=lambda item: (
            item[0][0],
            item[0][1],
            item[0][2],
            "" if item[0][3] is None else str(item[0][3]),
        ),
    )
    for (model_name, prompt_language, gold_intent, predicted_intent), count in ordered_items:
        rows.append(
            {
                "model_name": model_name,
                "prompt_language": prompt_language,
                "gold_intent": gold_intent,
                "predicted_intent": predicted_intent,
                "count": count,
            }
        )
    return rows
