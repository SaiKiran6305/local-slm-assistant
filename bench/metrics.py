"""
Scoring.

The single most important thing this module does is keep **conformance** and
**accuracy** apart.

    conformance  did the model produce something matching the schema?
    accuracy     were the values in it correct?

The repair ladder moves conformance and cannot move accuracy. Salvaging a
markdown fence off a well-formed object with a wrong priority label yields a
conformant, wrong answer. A benchmark that reports one number conflating both
will show repair "improving the model", which is false and is the most common
way this kind of evaluation misleads.

So every run reports:

    conformance_raw        parsed clean at L0, no help
    conformance_final      conformant after the ladder ran
    accuracy               field-level correctness among conformant answers
    end_to_end             conformant AND fully correct -- the number that
                           actually predicts whether a pipeline works
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.repair import RepairLevel, RepairResult
from bench.tasks import BenchTask


def _normalise(v: Any) -> Any:
    """Unwrap an Enum member to its value.

    Pydantic's `model_dump()` returns Enum members, not their values, so a
    correct `Category.BILLING` stringifies to "category.billing" and compares
    unequal to "billing". Left unhandled this silently scores every enum field
    as wrong, which understates accuracy on exactly the fields this task set was
    designed to stress. `model_dump(mode="json")` is the primary fix; this is
    the belt-and-braces for values arriving from anywhere else.
    """
    return v.value if isinstance(v, Enum) else v


def _values_match(got: Any, want: Any, acceptable: list[Any] | None = None) -> bool:
    """Compare one field, tolerantly where the label is genuinely debatable."""
    got, want = _normalise(got), _normalise(want)
    if acceptable:
        acceptable = [_normalise(a) for a in acceptable]
        norm = [str(a).strip().lower() if a is not None else None for a in acceptable]
        g = str(got).strip().lower() if got is not None else None
        return g in norm

    if want is None:
        # A model may express absence as null, "null", "none", or "n/a".
        # Treating those as failures measures phrasing, not comprehension.
        return got is None or str(got).strip().lower() in {"null", "none", "n/a", ""}

    if isinstance(want, bool):
        if isinstance(got, bool):
            return got == want
        return str(got).strip().lower() == str(want).lower()

    if isinstance(want, (int, float)) and not isinstance(want, bool):
        try:
            return abs(float(got) - float(want)) < 0.01
        except (TypeError, ValueError):
            return False

    if isinstance(want, list):
        # Lists here are free text (decisions, action items). Exact string
        # matching would measure phrasing. Count is the defensible proxy; the
        # report flags list fields so nobody reads this as semantic scoring.
        return isinstance(got, list) and len(got) == len(want)

    return str(got).strip().lower() == str(want).strip().lower()


@dataclass
class TaskScore:
    task_id: str
    schema_name: str
    conformant: bool
    level: RepairLevel
    fields_correct: int
    fields_total: int
    field_detail: dict[str, bool] = field(default_factory=dict)
    model_calls: int = 0
    ttft_ms: float = 0.0
    total_ms: float = 0.0
    tokens_per_second: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def fully_correct(self) -> bool:
        return self.conformant and self.fields_correct == self.fields_total

    @property
    def field_accuracy(self) -> float:
        return self.fields_correct / self.fields_total if self.fields_total else 0.0


def score_task(task: BenchTask, result: RepairResult) -> TaskScore:
    # Free-text fields are excluded from scoring: equality on a generated
    # summary measures phrasing, not correctness.
    scored_fields = [k for k in task.expected if k != "summary"]

    if not result.ok or result.value is None:
        return TaskScore(
            task_id=task.id, schema_name=task.schema_name,
            conformant=False, level=result.level,
            fields_correct=0, fields_total=len(scored_fields),
            model_calls=result.model_calls,
            ttft_ms=result.ttft_ms, total_ms=result.total_ms,
            errors=result.errors,
        )

    # mode="json" serialises Enum members to their values; the default
    # mode returns the members themselves and every enum field misscores.
    got = result.value.model_dump(mode="json")
    detail: dict[str, bool] = {}
    for key in scored_fields:
        want = task.expected[key]
        ok = _values_match(got.get(key), want, task.acceptable.get(key))
        detail[key] = ok

    gens = result.generations
    tps = statistics.mean([g.tokens_per_second for g in gens if g.tokens_per_second > 0]) if gens else 0.0

    return TaskScore(
        task_id=task.id, schema_name=task.schema_name,
        conformant=True, level=result.level,
        fields_correct=sum(detail.values()), fields_total=len(detail),
        field_detail=detail,
        model_calls=result.model_calls,
        ttft_ms=result.ttft_ms, total_ms=result.total_ms,
        tokens_per_second=tps,
    )


def _pct(xs: list[float]) -> dict[str, float]:
    if not xs:
        return {"p50": 0.0, "p95": 0.0, "mean": 0.0}
    ordered = sorted(xs)
    return {
        "p50": round(statistics.median(ordered), 1),
        "p95": round(ordered[max(0, int(len(ordered) * 0.95) - 1)], 1),
        "mean": round(statistics.mean(ordered), 1),
    }


def aggregate(scores: list[TaskScore], model: str, provider: str,
              grammar_first: bool = False) -> dict[str, Any]:
    n = len(scores) or 1

    raw_ok = sum(1 for s in scores if s.conformant and s.level == RepairLevel.PARSE)
    final_ok = sum(1 for s in scores if s.conformant)
    fully = sum(1 for s in scores if s.fully_correct)
    abstained = sum(1 for s in scores if s.level == RepairLevel.ABSTAIN)

    conformant = [s for s in scores if s.conformant]
    acc = statistics.mean([s.field_accuracy for s in conformant]) if conformant else 0.0

    level_counts = {lvl.label: 0 for lvl in RepairLevel}
    for s in scores:
        level_counts[s.level.label] += 1

    by_schema: dict[str, dict[str, Any]] = {}
    for name in sorted({s.schema_name for s in scores}):
        sub = [s for s in scores if s.schema_name == name]
        sub_conf = [s for s in sub if s.conformant]
        by_schema[name] = {
            "n": len(sub),
            "conformance_raw": round(sum(1 for s in sub if s.conformant and s.level == RepairLevel.PARSE) / len(sub), 3),
            "conformance_final": round(len(sub_conf) / len(sub), 3),
            "accuracy": round(statistics.mean([s.field_accuracy for s in sub_conf]), 3) if sub_conf else 0.0,
            "end_to_end": round(sum(1 for s in sub if s.fully_correct) / len(sub), 3),
        }

    return {
        "model": model,
        "provider": provider,
        "grammar_first": grammar_first,
        "n_tasks": len(scores),
        "conformance_raw": round(raw_ok / n, 3),
        "conformance_final": round(final_ok / n, 3),
        "accuracy": round(acc, 3),
        "end_to_end": round(fully / n, 3),
        "abstention_rate": round(abstained / n, 3),
        "repair_levels": level_counts,
        "avg_model_calls": round(statistics.mean([s.model_calls for s in scores]), 2),
        "ttft_ms": _pct([s.ttft_ms for s in scores if s.ttft_ms > 0]),
        "total_ms": _pct([s.total_ms for s in scores]),
        "tokens_per_second": round(
            statistics.mean([s.tokens_per_second for s in scores if s.tokens_per_second > 0]), 1
        ) if any(s.tokens_per_second > 0 for s in scores) else 0.0,
        "by_schema": by_schema,
        "failures": [
            {"task": s.task_id, "level": s.level.label, "errors": s.errors[:3]}
            for s in scores if not s.conformant
        ],
        "wrong_but_conformant": [
            {"task": s.task_id, "fields": [k for k, ok in s.field_detail.items() if not ok]}
            for s in scores if s.conformant and not s.fully_correct
        ],
    }
