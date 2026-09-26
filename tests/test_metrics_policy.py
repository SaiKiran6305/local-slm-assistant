"""
Tests for scoring and the escalation policy.

The central claim this file defends: **repair moves conformance and cannot move
accuracy.** If a change ever lets salvage improve field accuracy, something is
inventing data, and `test_repair_cannot_improve_accuracy` should fail loudly
rather than let the README keep asserting a separation that no longer holds.
"""

from __future__ import annotations

import pytest

from app.assistant import Assistant, Disposition, Policy
from app.providers.simulated import FailureProfile, SimulatedProvider
from app.repair import RepairLevel, structured_generate
from app.schemas.tasks import TicketTriage
from bench.metrics import TaskScore, _values_match, aggregate, score_task
from bench.tasks import TASKS, oracle_fn

CORRECT = {
    "category": "billing", "priority": "high",
    "sentiment": "negative", "requires_human": True,
}


def _prov(**kw):
    return SimulatedProvider(
        profile=FailureProfile(name="t", **kw), oracle=lambda _: dict(CORRECT)
    )


# --- field comparison -----------------------------------------------------

def test_absence_accepted_in_any_reasonable_spelling():
    """Models express "no value" as null, "null", "none" or "n/a". Counting
    those as wrong measures phrasing rather than comprehension."""
    for got in (None, "null", "None", "n/a", ""):
        assert _values_match(got, None)
    assert not _values_match("PO-123", None)


def test_numeric_comparison_tolerates_string_and_float_drift():
    assert _values_match("2855.00", 2855.0)
    assert _values_match(2855, 2855.0)
    assert not _values_match(2856, 2855.0)


def test_bool_accepts_stringified_form():
    assert _values_match("true", True)
    assert _values_match(True, True)
    assert not _values_match("false", True)


def test_acceptable_list_permits_debatable_labels():
    """Some labels are genuinely arguable. Forcing one right answer measures
    agreement with the task author's taste, not model capability."""
    assert _values_match("urgent", "high", acceptable=["high", "urgent"])
    assert not _values_match("low", "high", acceptable=["high", "urgent"])


# --- the conformance / accuracy separation --------------------------------

def test_conformant_but_wrong_is_counted_as_conformant_and_wrong():
    task = next(t for t in TASKS if t.id == "triage-01")

    class WrongValues(SimulatedProvider):
        pass

    p = WrongValues(profile=FailureProfile(name="t"),
                    oracle=lambda _: {**CORRECT, "category": "technical"})
    r = structured_generate(p, task.prompt, TicketTriage, model="t")
    s = score_task(task, r)

    assert s.conformant, "well-formed output is conformant even when wrong"
    assert not s.fully_correct, "a wrong value must not count as correct"
    assert s.field_detail["category"] is False
    assert s.field_detail["priority"] is True


def test_repair_cannot_improve_accuracy():
    """Same wrong answer, wrapped in a markdown fence. Salvage should recover
    the object and leave the wrongness exactly where it was."""
    task = next(t for t in TASKS if t.id == "triage-01")
    wrong = {**CORRECT, "category": "technical"}

    clean = SimulatedProvider(profile=FailureProfile(name="t"),
                              oracle=lambda _: dict(wrong))
    fenced = SimulatedProvider(profile=FailureProfile(name="t", markdown_fence=1.0),
                               oracle=lambda _: dict(wrong))

    s_clean = score_task(task, structured_generate(clean, task.prompt, TicketTriage, model="t"))
    s_fenced = score_task(task, structured_generate(fenced, task.prompt, TicketTriage, model="t"))

    assert s_clean.level is RepairLevel.PARSE
    assert s_fenced.level is RepairLevel.SALVAGE
    assert s_clean.field_accuracy == s_fenced.field_accuracy
    assert s_clean.field_detail == s_fenced.field_detail


def test_aggregate_separates_raw_and_final_conformance():
    scores = [
        TaskScore("a", "s", True, RepairLevel.PARSE, 4, 4),
        TaskScore("b", "s", True, RepairLevel.SALVAGE, 4, 4),
        TaskScore("c", "s", True, RepairLevel.REPROMPT, 2, 4),
        TaskScore("d", "s", False, RepairLevel.ABSTAIN, 0, 4),
    ]
    agg = aggregate(scores, model="m", provider="simulated")

    assert agg["conformance_raw"] == 0.25      # only the L0 one
    assert agg["conformance_final"] == 0.75    # three of four ended conformant
    assert agg["end_to_end"] == 0.5            # two were conformant AND fully right
    assert agg["abstention_rate"] == 0.25


def test_aggregate_accuracy_ignores_nonconformant_rows():
    """Averaging a 0 for an abstention into accuracy would conflate "refused to
    answer" with "answered wrongly", which are different failures."""
    scores = [
        TaskScore("a", "s", True, RepairLevel.PARSE, 4, 4),
        TaskScore("b", "s", False, RepairLevel.ABSTAIN, 0, 4),
    ]
    agg = aggregate(scores, model="m", provider="simulated")
    assert agg["accuracy"] == 1.0
    assert agg["end_to_end"] == 0.5


# --- escalation policy ----------------------------------------------------

def test_clean_local_answer_stays_on_device():
    a = Assistant(_prov(), "t", policy=Policy())
    r = a.run("x", TicketTriage)
    assert r.disposition is Disposition.LOCAL and r.stayed_local


def test_salvaged_answer_still_stays_local_by_default():
    """Formatting noise is not evidence the model misunderstood the task."""
    a = Assistant(_prov(markdown_fence=1.0), "t", policy=Policy())
    r = a.run("x", TicketTriage)
    assert r.disposition is Disposition.LOCAL


def test_reprompt_triggers_escalation():
    a = Assistant(
        _prov(dropped_field=1.0), "t",
        escalation_provider=_prov(), escalation_model="big",
        policy=Policy(escalate_above_level=RepairLevel.SALVAGE),
    )
    r = a.run("x", TicketTriage)
    assert r.disposition is Disposition.ESCALATED
    assert r.escalated is not None and r.value is not None


def test_never_leave_device_refuses_instead_of_escalating():
    """The hard privacy switch must win over every other rule, including a
    configured and reachable escalation backend."""
    a = Assistant(
        _prov(dropped_field=1.0), "t",
        escalation_provider=_prov(), escalation_model="big",
        policy=Policy(never_leave_device=True),
    )
    r = a.run("x", TicketTriage)
    assert r.disposition is Disposition.REFUSED
    assert r.value is None
    assert "never_leave_device" in r.reason


def test_refuses_when_escalation_is_not_configured():
    a = Assistant(_prov(dropped_field=1.0), "t", policy=Policy())
    r = a.run("x", TicketTriage)
    assert r.disposition is Disposition.REFUSED
    assert "no escalation backend" in r.reason


def test_local_share_is_measured_not_claimed():
    a = Assistant(_prov(), "t", policy=Policy())
    for _ in range(4):
        a.run("x", TicketTriage)
    assert a.local_share == 1.0
    assert a.stats["local"] == 4


# --- end to end on the real task set --------------------------------------

def test_clean_profile_answers_every_task_correctly():
    """A sanity floor: with no corruption the harness must score perfectly.
    If this drops, the bug is in scoring or the oracle, not in a model."""
    p = SimulatedProvider(profile="sim-clean", oracle=oracle_fn())
    scores = [
        score_task(t, structured_generate(p, t.prompt, t.schema_cls, model="sim-clean"))
        for t in TASKS
    ]
    agg = aggregate(scores, model="sim-clean", provider="simulated")
    assert agg["conformance_final"] == 1.0
    assert agg["end_to_end"] == 1.0, f"unexpected misses: {agg['wrong_but_conformant']}"
