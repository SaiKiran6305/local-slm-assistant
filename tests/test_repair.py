"""
Tests for the repair ladder.

The salvage tests use real malformed output shapes rather than synthetic
strings, because the failure modes worth handling are the ones models actually
produce, and a test built from an imagined failure tends to pass while the real
one still breaks.

The most important test here is `test_salvage_never_invents_data`. If salvage
could supply a missing field, the conformance metric would be measuring the
repairer rather than the model, and the whole benchmark would be meaningless.
"""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from app.providers.simulated import PROFILES, FailureProfile, SimulatedProvider
from app.repair import (
    RepairLevel, close_truncated, extract_first_object, fix_single_quotes,
    fix_trailing_commas, gbnf_from_schema, quote_bare_keys, salvage,
    strip_fences, structured_generate,
)
from app.schemas.tasks import InvoiceExtract, MeetingNotes, TicketTriage


# --- individual salvage transforms ---------------------------------------

def test_strip_markdown_fence():
    text = '```json\n{"a": 1}\n```'
    out, changed = strip_fences(text)
    assert changed and json.loads(out) == {"a": 1}


def test_strip_fence_without_language_tag():
    out, changed = strip_fences('```\n{"a": 1}\n```')
    assert changed and json.loads(out) == {"a": 1}


def test_extract_object_from_prose():
    text = 'Sure! Here is the JSON:\n\n{"a": 1, "b": 2}\n\nLet me know if you need more!'
    out, changed = extract_first_object(text)
    assert changed and json.loads(out) == {"a": 1, "b": 2}


def test_extract_object_handles_nesting():
    """Brace counting, not regex: a regex cannot balance nested objects."""
    text = 'preamble {"outer": {"inner": {"deep": 1}}, "x": 2} trailing'
    out, _ = extract_first_object(text)
    assert json.loads(out) == {"outer": {"inner": {"deep": 1}}, "x": 2}


def test_extract_object_ignores_braces_inside_strings():
    text = '{"note": "use {curly} braces", "n": 1}'
    out, _ = extract_first_object(text)
    assert json.loads(out)["note"] == "use {curly} braces"


def test_trailing_comma_object_and_array():
    out, changed = fix_trailing_commas('{"a": [1, 2,], "b": 3,}')
    assert changed and json.loads(out) == {"a": [1, 2], "b": 3}


def test_trailing_comma_leaves_clean_json_alone():
    clean = '{"a": 1, "b": 2}'
    out, changed = fix_trailing_commas(clean)
    assert not changed and out == clean


def test_single_quotes_only_when_no_double_quotes():
    out, changed = fix_single_quotes("{'a': 1}")
    assert changed and json.loads(out) == {"a": 1}

    # An apostrophe inside a correctly quoted string must survive untouched.
    safe = '{"note": "it\'s fine"}'
    out2, changed2 = fix_single_quotes(safe)
    assert not changed2 and out2 == safe


def test_quote_bare_keys():
    out, changed = quote_bare_keys('{a: 1, b: "two"}')
    assert changed and json.loads(out) == {"a": 1, "b": "two"}


def test_close_truncated_object():
    out, changed = close_truncated('{"a": 1, "b": {"c": 2')
    assert changed and json.loads(out) == {"a": 1, "b": {"c": 2}}


def test_close_truncated_drops_dangling_key():
    out, changed = close_truncated('{"a": 1, "b":')
    assert changed and json.loads(out) == {"a": 1}


def test_salvage_reports_which_transforms_fired():
    text = 'Here you go:\n```json\n{category: "billing", "priority": "high",}\n```'
    out, applied = salvage(text)
    assert json.loads(out) == {"category": "billing", "priority": "high"}
    assert "strip_fences" in applied
    assert "quote_bare_keys" in applied
    assert "fix_trailing_commas" in applied


def test_salvage_never_invents_data():
    """The load-bearing guarantee.

    Salvage fixes syntax. If it could supply a missing required field, the
    conformance number would measure the repairer, not the model.
    """
    out, _ = salvage('```json\n{"category": "billing"}\n```')
    parsed = json.loads(out)
    assert parsed == {"category": "billing"}
    assert "priority" not in parsed
    # And the schema must still reject it.
    with pytest.raises(Exception):
        TicketTriage.model_validate(parsed)


# --- grammar generation ---------------------------------------------------

def test_grammar_pins_enum_values():
    """Pydantic hides enums behind $ref; if that is not resolved the grammar
    silently degrades to a bare string and stops constraining the vocabulary."""
    g = gbnf_from_schema(TicketTriage)
    assert '"\\"billing\\""' in g
    assert '"\\"urgent\\""' in g
    assert "category-val ::=" in g


def test_grammar_pins_scalar_types():
    g = gbnf_from_schema(InvoiceExtract)
    assert "integer ::=" in g and "number ::=" in g
    assert "ws integer" in g


def test_grammar_allows_null_for_optional_fields():
    """An optional field must be able to emit null, or the model is forced to
    invent a value for a field that genuinely has none."""
    g = gbnf_from_schema(InvoiceExtract)
    assert 'purchase_order-val ::= string | "null"' in g


def test_grammar_handles_arrays():
    g = gbnf_from_schema(MeetingNotes)
    assert "decisions-val ::=" in g
    assert '"[" ws (string (ws "," ws string)*)? ws "]"' in g


# --- the ladder -----------------------------------------------------------

def _prov(**overrides):
    profile = FailureProfile(name="test", **overrides)
    return SimulatedProvider(
        profile=profile,
        oracle=lambda _: {
            "category": "billing", "priority": "high",
            "sentiment": "negative", "requires_human": True,
        },
    )


def test_l0_clean_output_parses_directly():
    r = structured_generate(_prov(), "x", TicketTriage, model="test")
    assert r.ok and r.level is RepairLevel.PARSE and r.model_calls == 1


def test_l1_salvage_recovers_fenced_output():
    r = structured_generate(_prov(markdown_fence=1.0), "x", TicketTriage, model="test")
    assert r.ok and r.level is RepairLevel.SALVAGE
    assert "strip_fences" in r.transforms
    assert r.model_calls == 1, "salvage is string work; it must not cost a model call"


def test_l2_reprompt_when_a_field_is_dropped():
    """A missing required field is not syntactic, so salvage cannot fix it and
    the ladder must escalate to a re-prompt."""
    r = structured_generate(_prov(dropped_field=1.0), "x", TicketTriage, model="test")
    assert r.level >= RepairLevel.REPROMPT
    assert r.model_calls >= 2


def test_ladder_respects_max_level():
    r = structured_generate(_prov(markdown_fence=1.0), "x", TicketTriage,
                            model="test", max_level=RepairLevel.PARSE)
    assert not r.ok and r.level is RepairLevel.ABSTAIN
    assert r.model_calls == 1


def test_abstain_returns_no_value_rather_than_a_guess():
    r = structured_generate(_prov(dropped_field=1.0), "x", TicketTriage,
                            model="test", max_level=RepairLevel.SALVAGE)
    assert not r.ok and r.value is None and r.errors


def test_grammar_first_lifts_l0_conformance():
    """The headline comparison: constraining upfront should remove the syntactic
    failures entirely rather than repairing them after the fact."""
    p = _prov(markdown_fence=1.0, prose_preamble=1.0, trailing_comma=1.0)

    after = structured_generate(p, "x", TicketTriage, model="test")
    assert after.level is RepairLevel.SALVAGE

    first = structured_generate(p, "y", TicketTriage, model="test", grammar_first=True)
    assert first.ok and first.level is RepairLevel.PARSE


def test_grammar_first_ignored_when_backend_cannot_constrain():
    """Claiming a constraint the backend never applied would make the benchmark
    arm a lie, so the flag must be a no-op on unsupported providers."""
    class NoGrammar(SimulatedProvider):
        @property
        def supports_grammar(self) -> bool:
            return False

    p = NoGrammar(profile=FailureProfile(name="t", markdown_fence=1.0),
                  oracle=lambda _: {"category": "billing", "priority": "high",
                                    "sentiment": "negative", "requires_human": True})
    r = structured_generate(p, "x", TicketTriage, model="t", grammar_first=True)
    assert r.level is RepairLevel.SALVAGE, "fence should still be present and salvaged"


def test_reprompt_gets_a_fresh_draw():
    """If a retry returned byte-identical output the re-prompt rung could never
    succeed, and the harness would under-report what re-prompting achieves."""
    p = _prov(markdown_fence=0.5)
    a = p.generate("same prompt", model="test")
    b = p.generate("same prompt", model="test")
    assert a.raw["attempt"] == 0 and b.raw["attempt"] == 1


# --- provider contract ----------------------------------------------------

def test_generation_throughput_excludes_prompt_processing():
    from app.providers.base import Generation
    g = Generation(text="x", model="m", completion_tokens=100,
                   ttft_ms=200.0, total_ms=1200.0)
    # 100 tokens over the 1000ms decode window, not over the full 1200ms.
    assert g.tokens_per_second == pytest.approx(100.0, rel=0.01)


def test_throughput_is_zero_when_nothing_was_generated():
    from app.providers.base import Generation
    assert Generation(text="", model="m", completion_tokens=0).tokens_per_second == 0.0


def test_grammar_costs_throughput_in_simulator():
    """Constrained sampling is not free; if it were, grammar-first would be an
    unconditional win and there would be nothing to measure."""
    p = SimulatedProvider(profile="sim-7b-q4", oracle=lambda _: {"a": 1})
    free = p.generate("p", model="sim-7b-q4")
    constrained = p.generate("p", model="sim-7b-q4", grammar="root ::= x")
    assert constrained.tokens_per_second < free.tokens_per_second


@pytest.mark.parametrize("name", [n for n in PROFILES])
def test_every_profile_is_usable(name):
    p = SimulatedProvider(profile=name, oracle=lambda _: {"a": 1})
    g = p.generate("hello", model=name)
    assert g.text and g.total_ms > 0 and not g.error
