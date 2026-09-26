"""
The structured output repair ladder.

The problem this exists for: a 7B model that emits schema-valid JSON 73% of the
time is unusable in a pipeline, and the gap between 73% and 99% is engineering,
not a bigger model. This is that engineering.

Five rungs, tried in order, each more expensive than the last:

    L0  PARSE       parse the raw output as-is
    L1  SALVAGE     strip fences and prose, fix commas and quotes, extract the
                    first balanced object. Pure string work: no model call, no
                    added latency worth measuring.
    L2  REPROMPT    hand the validation error back to the model and ask again.
                    Costs one more generation.
    L3  GRAMMAR     regenerate under a GBNF grammar derived from the schema, so
                    invalid syntax is unrepresentable rather than discouraged.
                    Only attempted when the provider actually supports it.
    L4  ABSTAIN     give up, and say so, rather than return a guess.

Two things this design gets right that are easy to get wrong:

**The ladder stops at the first success.** Reporting the distribution of rungs
reached is the whole point -- "L0 61%, L1 34%, L2 4%, abstain 1%" tells you
that cheap string repair is carrying the system and a bigger model would be
wasted money. An average conformance rate tells you none of that.

**Salvage never invents data.** Every L1 transform is syntactic. It will strip a
markdown fence and fix a trailing comma; it will not supply a missing required
field or coerce a wrong value into a right one. If repair could invent values,
the conformance metric would measure the repairer rather than the model.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Type, TypeVar

from pydantic import BaseModel, ValidationError

from app.providers.base import Generation, Provider

T = TypeVar("T", bound=BaseModel)


class RepairLevel(IntEnum):
    PARSE = 0
    SALVAGE = 1
    REPROMPT = 2
    GRAMMAR = 3
    ABSTAIN = 4

    @property
    def label(self) -> str:
        return {
            0: "L0 parsed clean",
            1: "L1 salvaged",
            2: "L2 re-prompted",
            3: "L3 grammar constrained",
            4: "L4 abstained",
        }[int(self)]


@dataclass
class RepairResult:
    """Outcome of one structured generation attempt."""
    ok: bool
    level: RepairLevel
    value: BaseModel | None = None
    raw_text: str = ""
    final_text: str = ""
    errors: list[str] = field(default_factory=list)
    transforms: list[str] = field(default_factory=list)   # which L1 fixes fired
    generations: list[Generation] = field(default_factory=list)

    @property
    def model_calls(self) -> int:
        return len(self.generations)

    @property
    def total_ms(self) -> float:
        return sum(g.total_ms for g in self.generations)

    @property
    def ttft_ms(self) -> float:
        return self.generations[0].ttft_ms if self.generations else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "level": int(self.level),
            "level_label": self.level.label,
            "value": self.value.model_dump() if self.value else None,
            "errors": self.errors,
            "transforms": self.transforms,
            "model_calls": self.model_calls,
            "total_ms": round(self.total_ms, 1),
            "ttft_ms": round(self.ttft_ms, 1),
            "raw_text": self.raw_text,
        }


# ---------------------------------------------------------------------------
# L1: syntactic salvage
# ---------------------------------------------------------------------------

_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


def strip_fences(text: str) -> tuple[str, bool]:
    m = _FENCE.search(text)
    return (m.group(1).strip(), True) if m else (text, False)


def extract_first_object(text: str) -> tuple[str, bool]:
    """Pull the first balanced {...} out of surrounding prose.

    Brace counting rather than a regex, because regexes cannot balance nested
    braces and small models nest objects constantly. String state is tracked so
    a brace inside a quoted value does not end the object early.
    """
    start = text.find("{")
    if start == -1:
        return text, False

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if c == "\\":
            escape = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start:i + 1]
                return candidate, candidate != text
    # Unbalanced: truncated mid-object.
    return text[start:], True


def fix_trailing_commas(text: str) -> tuple[str, bool]:
    fixed = re.sub(r",(\s*[}\]])", r"\1", text)
    return fixed, fixed != text


def fix_single_quotes(text: str) -> tuple[str, bool]:
    """Only when the text contains no double quotes at all.

    Guarded deliberately: a blanket quote swap would corrupt any legitimate
    apostrophe inside a correctly quoted string value.
    """
    if '"' in text or "'" not in text:
        return text, False
    return text.replace("'", '"'), True


def quote_bare_keys(text: str) -> tuple[str, bool]:
    fixed = re.sub(r"([{,]\s*)([A-Za-z_]\w*)(\s*:)", r'\1"\2"\3', text)
    return fixed, fixed != text


def close_truncated(text: str) -> tuple[str, bool]:
    """Close an object cut off by the token limit.

    Recovers the fields that did arrive. Anything the model never emitted stays
    missing, so schema validation still fails if a required field was lost --
    which is correct. This salvages a truncation, it does not paper over one.
    """
    s = text.rstrip().rstrip(",")
    if not s.startswith("{"):
        return text, False

    depth, in_string, escape = 0, False, False
    for c in s:
        if escape:
            escape = False
            continue
        if c == "\\":
            escape = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1

    if depth <= 0 and not in_string:
        return text, False

    repaired = s
    if in_string:
        repaired += '"'
    # Drop a dangling `"key":` with no value before closing.
    repaired = re.sub(r',?\s*"[^"]*"\s*:\s*$', "", repaired)
    repaired += "}" * max(depth, 0)
    return repaired, True


SALVAGE_STEPS = [
    ("strip_fences", strip_fences),
    ("extract_first_object", extract_first_object),
    ("quote_bare_keys", quote_bare_keys),
    ("fix_single_quotes", fix_single_quotes),
    ("fix_trailing_commas", fix_trailing_commas),
    ("close_truncated", close_truncated),
]


def salvage(text: str) -> tuple[str, list[str]]:
    """Apply every syntactic fix in order, recording which ones fired."""
    applied: list[str] = []
    current = text
    for name, fn in SALVAGE_STEPS:
        current, changed = fn(current)
        if changed:
            applied.append(name)
    return current, applied


# ---------------------------------------------------------------------------
# L3: GBNF grammar from a Pydantic schema
# ---------------------------------------------------------------------------

def gbnf_from_schema(schema_cls: Type[BaseModel]) -> str:
    """Build a GBNF grammar pinning the object's keys and value types.

    Keys are emitted as literals in schema order, so the sampler cannot produce
    a missing key, an extra key, or a key out of order. That eliminates every
    syntactic and structural failure mode by construction. It does not
    constrain the *content* of a string, so a wrong value remains perfectly
    representable -- which is why the benchmark reports conformance and accuracy
    as separate numbers.
    """
    schema = schema_cls.model_json_schema()
    props: dict[str, Any] = schema.get("properties", {})
    defs: dict[str, Any] = schema.get("$defs", {})

    def resolve(spec: dict[str, Any]) -> dict[str, Any]:
        """Follow $ref into $defs, and collapse the anyOf Pydantic emits for optionals.

        Pydantic puts every Enum behind a $ref rather than inlining it, so
        without this the enum rung degrades to a bare `string` rule and the
        grammar stops pinning the vocabulary -- which is most of the reason to
        use a grammar at all. Optionals arrive as anyOf[T, null]; we grammar the
        non-null branch, since a null is already representable.
        """
        seen = 0
        while "$ref" in spec and seen < 10:          # bounded: guards a cyclic schema
            ref = spec["$ref"].rsplit("/", 1)[-1]
            merged = dict(defs.get(ref, {}))
            merged.update({k: v for k, v in spec.items() if k != "$ref"})
            spec = merged
            seen += 1
        if "anyOf" in spec:
            branches = [b for b in spec["anyOf"] if b.get("type") != "null"]
            nullable = len(branches) != len(spec["anyOf"])
            if branches:
                spec = resolve(branches[0])
                if nullable:
                    spec = {**spec, "__nullable__": True}
        return spec

    def value_rule(spec: dict[str, Any], name: str) -> tuple[str, list[str]]:
        spec = resolve(spec)
        # An optional field must be able to emit null. Without this the grammar
        # forbids the very value the schema declares legal, and the model is
        # forced to invent a string for a field that genuinely has no value --
        # turning "absent" into a hallucination by construction.
        if spec.pop("__nullable__", False):
            inner, extra = value_rule(spec, f"{name}-inner")
            rule = f"{name}-val"
            return rule, extra + [f'{rule} ::= {inner} | "null"']
        t = spec.get("type")
        extra: list[str] = []
        if "enum" in spec:
            opts = " | ".join(f'"\\"{v}\\""' for v in spec["enum"])
            rule = f"{name}-val"
            extra.append(f"{rule} ::= {opts}")
            return rule, extra
        if t == "integer":
            return "integer", extra
        if t == "number":
            return "number", extra
        if t == "boolean":
            return "boolean", extra
        if t == "array":
            item = spec.get("items", {}) or {}
            item_rule, more = value_rule(item, f"{name}-item")
            extra += more
            rule = f"{name}-val"
            extra.append(f'{rule} ::= "[" ws ({item_rule} (ws "," ws {item_rule})*)? ws "]"')
            return rule, extra
        return "string", extra

    lines: list[str] = []
    pair_rules: list[str] = []
    for key, spec in props.items():
        safe = re.sub(r"\W", "-", key)
        vrule, extra = value_rule(spec, safe)
        lines += extra
        pair_rules.append(f'"\\"{key}\\"" ws ":" ws {vrule}')

    body = ' ws "," ws '.join(pair_rules) if pair_rules else ""
    grammar = [
        f'root ::= "{{" ws {body} ws "}}"' if body else 'root ::= "{" ws "}"',
        *lines,
        'string ::= "\\"" char* "\\""',
        'char ::= [^"\\\\] | "\\\\" ["\\\\/bfnrt]',
        'integer ::= "-"? ("0" | [1-9] [0-9]*)',
        'number ::= integer ("." [0-9]+)?',
        'boolean ::= "true" | "false"',
        'ws ::= [ \\t\\n]*',
    ]
    return "\n".join(grammar)


# ---------------------------------------------------------------------------
# The ladder
# ---------------------------------------------------------------------------

def _validate(text: str, schema_cls: Type[T]) -> tuple[T | None, list[str]]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        return None, [f"json: {e.msg} at line {e.lineno} col {e.colno}"]
    if not isinstance(data, dict):
        return None, [f"json: expected object, got {type(data).__name__}"]
    try:
        return schema_cls.model_validate(data), []
    except ValidationError as e:
        return None, [
            f"{'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}"
            for err in e.errors()
        ]


def _describe_errors(errors: list[str]) -> str:
    return "; ".join(errors[:5])


def structured_generate(
    provider: Provider,
    prompt: str,
    schema_cls: Type[T],
    *,
    model: str,
    system: str | None = None,
    max_level: RepairLevel = RepairLevel.GRAMMAR,
    temperature: float = 0.0,
    max_tokens: int = 512,
    grammar_first: bool = False,
) -> RepairResult:
    """Generate JSON conforming to `schema_cls`, climbing the ladder as needed.

    `grammar_first` inverts the strategy: constrain the very first call instead
    of generating freely and repairing after. That is the real deployment
    choice, and the two arms have opposite cost profiles.

        repair-after    fast unconstrained decode, occasional second call
        grammar-first   slower constrained decode, but L0 succeeds every time

    Which wins depends on the model's raw conformance rate and on how much the
    constrained sampler costs in throughput -- both of which vary enough by
    model that the honest answer is to measure it rather than assume. Running
    the benchmark both ways is what the --grammar-first flag is for.
    """

    schema_json = json.dumps(schema_cls.model_json_schema(), indent=2)
    base_system = system or (
        "You output only JSON. No prose, no markdown fences, no explanation. "
        "The JSON must match the supplied schema exactly: every required field "
        "present, no additional fields."
    )
    full_prompt = f"{prompt}\n\nRespond with JSON matching this schema:\n{schema_json}"

    generations: list[Generation] = []

    # Only constrain upfront if the backend can actually enforce it; otherwise
    # the flag would silently do nothing and the arm would be mislabelled.
    use_grammar_first = grammar_first and provider.supports_grammar
    first_grammar = gbnf_from_schema(schema_cls) if use_grammar_first else None

    # -- L0: parse as-is ---------------------------------------------------
    gen = provider.generate(
        full_prompt, model=model, system=base_system,
        temperature=temperature, max_tokens=max_tokens, grammar=first_grammar,
    )
    generations.append(gen)
    raw_text = gen.text

    if gen.error:
        return RepairResult(False, RepairLevel.ABSTAIN, None, raw_text, raw_text,
                            [f"provider error: {gen.error}"], [], generations)

    value, errors = _validate(raw_text, schema_cls)
    if value is not None:
        return RepairResult(True, RepairLevel.PARSE, value, raw_text, raw_text,
                            [], [], generations)

    # -- L1: syntactic salvage --------------------------------------------
    if max_level >= RepairLevel.SALVAGE:
        salvaged, transforms = salvage(raw_text)
        if transforms:
            value, s_errors = _validate(salvaged, schema_cls)
            if value is not None:
                return RepairResult(True, RepairLevel.SALVAGE, value, raw_text,
                                    salvaged, [], transforms, generations)
            errors = s_errors

    # -- L2: re-prompt with the validation error ---------------------------
    if max_level >= RepairLevel.REPROMPT:
        retry_prompt = (
            f"{full_prompt}\n\n"
            f"Your previous response was rejected.\n"
            f"Previous response:\n{raw_text[:800]}\n\n"
            f"Validation errors: {_describe_errors(errors)}\n\n"
            f"Return only corrected JSON. No prose, no fences."
        )
        gen2 = provider.generate(
            retry_prompt, model=model, system=base_system,
            temperature=temperature, max_tokens=max_tokens,
        )
        generations.append(gen2)
        if not gen2.error:
            candidate, transforms2 = salvage(gen2.text)
            value, r_errors = _validate(candidate, schema_cls)
            if value is not None:
                return RepairResult(True, RepairLevel.REPROMPT, value, raw_text,
                                    candidate, [], transforms2, generations)
            errors = r_errors

    # -- L3: constrained decoding -----------------------------------------
    # Skipped when the provider cannot constrain the sampler. Attempting it
    # anyway would record a grammar success the backend never enforced.
    if max_level >= RepairLevel.GRAMMAR and provider.supports_grammar:
        grammar = gbnf_from_schema(schema_cls)
        gen3 = provider.generate(
            full_prompt, model=model, system=base_system,
            temperature=temperature, max_tokens=max_tokens, grammar=grammar,
        )
        generations.append(gen3)
        if not gen3.error:
            candidate, transforms3 = salvage(gen3.text)
            value, g_errors = _validate(candidate, schema_cls)
            if value is not None:
                return RepairResult(True, RepairLevel.GRAMMAR, value, raw_text,
                                    candidate, [], transforms3, generations)
            errors = g_errors

    # -- L4: abstain -------------------------------------------------------
    return RepairResult(False, RepairLevel.ABSTAIN, None, raw_text,
                        raw_text, errors, [], generations)
