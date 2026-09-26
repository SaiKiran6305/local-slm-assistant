"""
Deterministic failure-mode simulator.

WHAT THIS IS FOR, precisely: it is a test fixture for the repair ladder and the
benchmark harness. It lets `pytest` and `python -m bench.run` execute end to end
on a machine with no Ollama, no GPU and no model files, so the harness can be
proven correct before it is pointed at a real model.

WHAT IT IS NOT: a source of benchmark numbers. The failure probabilities below
are plausible, not measured. Any figure produced with this provider is a
statement about the harness, never about llama3 or mistral. The benchmark report
labels every run with its provider for exactly this reason, and the README says
so in the results section.

The failure modes reproduced here are the ones that actually break structured
output from small models in practice:

    markdown fences        ```json { ... } ```
    prose preamble         "Sure! Here is the JSON you requested:"
    trailing commas        {"a": 1,}
    single quotes          {'a': 1}
    unquoted keys          {a: 1}
    explanatory suffix      { ... }  "Let me know if you need anything else!"
    hallucinated fields    extra keys not in the schema
    dropped fields         a required key simply missing
    type drift             "5" where the schema says int
    truncation             output stops mid-object at the token limit

Syntactic failures are repairable. Semantic failures -- a wrong value in a
well-formed object -- are not, by any amount of parsing. The simulator models
both, separately, because conflating them is how a benchmark ends up claiming
repair fixed accuracy when all it fixed was punctuation.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from app.providers.base import Generation, ModelInfo, Provider


@dataclass
class FailureProfile:
    """Per-model likelihood of each failure mode. All values are probabilities."""
    name: str

    # Syntactic -- recoverable by the repair ladder.
    markdown_fence: float = 0.0
    prose_preamble: float = 0.0
    explanatory_suffix: float = 0.0
    trailing_comma: float = 0.0
    single_quotes: float = 0.0
    unquoted_keys: float = 0.0
    truncation: float = 0.0

    # Structural -- recoverable only by re-prompting, not by parsing.
    hallucinated_field: float = 0.0
    dropped_field: float = 0.0
    type_drift: float = 0.0

    # Semantic -- not recoverable at all. A wrong value, correctly formatted.
    wrong_value: float = 0.0

    # Rough throughput characteristics, for plausible timing in the harness.
    ttft_ms_mean: float = 180.0
    tokens_per_second: float = 45.0

    # Constrained decoding is not free. Checking every candidate token against
    # the grammar costs throughput, and that cost is the whole reason
    # grammar-first is not automatically the right default. Modelled as a
    # multiplier on tokens_per_second when a grammar is supplied.
    grammar_throughput_penalty: float = 0.72


# Presets loosely shaped like the models this project targets. Ordering
# reflects the consistent finding that structured-output reliability tracks
# parameter count more strongly than it tracks benchmark scores, and that
# aggressive quantisation costs format adherence before it costs knowledge.
PROFILES: dict[str, FailureProfile] = {
    "sim-3b-q4": FailureProfile(
        name="sim-3b-q4",
        markdown_fence=0.42, prose_preamble=0.30, explanatory_suffix=0.18,
        trailing_comma=0.12, single_quotes=0.08, unquoted_keys=0.05,
        truncation=0.04,
        hallucinated_field=0.14, dropped_field=0.10, type_drift=0.12,
        wrong_value=0.17,
        ttft_ms_mean=120.0, tokens_per_second=78.0,
    ),
    "sim-7b-q4": FailureProfile(
        name="sim-7b-q4",
        markdown_fence=0.28, prose_preamble=0.16, explanatory_suffix=0.09,
        trailing_comma=0.06, single_quotes=0.03, unquoted_keys=0.01,
        truncation=0.02,
        hallucinated_field=0.07, dropped_field=0.04, type_drift=0.05,
        wrong_value=0.09,
        ttft_ms_mean=210.0, tokens_per_second=42.0,
    ),
    "sim-7b-q8": FailureProfile(
        name="sim-7b-q8",
        markdown_fence=0.21, prose_preamble=0.11, explanatory_suffix=0.06,
        trailing_comma=0.03, single_quotes=0.01, unquoted_keys=0.0,
        truncation=0.01,
        hallucinated_field=0.04, dropped_field=0.02, type_drift=0.03,
        wrong_value=0.06,
        ttft_ms_mean=260.0, tokens_per_second=24.0,
    ),
    "sim-1b-q4": FailureProfile(
        name="sim-1b-q4",
        markdown_fence=0.55, prose_preamble=0.44, explanatory_suffix=0.26,
        trailing_comma=0.18, single_quotes=0.14, unquoted_keys=0.09,
        truncation=0.07,
        hallucinated_field=0.22, dropped_field=0.18, type_drift=0.20,
        wrong_value=0.31,
        ttft_ms_mean=70.0, tokens_per_second=130.0,
    ),
    "sim-clean": FailureProfile(name="sim-clean", ttft_ms_mean=200.0, tokens_per_second=50.0),
}


class SimulatedProvider(Provider):
    """Emits a known-correct answer, then corrupts it according to a profile.

    Determinism is keyed on (seed, prompt, model, attempt) so a re-prompt after
    a failure produces a *different* draw rather than the identical broken
    output -- otherwise the re-prompt rung of the repair ladder could never
    succeed, and the harness would under-report what real re-prompting achieves.
    """

    name = "simulated"

    def __init__(
        self,
        profile: str | FailureProfile = "sim-7b-q4",
        seed: int = 17,
        oracle: Callable[[str], dict[str, Any]] | None = None,
        realtime: bool = False,
    ):
        self.profile = PROFILES[profile] if isinstance(profile, str) else profile
        self.seed = seed
        self.oracle = oracle
        self.realtime = realtime          # actually sleep, for UI demos
        self._attempts: dict[str, int] = {}

    @property
    def supports_grammar(self) -> bool:
        return True

    def available(self) -> bool:
        return True

    def list_models(self) -> list[ModelInfo]:
        return [
            ModelInfo(name=n, parameter_count=n.split("-")[1].upper() if "-" in n else None,
                      quantization=(n.split("-")[2].upper() if n.count("-") >= 2 else None))
            for n in PROFILES
        ]

    # -- generation --------------------------------------------------------
    def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        grammar: str | None = None,
        stop: list[str] | None = None,
        **_,
    ) -> Generation:
        profile = PROFILES.get(model, self.profile) if model else self.profile

        key = hashlib.sha1(f"{prompt}|{model}".encode()).hexdigest()
        attempt = self._attempts.get(key, 0)
        self._attempts[key] = attempt + 1
        rng = random.Random(f"{self.seed}|{key}|{attempt}")

        payload = self.oracle(prompt) if self.oracle else {"answer": "simulated response"}
        payload = dict(payload)

        # --- semantic corruption: a wrong value, correctly formatted --------
        if rng.random() < profile.wrong_value:
            payload = _corrupt_value(payload, rng)

        # --- structural corruption: schema-shaped but wrong -----------------
        # Suppressed under a grammar, and this is a fidelity point rather than a
        # convenience: the GBNF we generate emits the key sequence as literals
        # and pins each value's type, so an extra key, a missing key and a
        # stringified integer are all unrepresentable by the sampler. Letting
        # them through here would understate what constrained decoding buys and
        # make the grammar-first arm look worse than it is.
        if grammar is None:
            if rng.random() < profile.hallucinated_field:
                payload["confidence_note"] = "high"
            if rng.random() < profile.dropped_field and len(payload) > 1:
                payload.pop(rng.choice(list(payload.keys())))
            if rng.random() < profile.type_drift:
                payload = _drift_types(payload, rng)

        text = json.dumps(payload, indent=2)

        # --- syntactic corruption -------------------------------------------
        # Skipped under a grammar for the same reason as the structural block
        # above: malformed syntax is unrepresentable once the sampler is
        # constrained. Semantic corruption is the only kind that survives a
        # grammar, because a grammar constrains shape and not truth -- which is
        # exactly why the benchmark reports conformance and accuracy apart.
        if grammar is None:
            if rng.random() < profile.single_quotes:
                text = text.replace('"', "'")
            if rng.random() < profile.unquoted_keys:
                text = _unquote_keys(text)
            if rng.random() < profile.trailing_comma:
                text = _add_trailing_comma(text, rng)
            if rng.random() < profile.markdown_fence:
                text = f"```json\n{text}\n```"
            if rng.random() < profile.prose_preamble:
                text = rng.choice([
                    "Sure! Here is the JSON you requested:\n\n",
                    "Here's the structured output:\n\n",
                    "Based on the text provided, here is the extraction:\n\n",
                ]) + text
            if rng.random() < profile.explanatory_suffix:
                text += rng.choice([
                    "\n\nLet me know if you need anything else!",
                    "\n\nI hope this helps.",
                    "\n\nNote: this is based on the information given.",
                ])
            if rng.random() < profile.truncation:
                text = text[: int(len(text) * rng.uniform(0.55, 0.85))]

        completion_tokens = max(1, len(text) // 4)
        # Constrained sampling costs throughput; that trade is the reason
        # grammar-first is a measurable decision rather than a free win.
        tps = profile.tokens_per_second * (
            profile.grammar_throughput_penalty if grammar is not None else 1.0
        )
        decode_ms = (completion_tokens / max(tps, 1e-6)) * 1000
        ttft = profile.ttft_ms_mean * rng.uniform(0.75, 1.35)

        if self.realtime:
            time.sleep(min((ttft + decode_ms) / 1000.0, 2.0))

        return Generation(
            text=text,
            model=profile.name,
            prompt_tokens=max(1, len(prompt) // 4),
            completion_tokens=completion_tokens,
            ttft_ms=ttft,
            total_ms=ttft + decode_ms,
            stop_reason="length" if len(text) >= max_tokens * 4 else "stop",
            raw={"simulated": True, "profile": profile.name, "attempt": attempt},
        )


# --- corruption helpers ---------------------------------------------------

def _corrupt_value(payload: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """Change one value to something wrong but plausibly typed."""
    out = dict(payload)
    keys = [k for k, v in out.items() if isinstance(v, (str, int, float, bool))]
    if not keys:
        return out
    k = rng.choice(keys)
    v = out[k]
    if isinstance(v, bool):
        out[k] = not v
    elif isinstance(v, (int, float)):
        out[k] = type(v)(v * rng.choice([0.5, 2, 10]) + rng.choice([-1, 1]))
    elif isinstance(v, str):
        out[k] = rng.choice(["unknown", "not specified", v[::-1] if len(v) > 3 else "n/a"])
    return out


def _drift_types(payload: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """Stringify a number, or stringify a bool. The classic small-model tic."""
    out = dict(payload)
    for k, v in list(out.items()):
        if isinstance(v, bool) and rng.random() < 0.5:
            out[k] = "true" if v else "false"
        elif isinstance(v, (int, float)) and rng.random() < 0.5:
            out[k] = str(v)
    return out


def _unquote_keys(text: str) -> str:
    import re
    return re.sub(r'"(\w+)":', r"\1:", text)


def _add_trailing_comma(text: str, rng: random.Random) -> str:
    idx = text.rfind("}")
    if idx <= 0:
        return text
    return text[:idx].rstrip().rstrip(",") + ",\n" + text[idx:]
