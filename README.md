# local-slm-assistant

**Small local models often know the answer but fail to return valid JSON. This project measures those two failures separately and closes the gap with engineering, not a bigger model.**

> **Status:** the benchmark harness is complete, but every number below comes from a *simulated* provider that models known failure modes. No real model has been benchmarked yet; `make bench-real` runs the same benchmark against models served by Ollama.

Structured output from small local models, with a five rung repair ladder, grammar constrained decoding, and a benchmark that keeps *conformance* and *accuracy* strictly apart.

---

## The problem

Run a 1B or 3B model locally and ask it for JSON matching a schema. It will mostly know the answer. It will also wrap it in a markdown fence, prepend "Sure! Here's the JSON:", leave a trailing comma, stringify an integer, or invent a field you did not ask for. In a pipeline, all of those are total failures — the parse throws and the answer is lost, including the parts the model got right.

The instinct is to reach for a bigger model. That is usually the wrong fix, because the two failures are unrelated:

| | what it means | fixed by |
|---|---|---|
| **conformance** | did it produce something matching the schema? | engineering |
| **accuracy** | were the values in it correct? | a better model |

Conflating them is how this kind of evaluation misleads. Salvaging a markdown fence off a well formed object with a wrong priority label yields a conformant, wrong answer — and a benchmark reporting one blended number will show repair "improving the model", which is false.

So this project measures them separately, always.

## The repair ladder

Five rungs, tried in order, each more expensive than the last. The ladder stops at the first success and records which rung it reached.

```
L0  PARSE      parse the raw output as-is                      free
L1  SALVAGE    strip fences and prose, fix commas and quotes,   free
               extract the first balanced object                (no model call)
L2  REPROMPT   hand the validation error back and ask again     one more call
L3  GRAMMAR    regenerate under a GBNF grammar derived from     one more call,
               the schema                                       slower decode
L4  ABSTAIN    return nothing rather than a guess               —
```

**Reporting the distribution of rungs reached is the whole point.** `L0 8%, L1 75%, L2 17%` tells you cheap string repair is carrying the system and a bigger model would be wasted money. An average conformance figure tells you none of that.

**Salvage never invents data.** Every L1 transform is syntactic. It will strip a fence and fix a trailing comma; it will not supply a missing required field or coerce a wrong value into a right one. If repair could invent values, the conformance metric would be measuring the repairer rather than the model — `test_salvage_never_invents_data` pins this.

## Repair after, or constrain first?

That is the real deployment decision, and the two arms have opposite cost profiles:

- **repair-after** — fast unconstrained decode, occasional second call
- **grammar-first** — slower constrained decode, but L0 succeeds every time

The GBNF grammar is generated from the Pydantic schema: key sequence emitted as literals, value types pinned, enums pinned to their exact vocabulary, optionals allowed to emit `null`. That makes a missing key, an extra key, a stringified integer and an invented enum value all *unrepresentable by the sampler* rather than merely discouraged by the prompt.

What it does not constrain is truth. A wrong value is still perfectly representable, which is exactly why the two metrics stay separate.

```bash
python -m bench.run --provider ollama --models llama3.2:3b                  # repair-after
python -m bench.run --provider ollama --models llama3.2:3b --grammar-first  # grammar-first
```

## Results

> **The numbers below come from the simulated provider and describe the harness, not any real model.** Ollama's registry and HuggingFace were both unreachable from the environment this was built in, so no real inference has been run yet. Reproduce with real models using `make bench-real` — the report labels every run with its provider so the two can never be confused.

Simulated, repair-after, 12 tasks:

| profile | conf raw | conf final | accuracy | end to end | calls | tok/s |
|---|---:|---:|---:|---:|---:|---:|
| sim-1b-q4 | 0.083 | **1.000** | 0.931 | 0.750 | 1.17 | 130 |
| sim-3b-q4 | 0.167 | **1.000** | 0.951 | 0.750 | 1.25 | 78 |
| sim-7b-q4 | 0.583 | **1.000** | 0.965 | 0.833 | 1.00 | 42 |
| sim-7b-q8 | 0.667 | **1.000** | 1.000 | 1.000 | 1.08 | 24 |

Repair level distribution — where the work actually happens:

```
sim-1b-q4    L0:1   L1:9   L2:2
sim-3b-q4    L0:2   L1:7   L2:3
sim-7b-q4    L0:7   L1:5
sim-7b-q8    L0:8   L1:3   L2:1
```

Grammar-first on the same tasks lifts raw conformance to 1.000 and removes the extra model calls, at roughly 28% of decode throughput.

**Read the accuracy column against the conformance column.** Accuracy barely moves across the range while raw conformance moves by 8x. The small models here largely know the answers; what they cannot do reliably is emit them in the requested shape. That is the case for spending effort on the output layer rather than on parameters.

## Running it

```bash
pip install -r requirements.txt
python -m pytest tests/ -q                       # 48 tests
python -m bench.run --provider simulated         # harness check, no model needed
uvicorn app.main:app --port 8000                 # UI at localhost:8000
```

Nothing above downloads a model or needs an API key. For real inference:

```bash
ollama serve
ollama pull llama3.2:3b && ollama pull mistral:7b
make bench-real
PROVIDER=ollama LOCAL_MODEL=llama3.2:3b uvicorn app.main:app
```

The UI shows the raw model output beside the validated object, which rung was reached, which repairs fired, and live TTFT and throughput. Toggle grammar-first and watch conformance and speed trade against each other.

## About the simulated provider

It is a **test fixture**, not a model. It emits a known-correct answer then corrupts it according to a per-profile probability table covering the failure modes that actually break small model output: markdown fences, prose preamble, trailing commas, single quotes, bare keys, truncation, dropped and hallucinated fields, type drift, and wrong values.

It exists so the harness, the ladder and the scorer can be proven correct — and so `pytest` and the benchmark run end to end on a clean checkout with no GPU, no model file and no network. The failure probabilities are plausible, not measured. Any figure it produces is a statement about this code.

Syntactic corruption is suppressed when a grammar is supplied, and so is structural corruption, because the generated grammar genuinely makes both unrepresentable. Semantic corruption survives a grammar, because a grammar constrains shape and not truth.

## Escalation policy

A local assistant is only useful if it knows when it is out of its depth. The policy here routes on **observed process signals** rather than a self reported confidence score — small models are reliably overconfident, and a confidence field is just another value they can get wrong.

```
reached L0 or L1     the model found the shape, formatting noise at worst   → local
reached L2 or L3     it failed and had to be corrected                      → escalate
abstained            no valid output after every rung                       → escalate
never_leave_device   hard privacy switch; a failed local answer refuses     → refuse
```

`local_share` is tracked as a counter, so "most of this stayed on device" is a measurement rather than a claim.

## Limitations

**No real model numbers yet.** The headline table is simulated. This is the single thing to fix first, and `make bench-real` does it.

**The task set is 12 items.** Enough to exercise every schema shape, not enough for tight confidence intervals. Differences under roughly 10 points between adjacent profiles are noise.

**List fields are scored by length, not content.** Comparing generated action items by string equality would measure phrasing. Count is the defensible proxy and the report flags it, but it is a proxy.

**Ollama exposes JSON mode, not arbitrary GBNF.** `OllamaProvider.supports_grammar` therefore returns `False`, so the L3 rung is skipped there rather than silently recording a constraint that was never enforced. JSON mode is still used, and it guarantees valid JSON but not schema conformance. For true grammar constrained decoding, llama.cpp with `--grammar` is the backend to add next.

## What I would build next

A llama.cpp provider, for real GBNF support and a genuine grammar-first arm. A cost model putting local throughput against API pricing so the escalation threshold can be tuned against a budget rather than guessed. And a quantisation sweep on one model family — Q4 through Q8 on identical tasks — to find where the format adherence cliff actually sits, which is the question the profiles here are only gesturing at.

## Layout

```
app/
  providers/base.py       Provider interface, Generation with TTFT and throughput
  providers/ollama.py     real backend, NDJSON streaming, server-side timings
  providers/simulated.py  deterministic failure-mode fixture
  repair.py               the five rung ladder, salvage transforms, GBNF generation
  assistant.py            local-first assistant, explicit escalation policy
  schemas/tasks.py        Pydantic schemas built to stress enums, ints, arrays, optionals
  main.py                 FastAPI, single deployable
bench/
  tasks.py                12 tasks with hand-written ground truth
  metrics.py              conformance and accuracy, kept apart
  run.py                  runner, comparison report, CI smoke
static/index.html         raw output vs validated object, with the ladder visualised
tests/                    48 tests
```
