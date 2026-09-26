"""
Benchmark runner.

    # simulated: proves the harness, needs nothing installed
    python -m bench.run --provider simulated

    # real: needs ollama running with the models pulled
    python -m bench.run --provider ollama --models llama3.2:3b,mistral:7b,llama3.1:8b

    # isolate what repair is worth
    python -m bench.run --provider simulated --max-level 0   # no repair at all
    python -m bench.run --provider simulated --max-level 1   # salvage only

Every report records which provider produced it. Numbers from the simulated
provider describe this harness and nothing else; only `--provider ollama`
produces a statement about a model.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

from app.providers.base import Provider
from app.providers.ollama import OllamaProvider
from app.providers.simulated import PROFILES, SimulatedProvider
from app.repair import RepairLevel, structured_generate
from bench.metrics import aggregate, score_task
from bench.tasks import TASKS, oracle_fn


def build_provider(kind: str, seed: int, realtime: bool = False) -> Provider:
    if kind == "ollama":
        return OllamaProvider()
    return SimulatedProvider(seed=seed, oracle=oracle_fn(), realtime=realtime)


def run_model(provider: Provider, model: str, max_level: RepairLevel,
              verbose: bool = False, grammar_first: bool = False) -> dict[str, Any]:
    scores = []
    for task in TASKS:
        result = structured_generate(
            provider, task.prompt, task.schema_cls,
            model=model, max_level=max_level, grammar_first=grammar_first,
        )
        s = score_task(task, result)
        scores.append(s)
        if verbose:
            mark = "ok " if s.fully_correct else ("~  " if s.conformant else "X  ")
            print(f"    {mark} {task.id:12} {s.level.label:24} "
                  f"{s.fields_correct}/{s.fields_total} fields")
    return aggregate(scores, model=model, provider=provider.name,
                     grammar_first=grammar_first)


def print_report(reports: list[dict[str, Any]], max_level: RepairLevel) -> None:
    print(f"\n{'=' * 86}")
    print("  LOCAL SLM STRUCTURED OUTPUT BENCHMARK")
    print(f"{'=' * 86}")
    prov = reports[0]["provider"] if reports else "?"
    strategy = "grammar-first" if reports[0].get("grammar_first") else "repair-after"
    print(f"  provider: {prov}   strategy: {strategy}   "
          f"repair ceiling: {max_level.label}   tasks: {reports[0]['n_tasks']}")
    if prov == "simulated":
        print("  NOTE: simulated provider. These numbers describe the harness, not any model.")
    print()

    hdr = (f"  {'model':<22} {'conf raw':>9} {'conf fin':>9} {'accuracy':>9} "
           f"{'end2end':>9} {'calls':>6} {'ttft p50':>9} {'tok/s':>7}")
    print(hdr)
    print(f"  {'-' * 84}")
    for r in reports:
        print(f"  {r['model']:<22} {r['conformance_raw']:>9.3f} {r['conformance_final']:>9.3f} "
              f"{r['accuracy']:>9.3f} {r['end_to_end']:>9.3f} {r['avg_model_calls']:>6.2f} "
              f"{r['ttft_ms']['p50']:>8.0f}m {r['tokens_per_second']:>7.1f}")

    print()
    print("  repair level distribution")
    print(f"  {'-' * 84}")
    for r in reports:
        dist = "  ".join(f"{k.split()[0]}:{v}" for k, v in r["repair_levels"].items() if v)
        print(f"  {r['model']:<22} {dist}")

    print()
    print("  conformance gained by repair")
    print(f"  {'-' * 84}")
    for r in reports:
        gain = r["conformance_final"] - r["conformance_raw"]
        bar = "#" * int(gain * 40)
        print(f"  {r['model']:<22} {r['conformance_raw']:.0%} -> {r['conformance_final']:.0%} "
              f"(+{gain:.0%}) {bar}")

    # The point of separating the two metrics.
    print()
    print("  accuracy is unchanged by repair, by construction: salvage fixes syntax,")
    print("  never values. Any model below shows conformant-but-wrong answers here:")
    print(f"  {'-' * 84}")
    for r in reports:
        wrong = r["wrong_but_conformant"]
        if wrong:
            fields = sorted({f for w in wrong for f in w["fields"]})
            print(f"  {r['model']:<22} {len(wrong)} conformant but wrong  fields: {', '.join(fields[:6])}")
        else:
            print(f"  {r['model']:<22} none")
    print()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--provider", choices=["simulated", "ollama"], default="simulated")
    p.add_argument("--models", default=None,
                   help="comma separated. defaults to all simulated profiles, "
                        "or every model ollama has pulled")
    p.add_argument("--max-level", type=int, default=3,
                   help="repair ceiling: 0 parse, 1 salvage, 2 reprompt, 3 grammar")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--out", type=Path, default=Path("bench_report.json"))
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--grammar-first", action="store_true",
                   help="constrain the first call instead of repairing after")
    a = p.parse_args()

    provider = build_provider(a.provider, a.seed)

    if not provider.available():
        print(f"provider '{a.provider}' is not available.", file=sys.stderr)
        if a.provider == "ollama":
            print("Start it with `ollama serve`, then pull a model: "
                  "`ollama pull llama3.2:3b`", file=sys.stderr)
        raise SystemExit(1)

    if a.models:
        models = [m.strip() for m in a.models.split(",") if m.strip()]
    elif a.provider == "simulated":
        models = [m for m in PROFILES if m != "sim-clean"]
    else:
        models = [m.name for m in provider.list_models()]
        if not models:
            print("ollama has no models pulled. Try `ollama pull llama3.2:3b`.", file=sys.stderr)
            raise SystemExit(1)

    max_level = RepairLevel(a.max_level)
    reports = []
    for m in models:
        print(f"  running {m} ...")
        reports.append(run_model(provider, m, max_level, verbose=a.verbose,
                                 grammar_first=a.grammar_first))

    print_report(reports, max_level)

    payload = {
        "provider": provider.name,
        "strategy": "grammar-first" if a.grammar_first else "repair-after",
        "max_repair_level": int(max_level),
        "max_repair_label": max_level.label,
        "seed": a.seed,
        "simulated_warning": (
            "Simulated provider: these figures validate the harness and do not "
            "describe any real model."
        ) if provider.name == "simulated" else None,
        "reports": reports,
    }
    a.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  full report -> {a.out}\n")


if __name__ == "__main__":
    main()
