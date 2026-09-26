"""
FastAPI app. Single deployable: API on /api/*, UI on /.

Defaults to the simulated provider so `uvicorn app.main:app` works on a clean
checkout with nothing installed and no model pulled. Point it at a real model
with:

    PROVIDER=ollama LOCAL_MODEL=llama3.2:3b uvicorn app.main:app
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.assistant import Assistant, Policy
from app.providers.ollama import OllamaProvider
from app.providers.simulated import PROFILES, SimulatedProvider
from app.repair import RepairLevel, gbnf_from_schema
from app.schemas.tasks import SCHEMAS
from bench.tasks import TASKS, oracle_fn

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "static"

state: dict[str, Any] = {}


def _build_provider():
    kind = os.environ.get("PROVIDER", "simulated")
    if kind == "ollama":
        p = OllamaProvider()
        if p.available():
            return p, os.environ.get("LOCAL_MODEL", "llama3.2:3b")
        # Fall back rather than serving 503s, and say so on /api/health so the
        # UI never implies a real model produced these numbers.
    profile = os.environ.get("SIM_PROFILE", "sim-3b-q4")
    return SimulatedProvider(profile=profile, oracle=oracle_fn()), profile


@asynccontextmanager
async def lifespan(app: FastAPI):
    provider, model = _build_provider()
    state["provider"] = provider
    state["model"] = model
    state["assistant"] = Assistant(
        provider, model,
        policy=Policy(escalate_above_level=RepairLevel.SALVAGE),
        grammar_first=os.environ.get("GRAMMAR_FIRST", "").lower() in {"1", "true", "yes"},
    )
    yield
    state.clear()


app = FastAPI(
    title="Local SLM Assistant",
    description="Structured output from small local models, with a repair ladder "
                "and an explicit escalation policy.",
    version="1.0.0",
    lifespan=lifespan,
)


class RunRequest(BaseModel):
    prompt: str
    schema_name: str
    grammar_first: bool | None = None


def _assistant() -> Assistant:
    a = state.get("assistant")
    if a is None:
        raise HTTPException(503, "assistant not initialised")
    return a


@app.get("/api/health")
def health() -> dict[str, Any]:
    prov = state.get("provider")
    if prov is None:
        return {"status": "starting"}
    return {
        "status": "ok",
        "provider": prov.name,
        "model": state.get("model"),
        "supports_grammar": prov.supports_grammar,
        "simulated": prov.name == "simulated",
        "note": (
            "Simulated provider: outputs come from a deterministic failure-mode "
            "fixture, not a real model. Set PROVIDER=ollama for real inference."
        ) if prov.name == "simulated" else None,
        "available_models": [m.name for m in prov.list_models()],
        "schemas": list(SCHEMAS),
    }


@app.post("/api/run")
def run(req: RunRequest) -> dict[str, Any]:
    """Run one task and return the full repair trace, not just the answer."""
    a = _assistant()
    schema_cls = SCHEMAS.get(req.schema_name)
    if schema_cls is None:
        raise HTTPException(400, f"unknown schema '{req.schema_name}'. "
                                 f"Options: {list(SCHEMAS)}")
    if not req.prompt.strip():
        raise HTTPException(400, "empty prompt")

    if req.grammar_first is not None:
        a.grammar_first = req.grammar_first

    result = a.run(req.prompt, schema_cls)
    out = result.to_dict()
    out["schema"] = req.schema_name
    out["grammar_first"] = a.grammar_first
    out["local_share"] = round(a.local_share, 3)
    return out


@app.get("/api/grammar/{schema_name}")
def grammar(schema_name: str) -> dict[str, str]:
    schema_cls = SCHEMAS.get(schema_name)
    if schema_cls is None:
        raise HTTPException(404, f"unknown schema '{schema_name}'")
    return {"schema": schema_name, "gbnf": gbnf_from_schema(schema_cls)}


@app.get("/api/examples")
def examples() -> list[dict[str, str]]:
    return [
        {
            "id": t.id,
            "schema": t.schema_name,
            "label": f"{t.schema_name.replace('_', ' ')} -- {t.id}",
            "prompt": t.prompt,
        }
        for t in TASKS
    ]


@app.get("/api/report")
def report() -> JSONResponse:
    p = ROOT / "bench_report.json"
    if not p.exists():
        raise HTTPException(404, "no benchmark report; run `python -m bench.run` first")
    import json
    return JSONResponse(content=json.loads(p.read_text(encoding="utf-8")))


@app.get("/api/stats")
def stats() -> dict[str, Any]:
    a = _assistant()
    return {"counts": a.stats, "local_share": round(a.local_share, 3)}


if STATIC.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(str(STATIC / "index.html"))
