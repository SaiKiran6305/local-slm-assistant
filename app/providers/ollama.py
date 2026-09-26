"""
Ollama backend. The real path.

Streams by default, because TTFT cannot be measured any other way: a
non-streaming call gives you one timestamp at the end, which tells you nothing
about whether the assistant felt responsive. Ollama's NDJSON stream lets us
stamp the first token as it arrives.

Ollama reports its own timings in nanoseconds on the final chunk
(`prompt_eval_duration`, `eval_duration`). Those are preferred over wall clock
where present, since they exclude HTTP and client overhead, and wall clock is
kept as the fallback.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Iterator

from app.providers.base import Generation, ModelInfo, Provider, Timer

DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


class OllamaProvider(Provider):
    name = "ollama"

    def __init__(self, host: str = DEFAULT_HOST, timeout: float = 300.0):
        self.host = host.rstrip("/")
        self.timeout = timeout

    # -- capability --------------------------------------------------------
    @property
    def supports_grammar(self) -> bool:
        """Ollama exposes JSON mode, not arbitrary GBNF.

        Reported as False deliberately. `format: json` constrains output to
        *some* valid JSON, not to our schema, so it cannot stand in for the
        grammar rung of the repair ladder. Claiming otherwise would let the
        ladder record a constraint it never actually applied. JSON mode is
        still used below -- it is simply not the same guarantee.
        """
        return False

    def available(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=3) as r:
                return r.status == 200
        except Exception:
            return False

    def list_models(self) -> list[ModelInfo]:
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=10) as r:
                data = json.loads(r.read())
        except Exception:
            return []

        out: list[ModelInfo] = []
        for m in data.get("models", []):
            details = m.get("details", {}) or {}
            out.append(ModelInfo(
                name=m.get("name", "?"),
                parameter_count=details.get("parameter_size"),
                quantization=details.get("quantization_level"),
                size_bytes=m.get("size"),
                context_length=details.get("context_length"),
            ))
        return out

    # -- generation --------------------------------------------------------
    def generate(
        self,
        prompt: str,
        *,
        model: str,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        grammar: str | None = None,
        stop: list[str] | None = None,
        json_mode: bool = False,
    ) -> Generation:
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if system:
            payload["system"] = system
        if stop:
            payload["options"]["stop"] = stop
        if json_mode:
            # Constrains to syntactically valid JSON. Not schema conformance --
            # the model can still emit well-formed JSON with wrong fields.
            payload["format"] = "json"

        timer = Timer()
        chunks: list[str] = []
        final: dict[str, Any] = {}

        req = urllib.request.Request(
            f"{self.host}/api/generate",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                for raw_line in resp:
                    line = raw_line.decode().strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    piece = obj.get("response", "")
                    if piece:
                        timer.mark_first_token()
                        chunks.append(piece)
                    if obj.get("done"):
                        final = obj
                        break
        except urllib.error.URLError as e:
            return Generation(
                text="", model=model, stop_reason="error",
                error=f"ollama unreachable at {self.host}: {e}",
                total_ms=timer.total_ms,
            )
        except Exception as e:  # noqa: BLE001
            return Generation(
                text="", model=model, stop_reason="error",
                error=f"{type(e).__name__}: {e}", total_ms=timer.total_ms,
            )

        # Server-side nanosecond timings beat wall clock: they exclude HTTP and
        # client-loop overhead. Fall back to wall clock when absent.
        prompt_ns = final.get("prompt_eval_duration") or 0
        eval_ns = final.get("eval_duration") or 0
        ttft_ms = (prompt_ns / 1e6) if prompt_ns else timer.ttft_ms
        total_ms = ((prompt_ns + eval_ns) / 1e6) if (prompt_ns and eval_ns) else timer.total_ms

        return Generation(
            text="".join(chunks),
            model=model,
            prompt_tokens=final.get("prompt_eval_count", 0),
            completion_tokens=final.get("eval_count", 0),
            ttft_ms=ttft_ms,
            total_ms=total_ms,
            stop_reason="length" if final.get("done_reason") == "length" else "stop",
            raw=final,
        )

    def stream(self, prompt: str, *, model: str, system: str | None = None,
               temperature: float = 0.0, max_tokens: int = 512, **_) -> Iterator[str]:
        payload: dict[str, Any] = {
            "model": model, "prompt": prompt, "stream": True,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if system:
            payload["system"] = system

        req = urllib.request.Request(
            f"{self.host}/api/generate",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            for raw_line in resp:
                line = raw_line.decode().strip()
                if not line:
                    continue
                obj = json.loads(line)
                if obj.get("response"):
                    yield obj["response"]
                if obj.get("done"):
                    return
