"""
Provider interface for local model backends.

Everything downstream -- the repair ladder, the router, the benchmark harness --
talks to this interface and nothing else. Swapping Ollama for llama.cpp, for an
OpenAI-compatible endpoint, or for the simulator changes one constructor call
and nothing else.

`Generation` carries the timing fields that make a local-model project worth
doing at all. Total latency alone hides the thing that actually determines
whether a local assistant feels usable: time to first token. A model with worse
total latency but 200ms TTFT feels faster than one that sits silent for two
seconds and then dumps its answer. Both are recorded separately.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any, Iterator


@dataclass
class Generation:
    """One completion, with the measurements a deployment decision needs."""
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0

    # Timing, all milliseconds.
    ttft_ms: float = 0.0          # prompt submitted -> first token emitted
    total_ms: float = 0.0         # prompt submitted -> generation complete

    stop_reason: str = "stop"     # stop | length | error
    error: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def tokens_per_second(self) -> float:
        """Decode throughput, excluding prompt processing.

        Measured over the generation phase only. Including TTFT here would
        conflate prompt processing with decode speed and make long-prompt runs
        look like slow models.
        """
        decode_ms = self.total_ms - self.ttft_ms
        if decode_ms <= 0 or self.completion_tokens <= 0:
            return 0.0
        return self.completion_tokens / (decode_ms / 1000.0)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("raw", None)
        d["tokens_per_second"] = round(self.tokens_per_second, 2)
        d["ttft_ms"] = round(self.ttft_ms, 1)
        d["total_ms"] = round(self.total_ms, 1)
        return d


@dataclass
class ModelInfo:
    name: str
    parameter_count: str | None = None     # "7B"
    quantization: str | None = None        # "Q4_K_M"
    size_bytes: int | None = None
    context_length: int | None = None

    @property
    def size_gb(self) -> float | None:
        return round(self.size_bytes / 1e9, 2) if self.size_bytes else None


class Provider(ABC):
    """A local (or local-compatible) text generation backend."""

    name: str = "base"

    @abstractmethod
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
    ) -> Generation:
        """Run one completion.

        `grammar` is a GBNF grammar. Backends that support constrained decoding
        use it to make invalid output unrepresentable rather than merely
        discouraged; backends that do not must ignore it and say so via
        `supports_grammar`, so the repair ladder knows to skip that rung instead
        of silently believing it applied a constraint it did not.
        """

    @abstractmethod
    def list_models(self) -> list[ModelInfo]:
        ...

    @abstractmethod
    def available(self) -> bool:
        """Whether this backend can currently serve requests."""

    @property
    def supports_grammar(self) -> bool:
        return False

    def stream(self, prompt: str, **kwargs) -> Iterator[str]:
        """Token stream. Default falls back to a single non-streamed chunk."""
        yield self.generate(prompt, **kwargs).text


class Timer:
    """Wall-clock helper that records first-token time separately."""

    def __init__(self) -> None:
        self.start = time.perf_counter()
        self.first_token: float | None = None

    def mark_first_token(self) -> None:
        if self.first_token is None:
            self.first_token = time.perf_counter()

    @property
    def ttft_ms(self) -> float:
        if self.first_token is None:
            return 0.0
        return (self.first_token - self.start) * 1000

    @property
    def total_ms(self) -> float:
        return (time.perf_counter() - self.start) * 1000
