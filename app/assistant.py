"""
The assistant, and the decision that makes it worth running locally.

A local assistant is only interesting if it knows when it is out of its depth.
Running everything on a 3B model produces confident nonsense; escalating
everything to an API defeats the privacy and cost argument entirely. The value
is in the routing rule between those.

The rule here is deliberately built on **observed process signals**, not on a
self-reported confidence score. Asking a small model how confident it is
produces a number with almost no relationship to whether it was right -- small
models are reliably overconfident, and the field is just another value they can
get wrong. What does carry signal is how much work it took to get a valid
answer:

    reached L0 clean          the model found the shape unaided        trust
    needed L1 salvage         formatting noise only                    trust
    needed L2 re-prompt       it failed once and had to be corrected   suspect
    needed L3 grammar         it could not hold the shape at all       suspect
    abstained                 no valid output after every rung         escalate

Escalation policy is explicit and inspectable rather than a tuned threshold,
because the person deploying this needs to be able to state what leaves the
machine, and "the classifier decided" is not an answer to that question.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Type

from pydantic import BaseModel

from app.providers.base import Provider
from app.repair import RepairLevel, RepairResult, structured_generate


class Disposition(str, Enum):
    LOCAL = "local"                  # answered on device, nothing left the machine
    ESCALATED = "escalated"          # sent to a stronger model
    REFUSED = "refused"              # no valid answer and escalation unavailable


@dataclass
class Policy:
    """When to escalate. Every field is a stated rule, not a learned threshold."""

    # Escalate when the ladder had to climb past this rung.
    escalate_above_level: RepairLevel = RepairLevel.SALVAGE

    # Escalate on abstention. Off means refuse instead -- the right setting when
    # the whole point of running locally is that data must not leave.
    escalate_on_abstain: bool = True

    # Hard privacy switch. When set, nothing escalates, ever, regardless of the
    # rules above: a failed local answer becomes a refusal.
    never_leave_device: bool = False

    def decide(self, result: RepairResult) -> Disposition:
        if result.ok and result.level <= self.escalate_above_level:
            return Disposition.LOCAL
        if self.never_leave_device:
            return Disposition.REFUSED
        if not result.ok:
            return Disposition.ESCALATED if self.escalate_on_abstain else Disposition.REFUSED
        return Disposition.ESCALATED


@dataclass
class AssistantResult:
    disposition: Disposition
    value: BaseModel | None
    local: RepairResult
    escalated: RepairResult | None = None
    reason: str = ""

    @property
    def stayed_local(self) -> bool:
        return self.disposition is Disposition.LOCAL

    @property
    def total_ms(self) -> float:
        t = self.local.total_ms
        if self.escalated:
            t += self.escalated.total_ms
        return t

    def to_dict(self) -> dict[str, Any]:
        return {
            "disposition": self.disposition.value,
            "stayed_local": self.stayed_local,
            "reason": self.reason,
            "value": self.value.model_dump() if self.value else None,
            "total_ms": round(self.total_ms, 1),
            "local": self.local.to_dict(),
            "escalated": self.escalated.to_dict() if self.escalated else None,
        }


class Assistant:
    """Local-first structured assistant with an explicit escalation policy."""

    def __init__(
        self,
        local_provider: Provider,
        local_model: str,
        *,
        escalation_provider: Provider | None = None,
        escalation_model: str | None = None,
        policy: Policy | None = None,
        grammar_first: bool = False,
    ):
        self.local_provider = local_provider
        self.local_model = local_model
        self.escalation_provider = escalation_provider
        self.escalation_model = escalation_model
        self.policy = policy or Policy()
        self.grammar_first = grammar_first

        # Counters, so the privacy claim is a measurement rather than a promise.
        self.stats: dict[str, int] = {"local": 0, "escalated": 0, "refused": 0}

    def run(
        self,
        prompt: str,
        schema_cls: Type[BaseModel],
        *,
        system: str | None = None,
        max_tokens: int = 512,
    ) -> AssistantResult:
        local = structured_generate(
            self.local_provider, prompt, schema_cls,
            model=self.local_model, system=system,
            max_tokens=max_tokens, grammar_first=self.grammar_first,
        )

        disposition = self.policy.decide(local)

        if disposition is Disposition.LOCAL:
            self.stats["local"] += 1
            return AssistantResult(
                disposition, local.value, local,
                reason=f"handled on device ({local.level.label})",
            )

        can_escalate = (
            self.escalation_provider is not None
            and self.escalation_model is not None
            and self.escalation_provider.available()
        )

        if disposition is Disposition.REFUSED or not can_escalate:
            self.stats["refused"] += 1
            reason = (
                "refused: never_leave_device is set and the local model could not "
                "produce a valid answer"
                if self.policy.never_leave_device
                else f"refused: escalation needed ({local.level.label}) but no "
                     f"escalation backend is configured"
            )
            return AssistantResult(Disposition.REFUSED, None, local, reason=reason)

        escalated = structured_generate(
            self.escalation_provider, prompt, schema_cls,
            model=self.escalation_model, system=system, max_tokens=max_tokens,
        )
        self.stats["escalated"] += 1
        return AssistantResult(
            Disposition.ESCALATED,
            escalated.value if escalated.ok else None,
            local, escalated,
            reason=f"escalated: local reached {local.level.label}",
        )

    @property
    def local_share(self) -> float:
        """Fraction handled on device. The privacy and cost argument, measured."""
        total = sum(self.stats.values())
        return self.stats["local"] / total if total else 0.0
