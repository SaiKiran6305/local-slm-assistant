"""
Benchmark task set with hand-written ground truth.

Every expected value here was written by reading the input text and deciding
the answer, not by running a model and accepting what it said. That distinction
is the difference between a benchmark and a regression fixture.

Some items are deliberately ambiguous at the margin (is a slow dashboard
"high" or "medium" priority?). Those are marked `tolerant`, and the scorer
accepts any value in `acceptable` for that field. Forcing a single right answer
on a genuinely debatable label measures agreement with the author's taste, not
model capability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Type

from pydantic import BaseModel

from app.schemas.tasks import (
    EmailIntent, InvoiceExtract, MeetingNotes, TicketTriage,
)


@dataclass
class BenchTask:
    id: str
    schema_name: str
    schema_cls: Type[BaseModel]
    instruction: str
    input_text: str
    expected: dict[str, Any]
    # field -> set of acceptable values, for genuinely debatable labels
    acceptable: dict[str, list[Any]] = field(default_factory=dict)

    @property
    def prompt(self) -> str:
        return f"{self.instruction}\n\n---\n{self.input_text}\n---"


TASKS: list[BenchTask] = [

    # ---- ticket triage ---------------------------------------------------
    BenchTask(
        id="triage-01",
        schema_name="ticket_triage",
        schema_cls=TicketTriage,
        instruction="Triage this customer support ticket.",
        input_text=(
            "Subject: Charged twice for October\n\n"
            "I was billed $49 twice on October 3rd. My bank shows both charges "
            "cleared. I need one refunded. This is the second time this has "
            "happened and I am losing patience."
        ),
        expected={
            "category": "billing",
            "priority": "high",
            "sentiment": "negative",
            "requires_human": True,
        },
        acceptable={"priority": ["high", "urgent"]},
    ),
    BenchTask(
        id="triage-02",
        schema_name="ticket_triage",
        schema_cls=TicketTriage,
        instruction="Triage this customer support ticket.",
        input_text=(
            "Subject: How do I export my data?\n\n"
            "Hi, quick question -- is there a way to export my reports to CSV? "
            "I looked in settings but could not find it. Thanks!"
        ),
        expected={
            "category": "technical",
            "priority": "low",
            "sentiment": "neutral",
            "requires_human": False,
        },
        acceptable={"sentiment": ["neutral", "positive"], "category": ["technical", "other"]},
    ),
    BenchTask(
        id="triage-03",
        schema_name="ticket_triage",
        schema_cls=TicketTriage,
        instruction="Triage this customer support ticket.",
        input_text=(
            "Subject: PRODUCTION DOWN\n\n"
            "Our entire team cannot log in. Every request returns a 503. "
            "We have 200 staff blocked and a client demo in one hour. "
            "Please escalate immediately."
        ),
        expected={
            "category": "technical",
            "priority": "urgent",
            "sentiment": "negative",
            "requires_human": True,
        },
    ),
    BenchTask(
        id="triage-04",
        schema_name="ticket_triage",
        schema_cls=TicketTriage,
        instruction="Triage this customer support ticket.",
        input_text=(
            "Subject: Dark mode?\n\n"
            "Love the product. Any plans for a dark theme? Would be easier on "
            "the eyes for those of us staring at it all day. No rush."
        ),
        expected={
            "category": "feature_request",
            "priority": "low",
            "sentiment": "positive",
            "requires_human": False,
        },
    ),

    # ---- invoice extraction ---------------------------------------------
    BenchTask(
        id="invoice-01",
        schema_name="invoice_extract",
        schema_cls=InvoiceExtract,
        instruction="Extract the invoice details.",
        input_text=(
            "INVOICE  #INV-2291\n"
            "Northwind Logistics LLC\n"
            "PO Reference: PO-88412\n\n"
            "1. Freight, Chicago to Dallas .......... $2,400.00\n"
            "2. Fuel surcharge ...................... $  310.00\n"
            "3. Liftgate service .................... $  145.00\n\n"
            "TOTAL DUE: USD $2,855.00\n"
        ),
        expected={
            "invoice_number": "INV-2291",
            "vendor_name": "Northwind Logistics LLC",
            "total_amount": 2855.00,
            "currency": "USD",
            "line_item_count": 3,
            "purchase_order": "PO-88412",
        },
    ),
    BenchTask(
        id="invoice-02",
        schema_name="invoice_extract",
        schema_cls=InvoiceExtract,
        instruction="Extract the invoice details.",
        input_text=(
            "Cascade Analytics Inc.\n"
            "Invoice 7734-B\n\n"
            "Consulting retainer, November ........ EUR 9,500.00\n\n"
            "Amount payable: EUR 9,500.00\n"
            "No purchase order on file.\n"
        ),
        expected={
            "invoice_number": "7734-B",
            "vendor_name": "Cascade Analytics Inc.",
            "total_amount": 9500.00,
            "currency": "EUR",
            "line_item_count": 1,
            "purchase_order": None,
        },
    ),
    BenchTask(
        id="invoice-03",
        schema_name="invoice_extract",
        schema_cls=InvoiceExtract,
        instruction="Extract the invoice details.",
        input_text=(
            "Ironvale Manufacturing Co.   |   Invoice: IV-0043\n"
            "Order ref PO-1190\n\n"
            "Steel bracket, 200 units ....... GBP 1,200.00\n"
            "Powder coating .................. GBP   380.00\n"
            "Expedited shipping .............. GBP   220.00\n"
            "Packaging ....................... GBP    65.00\n"
            "-------------------------------------------\n"
            "Total .......................... GBP 1,865.00\n"
        ),
        expected={
            "invoice_number": "IV-0043",
            "vendor_name": "Ironvale Manufacturing Co.",
            "total_amount": 1865.00,
            "currency": "GBP",
            "line_item_count": 4,
            "purchase_order": "PO-1190",
        },
    ),

    # ---- meeting notes ---------------------------------------------------
    BenchTask(
        id="meeting-01",
        schema_name="meeting_notes",
        schema_cls=MeetingNotes,
        instruction="Summarise this meeting transcript.",
        input_text=(
            "Present: Priya, Marcus, Dana, Tom\n\n"
            "Priya: The migration is behind. We said end of Q3, we are looking "
            "at mid Q4 now.\n"
            "Marcus: Agreed. I think we cut the reporting module from phase one.\n"
            "Dana: Fine by me, nobody is using the old reports anyway.\n"
            "Priya: Decided then -- reporting moves to phase two. Marcus, can you "
            "update the roadmap doc by Friday?\n"
            "Marcus: Yes.\n"
            "Tom: I will tell the client about the new date. I will draft the "
            "email today and send it to Priya for review."
        ),
        expected={
            "decisions": ["Reporting module moved to phase two"],
            "action_items": [
                "Marcus to update the roadmap doc by Friday",
                "Tom to draft client email about the new date and send to Priya for review",
            ],
            "attendee_count": 4,
        },
    ),
    BenchTask(
        id="meeting-02",
        schema_name="meeting_notes",
        schema_cls=MeetingNotes,
        instruction="Summarise this meeting transcript.",
        input_text=(
            "Present: Sam, Alex\n\n"
            "Sam: Just a sync. Anything blocking you?\n"
            "Alex: Nothing blocking. The API work is on track for next week.\n"
            "Sam: Good. Nothing from my side either.\n"
            "Alex: Shall we skip next week then?\n"
            "Sam: Sure, let us skip it."
        ),
        expected={
            "decisions": ["Skip next week's sync"],
            "action_items": [],
            "attendee_count": 2,
        },
    ),

    # ---- email intent ----------------------------------------------------
    BenchTask(
        id="email-01",
        schema_name="email_intent",
        schema_cls=EmailIntent,
        instruction="Classify the intent of this email.",
        input_text=(
            "Hi team,\n\nCould you send over the signed SOW before the 15th? "
            "Legal needs it on file before we can start work.\n\nThanks,\nRina"
        ),
        expected={
            "intent": "request",
            "priority": "medium",
            "needs_reply": True,
            "deadline_mentioned": "the 15th",
        },
        acceptable={
            "priority": ["medium", "high"],
            "deadline_mentioned": ["the 15th", "15th", "before the 15th"],
        },
    ),
    BenchTask(
        id="email-02",
        schema_name="email_intent",
        schema_cls=EmailIntent,
        instruction="Classify the intent of this email.",
        input_text=(
            "This is an automated notification. Your scheduled backup completed "
            "successfully at 03:00 UTC. No action is required."
        ),
        expected={
            "intent": "notification",
            "priority": "low",
            "needs_reply": False,
            "deadline_mentioned": None,
        },
    ),
    BenchTask(
        id="email-03",
        schema_name="email_intent",
        schema_cls=EmailIntent,
        instruction="Classify the intent of this email.",
        input_text=(
            "I have now asked three times for someone to look at the billing "
            "discrepancy and have had no response. This is unacceptable. "
            "I expect an answer by end of day tomorrow or we are cancelling."
        ),
        expected={
            "intent": "complaint",
            "priority": "urgent",
            "needs_reply": True,
            "deadline_mentioned": "end of day tomorrow",
        },
        acceptable={
            "priority": ["urgent", "high"],
            "deadline_mentioned": ["end of day tomorrow", "tomorrow", "EOD tomorrow"],
        },
    ),
]


def build_oracle() -> dict[str, dict[str, Any]]:
    """Prompt -> correct answer, for the simulator.

    The simulator emits this and then corrupts it. That is what makes repair
    measurable: when salvage recovers the object, we can check it recovered the
    *right* object rather than merely a parseable one.
    """
    oracle: dict[str, dict[str, Any]] = {}
    for t in TASKS:
        payload = dict(t.expected)
        # MeetingNotes needs a summary; the tasks omit it since free text cannot
        # be scored by equality. Supply a placeholder so the object validates.
        if t.schema_name == "meeting_notes" and "summary" not in payload:
            payload["summary"] = "Team discussed timeline and agreed next steps."
        oracle[t.prompt] = payload
    return oracle


def oracle_fn():
    """Callable for SimulatedProvider, matching prompts by containment.

    Containment rather than equality because the repair ladder appends schema
    text and error feedback to the prompt on retries, so an exact-match lookup
    would miss every re-prompt and return the generic stub instead.
    """
    table = build_oracle()
    items = sorted(table.items(), key=lambda kv: -len(kv[0]))

    def _lookup(prompt: str) -> dict[str, Any]:
        for key, payload in items:
            if key in prompt:
                return payload
        return {"answer": "no oracle entry for this prompt"}

    return _lookup


def tasks_by_schema(name: str) -> list[BenchTask]:
    return [t for t in TASKS if t.schema_name == name]
