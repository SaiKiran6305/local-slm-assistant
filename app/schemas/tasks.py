"""
Task schemas.

These are deliberately the shapes that small models struggle with, rather than
the ones that flatter them:

  enums          a constrained vocabulary the model must not invent outside of
  integers       where models love to emit "3" instead of 3
  booleans       where models love to emit "true" instead of true
  arrays         where models drop the brackets or trail a comma
  optionals      where models emit the string "null" or omit the key entirely

A schema of three free-text strings would conform at nearly 100% on any model
and measure nothing.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class Category(str, Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    ACCOUNT = "account"
    FEATURE_REQUEST = "feature_request"
    OTHER = "other"


class TicketTriage(BaseModel):
    """Classify a support ticket. Three enums plus a bool: maximum enum pressure."""
    category: Category = Field(description="Which team this belongs to")
    priority: Priority = Field(description="How urgent this is")
    sentiment: Sentiment = Field(description="Customer's tone")
    requires_human: bool = Field(description="True if this cannot be auto-resolved")


class InvoiceExtract(BaseModel):
    """Extract invoice fields. Numeric types plus an optional."""
    invoice_number: str
    vendor_name: str
    total_amount: float = Field(description="Total in the invoice currency")
    currency: str = Field(description="Three letter ISO code, uppercase")
    line_item_count: int = Field(description="How many line items appear")
    purchase_order: str | None = Field(default=None, description="PO number, or null if absent")


class MeetingNotes(BaseModel):
    """Summarise a meeting. Arrays, which small models frequently malform."""
    summary: str = Field(description="Two sentences maximum")
    decisions: list[str] = Field(description="Decisions actually made")
    action_items: list[str] = Field(description="Tasks assigned to someone")
    attendee_count: int


class EmailIntent(BaseModel):
    """Route an email. Enum plus bool plus optional string."""
    intent: str = Field(description="One of: question, complaint, request, notification")
    priority: Priority
    needs_reply: bool
    deadline_mentioned: str | None = Field(default=None, description="Date if stated, else null")


SCHEMAS: dict[str, type[BaseModel]] = {
    "ticket_triage": TicketTriage,
    "invoice_extract": InvoiceExtract,
    "meeting_notes": MeetingNotes,
    "email_intent": EmailIntent,
}
