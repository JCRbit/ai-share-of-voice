"""Graph event contract: what the activity page and the run log show (owner: J)."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Self

from pydantic import AwareDatetime, Field, model_validator

from ai_share_of_voice.schemas.base import Contract


class EventStatus(StrEnum):
    """Lifecycle of a node, an agent step or a tool call."""

    STARTED = "started"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    WAITING_APPROVAL = "waiting_approval"
    APPROVED = "approved"
    REJECTED = "rejected"


class GraphEvent(Contract):
    """Which agent did what, with which tool, with what result and why."""

    run_id: str = Field(min_length=1)
    timestamp: AwareDatetime = Field(default_factory=lambda: datetime.now(UTC))
    node: str = Field(min_length=1)
    agent: str | None = None
    tool: str | None = None
    status: EventStatus
    result: str | None = Field(default=None, max_length=2000)
    reason: str | None = None
    cumulative_spend_usd: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def failures_need_a_reason(self) -> Self:
        if self.status in {EventStatus.FAILED, EventStatus.REJECTED} and not self.reason:
            raise ValueError("failed and rejected events must state a reason")
        return self
