"""Shared graph state (owner: J).

The state is checkpointed after every step, so it only holds small, checkpoint-friendly
values: contracts made of plain fields, counters, paths and strings. Heavy or
non-primitive data (raw answers, source documents, images) belongs in the run folder
``runs/<run_id>/`` and the state keeps a reference to it.
"""

import operator
from typing import Annotated, Literal, TypedDict

from ai_share_of_voice.schemas.base import Contract
from ai_share_of_voice.schemas.brief import Brief
from ai_share_of_voice.schemas.events import GraphEvent
from ai_share_of_voice.schemas.questions import QuestionPanel
from ai_share_of_voice.schemas.stopping import BatchSummary, StopDecision

RunStatus = Literal["running", "completed", "partial", "rejected"]


class ApprovalResponse(Contract):
    """What the approval page sends to resume the graph after the human pause."""

    approved: bool
    comment: str | None = None


class GraphState(TypedDict, total=False):
    """Shared state that every node reads and updates.

    Fields with a reducer (``history``, ``events``) are appended to; the others are replaced.
    """

    # Input
    run_id: str
    brief: Brief

    # Sampling and metrics
    panel: QuestionPanel
    n_answers: int
    n_brand_mentions: int
    spend_usd: float
    history: Annotated[list[BatchSummary], operator.add]
    stop_decision: StopDecision

    # Evidence
    sources: list[str]
    findings: list[str]

    # Drafting and review
    draft: str
    draft_version: int
    objections: list[str]
    review_round: int  # how many times the Skeptic sent the draft back

    # Approval and outcome
    approved: bool
    status: RunStatus
    partial_reason: str
    report_path: str

    # Trace
    events: Annotated[list[GraphEvent], operator.add]
