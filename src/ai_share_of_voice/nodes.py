"""Fake graph nodes (owner: J).

They have the same signature as the real nodes, so each one can be replaced by its real
version (A2, A3, A5, A7, I3, I4, I5, D1) without touching ``graph.py``. Every node
emits ``GraphEvent`` objects both to the state and to the JSONL event log.
"""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any

from langgraph.types import interrupt

from ai_share_of_voice.event_log import EventLog
from ai_share_of_voice.schemas.events import EventStatus, GraphEvent
from ai_share_of_voice.schemas.questions import Question, QuestionPanel
from ai_share_of_voice.schemas.stopping import (
    BatchSummary,
    StopConfig,
    StopDecision,
    StopReason,
    StopRule,
)
from ai_share_of_voice.state import ApprovalResponse, GraphState

NodeFn = Callable[[GraphState], dict[str, Any]]

FAKE_QUESTION_COUNT = 4


@dataclass(frozen=True)
class NodeDeps:
    """What the nodes need from the outside; swap the stopping rule for the real one."""

    event_log: EventLog
    should_stop: StopRule


@dataclass(frozen=True)
class FakeBehavior:
    """Knobs that let tests drive the fake nodes down every path of the graph."""

    objection_rounds: int = 0  # the Skeptic objects while review_round < this value
    fail_at: str | None = None  # "sampling" or "research": close the run in a partial state
    batch_cost_usd: float = 0.05


def fake_should_stop(stop_after_batches: int = 2) -> StopRule:
    """Fake stopping rule: stop after a fixed number of batches. Replaced by D2."""

    def rule(history: Sequence[BatchSummary], config: StopConfig) -> StopDecision:
        if len(history) >= stop_after_batches:
            return StopDecision(
                stop=True,
                reason=StopReason.PRECISION_REACHED,
                explanation=f"Fake rule: stopped after {len(history)} batches",
            )
        return StopDecision(
            stop=False, reason=StopReason.CONTINUE, explanation="Fake rule: more batches needed"
        )

    return rule


def _emit(
    deps: NodeDeps,
    state: GraphState,
    node: str,
    status: EventStatus,
    *,
    agent: str | None = None,
    tool: str | None = None,
    result: str | None = None,
    reason: str | None = None,
    spend: float | None = None,
) -> GraphEvent:
    """Build an event, write it to the JSONL log and return it for the state."""
    event = GraphEvent(
        run_id=state["run_id"],
        node=node,
        agent=agent,
        tool=tool,
        status=status,
        result=result,
        reason=reason,
        cumulative_spend_usd=state.get("spend_usd") if spend is None else spend,
    )
    deps.event_log.append(event)
    return event


def make_fake_nodes(deps: NodeDeps, behavior: FakeBehavior | None = None) -> dict[str, NodeFn]:
    """Return the fake version of every node of the graph, keyed by node name."""
    behavior = behavior or FakeBehavior()
    emit = partial(_emit, deps)

    def questions(state: GraphState) -> dict[str, Any]:
        agent = "question_generator"
        events = [emit(state, "questions", EventStatus.STARTED, agent=agent)]
        brief = state["brief"]
        panel = QuestionPanel(
            questions=tuple(
                Question(
                    id=f"q{index:02d}",
                    text=f"Recommend a {brief.sector} in {brief.city} (variant {index})",
                    intent="recommendation",
                )
                for index in range(1, FAKE_QUESTION_COUNT + 1)
            )
        )
        result = f"{len(panel.questions)} questions (simulated)"
        events.append(emit(state, "questions", EventStatus.SUCCEEDED, agent=agent, result=result))
        return {"panel": panel, "events": events}

    def sampling(state: GraphState) -> dict[str, Any]:
        tool = "llm_client"
        events = [emit(state, "sampling", EventStatus.STARTED, tool=tool)]
        if behavior.fail_at == "sampling":
            reason = "Simulated failure in sampling"
            events.append(emit(state, "sampling", EventStatus.FAILED, tool=tool, reason=reason))
            return {"partial_reason": reason, "events": events}
        batch = len(state.get("history", []))
        batch_answers = len(state["panel"].questions)
        spend = state.get("spend_usd", 0.0) + behavior.batch_cost_usd
        events.append(
            emit(
                state,
                "sampling",
                EventStatus.SUCCEEDED,
                tool=tool,
                result=f"batch {batch}: {batch_answers} answers (simulated)",
                spend=spend,
            )
        )
        return {
            "n_answers": state.get("n_answers", 0) + batch_answers,
            "n_brand_mentions": state.get("n_brand_mentions", 0) + (batch_answers + 1) // 2,
            "spend_usd": spend,
            "events": events,
        }

    def metrics(state: GraphState) -> dict[str, Any]:
        agent, tool = "analyst", "stats_tool"
        events = [emit(state, "metrics", EventStatus.STARTED, agent=agent, tool=tool)]
        brief = state["brief"]
        n_answers = state["n_answers"]
        share = state["n_brand_mentions"] / n_answers
        half_width = min(0.5, 0.5 / math.sqrt(n_answers))  # placeholder, replaced by D1
        summary = BatchSummary(
            batch_index=len(state.get("history", [])),
            n_answers=n_answers,
            brand_share=share,
            ci_low=max(0.0, share - half_width),
            ci_high=min(1.0, share + half_width),
            cumulative_spend_usd=state.get("spend_usd", 0.0),
        )
        history = [*state.get("history", []), summary]
        config = StopConfig(
            target_half_width=brief.target_half_width,
            min_batches=min(2, brief.max_batches),
            max_batches=brief.max_batches,
            max_spend_usd=brief.spend_cap_usd,
        )
        decision = deps.should_stop(history, config)
        if not decision.stop and len(history) >= brief.max_batches:
            decision = StopDecision(
                stop=True,
                reason=StopReason.MAX_BATCHES,
                explanation=f"Hard cap of {brief.max_batches} batches reached",
            )
        events.append(
            emit(
                state,
                "metrics",
                EventStatus.SUCCEEDED,
                agent=agent,
                tool=tool,
                result=f"share {share:.0%}, half-width {summary.half_width:.3f}",
                reason=decision.explanation,
            )
        )
        return {"history": [summary], "stop_decision": decision, "events": events}

    def research(state: GraphState) -> dict[str, Any]:
        agent, tool = "source_researcher", "mcp_sources"
        events = [emit(state, "research", EventStatus.STARTED, agent=agent, tool=tool)]
        if behavior.fail_at == "research":
            reason = "Simulated failure in research"
            events.append(
                emit(state, "research", EventStatus.FAILED, agent=agent, tool=tool, reason=reason)
            )
            return {"partial_reason": reason, "events": events}
        found = ["https://example.com/source-1", "https://example.com/source-2"]
        result = f"{len(found)} sources (simulated)"
        events.append(
            emit(state, "research", EventStatus.SUCCEEDED, agent=agent, tool=tool, result=result)
        )
        return {"sources": found, "events": events}

    def screenshots(state: GraphState) -> dict[str, Any]:
        agent, tool = "visual_analyst", "analyze_image"
        events = [emit(state, "screenshots", EventStatus.STARTED, agent=agent, tool=tool)]
        findings = ["Simulated finding: the client's bar is the lowest (hypothesis, not a cause)"]
        result = f"{len(findings)} findings (simulated)"
        events.append(
            emit(state, "screenshots", EventStatus.SUCCEEDED, agent=agent, tool=tool, result=result)
        )
        return {"findings": findings, "events": events}

    def write_draft(state: GraphState) -> dict[str, Any]:
        agent = "writer"
        events = [emit(state, "write_draft", EventStatus.STARTED, agent=agent)]
        version = state.get("draft_version", 0) + 1
        # A draft written while objections are pending is a revision: one more round back.
        review_round = state.get("review_round", 0) + (1 if state.get("objections") else 0)
        text = f"Simulated report draft v{version} based on {len(state.get('sources', []))} sources"
        events.append(
            emit(
                state,
                "write_draft",
                EventStatus.SUCCEEDED,
                agent=agent,
                result=f"draft v{version}",
            )
        )
        return {
            "draft": text,
            "draft_version": version,
            "review_round": review_round,
            "objections": [],
            "events": events,
        }

    def skeptic(state: GraphState) -> dict[str, Any]:
        agent = "skeptic"
        events = [emit(state, "skeptic", EventStatus.STARTED, agent=agent)]
        found: list[str] = []
        if state.get("review_round", 0) < behavior.objection_rounds:
            found = ["Simulated objection: claim 1 has no verifiable source"]
        events.append(
            emit(
                state,
                "skeptic",
                EventStatus.SUCCEEDED,
                agent=agent,
                result=f"{len(found)} objections",
            )
        )
        return {"objections": found, "events": events}

    def approval_request(state: GraphState) -> dict[str, Any]:
        pending = len(state.get("objections", []))
        version = state.get("draft_version", 1)
        result = f"Draft v{version} waits for approval ({pending} open objections)"
        event = emit(state, "approval_request", EventStatus.WAITING_APPROVAL, result=result)
        return {"events": [event]}

    def approval_gate(state: GraphState) -> dict[str, Any]:
        # Code before interrupt() runs again on resume, so nothing is emitted before it.
        answer = interrupt(
            {
                "run_id": state["run_id"],
                "draft": state.get("draft", ""),
                "unresolved_objections": state.get("objections", []),
            }
        )
        response = ApprovalResponse.model_validate(answer)
        if response.approved:
            event = emit(state, "approval_gate", EventStatus.APPROVED, result=response.comment)
        else:
            reason = response.comment or "Rejected without comment"
            event = emit(state, "approval_gate", EventStatus.REJECTED, reason=reason)
        return {"approved": response.approved, "events": [event]}

    def finalize(state: GraphState) -> dict[str, Any]:
        agent, tool = "writer", "report_writer"
        events = [emit(state, "finalize", EventStatus.STARTED, agent=agent, tool=tool)]
        report_path = f"runs/{state['run_id']}/report.md"  # the real node writes it (A7)
        events.append(
            emit(
                state,
                "finalize",
                EventStatus.SUCCEEDED,
                agent=agent,
                tool=tool,
                result=f"report and tracking CSV (simulated): {report_path}",
            )
        )
        return {"status": "completed", "report_path": report_path, "events": events}

    def close_rejected(state: GraphState) -> dict[str, Any]:
        event = emit(
            state, "close_rejected", EventStatus.SUCCEEDED, result="Closed without publishing"
        )
        return {"status": "rejected", "events": [event]}

    def close_partial(state: GraphState) -> dict[str, Any]:
        reason = state.get("partial_reason", "Unknown failure")
        event = emit(state, "close_partial", EventStatus.FAILED, reason=reason)
        return {"status": "partial", "events": [event]}

    return {
        "questions": questions,
        "sampling": sampling,
        "metrics": metrics,
        "research": research,
        "screenshots": screenshots,
        "write_draft": write_draft,
        "skeptic": skeptic,
        "approval_request": approval_request,
        "approval_gate": approval_gate,
        "finalize": finalize,
        "close_rejected": close_rejected,
        "close_partial": close_partial,
    }
