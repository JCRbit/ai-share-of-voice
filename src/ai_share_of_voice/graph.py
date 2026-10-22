"""Graph topology: explicit, auditable and bounded (owner: J).

Flow: questions -> sampling -> metrics -> (stop? else sampling again) -> research ->
screenshots -> write_draft -> skeptic -> (objections? back to write_draft, at most
``max_review_rounds`` times) -> approval request -> human pause -> finalize.
A failing step closes the run in a partial state; a rejection closes it without publishing.
"""

from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from ai_share_of_voice.nodes import NodeFn
from ai_share_of_voice.schemas.brief import Brief
from ai_share_of_voice.state import GraphState

REQUIRED_NODES = (
    "questions",
    "sampling",
    "metrics",
    "research",
    "screenshots",
    "write_draft",
    "skeptic",
    "approval_request",
    "approval_gate",
    "finalize",
    "close_rejected",
    "close_partial",
)


def route_unless_partial(next_node: str) -> Callable[[GraphState], str]:
    """Go to ``next_node`` unless a step recorded a failure."""

    def route(state: GraphState) -> str:
        return "close_partial" if state.get("partial_reason") else next_node

    return route


def route_after_metrics(state: GraphState) -> str:
    """Take another batch or move on, following the stopping decision."""
    return "research" if state["stop_decision"].stop else "sampling"


def make_route_after_skeptic(max_review_rounds: int) -> Callable[[GraphState], str]:
    """Send the draft back while there are objections and rounds left."""

    def route(state: GraphState) -> str:
        if state.get("objections") and state.get("review_round", 0) < max_review_rounds:
            return "write_draft"
        return "approval_request"

    return route


def route_after_approval(state: GraphState) -> str:
    """Finalize only with an explicit approval."""
    return "finalize" if state.get("approved") else "close_rejected"


def build_graph(
    nodes: Mapping[str, NodeFn],
    *,
    max_review_rounds: int = 2,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """Wire the nodes into the graph. A checkpointer is required for the human pause."""
    missing = [name for name in REQUIRED_NODES if name not in nodes]
    if missing:
        raise ValueError(f"missing graph nodes: {', '.join(missing)}")

    builder = StateGraph(GraphState)
    for name in REQUIRED_NODES:
        builder.add_node(name, nodes[name])

    builder.add_edge(START, "questions")
    builder.add_edge("questions", "sampling")
    builder.add_conditional_edges(
        "sampling",
        route_unless_partial("metrics"),
        {"metrics": "metrics", "close_partial": "close_partial"},
    )
    builder.add_conditional_edges(
        "metrics", route_after_metrics, {"sampling": "sampling", "research": "research"}
    )
    builder.add_conditional_edges(
        "research",
        route_unless_partial("screenshots"),
        {"screenshots": "screenshots", "close_partial": "close_partial"},
    )
    builder.add_edge("screenshots", "write_draft")
    builder.add_edge("write_draft", "skeptic")
    builder.add_conditional_edges(
        "skeptic",
        make_route_after_skeptic(max_review_rounds),
        {"write_draft": "write_draft", "approval_request": "approval_request"},
    )
    builder.add_edge("approval_request", "approval_gate")
    builder.add_conditional_edges(
        "approval_gate",
        route_after_approval,
        {"finalize": "finalize", "close_rejected": "close_rejected"},
    )
    builder.add_edge("finalize", END)
    builder.add_edge("close_rejected", END)
    builder.add_edge("close_partial", END)

    return builder.compile(checkpointer=checkpointer)


def new_run_id() -> str:
    """Unique, sortable run id, safe to use as a folder name."""
    return f"run-{datetime.now(UTC):%Y%m%d-%H%M%S}-{uuid4().hex[:6]}"


def initial_state(brief: Brief, run_id: str) -> GraphState:
    """State a run starts from."""
    return {"run_id": run_id, "brief": brief}


def run_config(run_id: str, brief: Brief, *, max_review_rounds: int = 2) -> dict[str, Any]:
    """Config for ``invoke``: the thread id resumes the run, the limit stops runaway loops."""
    steps = 2 * brief.max_batches + 2 * (max_review_rounds + 1) + 15
    return {"configurable": {"thread_id": run_id}, "recursion_limit": steps}
