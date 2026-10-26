from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from langgraph.types import Command

from ai_share_of_voice.checkpoint import memory_checkpointer
from ai_share_of_voice.errors import NoEvidence, ToolFailure
from ai_share_of_voice.event_log import EventLog
from ai_share_of_voice.exceptions import BudgetExceeded
from ai_share_of_voice.graph import build_graph, initial_state, run_config
from ai_share_of_voice.guard import DEFAULT_UNGUARDED, guard_nodes
from ai_share_of_voice.nodes import NodeDeps, NodeFn, fake_should_stop, make_fake_nodes
from ai_share_of_voice.schemas.brief import BrandSpec, Brief
from ai_share_of_voice.schemas.events import EventStatus
from ai_share_of_voice.state import GraphState

RUN_ID = "run-errors"
PAUSED = ("approval_gate",)

BRIEF = Brief(
    brand=BrandSpec(name="Acme"),
    competitors=(BrandSpec(name="Beta"), BrandSpec(name="Gamma")),
    sector="food delivery",
    city="Madrid",
    spend_cap_usd=2.0,
    max_batches=5,
)


class MustNotBeSwallowed(Exception):
    """Stands in for a safety error such as running the final action without approval."""


def raising(error: Exception) -> NodeFn:
    def node(state: GraphState) -> dict[str, Any]:
        raise error

    return node


def fail_on_call(real: NodeFn, call_number: int, error: Exception) -> NodeFn:
    """Behave like ``real`` except on one call, which raises ``error``."""
    calls = 0

    def node(state: GraphState) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == call_number:
            raise error
        return real(state)

    return node


def build(
    tmp_path: Path,
    *,
    replace: dict[str, NodeFn] | None = None,
    wrap: dict[str, Callable[[NodeFn], NodeFn]] | None = None,
    fatal: tuple[type[BaseException], ...] = (),
    unguarded: frozenset[str] = DEFAULT_UNGUARDED,
):
    log = EventLog(tmp_path / "runs")
    deps = NodeDeps(event_log=log, should_stop=fake_should_stop(2))
    nodes = make_fake_nodes(deps)
    for name, wrapper in (wrap or {}).items():
        nodes[name] = wrapper(nodes[name])
    nodes.update(replace or {})
    guarded = guard_nodes(nodes, log, unguarded=unguarded, fatal=fatal)
    graph = build_graph(guarded, checkpointer=memory_checkpointer())
    return graph, log


def start(graph) -> dict:
    config = run_config(RUN_ID, BRIEF)
    graph.invoke(initial_state(BRIEF, RUN_ID), config)
    return config


def values(graph, config: dict) -> dict:
    return graph.get_state(config).values


def next_nodes(graph, config: dict) -> tuple[str, ...]:
    return graph.get_state(config).next


# ---- the three required failure paths ---------------------------------------------------


@pytest.mark.parametrize(
    ("node", "error", "kind", "fragment"),
    [
        ("sampling", BudgetExceeded(spent_usd=2.5, cap_usd=2.0), "budget_exhausted", "2.5000"),
        ("research", ToolFailure("mcp_sources", "unreachable"), "tool_failure", "unreachable"),
        ("sampling", NoEvidence("no analyzable answers"), "no_evidence", "no analyzable"),
    ],
    ids=["budget_exhausted", "tool_failure", "no_evidence"],
)
def test_each_failure_kind_closes_the_run_in_a_partial_state(
    tmp_path: Path, node: str, error: Exception, kind: str, fragment: str
) -> None:
    graph, log = build(tmp_path, replace={node: raising(error)})
    config = start(graph)
    state = values(graph, config)

    assert state["status"] == "partial"
    assert state["failure_kind"] == kind
    assert fragment in state["partial_reason"]
    assert next_nodes(graph, config) == ()

    failed = [e for e in state["events"] if e.node == node and e.status is EventStatus.FAILED]
    assert failed
    assert failed[0].reason == state["partial_reason"]
    assert not [e for e in state["events"] if e.status is EventStatus.WAITING_APPROVAL]
    assert not [e for e in state["events"] if e.node == "finalize"]
    closing = state["events"][-1]
    assert (closing.node, closing.status) == ("close_partial", EventStatus.FAILED)
    assert closing.reason == state["partial_reason"]


@pytest.mark.parametrize(
    "node",
    ["questions", "sampling", "metrics", "research", "screenshots", "write_draft", "skeptic"],
)
def test_every_guarded_node_can_fail_and_close_the_run(tmp_path: Path, node: str) -> None:
    error = ToolFailure("some_tool", "down")
    graph, _ = build(tmp_path, replace={node: raising(error)})
    config = start(graph)
    state = values(graph, config)
    assert state["status"] == "partial"
    assert state["failure_kind"] == "tool_failure"
    assert next_nodes(graph, config) == ()


def test_the_budget_can_run_out_after_some_progress(tmp_path: Path) -> None:
    error = BudgetExceeded(spent_usd=2.1, cap_usd=2.0)
    graph, log = build(tmp_path, wrap={"sampling": lambda real: fail_on_call(real, 2, error)})
    config = start(graph)
    state = values(graph, config)

    assert state["status"] == "partial"
    assert state["failure_kind"] == "budget_exhausted"
    assert len(state["history"]) == 1  # the first batch is kept
    assert state["spend_usd"] > 0
    assert not (tmp_path / "runs" / RUN_ID / "report.md").exists()
    assert [(e.node, e.status) for e in log.read(RUN_ID)] == [
        (e.node, e.status) for e in state["events"]
    ]


def test_a_failure_after_the_approval_closes_the_run_in_a_partial_state(tmp_path: Path) -> None:
    error = ToolFailure("report_template", "disk full")
    graph, _ = build(tmp_path, replace={"finalize": raising(error)})
    config = start(graph)
    assert next_nodes(graph, config) == PAUSED

    graph.invoke(Command(resume={"approved": True}), config)
    state = values(graph, config)
    assert state["status"] == "partial"
    assert state["failure_kind"] == "tool_failure"
    assert "disk full" in state["partial_reason"]
    assert next_nodes(graph, config) == ()


# ---- what must not be absorbed ----------------------------------------------------------


def test_fatal_errors_are_raised_again_instead_of_closing_the_run(tmp_path: Path) -> None:
    graph, _ = build(
        tmp_path,
        replace={"finalize": raising(MustNotBeSwallowed("no approval"))},
        fatal=(MustNotBeSwallowed,),
    )
    config = start(graph)
    with pytest.raises(MustNotBeSwallowed):
        graph.invoke(Command(resume={"approved": True}), config)


def test_a_guarded_pause_still_pauses_and_resumes(tmp_path: Path) -> None:
    unguarded = frozenset({"approval_request", "close_rejected", "close_partial"})
    graph, _ = build(tmp_path, unguarded=unguarded)  # approval_gate is guarded this time
    config = start(graph)
    assert next_nodes(graph, config) == PAUSED
    assert "status" not in values(graph, config)

    graph.invoke(Command(resume={"approved": True}), config)
    assert values(graph, config)["status"] == "completed"


def test_guards_do_not_change_a_healthy_run(tmp_path: Path) -> None:
    graph, _ = build(tmp_path)
    config = start(graph)
    state = values(graph, config)
    assert next_nodes(graph, config) == PAUSED
    assert "partial_reason" not in state
    assert not [e for e in state["events"] if e.status is EventStatus.FAILED]


def test_a_failure_in_approval_request_closes_without_pausing(
    tmp_path: Path,
) -> None:
    error = ToolFailure("approval_service", "unreachable")
    graph, _ = build(
        tmp_path,
        replace={"approval_request": raising(error)},
    )
    config = start(graph)
    state = values(graph, config)

    assert state["status"] == "partial"
    assert state["failure_kind"] == "tool_failure"
    assert "unreachable" in state["partial_reason"]
    assert next_nodes(graph, config) == ()
    assert not [
        event
        for event in state["events"]
        if event.status is EventStatus.WAITING_APPROVAL
    ]
    assert not [
        event for event in state["events"] if event.node == "approval_gate"
    ]
