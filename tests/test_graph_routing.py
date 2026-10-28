from collections.abc import Sequence
from pathlib import Path

import pytest
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from pydantic import ValidationError

from ai_share_of_voice.checkpoint import memory_checkpointer, sqlite_checkpointer
from ai_share_of_voice.event_log import EventLog
from ai_share_of_voice.graph import build_graph, initial_state, run_config
from ai_share_of_voice.nodes import FakeBehavior, NodeDeps, fake_should_stop, make_fake_nodes
from ai_share_of_voice.schemas.brief import BrandSpec, Brief
from ai_share_of_voice.schemas.events import EventStatus
from ai_share_of_voice.schemas.stopping import (
    BatchSummary,
    StopConfig,
    StopDecision,
    StopReason,
    StopRule,
)

RUN_ID = "run-test"
PAUSED = ("approval_gate",)

BRIEF = Brief(
    brand=BrandSpec(name="Acme"),
    competitors=(BrandSpec(name="Beta"), BrandSpec(name="Gamma")),
    sector="food delivery",
    city="Madrid",
    spend_cap_usd=2.0,
    max_batches=5,
)


def always_continue(history: Sequence[BatchSummary], config: StopConfig) -> StopDecision:
    return StopDecision(stop=False, reason=StopReason.CONTINUE, explanation="never stops")


def build(
    tmp_path: Path,
    *,
    should_stop: StopRule | None = None,
    behavior: FakeBehavior | None = None,
    max_review_rounds: int = 2,
    checkpointer: BaseCheckpointSaver | None = None,
) -> tuple[CompiledStateGraph, EventLog]:
    log = EventLog(tmp_path / "runs")
    deps = NodeDeps(event_log=log, should_stop=should_stop or fake_should_stop(2))
    graph = build_graph(
        make_fake_nodes(deps, behavior),
        max_review_rounds=max_review_rounds,
        checkpointer=checkpointer or memory_checkpointer(),
    )
    return graph, log


def start(graph: CompiledStateGraph, max_review_rounds: int = 2) -> dict:
    config = run_config(RUN_ID, BRIEF, max_review_rounds=max_review_rounds)
    graph.invoke(initial_state(BRIEF, RUN_ID), config)
    return config


def values(graph: CompiledStateGraph, config: dict) -> dict:
    return graph.get_state(config).values


def next_nodes(graph: CompiledStateGraph, config: dict) -> tuple[str, ...]:
    return graph.get_state(config).next


def approve(graph: CompiledStateGraph, config: dict, **answer: object) -> None:
    graph.invoke(Command(resume=answer), config)


def nodes_with_status(state: dict, node: str, status: EventStatus) -> list:
    return [e for e in state["events"] if e.node == node and e.status is status]


# ---- continue or stop -------------------------------------------------------------------


def test_sampling_repeats_until_the_stopping_rule_stops(tmp_path: Path) -> None:
    graph, _ = build(tmp_path, should_stop=fake_should_stop(3))
    state = values(graph, start(graph))
    assert len(state["history"]) == 3
    assert state["stop_decision"].reason is StopReason.PRECISION_REACHED


def test_hard_cap_stops_a_rule_that_never_stops(tmp_path: Path) -> None:
    graph, _ = build(tmp_path, should_stop=always_continue)
    state = values(graph, start(graph))
    assert len(state["history"]) == BRIEF.max_batches
    assert state["stop_decision"].reason is StopReason.MAX_BATCHES


# ---- skeptic loop -----------------------------------------------------------------------


@pytest.mark.parametrize("max_rounds", [0, 1, 2])
def test_skeptic_sends_the_draft_back_at_most_max_rounds_times(
    tmp_path: Path, max_rounds: int
) -> None:
    graph, _ = build(
        tmp_path, behavior=FakeBehavior(objection_rounds=99), max_review_rounds=max_rounds
    )
    config = start(graph, max_rounds)
    state = values(graph, config)
    assert state["review_round"] == max_rounds
    assert state["draft_version"] == max_rounds + 1
    assert len(nodes_with_status(state, "skeptic", EventStatus.SUCCEEDED)) == max_rounds + 1
    assert state["objections"]  # unresolved objections go to the human
    assert next_nodes(graph, config) == PAUSED


def test_clean_review_goes_straight_to_approval(tmp_path: Path) -> None:
    graph, _ = build(tmp_path)
    state = values(graph, start(graph))
    assert state["review_round"] == 0
    assert state["draft_version"] == 1


def test_one_objection_triggers_exactly_one_revision(tmp_path: Path) -> None:
    graph, _ = build(tmp_path, behavior=FakeBehavior(objection_rounds=1))
    state = values(graph, start(graph))
    assert state["draft_version"] == 2
    assert state["objections"] == []


# ---- human pause ------------------------------------------------------------------------


def test_graph_pauses_before_the_final_action(tmp_path: Path) -> None:
    graph, _ = build(tmp_path)
    config = start(graph)
    state = values(graph, config)
    assert next_nodes(graph, config) == PAUSED
    assert "status" not in state
    assert state["events"][-1].status is EventStatus.WAITING_APPROVAL
    assert not nodes_with_status(state, "finalize", EventStatus.STARTED)


def test_approval_resumes_and_finalizes(tmp_path: Path) -> None:
    graph, _ = build(tmp_path)
    config = start(graph)
    approve(graph, config, approved=True)
    state = values(graph, config)
    assert state["status"] == "completed"
    assert state["report_path"].endswith("report.md")
    assert next_nodes(graph, config) == ()


def test_rejection_closes_without_finalizing(tmp_path: Path) -> None:
    graph, _ = build(tmp_path)
    config = start(graph)
    approve(graph, config, approved=False, comment="Numbers look off")
    state = values(graph, config)
    assert state["status"] == "rejected"
    assert not nodes_with_status(state, "finalize", EventStatus.STARTED)
    rejected = nodes_with_status(state, "approval_gate", EventStatus.REJECTED)
    assert rejected[0].reason == "Numbers look off"


@pytest.mark.parametrize("answer", [{"approved": "maybe"}, {"comment": "no decision"}])
def test_invalid_approval_answer_is_rejected(tmp_path: Path, answer: dict) -> None:
    graph, _ = build(tmp_path)
    config = start(graph)
    with pytest.raises(ValidationError):
        approve(graph, config, **answer)


# ---- partial close ----------------------------------------------------------------------


@pytest.mark.parametrize("fail_at", ["sampling", "research"])
def test_a_failing_step_closes_the_run_in_a_partial_state(tmp_path: Path, fail_at: str) -> None:
    graph, _ = build(tmp_path, behavior=FakeBehavior(fail_at=fail_at))
    config = start(graph)
    state = values(graph, config)
    assert state["status"] == "partial"
    assert state["failure_kind"] == "unexpected"
    assert fail_at in state["partial_reason"]
    assert next_nodes(graph, config) == ()
    assert not nodes_with_status(state, "approval_request", EventStatus.WAITING_APPROVAL)
    failed = [e for e in state["events"] if e.status is EventStatus.FAILED]
    assert failed
    assert all(event.reason for event in failed)


# ---- trace and persistence --------------------------------------------------------------


def test_every_event_is_in_the_state_and_in_the_jsonl_log(tmp_path: Path) -> None:
    graph, log = build(tmp_path)
    config = start(graph)
    approve(graph, config, approved=True)
    state = values(graph, config)
    logged = [(e.node, e.status) for e in log.read(RUN_ID)]
    assert logged == [(e.node, e.status) for e in state["events"]]
    assert all(e.run_id == RUN_ID for e in state["events"])


def test_run_resumes_from_a_new_sqlite_connection(tmp_path: Path) -> None:
    database = tmp_path / "checkpoints.db"
    first = sqlite_checkpointer(database)
    graph, _ = build(tmp_path, checkpointer=first)
    config = start(graph)
    assert next_nodes(graph, config) == PAUSED
    first.conn.close()

    second = sqlite_checkpointer(database)
    resumed, _ = build(tmp_path, checkpointer=second)
    approve(resumed, config, approved=True)
    assert values(resumed, config)["status"] == "completed"
    second.conn.close()


# ---- wiring -----------------------------------------------------------------------------


def test_build_graph_reports_missing_nodes(tmp_path: Path) -> None:
    deps = NodeDeps(event_log=EventLog(tmp_path), should_stop=fake_should_stop(2))
    nodes = make_fake_nodes(deps)
    del nodes["skeptic"]
    with pytest.raises(ValueError, match="skeptic"):
        build_graph(nodes, checkpointer=memory_checkpointer())
