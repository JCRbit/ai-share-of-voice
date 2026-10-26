from pathlib import Path

import pytest
from langgraph.errors import GraphBubbleUp

from ai_share_of_voice.errors import FailureKind, NoEvidence, ToolFailure, classify
from ai_share_of_voice.event_log import EventLog
from ai_share_of_voice.exceptions import BudgetExceeded
from ai_share_of_voice.guard import guard_node, guard_nodes
from ai_share_of_voice.nodes import NodeFn
from ai_share_of_voice.schemas.events import EventStatus

RUN_ID = "run-guard"


class MustNotBeSwallowed(Exception):
    """Stands in for a safety error such as running the final action without approval."""


def raising(error: BaseException) -> NodeFn:
    def node(state):
        raise error

    return node


# ---- classification ---------------------------------------------------------------------


def test_budget_exceeded_is_classified_with_its_numbers() -> None:
    failure = classify(BudgetExceeded(spent_usd=2.5, cap_usd=2.0))
    assert failure.kind is FailureKind.BUDGET_EXHAUSTED
    assert "2.5000" in failure.reason
    assert "cap" in failure.reason


def test_a_tool_failure_keeps_the_tool_name() -> None:
    failure = classify(ToolFailure("mcp_sources", "timeout"))
    assert failure.kind is FailureKind.TOOL_FAILURE
    assert failure.tool == "mcp_sources"
    assert "timeout" in failure.reason


def test_missing_evidence_has_a_default_reason() -> None:
    assert classify(NoEvidence("no analyzable answers")).reason == "no analyzable answers"
    assert classify(NoEvidence()).kind is FailureKind.NO_EVIDENCE
    assert classify(NoEvidence()).reason


def test_unexpected_errors_keep_their_type() -> None:
    failure = classify(ValueError("boom"))
    assert failure.kind is FailureKind.UNEXPECTED
    assert failure.reason == "ValueError: boom"


@pytest.mark.parametrize(
    "error",
    [RuntimeError("x" * 900), ToolFailure("tool", "y" * 900), NoEvidence("z" * 900)],
    ids=["unexpected", "tool_failure", "no_evidence"],
)
def test_long_messages_are_shortened_to_fit_the_event_reason(error: Exception) -> None:
    assert len(classify(error).reason) <= 500


# ---- the guard --------------------------------------------------------------------------


def test_guard_nodes_leaves_the_pause_and_the_closing_nodes_alone() -> None:
    log = EventLog("unused")
    nodes: dict[str, NodeFn] = {
        name: (lambda state: {})
        for name in ("questions", "approval_gate", "close_partial", "finalize")
    }
    guarded = guard_nodes(nodes, log)
    assert guarded["approval_gate"] is nodes["approval_gate"]
    assert guarded["close_partial"] is nodes["close_partial"]
    assert guarded["questions"] is not nodes["questions"]
    assert guarded["finalize"] is not nodes["finalize"]


def test_a_healthy_node_passes_through_untouched(tmp_path: Path) -> None:
    log = EventLog(tmp_path)
    node = guard_node("sampling", lambda state: {"n_answers": 4}, log)
    assert node({"run_id": RUN_ID}) == {"n_answers": 4}
    assert log.read(RUN_ID) == []


def test_a_failure_becomes_a_partial_update_with_a_failed_event(tmp_path: Path) -> None:
    log = EventLog(tmp_path)
    node = guard_node("research", raising(ToolFailure("mcp_sources", "down")), log)
    update = node({"run_id": RUN_ID, "spend_usd": 0.1})

    assert update["failure_kind"] == "tool_failure"
    assert "down" in update["partial_reason"]
    (event,) = update["events"]
    assert (event.node, event.tool, event.status) == ("research", "mcp_sources", EventStatus.FAILED)
    assert event.reason == update["partial_reason"]
    assert event.cumulative_spend_usd == pytest.approx(0.1)
    assert log.read(RUN_ID) == [event]


def test_an_unexpected_error_is_absorbed_with_its_type(tmp_path: Path) -> None:
    node = guard_node("skeptic", raising(RuntimeError("bug")), EventLog(tmp_path))
    update = node({"run_id": RUN_ID})
    assert update["failure_kind"] == "unexpected"
    assert update["partial_reason"] == "RuntimeError: bug"


def test_langgraph_control_flow_exceptions_pass_through(tmp_path: Path) -> None:
    node = guard_node("approval_gate", raising(GraphBubbleUp()), EventLog(tmp_path))
    with pytest.raises(GraphBubbleUp):
        node({"run_id": RUN_ID})


def test_fatal_errors_are_raised_again(tmp_path: Path) -> None:
    node = guard_node(
        "finalize",
        raising(MustNotBeSwallowed("no approval")),
        EventLog(tmp_path),
        fatal=(MustNotBeSwallowed,),
    )
    with pytest.raises(MustNotBeSwallowed):
        node({"run_id": RUN_ID})


def test_other_errors_are_still_absorbed_when_fatal_errors_are_set(tmp_path: Path) -> None:
    node = guard_node(
        "finalize", raising(ValueError("x")), EventLog(tmp_path), fatal=(MustNotBeSwallowed,)
    )
    assert node({"run_id": RUN_ID})["failure_kind"] == "unexpected"
