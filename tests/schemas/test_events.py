import pytest
from pydantic import ValidationError

from ai_share_of_voice.schemas.events import EventStatus, GraphEvent


def test_event_defaults_to_an_aware_timestamp() -> None:
    event = GraphEvent(run_id="run-1", node="sampling", status=EventStatus.STARTED)
    assert event.timestamp.tzinfo is not None
    assert event.agent is None


def test_succeeded_event_with_agent_and_tool() -> None:
    event = GraphEvent(
        run_id="run-1",
        node="metrics",
        agent="analyst",
        tool="stats_tool",
        status=EventStatus.SUCCEEDED,
        result="share computed",
        cumulative_spend_usd=0.12,
    )
    assert event.tool == "stats_tool"


@pytest.mark.parametrize("status", [EventStatus.FAILED, EventStatus.REJECTED])
def test_failed_and_rejected_events_need_a_reason(status: EventStatus) -> None:
    with pytest.raises(ValidationError):
        GraphEvent(run_id="run-1", node="sources", status=status)


def test_failed_event_with_reason_is_valid() -> None:
    event = GraphEvent(
        run_id="run-1", node="sources", status=EventStatus.FAILED, reason="url does not exist"
    )
    assert event.reason == "url does not exist"


def test_result_length_is_bounded() -> None:
    with pytest.raises(ValidationError):
        GraphEvent(run_id="run-1", node="draft", status=EventStatus.SUCCEEDED, result="x" * 2001)


def test_negative_spend_is_rejected() -> None:
    with pytest.raises(ValidationError):
        GraphEvent(
            run_id="run-1",
            node="draft",
            status=EventStatus.SUCCEEDED,
            cumulative_spend_usd=-0.01,
        )
