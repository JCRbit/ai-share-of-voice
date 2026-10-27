"""The real ``finalize`` node (owner: J).

After the human approval it writes the report and updates the tracking CSV. It replaces
the fake node without touching the graph::

    nodes = make_fake_nodes(deps)
    nodes["finalize"] = make_finalize_node(deps.event_log)
"""

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from ai_share_of_voice.automation.files import write_text_atomic
from ai_share_of_voice.automation.report import REPORT_FILE_NAME, build_report_data, render_report
from ai_share_of_voice.automation.tracking_csv import TRACKING_FILE_NAME, build_row, upsert_row
from ai_share_of_voice.event_log import EventLog
from ai_share_of_voice.schemas.events import EventStatus, GraphEvent
from ai_share_of_voice.state import GraphState

AGENT = "writer"
TOOL = "report_template"


class ApprovalRequired(Exception):
    """Raised when the node runs without an explicit approval."""


def _utc_now() -> datetime:
    return datetime.now(UTC)


def make_finalize_node(
    event_log: EventLog, *, clock: Callable[[], datetime] = _utc_now
) -> Callable[[GraphState], dict[str, Any]]:
    """Build the node. ``clock`` makes the report date injectable in tests."""

    def emit(
        state: GraphState,
        status: EventStatus,
        *,
        result: str | None = None,
        reason: str | None = None,
    ) -> GraphEvent:
        event = GraphEvent(
            run_id=state["run_id"],
            node="finalize",
            agent=AGENT,
            tool=TOOL,
            status=status,
            result=result,
            reason=reason,
            cumulative_spend_usd=state.get("spend_usd"),
        )
        event_log.append(event)
        return event

    def finalize(state: GraphState) -> dict[str, Any]:
        if state.get("approved") is not True:
            raise ApprovalRequired("the report and the tracking CSV need an explicit approval")
        events = [emit(state, EventStatus.STARTED)]
        try:
            data = build_report_data(state, clock())
            run_dir = event_log.path_for(state["run_id"]).parent
            report_path = run_dir / REPORT_FILE_NAME
            write_text_atomic(report_path, render_report(data))
            inserted = upsert_row(run_dir.parent / TRACKING_FILE_NAME, build_row(data, report_path))
        except Exception as error:
            emit(state, EventStatus.FAILED, reason=str(error)[:500] or type(error).__name__)
            raise
        action = "added" if inserted else "updated"
        result = f"report written to {report_path}; tracking row {action}"
        events.append(emit(state, EventStatus.SUCCEEDED, result=result))
        return {"status": "completed", "report_path": str(report_path), "events": events}

    return finalize
