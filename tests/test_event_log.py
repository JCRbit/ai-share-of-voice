from pathlib import Path

import pytest

from ai_share_of_voice.event_log import EventLog
from ai_share_of_voice.schemas.events import EventStatus, GraphEvent


def make_event(node: str = "sampling") -> GraphEvent:
    return GraphEvent(run_id="run-1", node=node, status=EventStatus.STARTED)


def test_events_round_trip_in_order(tmp_path: Path) -> None:
    log = EventLog(tmp_path)
    log.append(make_event("questions"))
    log.append(make_event("sampling"))
    assert [event.node for event in log.read("run-1")] == ["questions", "sampling"]


def test_each_event_is_one_json_line(tmp_path: Path) -> None:
    log = EventLog(tmp_path)
    log.append(make_event())
    lines = log.path_for("run-1").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1


def test_unknown_run_has_no_events(tmp_path: Path) -> None:
    assert EventLog(tmp_path).read("missing-run") == []


@pytest.mark.parametrize("run_id", ["../evil", "a/b", "..", "."])
def test_run_ids_cannot_leave_the_runs_folder(tmp_path: Path, run_id: str) -> None:
    with pytest.raises(ValueError, match="invalid run id"):
        EventLog(tmp_path).path_for(run_id)
