import importlib.util
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from ai_share_of_voice.exceptions import BudgetExceeded
from ai_share_of_voice.schemas.brief import BrandSpec, Brief
from ai_share_of_voice.schemas.events import EventStatus, GraphEvent
from ai_share_of_voice.ui import render
from ai_share_of_voice.ui.brief_tab import (
    build_brief,
    format_brand,
    parse_brand,
    parse_competitors,
)
from ai_share_of_voice.ui.mock_run import build_demo_brief
from ai_share_of_voice.ui.run_manager import (
    RunInProgress,
    RunManager,
    RunSnapshot,
    status_from_events,
)

ROOT = Path(__file__).parents[1]
FORM = {
    "brand": "Acme | AcmeCo, Acme Inc",
    "competitors": "Beta\nGamma",
    "sector": "food delivery",
    "city": "Madrid",
    "language": "en",
    "spend_cap_usd": 2.0,
    "max_batches": 5.0,
    "target_half_width": 0.05,
}


def event(node: str, status=EventStatus.SUCCEEDED, **fields) -> GraphEvent:
    return GraphEvent(run_id="run-1", node=node, status=status, **fields)


def closing_events(spend: float = 0.3) -> list[GraphEvent]:
    return [
        event("sampling", cumulative_spend_usd=spend),
        event("approval_request", EventStatus.WAITING_APPROVAL),
        event("finalize", result="report (simulated)", cumulative_spend_usd=spend),
    ]


def runner_of(events: list[GraphEvent]):
    def run(run_id: str, brief: Brief) -> Iterator[GraphEvent]:
        yield from events

    return run


def snapshot(status: str, **fields) -> RunSnapshot:
    values = {"run_id": "run-1", "events": (), "spent_usd": 0.0, "cap_usd": 2.0, "error": None}
    return RunSnapshot(status=status, **(values | fields))


# ---- spend bar and page content ------------------------------------------------------------


def test_the_spend_bar_shows_spend_against_the_cap() -> None:
    assert render.spend_bar(0.0, None) == "**Spend:** no run yet"
    empty = render.spend_bar(0.0, 2.0)
    assert "$0.00 of $2.00 (0%)" in empty and "█" not in empty
    half = render.spend_bar(1.0, 2.0)
    assert half.count("█") == 10 and half.count("░") == 10 and "(50%)" in half


def test_the_spend_bar_warns_near_the_cap_and_never_overflows() -> None:
    assert "near the cap" in render.spend_bar(1.7, 2.0)
    over = render.spend_bar(2.5, 2.0)
    assert over.count("█") == render.BAR_WIDTH and "cap reached" in over


def test_events_become_plain_rows() -> None:
    long_text = "word " * 100 + "\nend"
    first = event(
        "sampling", agent="sampler", tool="model", result=long_text, cumulative_spend_usd=0.5
    )
    failed = event("close_partial", EventStatus.FAILED, reason="tool failed")
    rows = render.events_rows([first, failed])
    assert len(rows) == 2 and all(len(row) == len(render.EVENT_HEADERS) for row in rows)
    time, node, agent, tool, status, detail, spend = rows[0]
    assert (node, agent, tool) == ("sampling", "sampler", "model")
    assert (status, spend) == ("succeeded", "0.50")
    assert len(detail) <= render.MAX_DETAIL_CHARS and "\n" not in detail and len(time) == 8
    assert rows[1][2:4] == ["", ""] and rows[1][5] == "tool failed" and rows[1][6] == ""


def test_markdown_is_escaped_before_it_is_shown() -> None:
    assert render.escape_markdown("**x** <b>[a](b)") == r"\*\*x\*\* \<b\>\[a\]\(b\)"
    text = render.results_text(snapshot("failed", error="unexpected: <script>*x*"))
    assert "<script>" not in text and r"\<script\>" in text


def test_the_status_and_results_follow_the_run() -> None:
    assert "no run yet" in render.status_line(snapshot("idle", run_id=None))
    assert "No run yet" in render.results_text(snapshot("idle", run_id=None))
    assert "in progress" in render.results_text(snapshot("running"))
    finished = snapshot("completed", events=tuple(closing_events()), spent_usd=0.3)
    done = render.results_text(finished)
    assert "completed" in done and "3 events" in done and "$0.30" in done


# ---- the Brief form ------------------------------------------------------------------------


def test_a_valid_form_becomes_a_brief() -> None:
    result = build_brief(**FORM)
    assert result.errors == () and result.brief is not None
    assert result.brief.brand == BrandSpec(name="Acme", variants=("AcmeCo", "Acme Inc"))
    assert [item.name for item in result.brief.competitors] == ["Beta", "Gamma"]
    assert result.brief.max_batches == 5 and result.brief.spend_cap_usd == 2.0


def test_the_demo_brief_survives_a_round_trip_through_the_form() -> None:
    demo = build_demo_brief()
    assert parse_brand(format_brand(demo.brand)) == demo.brand
    text = "\n".join(format_brand(item) for item in demo.competitors)
    assert parse_competitors(text) == demo.competitors


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        ({"brand": "  "}, "Brand:"),
        ({"competitors": "Beta"}, "Competitors:"),
        ({"competitors": "Beta\n\nGamma\n \n"}, None),
        ({"competitors": "Beta\nacme"}, "'acme' is used by both"),
        ({"sector": ""}, "Sector:"),
        ({"language": "english"}, "Language:"),
        ({"spend_cap_usd": None}, "Spend cap (USD):"),
        ({"spend_cap_usd": 500.0}, "Spend cap (USD):"),
        ({"max_batches": 1.0}, "Maximum batches:"),
        ({"max_batches": 5.5}, "Maximum batches:"),
        ({"target_half_width": None}, "Target half-width:"),
    ],
)
def test_a_bad_form_says_what_to_fix(change: dict, expected: str | None) -> None:
    result = build_brief(**(FORM | change))
    if expected is None:
        assert result.brief is not None
        return
    assert result.brief is None
    assert any(expected in message for message in result.errors)


# ---- the background run --------------------------------------------------------------------


def test_a_run_goes_through_the_events_and_ends_completed() -> None:
    manager = RunManager(runner_of(closing_events(0.3)))
    assert manager.snapshot().status == "idle"
    run_id = manager.start(build_demo_brief())
    manager.wait(5)
    state = manager.snapshot()
    assert state.run_id == run_id and state.status == "completed" and not state.running
    assert len(state.events) == 3 and state.spent_usd == 0.3 and state.cap_usd == 2.0


def test_the_closing_events_decide_the_status() -> None:
    assert status_from_events([event("close_partial", EventStatus.FAILED, reason="x")]) == "partial"
    assert status_from_events([event("close_rejected")]) == "rejected"
    assert status_from_events(closing_events()[:2]) == "waiting_approval"
    assert status_from_events([event("sampling")]) is None


def test_a_failing_run_is_reported_not_raised() -> None:
    def broken(run_id: str, brief: Brief) -> Iterator[GraphEvent]:
        raise BudgetExceeded(spent_usd=2.5, cap_usd=2.0)
        yield  # pragma: no cover

    manager = RunManager(broken)
    manager.start(build_demo_brief())
    manager.wait(5)
    state = manager.snapshot()
    assert state.status == "failed" and state.error.startswith("budget_exhausted:")


def test_a_run_without_a_closing_event_is_a_failure() -> None:
    manager = RunManager(runner_of([event("sampling")]))
    manager.start(build_demo_brief())
    manager.wait(5)
    assert manager.snapshot().status == "failed"
    assert "closing event" in manager.snapshot().error


def test_only_one_run_at_a_time_and_a_new_run_starts_clean() -> None:
    release = threading.Event()

    def slow(run_id: str, brief: Brief) -> Iterator[GraphEvent]:
        yield event("sampling")
        release.wait(5)
        yield event("finalize")

    manager = RunManager(slow)
    manager.start(build_demo_brief())
    with pytest.raises(RunInProgress):
        manager.start(build_demo_brief())
    assert manager.snapshot().running
    release.set()
    manager.wait(5)
    assert manager.snapshot().status == "completed"
    first = manager.snapshot().run_id
    manager.start(build_demo_brief())
    manager.wait(5)
    assert manager.snapshot().run_id != first and len(manager.snapshot().events) == 2


def test_the_simulated_graph_runs_in_the_background() -> None:
    pytest.importorskip("langgraph")
    manager = RunManager()
    manager.start(build_demo_brief())
    manager.wait(60)
    state = manager.snapshot()
    assert state.status == "completed" and state.events
    assert 0 <= state.spent_usd <= state.cap_usd


# ---- the app -------------------------------------------------------------------------------


def test_the_app_builds_with_its_three_tabs() -> None:
    gr = pytest.importorskip("gradio")
    spec = importlib.util.spec_from_file_location("app_under_test", ROOT / "app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    demo = module.build_app(RunManager(runner_of(closing_events())))
    assert isinstance(demo, gr.Blocks)
