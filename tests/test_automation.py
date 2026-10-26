import csv
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from langgraph.types import Command
from pydantic import ValidationError

from ai_share_of_voice.automation.files import write_text_atomic
from ai_share_of_voice.automation.finalize import ApprovalRequired, make_finalize_node
from ai_share_of_voice.automation.report import ReportError, build_report_data, render_report
from ai_share_of_voice.automation.tracking_csv import FIELDS, TrackingError, build_row, upsert_row
from ai_share_of_voice.event_log import EventLog
from ai_share_of_voice.schemas.brief import BrandSpec, Brief
from ai_share_of_voice.schemas.events import EventStatus, GraphEvent
from ai_share_of_voice.schemas.stopping import BatchSummary, StopDecision, StopReason

RUN_ID = "run-test"
GENERATED_AT = datetime(2026, 10, 23, 12, 0, tzinfo=UTC)

BRIEF = Brief(
    brand=BrandSpec(name="Acme"),
    competitors=(BrandSpec(name="Beta Eats"), BrandSpec(name="Gamma Bites")),
    sector="food delivery",
    city="Madrid",
    spend_cap_usd=2.0,
)


def make_state(**overrides: Any) -> dict[str, Any]:
    first = BatchSummary(
        batch_index=0,
        n_answers=4,
        brand_share=0.5,
        ci_low=0.25,
        ci_high=0.75,
        cumulative_spend_usd=0.05,
    )
    last = BatchSummary(
        batch_index=1,
        n_answers=8,
        brand_share=0.5,
        ci_low=0.32,
        ci_high=0.68,
        cumulative_spend_usd=0.10,
    )
    events = [
        GraphEvent(
            run_id=RUN_ID,
            timestamp=GENERATED_AT,
            node="sampling",
            tool="llm_client",
            status=EventStatus.SUCCEEDED,
            result="batch 0: 4 answers | simulated",
        ),
        GraphEvent(
            run_id=RUN_ID,
            timestamp=GENERATED_AT,
            node="research",
            agent="source_researcher",
            tool="mcp_sources",
            status=EventStatus.SUCCEEDED,
            result="2 sources",
        ),
    ]
    state: dict[str, Any] = {
        "run_id": RUN_ID,
        "brief": BRIEF,
        "n_answers": 8,
        "spend_usd": 0.10,
        "history": [first, last],
        "stop_decision": StopDecision(
            stop=True, reason=StopReason.PRECISION_REACHED, explanation="Interval is narrow enough"
        ),
        "sources": ["https://example.com/a", "https://example.org/b"],
        "findings": ["The client's bar is the lowest"],
        "draft": "Acme is mentioned in half of the answers.",
        "draft_version": 1,
        "objections": [],
        "review_round": 0,
        "approved": True,
        "events": events,
    }
    state.update(overrides)
    return state


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def tracking_row(run_id: str = RUN_ID, **overrides: str) -> dict[str, str]:
    row = build_row(build_report_data(make_state(run_id=run_id), GENERATED_AT), Path("report.md"))
    row.update(overrides)
    return row


# ---- report data ------------------------------------------------------------------------


def test_report_data_is_built_from_the_state() -> None:
    data = build_report_data(make_state(), GENERATED_AT)
    assert data.brand == "Acme"
    assert data.competitors == ("Beta Eats", "Gamma Bites")
    assert data.summary.n_answers == 8
    assert len(data.history) == 2
    assert len(data.sources) == 2


@pytest.mark.parametrize("missing", ["brief", "history", "stop_decision"])
def test_report_data_needs_the_essentials(missing: str) -> None:
    state = make_state()
    del state[missing]
    with pytest.raises(ReportError, match=missing):
        build_report_data(state, GENERATED_AT)


def test_an_empty_history_is_not_enough_for_a_report() -> None:
    with pytest.raises(ReportError, match="history"):
        build_report_data(make_state(history=[]), GENERATED_AT)


def test_the_report_date_must_be_timezone_aware() -> None:
    with pytest.raises(ValidationError):
        build_report_data(make_state(), datetime(2026, 10, 23))


# ---- report text ------------------------------------------------------------------------


def render(**overrides: Any) -> str:
    return render_report(build_report_data(make_state(**overrides), GENERATED_AT))


def test_the_report_states_it_is_a_dated_sample_and_not_market_share() -> None:
    text = render()
    assert "Dated sample, not market share" in text
    assert "2026-10-23" in text
    assert "Beta Eats, Gamma Bites" in text


def test_each_figure_names_the_tool_that_produced_it() -> None:
    text = render()
    assert "| Mention share of Acme | 50% | `stats_tool`, batch 2 |" in text
    assert "| Interval | 32% to 68% | `stats_tool`, batch 2 |" in text
    assert "| 1 | 4 | 50% | 25% to 75% | 0.05 |" in text
    assert "**precision_reached**" in text


def test_sources_are_listed_and_visual_findings_are_hypotheses() -> None:
    text = render()
    assert "1. <https://example.com/a>" in text
    assert "2. <https://example.org/b>" in text
    assert "Visual findings (hypotheses, not causes)" in text
    assert "The client's bar is the lowest" in text


def test_unsafe_urls_are_left_out() -> None:
    text = render(sources=["javascript:alert(1)", "https://example.com/a b"])
    assert "(invalid URL omitted)" in text
    assert "javascript:" not in text


def test_the_draft_is_included_and_open_objections_only_when_present() -> None:
    assert "Acme is mentioned in half of the answers." in render()
    assert "Open objections" not in render()
    with_objections = render(objections=["Claim 1 has no verifiable source"])
    assert "## Open objections from the Skeptic" in with_objections
    assert "Claim 1 has no verifiable source" in with_objections


def test_empty_sections_say_so() -> None:
    text = render(sources=[], findings=[], draft="")
    assert "No sources were recorded." in text
    assert "No visual findings were recorded." in text
    assert "No narrative was recorded." in text


def test_the_run_log_is_attached_with_escaped_cells() -> None:
    text = render()
    assert "## Run log (until approval)" in text
    row = "| 12:00:00 | research | source_researcher | mcp_sources | succeeded | 2 sources |"
    assert row in text
    assert "batch 0: 4 answers \\| simulated" in text


def test_a_pipe_in_the_brand_name_cannot_break_the_table() -> None:
    brief = BRIEF.model_copy(update={"brand": BrandSpec(name="Acme | Food")})
    assert "| Mention share of Acme \\| Food |" in render(brief=brief)


# ---- tracking csv -----------------------------------------------------------------------


def test_the_tracking_row_has_exactly_the_tracking_columns() -> None:
    row = tracking_row()
    assert set(row) == set(FIELDS)
    assert row["brand_share"] == "0.5000"
    assert row["batches"] == "2"
    assert row["stop_reason"] == "precision_reached"
    assert row["sources"] == "2"
    assert row["open_objections"] == "0"


def test_the_first_run_creates_the_file_with_its_header(tmp_path: Path) -> None:
    path = tmp_path / "tracking.csv"
    assert upsert_row(path, tracking_row()) is True
    assert path.read_text(encoding="utf-8").splitlines()[0] == ",".join(FIELDS)
    assert len(read_rows(path)) == 1


def test_repeating_a_run_does_not_duplicate_it(tmp_path: Path) -> None:
    path = tmp_path / "tracking.csv"
    upsert_row(path, tracking_row())
    assert upsert_row(path, tracking_row(brand_share="0.6000")) is False
    rows = read_rows(path)
    assert len(rows) == 1
    assert rows[0]["brand_share"] == "0.6000"


def test_a_new_run_adds_a_row(tmp_path: Path) -> None:
    path = tmp_path / "tracking.csv"
    upsert_row(path, tracking_row("run-1"))
    assert upsert_row(path, tracking_row("run-2")) is True
    assert [row["run_id"] for row in read_rows(path)] == ["run-1", "run-2"]


def test_formula_like_values_become_plain_text(tmp_path: Path) -> None:
    path = tmp_path / "tracking.csv"
    upsert_row(path, tracking_row(brand='=HYPERLINK("http://example.com")'))
    assert read_rows(path)[0]["brand"].startswith("'=")


def test_a_file_with_other_columns_is_not_touched(tmp_path: Path) -> None:
    path = tmp_path / "tracking.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(TrackingError, match="columns"):
        upsert_row(path, tracking_row())
    assert path.read_text(encoding="utf-8") == "a,b\n1,2\n"


def test_incomplete_rows_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(TrackingError):
        upsert_row(tmp_path / "tracking.csv", {"run_id": RUN_ID})


def test_no_temporary_files_are_left_next_to_the_csv(tmp_path: Path) -> None:
    upsert_row(tmp_path / "tracking.csv", tracking_row())
    assert [path.name for path in tmp_path.iterdir()] == ["tracking.csv"]


# ---- atomic writes ----------------------------------------------------------------------


def test_atomic_write_creates_folders_and_replaces_content(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "file.txt"
    write_text_atomic(target, "first\n")
    write_text_atomic(target, "second\n")
    assert target.read_text(encoding="utf-8") == "second\n"
    assert [path.name for path in target.parent.iterdir()] == ["file.txt"]


# ---- the finalize node ------------------------------------------------------------------


def make_node(tmp_path: Path):
    log = EventLog(tmp_path / "runs")
    return make_finalize_node(log, clock=lambda: GENERATED_AT), log


@pytest.mark.parametrize("approval", [False, None])
def test_nothing_is_written_without_an_explicit_approval(
    tmp_path: Path, approval: bool | None
) -> None:
    node, log = make_node(tmp_path)
    state = make_state(approved=approval)
    with pytest.raises(ApprovalRequired):
        node(state)
    assert not (tmp_path / "runs").exists()
    assert log.read(RUN_ID) == []


def test_an_approved_run_writes_the_report_and_the_tracking_row(tmp_path: Path) -> None:
    node, log = make_node(tmp_path)
    result = node(make_state())

    report_path = Path(result["report_path"])
    assert report_path == tmp_path / "runs" / RUN_ID / "report.md"
    assert "Dated sample, not market share" in report_path.read_text(encoding="utf-8")
    assert [row["run_id"] for row in read_rows(tmp_path / "runs" / "tracking.csv")] == [RUN_ID]

    assert result["status"] == "completed"
    statuses = [event.status for event in result["events"]]
    assert statuses == [EventStatus.STARTED, EventStatus.SUCCEEDED]
    assert [event.status for event in log.read(RUN_ID)] == statuses
    assert all(event.tool == "report_template" for event in result["events"])


def test_running_the_node_twice_gives_one_row_and_the_same_report(tmp_path: Path) -> None:
    node, _ = make_node(tmp_path)
    first = node(make_state())
    report_before = Path(first["report_path"]).read_text(encoding="utf-8")
    second = node(make_state())
    assert Path(second["report_path"]).read_text(encoding="utf-8") == report_before
    assert len(read_rows(tmp_path / "runs" / "tracking.csv")) == 1
    assert "tracking row updated" in second["events"][-1].result


def test_a_failure_is_logged_with_its_reason_and_raised(tmp_path: Path) -> None:
    node, log = make_node(tmp_path)
    with pytest.raises(ReportError):
        node(make_state(history=[]))
    events = log.read(RUN_ID)
    assert [event.status for event in events] == [EventStatus.STARTED, EventStatus.FAILED]
    assert "history" in events[-1].reason
    assert not (tmp_path / "runs" / RUN_ID / "report.md").exists()


# ---- inside the graph (skipped until the graph is on the branch) ------------------------


def build_graph_with_real_finalize(tmp_path: Path):
    nodes_module = pytest.importorskip("ai_share_of_voice.nodes")
    graph_module = pytest.importorskip("ai_share_of_voice.graph")
    checkpoint = pytest.importorskip("ai_share_of_voice.checkpoint")

    log = EventLog(tmp_path / "runs")
    deps = nodes_module.NodeDeps(event_log=log, should_stop=nodes_module.fake_should_stop(2))
    nodes = nodes_module.make_fake_nodes(deps)
    nodes["finalize"] = make_finalize_node(log, clock=lambda: GENERATED_AT)
    graph = graph_module.build_graph(nodes, checkpointer=checkpoint.memory_checkpointer())
    config = graph_module.run_config(RUN_ID, BRIEF)
    graph.invoke(graph_module.initial_state(BRIEF, RUN_ID), config)
    return graph, config


def test_the_graph_writes_the_report_only_after_approval(tmp_path: Path) -> None:
    graph, config = build_graph_with_real_finalize(tmp_path)
    assert not (tmp_path / "runs" / RUN_ID / "report.md").exists()
    assert not (tmp_path / "runs" / "tracking.csv").exists()

    graph.invoke(Command(resume={"approved": True}), config)
    state = graph.get_state(config).values
    assert state["status"] == "completed"
    text = Path(state["report_path"]).read_text(encoding="utf-8")
    assert "Dated sample, not market share" in text
    assert "sampling" in text  # the run log of the graph is attached
    assert len(read_rows(tmp_path / "runs" / "tracking.csv")) == 1


def test_a_rejection_writes_nothing(tmp_path: Path) -> None:
    graph, config = build_graph_with_real_finalize(tmp_path)
    graph.invoke(Command(resume={"approved": False, "comment": "Numbers look off"}), config)
    assert graph.get_state(config).values["status"] == "rejected"
    assert not (tmp_path / "runs" / RUN_ID / "report.md").exists()
    assert not (tmp_path / "runs" / "tracking.csv").exists()
