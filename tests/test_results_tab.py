import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ai_share_of_voice.automation.report import REPORT_FILE_NAME, ReportData
from ai_share_of_voice.automation.tracking_csv import FIELDS, TRACKING_FILE_NAME, upsert_row
from ai_share_of_voice.schemas.stopping import BatchSummary, StopDecision, StopReason
from ai_share_of_voice.stats.metrics import brand_metrics, load_answers
from ai_share_of_voice.tools.stats_tool import ANSWERS_FILE_NAME
from ai_share_of_voice.ui.results_tab import build_results_view, table_rows

FIXTURE = Path(__file__).parent / "fixtures" / "sample_answers.csv"
RUN_ID = "run-20261026-120000-abc123"
GENERATED_AT = datetime(2026, 10, 26, 12, 0, tzinfo=UTC)


def fixture_mention_rate(brand: str) -> float:
    metrics = brand_metrics(load_answers(FIXTURE))
    rows = metrics[(metrics["metric"] == "mention_rate") & (metrics["brand"] == brand)]
    return float(rows["estimate"].iloc[0])


def make_data(*, run_id: str = RUN_ID, brand_share: float | None = None) -> ReportData:
    share = fixture_mention_rate("Glovo") if brand_share is None else brand_share
    summary = BatchSummary(
        batch_index=0,
        n_answers=6,
        brand_share=share,
        ci_low=max(share - 0.2, 0.0),
        ci_high=min(share + 0.2, 1.0),
        cumulative_spend_usd=0.05,
        n_questions=3,
    )
    return ReportData(
        run_id=run_id,
        generated_at=GENERATED_AT,
        brand="Glovo",
        competitors=("Just Eat", "Uber Eats"),
        sector="food delivery",
        city="Madrid",
        history=(summary,),
        stop=StopDecision(stop=True, reason=StopReason.MAX_BATCHES, explanation="Batch limit"),
        spend_usd=0.05,
    )


@pytest.fixture
def runs_dir(tmp_path: Path) -> Path:
    (tmp_path / "runs" / RUN_ID).mkdir(parents=True)
    return tmp_path / "runs"


@pytest.fixture
def runs_with_answers(runs_dir: Path) -> Path:
    shutil.copy(FIXTURE, runs_dir / RUN_ID / ANSWERS_FILE_NAME)
    return runs_dir


# ---- layer A: the headline comes from the run state -----------------------------------------


def test_the_headline_comes_from_the_last_batch_even_without_answers(runs_dir: Path) -> None:
    view = build_results_view(make_data(), runs_dir)
    assert view.headline is not None
    assert view.headline.estimate == pytest.approx(fixture_mention_rate("Glovo"))
    assert view.headline.n_answers == 6
    assert view.headline.n_questions == 3
    assert view.table == ()


def test_the_sample_label_is_dated_and_never_says_market_share_as_a_claim(
    runs_dir: Path,
) -> None:
    label = build_results_view(make_data(), runs_dir).sample_label
    assert "2026-10-26" in label
    assert "not market share" in label


def test_the_convergence_chart_is_drawn_only_when_a_target_is_given(runs_dir: Path) -> None:
    without = build_results_view(make_data(), runs_dir)
    assert without.chart_paths == ()
    with_target = build_results_view(make_data(), runs_dir, target_half_width=0.1)
    chart = Path(with_target.chart_paths[0])
    assert chart.parent == runs_dir / RUN_ID / "charts"
    assert chart.exists()


# ---- layer B: the per-brand table comes from the vetted statistics --------------------------


def test_the_table_has_one_row_per_brand_with_the_vetted_figures(
    runs_with_answers: Path,
) -> None:
    view = build_results_view(make_data(), runs_with_answers)
    assert sorted(row.brand for row in view.table) == ["Glovo", "Just Eat", "Uber Eats"]
    by_brand = {row.brand: row for row in view.table}
    for brand in by_brand:
        assert by_brand[brand].estimate == pytest.approx(fixture_mention_rate(brand))
    assert Path(view.chart_paths[0]).name == "mention_rate_by_brand.png"


def test_too_few_questions_raises_a_visible_note(runs_with_answers: Path) -> None:
    view = build_results_view(make_data(), runs_with_answers)
    assert any("few questions" in note.lower() for note in view.notes)


def test_missing_answers_explain_why_the_table_is_empty(runs_dir: Path) -> None:
    view = build_results_view(make_data(), runs_dir)
    assert any("analyzed answers" in note for note in view.notes)


def test_a_headline_that_disagrees_with_the_table_is_flagged(runs_with_answers: Path) -> None:
    agree = build_results_view(make_data(), runs_with_answers)
    assert not any("differ" in note for note in agree.notes)
    disagree = build_results_view(make_data(brand_share=0.95), runs_with_answers)
    assert any("differ" in note for note in disagree.notes)


def test_a_client_brand_missing_from_the_table_is_flagged(runs_with_answers: Path) -> None:
    data = make_data().model_copy(update={"brand": "Unknown Brand"})
    view = build_results_view(data, runs_with_answers)
    assert any("Unknown Brand" in note for note in view.notes)


def test_the_table_shows_percentages_with_intervals_and_n_a(runs_with_answers: Path) -> None:
    view = build_results_view(make_data(), runs_with_answers, metric="recommendation_conversion")
    cells = [cell for row in table_rows(view) for cell in row]
    assert any("(" in cell and "%" in cell for cell in cells)
    never = view.model_copy(update={"table": ()})
    assert table_rows(never) == []


def test_an_undefined_metric_is_shown_as_n_a(runs_with_answers: Path) -> None:
    view = build_results_view(make_data(), runs_with_answers, metric="recommendation_conversion")
    undefined = view.table[0].model_copy(update={"estimate": None, "ci_low": None, "ci_high": None})
    shown = table_rows(view.model_copy(update={"table": (undefined,)}))
    assert shown[0][1] == "n/a"


# ---- layer C: downloads exist only after the approval ---------------------------------------


def test_there_are_no_downloads_before_the_report_exists(runs_dir: Path) -> None:
    view = build_results_view(make_data(), runs_dir)
    assert view.downloads == ()
    assert any("approved" in note for note in view.notes)


def test_after_approval_the_report_and_only_this_runs_tracking_row_are_offered(
    runs_dir: Path,
) -> None:
    run_dir = runs_dir / RUN_ID
    (run_dir / REPORT_FILE_NAME).write_text("# report", encoding="utf-8")
    other_row = dict.fromkeys(FIELDS, "x")
    other_row["run_id"] = "run-other-secret"
    upsert_row(runs_dir / TRACKING_FILE_NAME, other_row)

    view = build_results_view(make_data(), runs_dir)

    offered = [Path(path) for path in view.downloads]
    assert {path.name for path in offered} == {REPORT_FILE_NAME, "tracking_row.csv"}
    assert all(path.parent == run_dir for path in offered)
    row_text = (run_dir / "tracking_row.csv").read_text(encoding="utf-8")
    assert RUN_ID in row_text
    assert "run-other-secret" not in row_text


def test_the_tracking_row_is_not_duplicated_when_the_view_is_built_twice(
    runs_dir: Path,
) -> None:
    (runs_dir / RUN_ID / REPORT_FILE_NAME).write_text("# report", encoding="utf-8")
    build_results_view(make_data(), runs_dir)
    build_results_view(make_data(), runs_dir)
    lines = (runs_dir / RUN_ID / "tracking_row.csv").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2  # header and one row


@pytest.mark.parametrize("run_id", ["../escape", "a/b", "..", "run id"])
def test_an_unsafe_run_id_is_rejected_before_touching_the_disk(runs_dir: Path, run_id: str) -> None:
    with pytest.raises(ValueError, match="run_id"):
        build_results_view(make_data(run_id=run_id), runs_dir)
