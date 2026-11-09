import csv
import os
import re
import shutil
from pathlib import Path

import pandas as pd
import pytest

from ai_share_of_voice.event_log import EventLog
from ai_share_of_voice.schemas.events import EventStatus, GraphEvent
from ai_share_of_voice.stats.metrics import ANSWER_COLUMNS, METRICS, brand_metrics, load_answers
from ai_share_of_voice.ui import results_tab
from ai_share_of_voice.ui.results_tab import (
    ResultsError,
    ResultsPanel,
    ResultsView,
    format_estimate,
    format_interval,
    list_runs,
    load_results,
    safe_csv_text,
    show_results,
    summary_text,
    view_for_ended_run,
)
from ai_share_of_voice.ui.run_manager import RunSnapshot

FIXTURE = Path(__file__).parent / "fixtures" / "sample_answers.csv"
RUN = "run-20261101-120000-abc123"
BRANDS = ("Acme Food", "Beta Eats", "Never Seen")
PNG = b"\x89PNG\r\n\x1a\n"


def make_answers(brands=BRANDS, questions: int = 12) -> pd.DataFrame:
    """Acme is mentioned and recommended on even questions, Beta on odd ones, the third never."""
    rows = []
    for number in range(questions):
        for batch in range(2):
            response = f"q{number:02d}-b{batch}"
            for brand in brands:
                if brand == brands[0] and number % 2 == 0:
                    detail = (True, True, 1, "positive")
                elif brand == brands[1] and number % 2 == 1:
                    detail = (True, False, 1, "neutral")
                else:
                    detail = (False, False, None, None)
                rows.append([response, f"q{number:02d}", batch, brand, *detail])
    return pd.DataFrame(rows, columns=list(ANSWER_COLUMNS))


def make_run(runs: Path, run_id: str = RUN, frame: pd.DataFrame | None = None) -> Path:
    folder = runs / run_id
    folder.mkdir(parents=True)
    (frame if frame is not None else make_answers()).to_csv(folder / "answers.csv", index=False)
    return folder


def snapshot(status: str = "completed", run_id: str | None = RUN) -> RunSnapshot:
    return RunSnapshot(
        run_id=run_id, status=status, events=(), spent_usd=0.0, cap_usd=2.0, error=None
    )


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def percentages(text: str) -> list[float]:
    return [float(number) for number in re.findall(r"(-?\d+(?:\.\d+)?)%", text)]


# ---- which runs are listed -------------------------------------------------------------------


def test_only_runs_with_answers_are_listed_newest_first(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    make_run(runs, "run-20261101-100000-aaaaaa")
    make_run(runs, "run-20261102-100000-bbbbbb")
    (runs / "run-20261103-100000-cccccc").mkdir()
    make_run(runs, "bad name")
    (runs / "tracking.csv").write_text("run_id\n", encoding="utf-8")
    assert list_runs(runs) == ["run-20261102-100000-bbbbbb", "run-20261101-100000-aaaaaa"]
    assert list_runs(tmp_path / "missing") == []


# ---- figures and intervals -------------------------------------------------------------------


def test_a_run_becomes_a_table_with_every_metric_of_every_brand(tmp_path: Path) -> None:
    make_run(tmp_path)
    results = load_results(tmp_path, RUN)
    assert len(results.rows) == len(METRICS) * len(BRANDS)
    assert all(len(row) == len(results_tab.TABLE_HEADERS) for row in results.rows)
    assert [row[1] for row in results.rows[:3]] == list(BRANDS)
    metrics = list(dict.fromkeys(row[0] for row in results.rows))
    assert metrics[:3] == ["Mention rate", "Recommendation rate", "Top-1 rate"]
    assert (results.n_answers, results.n_questions, results.n_batches) == (24, 12, 2)
    assert results.few_questions is False


def test_the_estimates_are_right_and_the_interval_contains_them(tmp_path: Path) -> None:
    make_run(tmp_path)
    rows = {(row[0], row[1]): row for row in load_results(tmp_path, RUN).rows}
    acme = rows[("Mention rate", "Acme Food")]
    assert acme[2] == "50.0%" and rows[("Mention rate", "Never Seen")][2] == "0.0%"
    low, high = percentages(acme[3])
    assert low < 50.0 < high
    assert rows[("Recommendation rate", "Beta Eats")][2] == "0.0%"
    assert rows[("Net tone", "Acme Food")][2] == "+1.00"


def test_the_figures_match_the_vetted_statistics(tmp_path: Path) -> None:
    folder = make_run(tmp_path)
    table = brand_metrics(load_answers(folder / "answers.csv"))
    expected = table[(table["metric"] == "mention_rate") & (table["brand"] == "Acme Food")].iloc[0]
    row = load_results(tmp_path, RUN).rows[0]
    assert row[2] == format_estimate("mention_rate", expected["estimate"])
    assert row[3] == format_interval("mention_rate", expected["ci_low"], expected["ci_high"])


def test_a_metric_that_cannot_be_computed_is_not_defined_never_zero(tmp_path: Path) -> None:
    make_run(tmp_path)
    rows = {(row[0], row[1]): row for row in load_results(tmp_path, RUN).rows}
    never = rows[("Recommendation conversion", "Never Seen")]
    assert never[2] == "n/a" and never[3] == "n/a" and "not defined" in never[4]
    assert rows[("Mention rate", "Never Seen")][4] == ""


def test_rates_are_percentages_and_tone_is_a_signed_score() -> None:
    assert format_estimate("mention_rate", 0.5) == "50.0%"
    assert format_estimate("net_tone", 0.25) == "+0.25"
    assert format_estimate("net_tone", -1.0) == "-1.00"
    assert format_estimate("mention_rate", float("nan")) == "n/a"
    assert format_estimate("mention_rate", None) == "n/a"
    assert format_interval("mention_rate", 0.25, 0.75) == "25.0% to 75.0%"
    assert format_interval("net_tone", -0.5, 0.5) == "-0.50 to +0.50"
    assert format_interval("mention_rate", float("nan"), 0.5) == "n/a"


def test_too_few_questions_are_flagged_in_the_summary(tmp_path: Path) -> None:
    folder = tmp_path / RUN
    folder.mkdir(parents=True)
    shutil.copy(FIXTURE, folder / "answers.csv")
    results = load_results(tmp_path, RUN)
    assert results.few_questions is True and results.n_questions == 3
    text = summary_text(tmp_path, results)
    assert "Only 3 questions" in text and "too optimistic" in text


def test_the_summary_says_what_the_figures_are_and_are_not(tmp_path: Path) -> None:
    make_run(tmp_path)
    text = summary_text(tmp_path, load_results(tmp_path, RUN))
    assert "not a market share" in text and "coverage has not been validated" in text
    assert "24 answers to 12 questions, in 2 batches" in text and "Only" not in text


# ---- charts ----------------------------------------------------------------------------------


def test_every_metric_gets_its_chart_and_a_current_chart_is_not_redrawn(tmp_path: Path) -> None:
    folder = make_run(tmp_path)
    results = load_results(tmp_path, RUN)
    names = [Path(path).name for path, _ in results.charts]
    assert names == [f"{metric}_by_brand.png" for metric in METRICS]
    assert [caption for _, caption in results.charts][0] == "Mention rate"
    assert all(Path(path).read_bytes().startswith(PNG) for path, _ in results.charts)
    chart = folder / "charts" / "mention_rate_by_brand.png"
    before = chart.stat().st_mtime_ns
    load_results(tmp_path, RUN)
    assert chart.stat().st_mtime_ns == before


def test_a_chart_older_than_the_answers_is_drawn_again(tmp_path: Path) -> None:
    folder = make_run(tmp_path)
    load_results(tmp_path, RUN)
    chart = folder / "charts" / "mention_rate_by_brand.png"
    old = chart.stat().st_mtime - 100
    os.utime(chart, (old, old))
    load_results(tmp_path, RUN)
    assert chart.stat().st_mtime > old + 50


# ---- downloads -------------------------------------------------------------------------------


def test_the_downloads_come_in_a_fixed_order_and_only_if_they_exist(tmp_path: Path) -> None:
    folder = make_run(tmp_path)
    bare = [Path(p).name for p in load_results(tmp_path, RUN).downloads]
    assert bare == ["answers.csv", "metrics.csv", *[f"{m}_by_brand.png" for m in METRICS]]

    (folder / "report.md").write_text("# Report\n", encoding="utf-8")
    (tmp_path / "tracking.csv").write_text("run_id\n", encoding="utf-8")
    EventLog(tmp_path).append(GraphEvent(run_id=RUN, node="sampling", status=EventStatus.STARTED))
    full = [Path(p).name for p in load_results(tmp_path, RUN).downloads]
    assert full[:4] == ["report.md", "tracking.csv", "answers.csv", "metrics.csv"]
    assert full[-1] == "events.jsonl" and len(full) == len(bare) + 3


def test_a_file_that_is_really_somewhere_else_is_not_offered(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    folder = make_run(runs)
    secret = tmp_path / "secret.md"
    secret.write_text("private", encoding="utf-8")
    try:
        (folder / "report.md").symlink_to(secret)
    except OSError:
        pytest.skip("symbolic links are not available here")
    results = load_results(runs, RUN)
    offered = [Path(path).name for path in results.downloads]
    assert "report.md" not in offered and "secret.md" not in offered
    assert results.has_report is False


def test_the_csv_downloads_cannot_start_a_formula_and_the_originals_are_untouched(
    tmp_path: Path,
) -> None:
    brands = ('=HYPERLINK("http://evil.example")', "+34 Taxi", "@cmd")
    folder = make_run(tmp_path, frame=make_answers(brands))
    original = (folder / "answers.csv").read_text(encoding="utf-8")
    results = load_results(tmp_path, RUN)
    for name in ("answers.csv", "metrics.csv"):
        copy = folder / "downloads" / name
        assert str(copy) in results.downloads
        shown = {row["brand"] for row in csv_rows(copy)}
        assert shown == {"'" + brand for brand in brands}
    assert (folder / "answers.csv").read_text(encoding="utf-8") == original
    assert set(load_answers(folder / "answers.csv")["brand"]) == set(brands)
    numbers = csv_rows(folder / "downloads" / "metrics.csv")[0]
    assert 0 <= float(numbers["estimate"]) <= 1 and numbers["metric"] == "mention_rate"


def test_safe_csv_text_only_touches_text_cells() -> None:
    frame = pd.DataFrame({"brand": ["=x", "ok", "-1"], "estimate": [-0.5, 0.25, 1.0]})
    rows = list(csv.DictReader(safe_csv_text(frame).splitlines()))
    assert [r["brand"] for r in rows] == ["'=x", "ok", "'-1"]
    assert [r["estimate"] for r in rows] == ["-0.5", "0.25", "1.0"]


def test_looking_at_a_run_twice_does_not_rewrite_its_download_files(tmp_path: Path) -> None:
    folder = make_run(tmp_path)
    load_results(tmp_path, RUN)
    copy = folder / "downloads" / "answers.csv"
    old = (copy.stat().st_mtime_ns // 2_000_000_000 - 50) * 2_000_000_000
    os.utime(copy, ns=(old, old))
    load_results(tmp_path, RUN)
    assert copy.stat().st_mtime_ns == old


# ---- the metrics are computed once -----------------------------------------------------------


def test_the_metrics_are_computed_once_until_the_answers_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = make_run(tmp_path)
    calls: list[int] = []

    def counting(answers: pd.DataFrame) -> pd.DataFrame:
        calls.append(len(answers))
        return brand_metrics(answers)

    monkeypatch.setattr(results_tab, "brand_metrics", counting)
    results_tab._CACHE.clear()
    load_results(tmp_path, RUN)
    load_results(tmp_path, RUN)
    assert len(calls) == 1
    make_answers(questions=14).to_csv(folder / "answers.csv", index=False)
    assert load_results(tmp_path, RUN).n_questions == 14 and len(calls) == 2


# ---- how it ended ----------------------------------------------------------------------------


def test_the_summary_tells_how_the_run_ended_and_whether_it_has_a_report(tmp_path: Path) -> None:
    folder = make_run(tmp_path)
    assert "How it ended" not in summary_text(tmp_path, load_results(tmp_path, RUN))
    EventLog(tmp_path).append(
        GraphEvent(
            run_id=RUN, node="finalize", status=EventStatus.SUCCEEDED, cumulative_spend_usd=0.3
        )
    )
    text = summary_text(tmp_path, load_results(tmp_path, RUN))
    assert "How it ended: completed, spend $0.30." in text
    assert "No report was written" in text
    (folder / "report.md").write_text("# Report\n", encoding="utf-8")
    assert "approved report is in the downloads" in summary_text(
        tmp_path, load_results(tmp_path, RUN)
    )


# ---- problems become messages ----------------------------------------------------------------


@pytest.mark.parametrize("run_id", ["", ".", "..", "../outside", "a/b", " run", "run name"])
def test_a_run_id_that_is_not_a_plain_folder_name_is_refused(tmp_path: Path, run_id: str) -> None:
    make_run(tmp_path)
    with pytest.raises(ResultsError, match="choose a run"):
        load_results(tmp_path, run_id)
    assert show_results(tmp_path, run_id).rows == []


def test_a_run_without_answers_says_so(tmp_path: Path) -> None:
    (tmp_path / RUN).mkdir()
    with pytest.raises(ResultsError, match="no analyzed answers"):
        load_results(tmp_path, RUN)
    assert "no analyzed answers" in show_results(tmp_path, RUN).summary


def test_a_damaged_answers_file_is_reported_without_echoing_its_content(tmp_path: Path) -> None:
    folder = tmp_path / RUN
    folder.mkdir()
    (folder / "answers.csv").write_text("secret,other\nTOP-SECRET-VALUE,1\n", encoding="utf-8")
    view = show_results(tmp_path, RUN)
    assert "cannot be read" in view.summary and "TOP-SECRET-VALUE" not in view.summary
    assert view.rows == [] and view.charts == [] and view.files == []


def test_too_little_data_is_a_message_not_a_crash(tmp_path: Path) -> None:
    make_run(tmp_path, frame=make_answers(questions=1))
    assert "not enough data" in show_results(tmp_path, RUN).summary


def test_no_choice_gives_a_hint(tmp_path: Path) -> None:
    assert show_results(tmp_path, None).summary == "Choose a run to see its results."


# ---- what to show when a run ends ------------------------------------------------------------


def test_nothing_changes_while_a_run_is_going_or_before_one_starts(tmp_path: Path) -> None:
    make_run(tmp_path)
    assert view_for_ended_run(tmp_path, snapshot("running")) is None
    assert view_for_ended_run(tmp_path, snapshot("idle", run_id=None)) is None


def test_a_run_that_ended_with_figures_is_selected_and_shown(tmp_path: Path) -> None:
    make_run(tmp_path)
    selected, view = view_for_ended_run(tmp_path, snapshot("completed"))
    assert selected == RUN and len(view.rows) == len(METRICS) * len(BRANDS)
    assert view.charts and view.files


def test_a_run_that_ended_without_figures_says_why_and_selects_nothing(tmp_path: Path) -> None:
    selected, view = view_for_ended_run(tmp_path, snapshot("partial"))
    assert selected is None and view.rows == []
    assert "closed in a partial state" in view.summary and "no analyzed answers" in view.summary


def test_the_view_and_the_panel_agree_on_the_order_of_the_outputs() -> None:
    view = ResultsView("text", [["a"]], [("p.png", "c")], ["f.csv"])
    assert view.as_outputs() == ("text", [["a"]], [("p.png", "c")], ["f.csv"])
    panel = ResultsPanel(run="r", refresh="b", summary="s", table="t", charts="c", files="f")
    assert panel.outputs == ["s", "t", "c", "f"]


# ---- the components --------------------------------------------------------------------------


def test_the_tab_builds_its_components_with_the_newest_run_selected(tmp_path: Path) -> None:
    gr = pytest.importorskip("gradio")
    make_run(tmp_path, "run-20261101-100000-aaaaaa")
    make_run(tmp_path, "run-20261102-100000-bbbbbb")
    with gr.Blocks():
        panel = results_tab.build_results_tab(tmp_path, "hello")
    assert len(panel.outputs) == 4
    assert panel.run.value == "run-20261102-100000-bbbbbb"


def test_downloads_directory_symlink_cannot_write_outside_runs(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    folder = make_run(runs)
    external = tmp_path / "external_downloads"
    external.mkdir()

    try:
        (folder / "downloads").symlink_to(external, target_is_directory=True)
    except OSError:
        pytest.skip("directory symbolic links are not available here")

    load_results(runs, RUN)

    assert list(external.iterdir()) == []


def test_charts_directory_symlink_cannot_write_outside_runs(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    folder = make_run(runs)
    external = tmp_path / "external_charts"
    external.mkdir()

    try:
        (folder / "charts").symlink_to(external, target_is_directory=True)
    except OSError:
        pytest.skip("directory symbolic links are not available here")

    load_results(runs, RUN)

    assert list(external.iterdir()) == []
