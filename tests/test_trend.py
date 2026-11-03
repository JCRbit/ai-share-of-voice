"""Trend over time (R12): which runs are comparable, the change rule and the chart."""

import csv
from pathlib import Path

import pytest

from ai_share_of_voice.config import load_settings
from ai_share_of_voice.schemas.analysis import RawAnswer
from ai_share_of_voice.stats.charts import TrendChartText, TrendPoint, _trend_figure, trend_chart
from ai_share_of_voice.stats.trend import compare
from ai_share_of_voice.ui import components as ui
from ai_share_of_voice.ui.run_controller import MAX_TREND_POINTS, RunController

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SIMULATED = {"LLM_MODE": "simulated", "SPEND_CAP_USD": "0.5"}
TEXT = TrendChartText(
    title="¿Cómo evoluciona Glovo?", axis="% de respuestas", note="Banda: margen."
)
POINTS = [
    TrendPoint("11-01 10:00", (0.40, 0.30, 0.50)),
    TrendPoint("11-02 10:00", (0.55, 0.45, 0.65)),
    TrendPoint("11-03 10:00", (0.70, 0.60, 0.80)),
]


# --- the change rule ------------------------------------------------------------------------------


def test_a_change_is_clear_only_when_the_intervals_do_not_overlap():
    assert compare((0.40, 0.30, 0.50), (0.70, 0.60, 0.80)) == "up"
    assert compare((0.70, 0.60, 0.80), (0.40, 0.30, 0.50)) == "down"


@pytest.mark.parametrize(
    "latest",
    [(0.55, 0.45, 0.65), (0.60, 0.50, 0.70), (0.40, 0.30, 0.50), (0.30, 0.20, 0.30)],
)
def test_overlapping_or_touching_intervals_are_unclear(latest):
    assert compare((0.40, 0.30, 0.50), latest) == "unclear"


# --- the chart ------------------------------------------------------------------------------------


def test_the_trend_chart_shows_each_measurement_in_order():
    axes = _trend_figure(POINTS, TEXT).axes[0]

    assert [tick.get_text() for tick in axes.get_xticklabels()] == [
        "11-01 10:00",
        "11-02 10:00",
        "11-03 10:00",
    ]
    assert axes.get_title(loc="left") == "¿Cómo evoluciona Glovo?"
    assert list(axes.lines[0].get_ydata()) == [0.40, 0.55, 0.70]


def test_the_trend_chart_is_reproducible(tmp_path):
    first = trend_chart(POINTS, tmp_path / "a.png", TEXT)
    second = trend_chart(POINTS, tmp_path / "b.png", TEXT)

    assert first.read_bytes() == second.read_bytes()


def test_a_trend_needs_two_measurements(tmp_path):
    with pytest.raises(ValueError, match="at least two"):
        trend_chart(POINTS[:1], tmp_path / "x.png", TEXT)


# --- which runs are compared ----------------------------------------------------------------------

FIELDS = ["run_id", "generated_at", "brand", "sector", "city", "n_answers", "brand_share"]
FIELDS += ["ci_low", "ci_high", "batches", "spend_usd", "stop_reason", "sources"]
FIELDS += ["open_objections", "report_path"]


def past_run(runs: Path, run_id: str, *, model="gpt-oss-20b", brand="Glovo", city="Madrid"):
    day = run_id[8:10]
    raw = runs / run_id / "raw"
    raw.mkdir(parents=True)
    answer = RawAnswer(response_id="r1", question_id="q1", batch_index=0, model=model, text="x")
    (raw / "batch_000.jsonl").write_text(answer.model_dump_json() + "\n", encoding="utf-8")
    return {
        "run_id": run_id,
        "generated_at": f"2026-11-{day}T10:00:00+00:00",
        "brand": brand,
        "sector": "food delivery",
        "city": city,
        "n_answers": "30",
        "brand_share": "0.5000",
        "ci_low": "0.4000",
        "ci_high": "0.6000",
        "batches": "3",
        "spend_usd": "0.0100",
        "stop_reason": "max_batches",
        "sources": "0",
        "open_objections": "0",
        "report_path": "",
    }


@pytest.fixture
def runs(tmp_path):
    folder = tmp_path / "runs"
    rows = [
        past_run(folder, "run-2026110-a"),
        past_run(folder, "run-2026112-b", model="gemini-3.6-flash"),  # another AI
        past_run(folder, "run-2026113-c", city="Lisboa"),  # another city
        past_run(folder, "run-2026114-d", brand="glovo"),  # same brand, other casing
        past_run(folder, "run-2026115-e"),
        past_run(folder, "run-2026116-f"),  # newer than the run being looked at
    ]
    with (folder / "tracking.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return folder


def test_the_trend_compares_the_same_brand_place_and_ai_up_to_the_run(runs):
    controller = RunController(load_settings(SIMULATED), runs_dir=runs, data_dir=DATA_DIR)

    trend = controller.trend("run-2026115-e")

    assert [item.run_id for item in trend] == ["run-2026110-a", "run-2026114-d", "run-2026115-e"]


def test_a_run_that_is_not_approved_has_no_trend(runs):
    controller = RunController(load_settings(SIMULATED), runs_dir=runs, data_dir=DATA_DIR)

    assert controller.trend("run-not-in-tracking") == []


def test_the_trend_chart_is_drawn_only_with_two_comparable_runs(runs):
    controller = RunController(load_settings(SIMULATED), runs_dir=runs, data_dir=DATA_DIR)

    assert controller.trend_chart("run-2026110-a", "es") is None
    path = controller.trend_chart("run-2026115-e", "es")
    assert path is not None and Path(path).is_file()


# --- the caption ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("latest", "expected"),
    [((0.7, 0.6, 0.8), "cambio claro"), ((0.45, 0.35, 0.55), "puede deberse al azar")],
)
def test_the_caption_says_whether_the_change_is_clear(latest, expected):
    caption = ui.trend_caption("es", "Glovo", "Madrid", "gpt-oss-20b", (0.4, 0.3, 0.5), latest)

    assert expected in caption
    assert "40 %" in caption or "40%" in caption


def test_the_caption_says_when_the_figure_holds():
    caption = ui.trend_caption(
        "es", "Glovo", "Madrid", "gpt-oss-20b", (0.7, 0.4, 1.0), (0.7, 0.4, 1.0)
    )

    assert "Se mantiene en el 70%" in caption and "de 70% a 70%" not in caption


def test_the_trend_shows_at_most_the_last_measurements(tmp_path):
    folder = tmp_path / "runs"
    rows = [past_run(folder, f"run-202611{day:02d}-x") for day in range(1, 16)]
    with (folder / "tracking.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    controller = RunController(load_settings(SIMULATED), runs_dir=folder, data_dir=DATA_DIR)

    trend = controller.trend("run-20261115-x")

    assert len(trend) == MAX_TREND_POINTS
    assert trend[-1].run_id == "run-20261115-x" and trend[0].run_id == "run-20261104-x"
