"""Trend over time (R12): which runs are comparable, the change rule and the chart."""

import csv
from pathlib import Path

import pytest

from ai_share_of_voice.config import load_settings
from ai_share_of_voice.panel import panel_fingerprint, write_panel_file
from ai_share_of_voice.schemas.analysis import RawAnswer
from ai_share_of_voice.schemas.questions import Question, QuestionPanel
from ai_share_of_voice.stats.charts import TrendChartText, TrendPoint, _trend_figure, trend_chart
from ai_share_of_voice.stats.trend import compare
from ai_share_of_voice.ui import components as ui
from ai_share_of_voice.ui.run_controller import MAX_TREND_POINTS, RunController, panel_letters

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


PANEL_A = ("¿Cuál es la mejor app de reparto?", "¿Qué app me recomiendas para cenar?")
PANEL_B = ("¿Qué app de reparto es la más barata?", "¿Qué app me recomiendas para cenar?")


def make_panel(texts) -> QuestionPanel:
    return QuestionPanel(
        questions=tuple(
            Question(id=f"q{n:02d}", text=text, intent="recommendation")
            for n, text in enumerate(texts, start=1)
        )
    )


def past_run(
    runs: Path, run_id: str, *, model="gpt-oss-20b", brand="Glovo", city="Madrid", panel=PANEL_A
):
    day = run_id[8:10]
    raw = runs / run_id / "raw"
    raw.mkdir(parents=True)
    if panel is not None:
        write_panel_file(make_panel(panel), runs / run_id / "questions.csv")
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


# --- the question set behind each point -----------------------------------------------------------


def test_the_fingerprint_depends_on_the_questions_not_on_their_order_or_ids():
    same = make_panel(tuple(reversed(PANEL_A)))
    spaced = make_panel(("  ¿CUÁL es la mejor   app de reparto? ", PANEL_A[1]))

    assert panel_fingerprint(make_panel(PANEL_A)) == panel_fingerprint(same)
    assert panel_fingerprint(make_panel(PANEL_A)) == panel_fingerprint(spaced)
    assert panel_fingerprint(make_panel(PANEL_A)) != panel_fingerprint(make_panel(PANEL_B))


def write_tracking(folder: Path, rows) -> None:
    with (folder / "tracking.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_each_past_run_records_its_question_set(tmp_path):
    folder = tmp_path / "runs"
    write_tracking(
        folder, [past_run(folder, "run-2026111-a"), past_run(folder, "run-2026112-b", panel=None)]
    )
    controller = RunController(load_settings(SIMULATED), runs_dir=folder, data_dir=DATA_DIR)

    runs = {item.run_id: item for item in controller.history()}

    assert runs["run-2026111-a"].panel_id == panel_fingerprint(make_panel(PANEL_A))
    assert runs["run-2026111-a"].n_questions == 2
    assert runs["run-2026112-b"].panel_id == "" and runs["run-2026112-b"].n_questions == 0


def test_question_sets_get_letters_in_order_of_first_use(tmp_path):
    folder = tmp_path / "runs"
    write_tracking(
        folder,
        [
            past_run(folder, "run-2026111-a"),
            past_run(folder, "run-2026112-b", panel=PANEL_B),
            past_run(folder, "run-2026113-c"),
            past_run(folder, "run-2026114-d", panel=None),
        ],
    )
    controller = RunController(load_settings(SIMULATED), runs_dir=folder, data_dir=DATA_DIR)

    letters = panel_letters(controller.trend("run-2026114-d"))

    assert [letters[item.run_id] for item in controller.trend("run-2026114-d")] == [
        "A",
        "B",
        "A",
        "?",
    ]


def test_the_trend_file_lets_you_trace_every_point_to_its_questions(tmp_path):
    folder = tmp_path / "runs"
    write_tracking(
        folder,
        [past_run(folder, "run-2026111-a"), past_run(folder, "run-2026112-b", panel=PANEL_B)],
    )
    controller = RunController(load_settings(SIMULATED), runs_dir=folder, data_dir=DATA_DIR)

    path = controller.trend_file("run-2026112-b")

    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["run_id"] for row in rows] == ["run-2026111-a", "run-2026112-b"]
    assert [row["question_set"] for row in rows] == ["A", "B"]
    assert rows[1]["question_set_id"] == panel_fingerprint(make_panel(PANEL_B))
    assert rows[0]["questions_file"] == "run-2026111-a/questions.csv"
    assert rows[0]["model"] == "gpt-oss-20b" and rows[0]["n_questions"] == "2"


def test_the_questions_of_each_set_can_be_read(tmp_path):
    folder = tmp_path / "runs"
    write_tracking(
        folder,
        [past_run(folder, "run-2026111-a"), past_run(folder, "run-2026112-b", panel=PANEL_B)],
    )
    controller = RunController(load_settings(SIMULATED), runs_dir=folder, data_dir=DATA_DIR)

    sets = controller.trend_question_sets("run-2026112-b")

    assert sets == {"A": list(PANEL_A), "B": list(PANEL_B)}


def test_a_change_of_questions_is_never_called_a_clear_change():
    caption = ui.trend_caption(
        "es",
        "Glovo",
        "Madrid",
        "gpt-oss-20b",
        (0.4, 0.3, 0.5),
        (0.8, 0.7, 0.9),
        previous_set="A",
        latest_set="B",
    )

    assert "cambio claro" not in caption
    assert "las preguntas cambiaron" in caption and "A → B" in caption


def test_an_unknown_question_set_is_treated_as_a_change():
    caption = ui.trend_caption(
        "es",
        "Glovo",
        "Madrid",
        "m",
        (0.4, 0.3, 0.5),
        (0.8, 0.7, 0.9),
        previous_set="?",
        latest_set="?",
    )

    assert "cambio claro" not in caption


def test_the_trend_table_shows_each_point_with_its_question_set():
    html = ui.trend_table_html(
        "es",
        [("11-01 10:00", "A", 2, (0.4, 0.3, 0.5)), ("11-02 10:00", "B", 2, (0.6, 0.5, 0.7))],
        {"A": ["¿Cuál <b>es</b>?"], "B": ["¿Otra?"]},
    )

    assert "Preguntas A" in html and "Preguntas B" in html
    assert "&lt;b&gt;" in html and "<b>es</b>" not in html  # question text is escaped
    assert "<details" in html


def test_a_steady_figure_with_other_questions_says_it_is_not_an_exact_comparison():
    caption = ui.trend_caption(
        "es",
        "Glovo",
        "Madrid",
        "m",
        (0.7, 0.4, 1.0),
        (0.7, 0.4, 1.0),
        previous_set="C",
        latest_set="A",
    )

    assert "Se mantiene en el 70%" in caption and "no es una comparación exacta" in caption
    assert "de 70% a 70%" not in caption
