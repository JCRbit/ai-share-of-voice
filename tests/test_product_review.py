"""Improvements from the product review (R11): errors, progress, results, safety, history."""

from pathlib import Path

import pytest

from ai_share_of_voice.config import load_settings
from ai_share_of_voice.ui.components import (
    friendly_failure,
    glossary_html,
    insights_html,
    progress_html,
    results_html,
    safe_markdown,
)
from ai_share_of_voice.ui.run_controller import BriefForm, RunController, RunView

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SIMULATED = {"LLM_MODE": "simulated", "SPEND_CAP_USD": "0.5"}


def form(**overrides):
    values = {
        "brand": "Glovo",
        "competitors": "Just Eat, Uber Eats",
        "sector": "comida a domicilio",
        "city": "Madrid",
        "country": "España",
        "language": "es",
        "panel": "demo",
        "questions": 10,
        "batches": 2,
    }
    values.update(overrides)
    return BriefForm(**values)


@pytest.fixture
def controller(tmp_path):
    return RunController(load_settings(SIMULATED), runs_dir=tmp_path / "runs", data_dir=DATA_DIR)


def finished(controller):
    run_id = controller.start(form())
    controller.wait(run_id, timeout=60)
    controller.decide(run_id, approved=True)
    controller.wait(run_id, timeout=60)
    return run_id


# --- errors in plain words ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "reason", "expected"),
    [
        ("tool_failure", "RateLimitError: 429 ... tokens per day (TPD)", "cuota gratuita"),
        ("tool_failure", "QuotaExhausted: quota exceeded", "cuota gratuita"),
        ("budget_exhausted", "Spending cap exceeded", "gasto máximo"),
        ("tool_failure", "sources failed: search failed with HTTP 432", "buscar fuentes"),
        ("no_evidence", "no analyzed answers to measure", "respuestas suficientes"),
        ("unexpected", "KeyError: 'x'", "registro técnico"),
    ],
)
def test_a_failure_is_explained_in_plain_words(kind, reason, expected):
    assert expected in friendly_failure("es", kind, reason)


def test_the_progress_shows_the_round_and_the_friendly_failure():
    view = RunView(
        run_id="r",
        phase="partial",
        cap_usd=0.5,
        rounds_done=1,
        max_rounds=3,
        failure_kind="tool_failure",
        failure_reason="RateLimitError: tokens per day",
    )

    html = progress_html(view, "es")

    assert "Ronda 1 de 3" in html
    assert "cuota gratuita" in html
    assert "RateLimitError" not in html  # the technical text stays in the log


def test_the_view_reports_rounds_and_the_failure(controller):
    run_id = controller.start(form())
    view = controller.wait(run_id, timeout=60)

    assert (view.rounds_done, view.max_rounds) == (2, 2)
    assert view.failure_kind is None


# --- results: model, actions, glossary --------------------------------------------------------


def test_results_say_which_ai_was_measured(controller):
    run_id = finished(controller)

    bundle = controller.results(run_id)

    assert bundle.subject_model == "simulated"
    assert "simulated" in results_html("es", bundle)


def test_actions_follow_from_the_figures(controller):
    run_id = finished(controller)
    bundle = controller.results(run_id)

    html = insights_html("es", bundle)

    assert "Qué puedes hacer ahora" in html
    assert "Just Eat" in html  # the demo leader is not the client
    assert "Tienda Local Ejemplo" in html  # suggested as a competitor


def test_the_glossary_explains_the_terms_in_both_languages():
    assert "Margen de error" in glossary_html("es")
    assert "Margin of error" in glossary_html("en")


# --- the draft is formatted but safe ----------------------------------------------------------


def test_the_draft_markdown_has_no_html_and_no_images():
    text = "# Informe\n\n<script>x()</script> ![pixel](https://evil.example/p.gif) **ok**"

    safe = safe_markdown(text)

    assert "<script>" not in safe and "&lt;script&gt;" in safe
    assert "evil.example" not in safe and "![" not in safe
    assert "# Informe" in safe and "**ok**" in safe


# --- my measurements ----------------------------------------------------------------------------


def test_past_runs_are_listed_and_reopened_after_a_restart(controller, tmp_path):
    run_id = finished(controller)

    restarted = RunController(
        load_settings(SIMULATED), runs_dir=tmp_path / "runs", data_dir=DATA_DIR
    )
    history = restarted.history()

    assert [item.run_id for item in history] == [run_id]
    assert history[0].brand == "Glovo" and 0 <= history[0].share <= 1

    restarted.open_run(run_id)
    view = restarted.view(run_id)
    bundle = restarted.results(run_id)
    assert view.phase == "completed"
    assert bundle.view is not None and bundle.view.headline is not None
    assert restarted.example_answer(run_id) is not None


def test_an_unknown_run_cannot_be_opened(controller):
    with pytest.raises(KeyError):
        controller.open_run("run-does-not-exist")
