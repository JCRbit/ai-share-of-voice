import re
import string

import pytest

from ai_share_of_voice.ui.components import (
    comparison_html,
    discovered_html,
    draft_html,
    hero_html,
    kpis_html,
    progress_html,
    research_html,
    stage_states,
    summary_html,
)
from ai_share_of_voice.ui.i18n import EVENT_STATUS, KINDS, STAGES, TEXTS, note, t
from ai_share_of_voice.ui.run_controller import EventRow, RunView

# --- texts ----------------------------------------------------------------------------------


@pytest.mark.parametrize("table", [TEXTS, STAGES, EVENT_STATUS, KINDS])
def test_every_text_exists_in_both_languages(table):
    assert set(table["es"]) == set(table["en"])


def placeholders(text):
    return {name for _, name, _, _ in string.Formatter().parse(text) if name}


def test_both_languages_use_the_same_placeholders():
    for key in TEXTS["es"]:
        assert placeholders(TEXTS["es"][key]) == placeholders(TEXTS["en"][key]), key


def test_an_unknown_language_falls_back_to_spanish():
    assert t("fr", "start") == t("es", "start") == "Medir mi marca"
    assert t("en", "start") == "Measure my brand"


@pytest.mark.parametrize(
    ("english", "spanish_part"),
    [
        (
            "Few questions (5): the intervals are not reliable, read them as indicative only.",
            "Solo se usaron 5 preguntas",
        ),
        ("Downloads appear after the report is approved.", "cuando lo apruebes"),
        ("Glovo is not in the per-brand table: check the brand names.", "«Glovo»"),
    ],
)
def test_results_notes_are_translated(english, spanish_part):
    assert spanish_part in note("es", english)


def test_an_unknown_note_is_shown_as_it_is():
    assert note("es", "Something new.") == "Something new."


# --- safety: model and web content are escaped ----------------------------------------------


def test_a_draft_cannot_inject_html():
    html = draft_html("<script>alert(1)</script> **bold**", ("<img src=x onerror=1>",), "es")

    assert "<script>" not in html and "&lt;script&gt;" in html
    assert "<img" not in html


def test_only_http_links_become_links():
    research = {
        "sources": ["https://ok.example/a", "javascript:alert(1)"],
        "checks": [],
    }

    html = research_html("es", research)

    assert '<a href="https://ok.example/a"' in html
    assert 'href="javascript' not in html


def test_business_names_are_escaped():
    rows = [
        {
            "name": "<b>Evil</b>",
            "kind": "other",
            "rate": "0.5",
            "suggested": "True",
            "n_questions_with": "3",
        }
    ]

    html = discovered_html("es", rows)

    assert "&lt;b&gt;Evil&lt;/b&gt;" in html and "★" in html


# --- progress -------------------------------------------------------------------------------


def view(phase, *events):
    return RunView(
        run_id="run-x",
        phase=phase,
        events=tuple(
            EventRow(time="10:00:00", status=status, node=node) for node, status in events
        ),
        spend_usd=0.1,
        cap_usd=0.5,
    )


def test_stage_states_follow_the_events():
    states = stage_states(
        view(
            "waiting_approval",
            ("questions", "started"),
            ("questions", "succeeded"),
            ("sampling", "started"),
            ("sampling", "succeeded"),
            ("metrics", "started"),
            ("approval_request", "waiting_approval"),
        )
    )

    assert states["questions"] == states["sampling"] == "done"
    assert states["metrics"] == "active"
    assert states["approval_gate"] == "waiting"
    assert states["finalize"] == "pending"


def test_a_partial_run_marks_the_running_stage_as_failed():
    states = stage_states(view("partial", ("sampling", "started")))

    assert states["sampling"] == "failed"


def test_progress_shows_the_phase_in_plain_words_in_each_language():
    running = view("waiting_approval")

    assert "Esperando tu aprobación" in progress_html(running, "es")
    assert "Waiting for your approval" in progress_html(running, "en")
    assert "Preguntar a la IA" in progress_html(running, "es")


# --- results --------------------------------------------------------------------------------


def test_kpis_explain_the_interval_as_a_margin_of_error():
    html = kpis_html("es", mention=(0.7, 0.4, 1.0), recommendation=0.5, answers=30, spend_usd=0.01)

    assert "70%" in html and "Margen de error: entre 40% y 100%" in html
    assert "50%" in html and ">30<" in html


def test_the_summary_says_it_in_plain_words():
    rows = [("Glovo", 0.7), ("Just Eat", 0.8), ("Uber Eats", None)]

    html = summary_html("es", "Glovo", rows)

    assert re.search(r"<b>Glovo aparece en 7</b>", html)
    assert "<b>Just Eat</b>" in html  # the leader


def test_the_summary_says_when_the_client_leads():
    assert "Eres la marca más nombrada" in summary_html("es", "Glovo", [("Glovo", 0.9), ("X", 0.2)])


def test_the_comparison_highlights_the_client_and_sorts_by_rate():
    html = comparison_html("en", "Glovo", [("Glovo", 0.4, 0.2, 0.6), ("Just Eat", 0.8, 0.6, 1.0)])

    assert html.index("Just Eat") < html.index("Glovo")
    assert '<tr class="me"><td>Glovo' in html


def test_the_hero_explains_the_product_in_three_steps():
    html = hero_html("es", simulated=True)

    assert "¿Te recomienda la inteligencia artificial?" in html
    assert html.count('class="sov-step"') == 3
    assert "Modo demostración" in html
