import csv
import json
from pathlib import Path

import pytest

from ai_share_of_voice.agents.skeptic_checks import run_checks
from ai_share_of_voice.event_log import EventLog
from ai_share_of_voice.llm_client import Budget, SimulatedClient
from ai_share_of_voice.nodes import NodeDeps, fake_should_stop
from ai_share_of_voice.nodes_real import ANALYZED_FOLDER, DISCOVERED_FILE_NAME, make_real_nodes
from ai_share_of_voice.research import RESEARCH_FILE_NAME
from ai_share_of_voice.schemas.analysis import AnalyzedResponse, BrandMention
from ai_share_of_voice.schemas.brief import BrandSpec, Brief
from ai_share_of_voice.schemas.questions import Question, QuestionPanel
from ai_share_of_voice.writing import (
    DraftReport,
    collect_facts,
    facts_record,
    make_checked_skeptic_node,
    make_writer_node,
    template_draft,
)

RUN_ID = "run-w"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SOURCE = "https://www.example.com/apps-reparto"
EVIDENCE = "https://bolt.example.com/madrid"


def brief(language="es"):
    return Brief(
        brand=BrandSpec(name="Glovo"),
        competitors=(BrandSpec(name="Just Eat"), BrandSpec(name="Uber Eats")),
        sector="comida a domicilio",
        city="Madrid",
        language=language,
        spend_cap_usd=1.0,
        max_batches=3,
    )


def write_run(tmp_path):
    """Analyzed answers, discovery ranking and research of one finished batch."""
    run_dir = tmp_path / "runs" / RUN_ID
    items = []
    for index in range(1, 11):
        qid = f"q{index:02d}"
        glovo = index <= 7
        just_eat = index <= 8
        items.append(
            AnalyzedResponse(
                response_id=f"b000-{qid}",
                question_id=qid,
                batch_index=0,
                mentions=(
                    BrandMention(brand="Glovo", mentioned=glovo, recommended=glovo and index <= 5),
                    BrandMention(brand="Just Eat", mentioned=just_eat, recommended=just_eat),
                    BrandMention(brand="Uber Eats", mentioned=False),
                ),
            )
        )
    path = run_dir / ANALYZED_FOLDER / "batch_000.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("".join(item.model_dump_json() + "\n" for item in items), "utf-8")
    with (run_dir / DISCOVERED_FILE_NAME).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "name",
                "kind",
                "rate",
                "ci_low",
                "ci_high",
                "n_answers_with",
                "n_answers",
                "n_questions_with",
                "n_questions",
                "few_questions",
                "suggested",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "name": "Bolt Food",
                "kind": "app_or_platform",
                "rate": 0.3,
                "ci_low": 0.1,
                "ci_high": 0.6,
                "n_answers_with": 3,
                "n_answers": 10,
                "n_questions_with": 3,
                "n_questions": 10,
                "few_questions": False,
                "suggested": True,
            }
        )
    (run_dir / RESEARCH_FILE_NAME).write_text(
        json.dumps(
            {
                "sources": [SOURCE],
                "checks": [
                    {
                        "name": "Bolt Food",
                        "rate": 0.3,
                        "found_online": True,
                        "evidence_url": EVIDENCE,
                    }
                ],
            }
        ),
        "utf-8",
    )
    return run_dir


def finished_state(tmp_path, language="es"):
    """Run the real metrics node so the state and answers.csv are consistent."""
    run_dir = write_run(tmp_path)
    budget = Budget(1.0)
    client = SimulatedClient(budget)
    deps = NodeDeps(event_log=EventLog(tmp_path / "runs"), should_stop=fake_should_stop(1))
    nodes = make_real_nodes(
        deps,
        subject=client,
        agent=client,
        budget=budget,
        runs_dir=tmp_path / "runs",
        data_dir=DATA_DIR,
    )
    panel = QuestionPanel(
        questions=tuple(
            Question(id=f"q{i:02d}", text=f"Pregunta {i}", intent="price") for i in range(1, 11)
        )
    )
    state = {"run_id": RUN_ID, "brief": brief(language), "panel": panel, "spend_usd": 0.02}
    state.update(nodes["metrics"](state))
    state["sources"] = [SOURCE]
    return state, run_dir


# --- facts and template ---------------------------------------------------------------------


def test_facts_come_from_the_vetted_stats(tmp_path):
    state, run_dir = finished_state(tmp_path)

    facts = collect_facts(state, run_dir)

    rows = {row.brand: row for row in facts.brands}
    assert rows["Glovo"].mention[0] == pytest.approx(0.7)
    assert rows["Glovo"].recommendation[0] == pytest.approx(0.5)
    assert rows["Just Eat"].mention[0] == pytest.approx(0.8)
    assert facts.sources == (SOURCE, EVIDENCE)  # evidence links are citable sources too
    assert facts.discovered[0].name == "Bolt Food"


@pytest.mark.parametrize("language", ["es", "en"])
def test_the_template_passes_every_skeptic_check(tmp_path, language):
    state, run_dir = finished_state(tmp_path, language)
    facts = collect_facts(state, run_dir)

    draft = template_draft(facts)

    assert run_checks(draft, facts_record(facts)) == []


def test_the_spanish_template_reads_like_a_report(tmp_path):
    state, run_dir = finished_state(tmp_path)

    draft = template_draft(collect_facts(state, run_dir))

    assert draft.startswith("# Visibilidad de Glovo")
    assert "70%" in draft and "Just Eat" in draft and "Bolt Food" in draft
    assert "[S1]" in draft and SOURCE in draft
    assert "no es una cuota de mercado" in draft


def test_an_invented_figure_is_not_in_the_record(tmp_path):
    state, run_dir = finished_state(tmp_path)
    record = facts_record(collect_facts(state, run_dir))

    objections = run_checks("Glovo aparece en el 43% de las respuestas.", record)

    assert [item.kind.value for item in objections] == ["unrecorded_figure"]


# --- writer node ----------------------------------------------------------------------------


def writer(tmp_path, responder=None):
    budget = Budget(1.0)
    responders = {DraftReport: responder} if responder else {}
    client = SimulatedClient(budget, responders=responders)
    return make_writer_node(EventLog(tmp_path / "runs"), client, tmp_path / "runs"), budget


def test_the_writer_uses_the_model_rewrite(tmp_path):
    state, _ = finished_state(tmp_path)
    prompts = []

    def rewrite(prompt):
        prompts.append(prompt)
        return DraftReport(
            markdown="# Informe\n\nLa IA nombra a Glovo en el 70% de las respuestas."
        )

    node, _ = writer(tmp_path, rewrite)
    update = node(state)

    assert update["draft"].startswith("# Informe")
    assert update["draft_version"] == 1
    assert "never add" in prompts[0].casefold() and "[S1]" in prompts[0]


def test_the_writer_falls_back_to_the_template_when_the_model_fails(tmp_path):
    state, _ = finished_state(tmp_path)
    node, _ = writer(tmp_path)  # no responder: invalid answer

    update = node(state)

    assert update["draft"].startswith("# Visibilidad de Glovo")
    assert "template" in update["events"][-1].result


def test_a_revision_gets_the_objections_and_counts_a_round(tmp_path):
    state, _ = finished_state(tmp_path)
    state.update(draft="old draft", draft_version=1, objections=['[causal_claim] "x": y'])
    prompts = []

    def rewrite(prompt):
        prompts.append(prompt)
        return DraftReport(markdown="# Informe corregido\n\nSin afirmaciones causales.")

    node, _ = writer(tmp_path, rewrite)
    update = node(state)

    assert update["review_round"] == 1 and update["draft_version"] == 2
    assert "causal_claim" in prompts[0] and "old draft" in prompts[0]
    assert update["objections"] == []


# --- skeptic with the full record -----------------------------------------------------------


def test_the_skeptic_accepts_competitor_figures_from_the_stats(tmp_path):
    state, _ = finished_state(tmp_path)
    state["draft"] = "Just Eat aparece en el 80% de las respuestas y Glovo en el 70%."
    node = make_checked_skeptic_node(EventLog(tmp_path / "runs"), None, tmp_path / "runs")

    assert node(state)["objections"] == []


def test_the_skeptic_still_objects_to_invented_figures_and_market_share(tmp_path):
    state, _ = finished_state(tmp_path)
    state["draft"] = "Glovo tiene el 43% de la cuota de mercado."
    node = make_checked_skeptic_node(EventLog(tmp_path / "runs"), None, tmp_path / "runs")

    objections = node(state)["objections"]

    assert any("unrecorded_figure" in item for item in objections)
    assert any("market_share_claim" in item for item in objections)


# --- edge cases of the summary and the client report ---------------------------------------

from dataclasses import replace  # noqa: E402

from ai_share_of_voice.writing import (  # noqa: E402
    CLIENT_REPORT_FILE_NAME,
    BrandFacts,
    make_client_report_finalize,
)


def facts_with(tmp_path, *mentions):
    state, run_dir = finished_state(tmp_path)
    facts = collect_facts(state, run_dir)
    brands = tuple(
        BrandFacts(name, (value, 0.0, min(1.0, value + 0.2)), None)
        for name, value in zip(("Glovo", "Just Eat", "Uber Eats"), mentions, strict=True)
    )
    return replace(facts, brands=brands)


def test_no_brand_is_called_the_most_named_when_none_is_named(tmp_path):
    draft = template_draft(facts_with(tmp_path, 0.0, 0.0, 0.0))

    assert "más nombrada" not in draft
    assert "no nombra a ninguna" in draft


def test_a_tie_at_the_top_is_said_as_a_tie(tmp_path):
    draft = template_draft(facts_with(tmp_path, 0.7, 0.7, 0.2))

    assert "empatan" in draft and "Glovo" in draft and "Just Eat" in draft


def test_the_client_report_is_written_only_after_the_approved_finalize(tmp_path):
    state, run_dir = finished_state(tmp_path)
    state.update(draft="# Informe\n\nTexto aprobado.\n", approved=True)

    def finalize(current):
        return {"status": "completed", "report_path": str(run_dir / "report.md"), "events": []}

    update = make_client_report_finalize(finalize, tmp_path / "runs")(state)

    text = (run_dir / CLIENT_REPORT_FILE_NAME).read_text("utf-8")
    assert text.startswith("# Informe") and "Aprobado por una persona" in text
    assert update["status"] == "completed"


def test_no_client_report_when_the_finalize_fails(tmp_path):
    state, run_dir = finished_state(tmp_path)
    state.update(draft="# Informe\n", approved=True)

    def finalize(current):
        raise RuntimeError("disk full")

    with pytest.raises(RuntimeError):
        make_client_report_finalize(finalize, tmp_path / "runs")(state)
    assert not (run_dir / CLIENT_REPORT_FILE_NAME).exists()


def test_the_skeptic_model_gets_the_run_facts_and_the_method_rule(tmp_path):
    # Found live: without them it called "only 5 questions were used" unverifiable.
    from ai_share_of_voice.agents.skeptic import SkepticReview

    state, _ = finished_state(tmp_path)
    state["draft"] = "Solo se usaron 10 preguntas."
    prompts = []

    def review(prompt):
        prompts.append(prompt)
        return SkepticReview()

    client = SimulatedClient(Budget(1.0), responders={SkepticReview: review})
    make_checked_skeptic_node(EventLog(tmp_path / "runs"), client, tmp_path / "runs")(state)

    prompt = prompts[0]
    assert "Recorded run facts" in prompt and "questions: 10" in prompt
    assert "stop reason" in prompt.casefold()
    assert "describe the method" in prompt
    assert prompt.index("Recorded run facts") < prompt.index("<draft>")


def test_the_skeptic_event_lists_its_objections_for_the_audit(tmp_path):
    state, _ = finished_state(tmp_path)
    state["draft"] = "Glovo tiene el 43% de la cuota de mercado."
    node = make_checked_skeptic_node(EventLog(tmp_path / "runs"), None, tmp_path / "runs")

    result = node(state)["events"][-1].result

    assert "unrecorded_figure" in result and "market_share_claim" in result
