import json
import re
from pathlib import Path

import pytest

from ai_share_of_voice.errors import ToolFailure
from ai_share_of_voice.event_log import EventLog
from ai_share_of_voice.exceptions import BudgetExceeded
from ai_share_of_voice.llm_client import Budget, SimulatedClient
from ai_share_of_voice.nodes import NodeDeps, fake_should_stop
from ai_share_of_voice.nodes_real import (
    ANALYZED_FOLDER,
    RAW_FOLDER,
    RealNodeOptions,
    load_panel,
    make_real_nodes,
)
from ai_share_of_voice.schemas.analysis import AnalyzedResponse, RawAnswer
from ai_share_of_voice.schemas.brief import BrandSpec, Brief
from ai_share_of_voice.schemas.questions import Question, QuestionPanel
from ai_share_of_voice.tools.mention_tool import BrandJudgement, JudgementBatch

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RUN_ID = "run-test"


def make_brief(language: str = "es") -> Brief:
    return Brief(
        brand=BrandSpec(name="Glovo", variants=("GlovoApp",)),
        competitors=(BrandSpec(name="Just Eat"), BrandSpec(name="Uber Eats")),
        sector="food delivery",
        city="Madrid",
        language=language,
        spend_cap_usd=1.0,
        max_batches=3,
    )


def judge_all(prompt: str) -> JudgementBatch:
    """Recommend every brand the prompt asks about (they are already known to be mentioned)."""
    brands = json.loads(re.search(r"Brands to judge: (\[.*\])", prompt).group(1))
    return JudgementBatch(
        judgements=tuple(
            BrandJudgement(brand=brand, recommended=True, tone="positive") for brand in brands
        )
    )


def panel(*ids: str) -> QuestionPanel:
    return QuestionPanel(
        questions=tuple(
            Question(id=qid, text=f"Pregunta de prueba {qid}", intent="recommendation")
            for qid in ids
        )
    )


def build(tmp_path, *, text_responder, budget=None, judge=judge_all, options=None):
    budget = budget or Budget(10.0)
    subject = SimulatedClient(budget, text_responder=text_responder, cost_per_call_usd=0.01)
    agent = SimulatedClient(budget, responders={JudgementBatch: judge}, cost_per_call_usd=0.002)
    deps = NodeDeps(event_log=EventLog(tmp_path / "runs"), should_stop=fake_should_stop(2))
    nodes = make_real_nodes(
        deps,
        subject=subject,
        agent=agent,
        budget=budget,
        runs_dir=tmp_path / "runs",
        data_dir=DATA_DIR,
        options=options or RealNodeOptions(),
    )
    return nodes, budget


def state_for(questions: QuestionPanel, **extra):
    return {"run_id": RUN_ID, "brief": make_brief(), "panel": questions, **extra}


def read_jsonl(path: Path, model):
    return [model.model_validate_json(line) for line in path.read_text("utf-8").splitlines()]


# --- question panel -------------------------------------------------------------------------


@pytest.mark.parametrize(("language", "first_word"), [("es", "¿Cuál"), ("en", "What")])
def test_the_panel_comes_from_the_file_of_the_brief_language(language, first_word):
    loaded = load_panel(language, DATA_DIR)

    assert len(loaded.questions) == 30
    assert loaded.questions[0].id == "q01"
    assert loaded.questions[0].text.startswith(first_word)


def test_an_unsupported_language_falls_back_to_english():
    assert load_panel("fr", DATA_DIR).questions[0].text.startswith("What")


def test_the_questions_node_can_limit_the_panel_for_cheap_runs(tmp_path):
    nodes, _ = build(
        tmp_path, text_responder=lambda _: "x", options=RealNodeOptions(max_questions=5)
    )

    update = nodes["questions"]({"run_id": RUN_ID, "brief": make_brief()})

    assert [question.id for question in update["panel"].questions] == [
        "q01",
        "q02",
        "q03",
        "q04",
        "q05",
    ]


# --- sampling -------------------------------------------------------------------------------


def test_sampling_stores_raw_and_analyzed_answers_and_counts_mentions(tmp_path):
    answers = {
        "Pregunta de prueba q1": "Te recomiendo Glovo.",
        "Pregunta de prueba q2": "Usa Just Eat.",
    }
    nodes, budget = build(tmp_path, text_responder=answers.__getitem__)

    update = nodes["sampling"](state_for(panel("q1", "q2"), spend_usd=0.5))

    run_dir = tmp_path / "runs" / RUN_ID
    raw = read_jsonl(run_dir / RAW_FOLDER / "batch_000.jsonl", RawAnswer)
    analyzed = read_jsonl(run_dir / ANALYZED_FOLDER / "batch_000.jsonl", AnalyzedResponse)
    assert [answer.text for answer in raw] == ["Te recomiendo Glovo.", "Usa Just Eat."]
    assert [item.response_id for item in analyzed] == ["b000-q1", "b000-q2"]
    glovo = analyzed[0].mentions[0]
    assert (glovo.brand, glovo.mentioned, glovo.recommended) == ("Glovo", True, True)
    assert update["n_answers"] == 2
    assert update["n_brand_mentions"] == 1
    # two subject answers (0.01) and two judgements (0.002), on top of the previous spend
    assert update["spend_usd"] == pytest.approx(0.5 + 0.024)


def test_sampling_adds_to_the_counters_of_previous_batches(tmp_path):
    nodes, _ = build(tmp_path, text_responder=lambda _: "Glovo")
    history = [object()]  # one batch already done: this is batch 1

    update = nodes["sampling"](
        state_for(panel("q1"), history=history, n_answers=10, n_brand_mentions=4)
    )

    assert (update["n_answers"], update["n_brand_mentions"]) == (11, 5)
    assert (tmp_path / "runs" / RUN_ID / RAW_FOLDER / "batch_001.jsonl").is_file()


def test_rerunning_a_batch_overwrites_its_files(tmp_path):
    # LangGraph re-runs a node after a crash; the batch must not be stored twice.
    nodes, _ = build(tmp_path, text_responder=lambda _: "Glovo")
    state = state_for(panel("q1", "q2"))

    nodes["sampling"](state)
    nodes["sampling"](state)

    raw = tmp_path / "runs" / RUN_ID / RAW_FOLDER / "batch_000.jsonl"
    assert len(raw.read_text("utf-8").splitlines()) == 2


def test_a_few_failed_questions_are_skipped_and_reported(tmp_path):
    def flaky(prompt):
        if prompt.endswith("q01"):
            raise ToolFailure("llm_client", "503 high demand")
        return "Glovo"

    ids = [f"q{index:02d}" for index in range(1, 11)]
    nodes, _ = build(tmp_path, text_responder=flaky)

    update = nodes["sampling"](state_for(panel(*ids)))

    assert update["n_answers"] == 9
    assert "1 failed (q01)" in update["events"][-1].result


def test_too_many_failures_close_the_batch_as_a_tool_failure(tmp_path):
    def down(prompt):
        raise ToolFailure("llm_client", "503 high demand")

    nodes, _ = build(tmp_path, text_responder=down)

    with pytest.raises(ToolFailure, match="2 of 2 questions failed"):
        nodes["sampling"](state_for(panel("q1", "q2")))


def test_a_failed_judgement_is_retried_once(tmp_path):
    calls = []

    def judge_second_time(prompt):
        calls.append(prompt)
        if len(calls) == 1:
            return JudgementBatch(judgements=())  # misses Glovo: ExtractionError
        return judge_all(prompt)

    nodes, _ = build(tmp_path, text_responder=lambda _: "Glovo", judge=judge_second_time)

    update = nodes["sampling"](state_for(panel("q1")))

    assert len(calls) == 2
    assert update["n_brand_mentions"] == 1
    # one answer (0.01) and two paid judgements (0.002 each): the failed one counts too
    assert update["spend_usd"] == pytest.approx(0.014)


def test_an_exhausted_budget_stops_sampling(tmp_path):
    # BudgetExceeded is not a question failure: the guard closes the run as budget_exhausted.
    nodes, _ = build(tmp_path, text_responder=lambda _: "Glovo", budget=Budget(0.015))

    with pytest.raises(BudgetExceeded):
        nodes["sampling"](state_for(panel("q1", "q2")))
