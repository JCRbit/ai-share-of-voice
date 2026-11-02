import json
import re
from pathlib import Path

import pytest

from ai_share_of_voice.discovery import (
    BusinessKind,
    BusinessList,
    DiscoveryRecord,
    NamedBusiness,
)
from ai_share_of_voice.errors import NoEvidence, ToolFailure
from ai_share_of_voice.event_log import EventLog
from ai_share_of_voice.exceptions import BudgetExceeded
from ai_share_of_voice.llm_client import Budget, QuotaExhausted, SimulatedClient
from ai_share_of_voice.nodes import NodeDeps, fake_should_stop
from ai_share_of_voice.nodes_real import (
    ANALYZED_FOLDER,
    DISCOVERED_FILE_NAME,
    DISCOVERED_FOLDER,
    PANEL_FILE_NAME,
    RAW_FOLDER,
    RealNodeOptions,
    load_panel,
    make_real_nodes,
)
from ai_share_of_voice.panel import (
    DraftQuestion,
    DraftQuestions,
    read_panel_file,
    write_panel_file,
)
from ai_share_of_voice.schemas.analysis import AnalyzedResponse, BrandMention, RawAnswer
from ai_share_of_voice.schemas.brief import BrandSpec, Brief
from ai_share_of_voice.schemas.questions import Question, QuestionPanel
from ai_share_of_voice.schemas.stopping import StopDecision, StopReason
from ai_share_of_voice.stats.metrics import load_answers, wilson_interval
from ai_share_of_voice.stats.stopping import should_stop as real_should_stop
from ai_share_of_voice.tools.mention_tool import BrandJudgement, JudgementBatch
from ai_share_of_voice.tools.stats_tool import ANSWERS_FILE_NAME, StatsToolInput, run_stats

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


def build(tmp_path, *, text_responder, budget=None, judge=judge_all, options=None, discovery=None):
    """Without ``discovery`` the agent has no BusinessList responder: discovery fails."""
    budget = budget or Budget(10.0)
    subject = SimulatedClient(budget, text_responder=text_responder, cost_per_call_usd=0.01)
    responders = {JudgementBatch: judge}
    if discovery is not None:
        responders[BusinessList] = discovery
    agent = SimulatedClient(budget, responders=responders, cost_per_call_usd=0.002)
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


# --- metrics --------------------------------------------------------------------------------


def write_analyzed(tmp_path, batch_index, mentioned_by_question):
    """Store one analyzed batch: question id -> whether Glovo is mentioned."""
    items = []
    for qid, mentioned in mentioned_by_question.items():
        items.append(
            AnalyzedResponse(
                response_id=f"b{batch_index:03d}-{qid}",
                question_id=qid,
                batch_index=batch_index,
                mentions=(
                    BrandMention(
                        brand="Glovo",
                        mentioned=mentioned,
                        recommended=mentioned,
                        position=1 if mentioned else None,
                    ),
                    BrandMention(brand="Just Eat", mentioned=False),
                    BrandMention(brand="Uber Eats", mentioned=False),
                ),
            )
        )
    path = tmp_path / "runs" / RUN_ID / ANALYZED_FOLDER / f"batch_{batch_index:03d}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(item.model_dump_json() + "\n" for item in items), "utf-8")


def metrics_nodes(tmp_path, should_stop=None):
    budget = Budget(10.0)
    client = SimulatedClient(budget)
    deps = NodeDeps(
        event_log=EventLog(tmp_path / "runs"),
        should_stop=should_stop or real_should_stop,
    )
    return make_real_nodes(
        deps,
        subject=client,
        agent=client,
        budget=budget,
        runs_dir=tmp_path / "runs",
        data_dir=DATA_DIR,
    )


TEN = [f"q{index:02d}" for index in range(1, 11)]


def test_metrics_writes_answers_csv_that_the_stats_tool_can_read(tmp_path):
    write_analyzed(tmp_path, 0, {qid: qid in TEN[:4] for qid in TEN})

    metrics_nodes(tmp_path)["metrics"](state_for(panel(*TEN), spend_usd=0.1))

    answers = load_answers(tmp_path / "runs" / RUN_ID / ANSWERS_FILE_NAME)
    assert len(answers) == 10 * 3  # one row per answer and tracked brand
    result = run_stats(StatsToolInput(run_id=RUN_ID, chart=False), tmp_path / "runs")
    assert {row.brand for row in result.rows} == {"Glovo", "Just Eat", "Uber Eats"}


def test_metrics_summarizes_the_client_mention_rate_with_its_interval(tmp_path):
    write_analyzed(tmp_path, 0, {qid: qid in TEN[:4] for qid in TEN})

    update = metrics_nodes(tmp_path)["metrics"](state_for(panel(*TEN), spend_usd=0.1))

    (summary,) = update["history"]
    assert summary.batch_index == 0
    assert summary.n_answers == 10
    assert summary.n_questions == 10
    assert summary.brand_share == pytest.approx(0.4)
    assert summary.ci_low < 0.4 < summary.ci_high
    # The headline interval is never narrower than Wilson alone.
    wilson_low, wilson_high = wilson_interval(4, 10)
    assert summary.ci_low <= wilson_low + 1e-9 and summary.ci_high >= wilson_high - 1e-9
    assert summary.cumulative_spend_usd == pytest.approx(0.1)


def test_metrics_pools_every_batch_so_far(tmp_path):
    write_analyzed(tmp_path, 0, {qid: qid in TEN[:4] for qid in TEN})
    write_analyzed(tmp_path, 1, {qid: qid in TEN[:6] for qid in TEN})
    previous = metrics_nodes(tmp_path)["metrics"](state_for(panel(*TEN)))["history"]

    update = metrics_nodes(tmp_path)["metrics"](state_for(panel(*TEN), history=previous))

    (summary,) = update["history"]
    assert summary.batch_index == 1
    assert summary.n_answers == 20
    assert summary.n_questions == 10  # answers to one question stay one cluster
    assert summary.brand_share == pytest.approx(0.5)


def test_metrics_asks_the_real_stopping_rule(tmp_path):
    write_analyzed(tmp_path, 0, {qid: qid in TEN[:4] for qid in TEN})

    update = metrics_nodes(tmp_path)["metrics"](state_for(panel(*TEN)))

    decision = update["stop_decision"]
    assert decision.stop is False  # one batch is below min_batches
    assert decision.reason is StopReason.CONTINUE


def test_metrics_stops_at_the_max_batches_of_the_brief(tmp_path):
    def never_stop(history, config):
        return StopDecision(stop=False, reason=StopReason.CONTINUE, explanation="more")

    for batch_index in range(3):  # make_brief() has max_batches=3
        write_analyzed(tmp_path, batch_index, {qid: True for qid in TEN})
    nodes = metrics_nodes(tmp_path, should_stop=never_stop)
    history = []
    for _ in range(3):
        update = nodes["metrics"](state_for(panel(*TEN), history=history))
        history = [*history, *update["history"]]

    assert update["stop_decision"].reason is StopReason.MAX_BATCHES


def test_metrics_without_analyzed_answers_is_no_evidence(tmp_path):
    with pytest.raises(NoEvidence):
        metrics_nodes(tmp_path)["metrics"](state_for(panel(*TEN)))


# --- discovery (R7) -------------------------------------------------------------------------


def list_bolt(prompt):
    return BusinessList(
        businesses=[
            NamedBusiness(name="Bolt Food", kind=BusinessKind.APP_OR_PLATFORM),
            NamedBusiness(name="Glovo", kind=BusinessKind.APP_OR_PLATFORM),
        ]
    )


def test_sampling_stores_what_every_answer_names(tmp_path):
    nodes, _ = build(tmp_path, text_responder=lambda _: "Glovo o Bolt Food", discovery=list_bolt)

    update = nodes["sampling"](state_for(panel("q1", "q2")))

    path = tmp_path / "runs" / RUN_ID / DISCOVERED_FOLDER / "batch_000.jsonl"
    records = read_jsonl(path, DiscoveryRecord)
    assert [record.question_id for record in records] == ["q1", "q2"]
    assert all(record.ok for record in records)
    assert {item.name: item.tracked_brand for item in records[0].businesses} == {
        "Bolt Food": None,
        "Glovo": "Glovo",
    }
    assert "discovery" not in update["events"][-1].result  # nothing failed


def test_a_failed_discovery_never_drops_the_answer(tmp_path):
    # Discovery is secondary: the mention metrics must not lose answers because of it.
    nodes, _ = build(tmp_path, text_responder=lambda _: "Glovo")  # no discovery responder

    update = nodes["sampling"](state_for(panel("q1", "q2")))

    assert update["n_answers"] == 2
    assert "discovery failed for 2" in update["events"][-1].result
    records = read_jsonl(
        tmp_path / "runs" / RUN_ID / DISCOVERED_FOLDER / "batch_000.jsonl", DiscoveryRecord
    )
    assert [record.ok for record in records] == [False, False]


def test_discovery_can_be_switched_off(tmp_path):
    nodes, _ = build(
        tmp_path,
        text_responder=lambda _: "Glovo",
        discovery=list_bolt,
        options=RealNodeOptions(discover=False),
    )

    nodes["sampling"](state_for(panel("q1")))

    assert not (tmp_path / "runs" / RUN_ID / DISCOVERED_FOLDER).exists()


def test_metrics_writes_the_ranking_of_untracked_businesses(tmp_path):
    nodes, _ = build(tmp_path, text_responder=lambda _: "Glovo o Bolt Food", discovery=list_bolt)
    state = state_for(panel("q1", "q2"))
    nodes["sampling"](state)

    nodes["metrics"](state)

    ranking = (tmp_path / "runs" / RUN_ID / DISCOVERED_FILE_NAME).read_text("utf-8")
    header, *rows = ranking.splitlines()
    assert header.startswith("name,kind,rate")
    assert len(rows) == 1 and rows[0].startswith("Bolt Food,app_or_platform,1.0")


def test_an_exhausted_quota_stops_sampling_at_once(tmp_path):
    calls = []

    def quota(prompt):
        calls.append(prompt)
        raise QuotaExhausted("llm_client", "429 requests per day exceeded")

    nodes, _ = build(tmp_path, text_responder=quota)

    with pytest.raises(QuotaExhausted):
        nodes["sampling"](state_for(panel("q1", "q2", "q3")))
    assert len(calls) == 1  # no point asking the other questions today


# --- question panel source (R6) -------------------------------------------------------------


def test_the_panel_in_use_is_always_saved_with_the_run(tmp_path):
    nodes, _ = build(tmp_path, text_responder=lambda _: "x")

    update = nodes["questions"]({"run_id": RUN_ID, "brief": make_brief()})

    saved = read_panel_file(tmp_path / "runs" / RUN_ID / PANEL_FILE_NAME)
    assert saved == update["panel"]


def test_an_approved_panel_file_is_reused_as_it_is(tmp_path):
    approved = write_panel_file(panel("a1", "a2"), tmp_path / "approved.csv")
    nodes, _ = build(
        tmp_path, text_responder=lambda _: "x", options=RealNodeOptions(panel_file=approved)
    )

    update = nodes["questions"]({"run_id": RUN_ID, "brief": make_brief()})

    assert [question.id for question in update["panel"].questions] == ["a1", "a2"]
    assert "approved panel" in update["events"][-1].result


def test_a_panel_can_be_generated_for_the_brief(tmp_path):
    texts = [f"¿Qué tienda de ropa me recomiendas número {index}?" for index in range(1, 13)]

    def draft(prompt):
        return DraftQuestions(
            questions=[DraftQuestion(text=text, intent="recommendation") for text in texts]
        )

    budget = Budget(10.0)
    agent = SimulatedClient(budget, responders={DraftQuestions: draft})
    deps = NodeDeps(event_log=EventLog(tmp_path / "runs"), should_stop=fake_should_stop(2))
    nodes = make_real_nodes(
        deps,
        subject=agent,
        agent=agent,
        budget=budget,
        runs_dir=tmp_path / "runs",
        data_dir=DATA_DIR,
        options=RealNodeOptions(generate_panel=True, panel_size=12),
    )

    update = nodes["questions"]({"run_id": RUN_ID, "brief": make_brief()})

    assert len(update["panel"].questions) == 12
    assert "generated" in update["events"][-1].result
    assert (tmp_path / "runs" / RUN_ID / PANEL_FILE_NAME).is_file()
