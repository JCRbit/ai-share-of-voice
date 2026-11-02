"""Real graph nodes for the local end-to-end integration (E).

``make_real_nodes`` starts from the fake nodes and replaces the ones that already have a
real version, so the graph always runs end to end while the swap goes one node at a time.
Real so far:

* ``questions``: an approved panel file if given, else a panel generated for the brief
  (``panel.py``) if asked, else the demo panel of ``Brief.language``. The panel in use is
  always saved as ``runs/<id>/questions.csv`` so a later run can reuse it.
* ``sampling``: asks the measured model every question once and analyzes each answer
  with the mention tool right away. Raw and analyzed answers go to the run folder, one
  file per batch, so a re-run of the node overwrites its batch instead of duplicating it.
* ``metrics``: no model. Pools every analyzed batch into ``answers.csv`` (the file the stats
  tool and the results tab read), computes the client's mention rate with the headline
  interval of ``stats/metrics.py`` and asks the stopping rule in ``deps.should_stop``.
  It also ranks the untracked businesses found by discovery (``discovered_businesses.csv``).

* ``research`` (when a sources client is given): searches, reads and checks businesses
  through the MCP sources server (``research.py``).

* ``screenshots`` (when a vision client is given): the vision model reads our chart and
  code keeps the reading only if the figures support it (``chart_vision.py``).
* ``write_draft`` and ``skeptic``: the code draft rewritten by the agent model, reviewed
  by J's Skeptic over the full run record (``writing.py``).

Discovery (R7) runs in ``sampling`` after the mention analysis: every business each answer
names goes to ``discovered/``. It is secondary, so its failures never drop an answer.

Failures follow the guard (``guard.py``): ``BudgetExceeded`` and ``ToolFailure`` propagate
and close the run in a partial state. A few failed questions are tolerated and reported,
because dropping answers can bias the estimates; too many close the batch.
"""

import csv
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ai_share_of_voice.chart_vision import make_vision_node
from ai_share_of_voice.discovery import (
    DiscoveryRecord,
    discover_businesses,
    discovery_ranking,
    ranking_rows,
)
from ai_share_of_voice.errors import NoEvidence, ToolFailure
from ai_share_of_voice.llm_client import Budget, QuotaExhausted
from ai_share_of_voice.nodes import NodeDeps, NodeFn, _emit, make_fake_nodes
from ai_share_of_voice.panel import (
    DEFAULT_PANEL_SIZE,
    generate_panel,
    read_panel_file,
    write_panel_file,
)
from ai_share_of_voice.research import research
from ai_share_of_voice.schemas.analysis import AnalyzedResponse, RawAnswer
from ai_share_of_voice.schemas.brief import Brief
from ai_share_of_voice.schemas.events import EventStatus
from ai_share_of_voice.schemas.llm import ModelClient, Usage
from ai_share_of_voice.schemas.questions import Question, QuestionPanel
from ai_share_of_voice.schemas.stopping import (
    BatchSummary,
    StopConfig,
    StopDecision,
    StopReason,
    SummaryMetric,
)
from ai_share_of_voice.state import GraphState
from ai_share_of_voice.stats.metrics import brand_metrics, responses_to_frame
from ai_share_of_voice.tools.mcp_sources_client import McpSourcesClient
from ai_share_of_voice.tools.mention_tool import (
    ExtractionError,
    MentionExtractionInput,
    extract_mentions,
)
from ai_share_of_voice.tools.stats_tool import ANSWERS_FILE_NAME
from ai_share_of_voice.writing import make_checked_skeptic_node, make_writer_node

RAW_FOLDER = "raw"
DISCOVERED_FOLDER = "discovered"
DISCOVERED_FILE_NAME = "discovered_businesses.csv"
PANEL_FILE_NAME = "questions.csv"
ANALYZED_FOLDER = "analyzed"
QUESTION_FILES = {"es": "questions_es.csv", "en": "questions.csv"}
DEFAULT_QUESTION_FILE = "questions.csv"


@dataclass(frozen=True)
class RealNodeOptions:
    """Knobs of the real nodes."""

    max_questions: int | None = None  # ask only the first N questions: cheap live checks
    max_failed_share: float = 0.1  # above this share of failed questions, the batch fails
    extraction_attempts: int = 2
    discover: bool = True  # one more agent call per answer
    panel_file: Path | None = None  # an approved panel, reused as it is
    generate_panel: bool = False  # draft a panel for the brief when there is no file
    panel_size: int = DEFAULT_PANEL_SIZE
    location: str | None = None  # e.g. "Valencia, España": disambiguates searches


def load_panel(language: str, data_dir: Path) -> QuestionPanel:
    """Load the question panel of ``language``; English when there is no file for it."""
    path = Path(data_dir) / QUESTION_FILES.get(language, DEFAULT_QUESTION_FILE)
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return QuestionPanel(questions=tuple(Question.model_validate(row) for row in rows))


def batch_file(run_dir: Path, folder: str, batch_index: int) -> Path:
    return run_dir / folder / f"batch_{batch_index:03d}.jsonl"


def _write_jsonl(path: Path, items: list[Any]) -> None:
    """Write the whole batch at once: a re-run replaces it instead of appending."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(item.model_dump_json() + "\n" for item in items), encoding="utf-8")


def load_analyzed(run_dir: Path) -> list[AnalyzedResponse]:
    """Every analyzed answer of the run, batch by batch."""
    return [
        AnalyzedResponse.model_validate_json(line)
        for path in sorted((run_dir / ANALYZED_FOLDER).glob("batch_*.jsonl"))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_ranking(run_dir: Path) -> None:
    """Rank the untracked businesses of every discovered batch so far."""
    records = [
        DiscoveryRecord.model_validate_json(line)
        for path in sorted((run_dir / DISCOVERED_FOLDER).glob("batch_*.jsonl"))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = ranking_rows(discovery_ranking(records))
    if not rows:
        return
    with (run_dir / DISCOVERED_FILE_NAME).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def stop_config(brief: Brief) -> StopConfig:
    """The limits of the brief, as the stopping rule reads them."""
    return StopConfig(
        target_half_width=brief.target_half_width,
        min_batches=min(2, brief.max_batches),
        max_batches=brief.max_batches,
        max_spend_usd=brief.spend_cap_usd,
    )


def _analyze(
    answer: RawAnswer, brief: Brief, agent: ModelClient, attempts: int
) -> AnalyzedResponse:
    """Run the mention tool, retrying an inconsistent or invalid judgement."""
    data = MentionExtractionInput.from_answer(answer, brief)
    for attempt in range(1, attempts + 1):
        try:
            return extract_mentions(data, agent).analysis
        except (ExtractionError, ValueError):
            if attempt == attempts:
                raise
    raise ValueError("extraction_attempts must be at least 1")


def make_real_nodes(
    deps: NodeDeps,
    *,
    subject: Any,
    agent: ModelClient,
    budget: Budget,
    runs_dir: Path,
    data_dir: Path,
    options: RealNodeOptions | None = None,
    sources: Callable[[], AbstractContextManager[McpSourcesClient]] | None = None,
    vision: Any = None,
) -> dict[str, NodeFn]:
    """Return every node of the graph: real where ready, fake otherwise.

    ``budget`` is the one shared by ``subject`` and ``agent``. A node's cost is what the
    budget spent while it ran, so paid calls whose answer was unusable are counted too.
    """
    options = options or RealNodeOptions()
    emit = partial(_emit, deps)
    nodes = make_fake_nodes(deps)

    def questions(state: GraphState) -> dict[str, Any]:
        agent_name = "question_generator"
        events = [emit(state, "questions", EventStatus.STARTED, agent=agent_name)]
        brief = state["brief"]
        if options.panel_file is not None:
            panel, source = read_panel_file(options.panel_file), "approved panel"
        elif options.generate_panel:
            panel, source = generate_panel(brief, agent, options.panel_size), "generated panel"
        else:
            panel, source = load_panel(brief.language, data_dir), "demo panel"
        if options.max_questions is not None:
            panel = QuestionPanel(questions=panel.questions[: options.max_questions])
        write_panel_file(panel, Path(runs_dir) / state["run_id"] / PANEL_FILE_NAME)
        result = f"{len(panel.questions)} questions ({brief.language}) from the {source}"
        events.append(
            emit(state, "questions", EventStatus.SUCCEEDED, agent=agent_name, result=result)
        )
        return {"panel": panel, "events": events}

    def sampling(state: GraphState) -> dict[str, Any]:
        tool = "llm_client"
        events = [emit(state, "sampling", EventStatus.STARTED, tool=tool)]
        brief = state["brief"]
        batch_index = len(state.get("history", []))
        run_dir = Path(runs_dir) / state["run_id"]

        raw: list[RawAnswer] = []
        analyzed: list[AnalyzedResponse] = []
        failed: list[str] = []
        discovered: list[DiscoveryRecord] = []
        spent_before = budget.spent_usd
        for question in state["panel"].questions:
            try:
                text, usage = subject.answer(question.text)
                answer = RawAnswer(
                    response_id=f"b{batch_index:03d}-{question.id}",
                    question_id=question.id,
                    batch_index=batch_index,
                    model=_model_name(subject, usage),
                    text=text,
                )
                analysis = _analyze(answer, brief, agent, options.extraction_attempts)
            except QuotaExhausted:
                raise  # the rest of the panel would fail the same way today
            except (ToolFailure, ExtractionError, ValidationError, ValueError):
                failed.append(question.id)
                continue
            raw.append(answer)
            if options.discover:
                discovered.append(discover_businesses(answer, brief, agent))
            analyzed.append(analysis)

        asked = len(state["panel"].questions)
        if not analyzed or len(failed) > options.max_failed_share * asked:
            raise ToolFailure(
                tool, f"{len(failed)} of {asked} questions failed in batch {batch_index}"
            )

        _write_jsonl(batch_file(run_dir, RAW_FOLDER, batch_index), raw)
        _write_jsonl(batch_file(run_dir, ANALYZED_FOLDER, batch_index), analyzed)
        if options.discover:
            _write_jsonl(batch_file(run_dir, DISCOVERED_FOLDER, batch_index), discovered)

        client_brand = brief.brand.name.casefold()
        brand_mentions = sum(
            1
            for item in analyzed
            for mention in item.mentions
            if mention.brand.casefold() == client_brand and mention.mentioned
        )
        spend = state.get("spend_usd", 0.0) + (budget.spent_usd - spent_before)
        result = f"batch {batch_index}: {len(analyzed)} answers analyzed"
        discovery_failed = sum(1 for record in discovered if not record.ok)
        if discovery_failed:
            result += f", discovery failed for {discovery_failed}"
        if failed:
            result += f", {len(failed)} failed ({', '.join(failed)})"
        events.append(
            emit(state, "sampling", EventStatus.SUCCEEDED, tool=tool, result=result, spend=spend)
        )
        return {
            "n_answers": state.get("n_answers", 0) + len(analyzed),
            "n_brand_mentions": state.get("n_brand_mentions", 0) + brand_mentions,
            "spend_usd": spend,
            "events": events,
        }

    def metrics(state: GraphState) -> dict[str, Any]:
        agent_name, tool = "analyst", "stats_tool"
        events = [emit(state, "metrics", EventStatus.STARTED, agent=agent_name, tool=tool)]
        brief = state["brief"]
        run_dir = Path(runs_dir) / state["run_id"]
        responses = load_analyzed(run_dir)
        if not responses:
            raise NoEvidence("no analyzed answers to measure")

        brands = [brand.name for brand in (brief.brand, *brief.competitors)]
        frame = responses_to_frame(responses, brands)
        frame.to_csv(run_dir / ANSWERS_FILE_NAME, index=False)  # rewritten whole: idempotent
        _write_ranking(run_dir)
        table = brand_metrics(frame)
        row = table[
            (table["brand"] == brief.brand.name)
            & (table["metric"] == SummaryMetric.MENTION_RATE.value)
        ].iloc[0]

        history = state.get("history", [])
        summary = BatchSummary(
            batch_index=len(history),
            n_answers=len(responses),
            brand_share=float(row["estimate"]),
            ci_low=float(row["ci_low"]),
            ci_high=float(row["ci_high"]),
            cumulative_spend_usd=state.get("spend_usd", 0.0),
            n_questions=int(row["n_questions"]),
        )
        decision = deps.should_stop([*history, summary], stop_config(brief))
        if not decision.stop and len(history) + 1 >= brief.max_batches:
            decision = StopDecision(
                stop=True,
                reason=StopReason.MAX_BATCHES,
                explanation=f"Hard cap of {brief.max_batches} batches reached",
            )
        result = (
            f"{brief.brand.name} mention rate {summary.brand_share:.0%} "
            f"[{summary.ci_low:.0%}, {summary.ci_high:.0%}], "
            f"{summary.n_answers} answers, {summary.n_questions} questions"
        )
        events.append(
            emit(
                state,
                "metrics",
                EventStatus.SUCCEEDED,
                agent=agent_name,
                tool=tool,
                result=result,
                reason=decision.explanation,
            )
        )
        return {"history": [summary], "stop_decision": decision, "events": events}

    def research_node(state: GraphState) -> dict[str, Any]:
        agent_name, tool = "source_researcher", "mcp_sources"
        events = [emit(state, "research", EventStatus.STARTED, agent=agent_name, tool=tool)]
        with sources() as client:
            found = research(
                state["brief"],
                Path(runs_dir) / state["run_id"],
                client,
                location=options.location,
            )
        result = (
            f"{len(found.sources)} sources read, {len(found.skipped)} skipped, "
            f"{found.flagged} flagged for injection"
        )
        if found.checks:
            missing = sum(not check.found_online for check in found.checks)
            result += f"; {len(found.checks)} businesses checked, {missing} without evidence online"
        events.append(
            emit(
                state, "research", EventStatus.SUCCEEDED, agent=agent_name, tool=tool, result=result
            )
        )
        return {"sources": found.sources, "events": events}

    nodes.update(
        questions=questions,
        sampling=sampling,
        metrics=metrics,
        write_draft=make_writer_node(deps.event_log, agent, Path(runs_dir)),
        skeptic=make_checked_skeptic_node(deps.event_log, agent, Path(runs_dir)),
    )
    if sources is not None:
        nodes["research"] = research_node
    if vision is not None:
        nodes["screenshots"] = make_vision_node(deps.event_log, vision, Path(runs_dir))
    return nodes


def _model_name(client: Any, usage: Usage) -> str:
    return usage.model or getattr(client, "model", "unknown")
