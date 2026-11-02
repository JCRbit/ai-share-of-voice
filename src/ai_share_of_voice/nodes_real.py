"""Real graph nodes for the local end-to-end integration (E).

``make_real_nodes`` starts from the fake nodes and replaces the ones that already have a
real version, so the graph always runs end to end while the swap goes one node at a time.
Real so far:

* ``questions``: the panel from ``data/questions_es.csv`` or ``data/questions.csv``,
  picked by ``Brief.language``.
* ``sampling``: asks the measured model every question once and analyzes each answer
  with the mention tool right away. Raw and analyzed answers go to the run folder, one
  file per batch, so a re-run of the node overwrites its batch instead of duplicating it.

Failures follow the guard (``guard.py``): ``BudgetExceeded`` and ``ToolFailure`` propagate
and close the run in a partial state. A few failed questions are tolerated and reported,
because dropping answers can bias the estimates; too many close the batch.
"""

import csv
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ai_share_of_voice.errors import ToolFailure
from ai_share_of_voice.llm_client import Budget
from ai_share_of_voice.nodes import NodeDeps, NodeFn, _emit, make_fake_nodes
from ai_share_of_voice.schemas.analysis import AnalyzedResponse, RawAnswer
from ai_share_of_voice.schemas.brief import Brief
from ai_share_of_voice.schemas.events import EventStatus
from ai_share_of_voice.schemas.llm import ModelClient, Usage
from ai_share_of_voice.schemas.questions import Question, QuestionPanel
from ai_share_of_voice.state import GraphState
from ai_share_of_voice.tools.mention_tool import (
    ExtractionError,
    MentionExtractionInput,
    extract_mentions,
)

RAW_FOLDER = "raw"
ANALYZED_FOLDER = "analyzed"
QUESTION_FILES = {"es": "questions_es.csv", "en": "questions.csv"}
DEFAULT_QUESTION_FILE = "questions.csv"


@dataclass(frozen=True)
class RealNodeOptions:
    """Knobs of the real nodes."""

    max_questions: int | None = None  # ask only the first N questions: cheap live checks
    max_failed_share: float = 0.1  # above this share of failed questions, the batch fails
    extraction_attempts: int = 2


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
        panel = load_panel(brief.language, data_dir)
        if options.max_questions is not None:
            panel = QuestionPanel(questions=panel.questions[: options.max_questions])
        result = f"{len(panel.questions)} questions ({brief.language}) from the panel file"
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
            except (ToolFailure, ExtractionError, ValidationError, ValueError):
                failed.append(question.id)
                continue
            raw.append(answer)
            analyzed.append(analysis)

        asked = len(state["panel"].questions)
        if not analyzed or len(failed) > options.max_failed_share * asked:
            raise ToolFailure(
                tool, f"{len(failed)} of {asked} questions failed in batch {batch_index}"
            )

        _write_jsonl(batch_file(run_dir, RAW_FOLDER, batch_index), raw)
        _write_jsonl(batch_file(run_dir, ANALYZED_FOLDER, batch_index), analyzed)

        client_brand = brief.brand.name.casefold()
        brand_mentions = sum(
            1
            for item in analyzed
            for mention in item.mentions
            if mention.brand.casefold() == client_brand and mention.mentioned
        )
        spend = state.get("spend_usd", 0.0) + (budget.spent_usd - spent_before)
        result = f"batch {batch_index}: {len(analyzed)} answers analyzed"
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

    nodes.update(questions=questions, sampling=sampling)
    return nodes


def _model_name(client: Any, usage: Usage) -> str:
    return usage.model or getattr(client, "model", "unknown")
