"""Runs behind the app: start, watch, approve and read results (I6/A4, local integration).

The UI never touches the graph directly. ``RunController`` starts each run in a background
thread, pauses at the human approval (LangGraph ``interrupt`` with a SQLite checkpointer),
resumes it with the decision and builds the results. Everything here is plain data, so it
is tested without a UI; ``app.py`` only draws it.
"""

import csv
import json
import random
import threading
import time
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from langgraph.types import Command
from pydantic import Field, field_validator

from ai_share_of_voice.automation.finalize import ApprovalRequired, make_finalize_node
from ai_share_of_voice.automation.report import build_report_data
from ai_share_of_voice.checkpoint import sqlite_checkpointer
from ai_share_of_voice.config import Settings
from ai_share_of_voice.event_log import EventLog
from ai_share_of_voice.graph import build_graph, initial_state, new_run_id, run_config
from ai_share_of_voice.guard import guard_nodes
from ai_share_of_voice.llm_client import build_clients
from ai_share_of_voice.nodes import NodeDeps
from ai_share_of_voice.nodes_real import (
    DISCOVERED_FILE_NAME,
    PANEL_FILE_NAME,
    RAW_FOLDER,
    RealNodeOptions,
    load_analyzed,
    load_panel,
    make_real_nodes,
)
from ai_share_of_voice.panel import generate_panel, read_panel_file, write_panel_file
from ai_share_of_voice.research import RESEARCH_FILE_NAME
from ai_share_of_voice.schemas.analysis import BrandMention, RawAnswer
from ai_share_of_voice.schemas.base import Contract
from ai_share_of_voice.schemas.brief import BrandSpec, Brief
from ai_share_of_voice.schemas.questions import QuestionPanel
from ai_share_of_voice.stats.charts import (
    BrandRates,
    PrecisionChartText,
    RatesChartText,
    precision_chart,
    rates_chart,
)
from ai_share_of_voice.stats.stopping import should_stop
from ai_share_of_voice.tools.mcp_sources_client import McpSourcesClient
from ai_share_of_voice.tools.stats_tool import (
    CHARTS_FOLDER,
    StatsToolError,
    StatsToolInput,
    run_stats,
)
from ai_share_of_voice.ui.demo_responders import demo_responders
from ai_share_of_voice.ui.i18n import t
from ai_share_of_voice.ui.results_tab import ResultsView, build_results_view
from ai_share_of_voice.writing import CLIENT_REPORT_FILE_NAME, make_client_report_finalize

Phase = Literal["running", "waiting_approval", "completed", "partial", "rejected", "error"]
CHECKPOINT_FILE_NAME = "checkpoints.sqlite"
APPROVED_PANEL_FILE = "approved_questions.csv"  # the user's panel, before the run copies it


class BriefForm(Contract):
    """What the Brief tab sends. Validated before any thread or call starts."""

    brand: str = Field(min_length=1, max_length=100)
    competitors: str  # comma separated, as typed in the form
    sector: str = Field(min_length=2, max_length=100)
    city: str = Field(min_length=2, max_length=100)
    country: str = ""
    language: Literal["es", "en"] = "es"
    panel: Literal["demo", "generate"] = "demo"
    questions: int = Field(default=10, ge=3, le=30)
    batches: int = Field(default=3, ge=2, le=10)

    @field_validator("competitors")
    @classmethod
    def two_or_more(cls, value: str) -> str:
        names = [name.strip() for name in value.split(",") if name.strip()]
        if not 2 <= len(names) <= 5:
            raise ValueError("write between two competitors and five, separated by commas")
        return ", ".join(names)

    def to_brief(self, cap_usd: float) -> Brief:
        return Brief(
            brand=BrandSpec(name=self.brand),
            competitors=tuple(BrandSpec(name=name) for name in self.competitors.split(", ")),
            sector=self.sector,
            city=self.city,
            language=self.language,
            spend_cap_usd=cap_usd,
            max_batches=self.batches,
        )

    @property
    def location(self) -> str | None:
        return f"{self.city}, {self.country}" if self.country.strip() else None


class EventRow(Contract):
    time: str
    status: str
    node: str
    detail: str | None = None


class RunView(Contract):
    run_id: str
    phase: Phase
    events: tuple[EventRow, ...] = ()
    spend_usd: float = 0.0
    cap_usd: float
    draft: str = ""
    objections: tuple[str, ...] = ()
    error: str | None = None


@dataclass(frozen=True)
class AnswerExample:
    """A real answer of the measured AI, its analysis and the questions to pick from."""

    question_id: str
    question: str
    answer: str
    mentions: tuple[BrandMention, ...]
    choices: tuple[tuple[str, str], ...]  # (question id, question text)


@dataclass
class ResultsBundle:
    view: ResultsView | None
    recommendation: float | None = None  # client brand recommendation rate
    spend_usd: float = 0.0
    brand: str = ""
    sector: str = ""
    city: str = ""
    date: str = ""  # YYYY-MM-DD of the results
    discovered: list[dict[str, str]] = field(default_factory=list)
    research: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class _Run:
    graph: Any = None
    config: dict[str, Any] = field(default_factory=dict)
    budget: Any = None
    thread: threading.Thread | None = None
    error: str | None = None
    target_half_width: float = 0.05


SourcesFactory = Callable[[], AbstractContextManager[McpSourcesClient]]


class RunController:
    def __init__(
        self,
        settings: Settings,
        *,
        runs_dir: Path,
        data_dir: Path,
        clients_factory: Callable[..., Any] = build_clients,
        sources_factory: SourcesFactory | None = None,
    ) -> None:
        self.settings = settings
        self.runs_dir = Path(runs_dir)
        self.data_dir = Path(data_dir)
        self._clients_factory = clients_factory
        self._sources_factory = sources_factory
        self._log = EventLog(self.runs_dir)
        self._checkpointer = sqlite_checkpointer(self.runs_dir / CHECKPOINT_FILE_NAME)
        self._runs: dict[str, _Run] = {}
        self._lock = threading.Lock()

    # --- start ------------------------------------------------------------------------------

    def draft_panel(self, form: BriefForm) -> QuestionPanel:
        """The questions proposed for review (R9): the demo panel or one generated now.

        Generating costs one agent call, on a small budget of its own: the run's cap starts
        when the run starts.
        """
        brief = form.to_brief(self.settings.spend_cap_usd)
        if form.panel == "generate":
            responders, text = demo_responders(brief)
            clients = self._clients_factory(
                self.settings, responders=responders, text_responder=text
            )
            return generate_panel(brief, clients.agent, form.questions)
        panel = load_panel(brief.language, self.data_dir)
        return QuestionPanel(questions=panel.questions[: form.questions])

    def start(self, form: BriefForm, panel: QuestionPanel | None = None) -> str:
        """Start a run; ``panel`` is the one the user approved, asked in every batch."""
        brief = form.to_brief(self.settings.spend_cap_usd)
        run_id = new_run_id()
        run = _Run(target_half_width=brief.target_half_width)
        with self._lock:
            self._runs[run_id] = run
        approved = None
        if panel is not None:
            approved = write_panel_file(panel, self.runs_dir / run_id / APPROVED_PANEL_FILE)

        def work() -> None:
            responders, text = demo_responders(brief)
            clients = self._clients_factory(
                self.settings,
                usage_log=self.runs_dir / run_id / "usage.jsonl",
                responders=responders,
                text_responder=text,
            )
            run.budget = clients.budget
            deps = NodeDeps(event_log=self._log, should_stop=should_stop)
            nodes = make_real_nodes(
                deps,
                subject=clients.subject,
                agent=clients.agent,
                budget=clients.budget,
                runs_dir=self.runs_dir,
                data_dir=self.data_dir,
                options=RealNodeOptions(
                    max_questions=None if approved else form.questions,
                    panel_file=approved,
                    generate_panel=approved is None and form.panel == "generate",
                    panel_size=form.questions,
                    location=form.location,
                ),
                sources=self._sources_factory,
                vision=clients.vision,
            )
            nodes["finalize"] = make_client_report_finalize(
                make_finalize_node(self._log), self.runs_dir
            )
            guarded = guard_nodes(nodes, self._log, fatal=(ApprovalRequired,))
            run.graph = build_graph(guarded, checkpointer=self._checkpointer)
            run.config = run_config(run_id, brief)
            run.graph.invoke(initial_state(brief, run_id), run.config)

        self._launch(run, work)
        return run_id

    def _launch(self, run: _Run, work: Callable[[], None]) -> None:
        def target() -> None:
            try:
                work()
            except Exception as error:  # shown in the UI; the guard handles node failures
                run.error = f"{type(error).__name__}: {error}"

        run.error = None
        run.thread = threading.Thread(target=target, daemon=True)
        run.thread.start()

    # --- watch ------------------------------------------------------------------------------

    def view(self, run_id: str) -> RunView:
        run = self._runs[run_id]
        events = tuple(
            EventRow(
                time=f"{event.timestamp:%H:%M:%S}",
                status=event.status.value,
                node=event.node,
                detail=" | ".join(part for part in (event.result, event.reason) if part) or None,
            )
            for event in self._log.read(run_id)
        )
        values: dict[str, Any] = {}
        phase: Phase = "running"
        if run.error:
            phase = "error"
        elif run.thread is not None and not run.thread.is_alive() and run.graph is not None:
            state = run.graph.get_state(run.config)
            values = state.values
            if state.next == ("approval_gate",):
                phase = "waiting_approval"
            else:
                phase = values.get("status", "error")
        spend = run.budget.spent_usd if run.budget is not None else 0.0
        return RunView(
            run_id=run_id,
            phase=phase,
            events=events,
            spend_usd=spend,
            cap_usd=self.settings.spend_cap_usd,
            draft=values.get("draft", ""),
            objections=tuple(values.get("objections", [])),
            error=run.error,
        )

    def wait(self, run_id: str, timeout: float = 600.0) -> RunView:
        """Block until the run leaves the running phase (tests and scripts)."""
        deadline = time.monotonic() + timeout
        while True:
            view = self.view(run_id)
            if view.phase != "running" or time.monotonic() > deadline:
                return view
            time.sleep(0.1)

    # --- approve ----------------------------------------------------------------------------

    def decide(self, run_id: str, *, approved: bool, comment: str | None = None) -> None:
        if self.view(run_id).phase != "waiting_approval":
            raise ValueError(f"run {run_id} is not waiting for approval")
        run = self._runs[run_id]
        decision = {"approved": approved, "comment": (comment or "").strip() or None}
        self._launch(run, lambda: run.graph.invoke(Command(resume=decision), run.config))

    # --- results ----------------------------------------------------------------------------

    def example_answer(
        self, run_id: str, question_id: str | None = None, rng: random.Random | None = None
    ) -> AnswerExample | None:
        """One real answer of the measured AI with its analysis (R10); random by default."""
        run_dir = self.runs_dir / run_id
        panel_path = run_dir / PANEL_FILE_NAME
        raw = sorted((run_dir / RAW_FOLDER).glob("batch_*.jsonl"))
        if not panel_path.is_file() or not raw:
            return None
        questions = {q.id: q.text for q in read_panel_file(panel_path).questions}
        answers = [
            RawAnswer.model_validate_json(line)
            for line in raw[0].read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        answered = [a for a in answers if a.question_id in questions]
        if not answered:
            return None
        chosen = next((a for a in answered if a.question_id == question_id), None)
        if chosen is None:
            chosen = (rng or random.Random()).choice(answered)
        analyzed = {item.response_id: item for item in load_analyzed(run_dir)}
        analysis = analyzed.get(chosen.response_id)
        return AnswerExample(
            question_id=chosen.question_id,
            question=questions[chosen.question_id],
            answer=chosen.text,
            mentions=analysis.mentions if analysis else (),
            choices=tuple((a.question_id, questions[a.question_id]) for a in answered),
        )

    def charts(self, run_id: str, lang: str) -> tuple[str | None, str | None]:
        """Paths of the rates and precision charts in ``lang`` (``None`` when not ready)."""
        run = self._runs[run_id]
        if run.graph is None:
            return None, None
        state = run.graph.get_state(run.config).values
        return _build_charts(state, self.runs_dir, run_id, run.target_half_width, lang)

    def results(self, run_id: str) -> ResultsBundle:
        run = self._runs[run_id]
        run_dir = self.runs_dir / run_id
        bundle = ResultsBundle(view=None)
        if run.graph is not None:
            state = run.graph.get_state(run.config).values
            try:
                data = build_report_data(state, generated_at=datetime.now(UTC))
            except Exception as error:  # not enough data yet: say so instead of failing
                bundle.notes.append(f"No results yet: {error}")
            else:
                bundle.view = build_results_view(
                    data, self.runs_dir, target_half_width=run.target_half_width
                )
                client_report = run_dir / CLIENT_REPORT_FILE_NAME
                if client_report.is_file():  # the client's report first, the audit after
                    downloads = (str(client_report), *bundle.view.downloads)
                    bundle.view = bundle.view.model_copy(update={"downloads": downloads})
                bundle.spend_usd = data.spend_usd
                bundle.brand, bundle.sector, bundle.city = data.brand, data.sector, data.city
                bundle.date = f"{data.generated_at:%Y-%m-%d}"
                bundle.recommendation = _client_rate(
                    run_id, data.brand, self.runs_dir, "recommendation_rate"
                )
        ranking = run_dir / DISCOVERED_FILE_NAME
        if ranking.is_file():
            with ranking.open(encoding="utf-8", newline="") as handle:
                bundle.discovered = list(csv.DictReader(handle))
        research = run_dir / RESEARCH_FILE_NAME
        if research.is_file():
            bundle.research = json.loads(research.read_text(encoding="utf-8"))
        return bundle


def _client_rate(run_id: str, brand: str, runs_dir: Path, metric: str) -> float | None:
    """One metric of the client brand from the vetted stats tool (``None`` if unavailable)."""
    try:
        result = run_stats(StatsToolInput(run_id=run_id, metric=metric, chart=False), runs_dir)
    except StatsToolError:
        return None
    row = next((row for row in result.rows if row.brand == brand), None)
    return None if row is None else row.estimate


def _build_charts(
    state: dict[str, Any], runs_dir: Path, run_id: str, target_half_width: float, lang: str
) -> tuple[str | None, str | None]:
    """The two app charts of a run in ``lang``: (rates chart, precision chart) paths."""
    history = state.get("history", [])
    brief = state.get("brief")
    folder = runs_dir / run_id / CHARTS_FOLDER
    rates_path = precision_path = None
    try:
        mention = run_stats(
            StatsToolInput(run_id=run_id, metric="mention_rate", chart=False), runs_dir
        )
        recommendation = run_stats(
            StatsToolInput(run_id=run_id, metric="recommendation_rate", chart=False), runs_dir
        )
    except StatsToolError:
        mention = recommendation = None
    if mention is not None and brief is not None:
        recommended = {row.brand: row for row in recommendation.rows}
        rows = [
            BrandRates(
                brand=row.brand,
                mention=(row.estimate or 0.0, row.ci_low or 0.0, row.ci_high or 0.0),
                recommendation=(
                    (rec.estimate or 0.0, rec.ci_low or 0.0, rec.ci_high or 0.0)
                    if (rec := recommended.get(row.brand)) is not None
                    else None
                ),
            )
            for row in mention.rows
        ]
        text = RatesChartText(
            title=t(lang, "chart_rates_title"),
            mention=t(lang, "chart_named"),
            recommendation=t(lang, "chart_recommended"),
            axis=t(lang, "chart_axis"),
            note=t(lang, "chart_rates_note"),
        )
        rates_path = str(
            rates_chart(rows, brief.brand.name, folder / f"app_rates_{lang}.png", text)
        )
    if history:
        text = PrecisionChartText(
            title=t(lang, "chart_precision_title"),
            rounds=t(lang, "chart_rounds"),
            margin=t(lang, "chart_margin"),
            target=t(lang, "chart_target"),
            note=t(lang, "chart_precision_note"),
        )
        precision_path = str(
            precision_chart(history, target_half_width, folder / f"app_precision_{lang}.png", text)
        )
    return rates_path, precision_path
