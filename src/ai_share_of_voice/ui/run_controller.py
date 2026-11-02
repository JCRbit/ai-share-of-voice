"""Runs behind the app: start, watch, approve and read results (I6/A4, local integration).

The UI never touches the graph directly. ``RunController`` starts each run in a background
thread, pauses at the human approval (LangGraph ``interrupt`` with a SQLite checkpointer),
resumes it with the decision and builds the results. Everything here is plain data, so it
is tested without a UI; ``app.py`` only draws it.
"""

import csv
import json
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
from ai_share_of_voice.nodes_real import DISCOVERED_FILE_NAME, RealNodeOptions, make_real_nodes
from ai_share_of_voice.research import RESEARCH_FILE_NAME
from ai_share_of_voice.schemas.base import Contract
from ai_share_of_voice.schemas.brief import BrandSpec, Brief
from ai_share_of_voice.stats.stopping import should_stop
from ai_share_of_voice.tools.mcp_sources_client import McpSourcesClient
from ai_share_of_voice.tools.stats_tool import StatsToolError, StatsToolInput, run_stats
from ai_share_of_voice.ui.demo_responders import demo_responders
from ai_share_of_voice.ui.results_tab import ResultsView, build_results_view

Phase = Literal["running", "waiting_approval", "completed", "partial", "rejected", "error"]
CHECKPOINT_FILE_NAME = "checkpoints.sqlite"


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


@dataclass
class ResultsBundle:
    view: ResultsView | None
    recommendation: float | None = None  # client brand recommendation rate
    spend_usd: float = 0.0
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

    def start(self, form: BriefForm) -> str:
        brief = form.to_brief(self.settings.spend_cap_usd)
        run_id = new_run_id()
        run = _Run(target_half_width=brief.target_half_width)
        with self._lock:
            self._runs[run_id] = run

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
                    max_questions=form.questions,
                    generate_panel=form.panel == "generate",
                    panel_size=form.questions,
                    location=form.location,
                ),
                sources=self._sources_factory,
            )
            nodes["finalize"] = make_finalize_node(self._log)
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
                bundle.spend_usd = data.spend_usd
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
