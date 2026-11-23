"""HTTP API for the chat frontend (``web/``): the same runs as the Gradio app, as JSON.

Run it with ``python serve.py`` (add ``--simulated`` for the free demo). Everything goes through
``RunController``, so the graph, the spending caps, the human approval and the files written are
exactly those of the Gradio app; this module only translates them for the browser.

Rules kept here (team review of the redesign):

* Every endpoint that can call a model asks for the demo password (``ui/access.AccessGate``,
  header ``X-Demo-Password``) in live mode, and every call goes through the budgeted clients.
* Figures shown or said in the chat come from the statistics, never from free model text.
  Intervals are the per-question (cluster) intervals of ``stats/``; the funnel shows the
  stopping rule's own reason.
* Downloads are a fixed list of names inside the run folder; requests are rate limited.

The server listens on this computer only.
"""

import os
import re
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError

from ai_share_of_voice.api import concierge
from ai_share_of_voice.config import LlmMode, Settings
from ai_share_of_voice.event_log import EventLog
from ai_share_of_voice.llm_client import build_clients
from ai_share_of_voice.nodes_real import PANEL_FILE_NAME, RAW_FOLDER, load_analyzed
from ai_share_of_voice.panel import check_user_panel, read_panel_file
from ai_share_of_voice.schemas.analysis import RawAnswer
from ai_share_of_voice.schemas.brief import BrandSpec, Brief
from ai_share_of_voice.schemas.stopping import StopConfig
from ai_share_of_voice.tools.stats_tool import StatsToolError, StatsToolInput, run_stats
from ai_share_of_voice.ui import catalog
from ai_share_of_voice.ui.access import Access, AccessGate, gate_from_env
from ai_share_of_voice.ui.demo_responders import demo_responders
from ai_share_of_voice.ui.run_controller import BriefForm, RunController
from ai_share_of_voice.voice import event_line, headline, objections_intro, overlaps, stage_line
from ai_share_of_voice.writing import CLIENT_REPORT_FILE_NAME

DOWNLOADS = {
    "client_report.md": "text/markdown; charset=utf-8",
    "report.md": "text/markdown; charset=utf-8",
    "answers.csv": "text/csv; charset=utf-8",
    "tracking_row.csv": "text/csv; charset=utf-8",
    "discovered_businesses.csv": "text/csv; charset=utf-8",
    "questions.csv": "text/csv; charset=utf-8",
}
CHART_NAME = re.compile(r"^[A-Za-z0-9_-]{1,80}\.png$")
PLANS = {"quick": (5, 2), "standard": (10, 3), "deep": (20, 5)}
LIVE_SECONDS_PER_ANSWER = 25  # measured on live runs (Groq, gpt-oss-120b)
SIMULATED_SECONDS_PER_ANSWER = 0.2
DEMO_SECTOR = "food_delivery"  # the only sector with a ready-made question panel
RATE_LIMIT = (240, 60.0)  # requests per window (seconds) per client, all endpoints
DOWNLOAD_LIMIT = (30, 60.0)
AGENTS = {
    "questions": ("Diseñador de preguntas", "Question designer"),
    "sampling": ("Muestreo", "Sampling"),
    "metrics": ("Analista", "Analyst"),
    "research": ("Investigador de fuentes", "Source researcher"),
    "screenshots": ("Analista visual", "Visual analyst"),
    "write_draft": ("Redactor", "Writer"),
    "skeptic": ("Escéptico", "Sceptic"),
    "approval_request": ("Tu aprobación", "Your approval"),
    "approval_gate": ("Tu aprobación", "Your approval"),
    "finalize": ("Informe y descargas", "Report and downloads"),
    "close_partial": ("Cierre parcial", "Partial close"),
    "close_rejected": ("Cierre", "Closed"),
}
STAGES = [
    "questions",
    "sampling",
    "metrics",
    "research",
    "screenshots",
    "write_draft",
    "skeptic",
    "approval_request",
    "finalize",
]
STOP_TEXT = {
    "es": {
        "continue": "Sigo: el margen aún puede mejorar lo bastante y queda presupuesto.",
        "precision_reached": "Paré porque el margen de error ya es tan estrecho como pediste.",
        "marginal_return": (
            "Paré porque otra ronda apenas estrechaba el margen: no compensaba el gasto."
        ),
        "budget_exhausted": "Paré porque otra ronda se pasaría del presupuesto de la medición.",
        "max_batches": "Paré porque llegué al número máximo de rondas que elegiste.",
    },
    "en": {
        "continue": "Continuing: the margin can still improve enough and there is budget left.",
        "precision_reached": "I stopped because the margin is as narrow as you asked.",
        "marginal_return": (
            "I stopped because another round barely narrowed the margin: not worth it."
        ),
        "budget_exhausted": "I stopped because another round would go over this run's budget.",
        "max_batches": "I stopped because I reached the maximum number of rounds you chose.",
    },
}


class ParseRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    lang: Literal["es", "en"] = "es"


class BriefIn(BaseModel):
    brand: str = Field(max_length=100)
    competitors: list[str] = Field(max_length=5)
    sector: str = Field(max_length=100)  # a catalog key, or the sector typed by the person
    city: str = Field(max_length=100)
    country: str = Field(default="", max_length=60)
    language: Literal["es", "en"] = "es"
    plan: Literal["quick", "standard", "deep"] = "standard"

    def sector_text(self) -> str:
        names = catalog.SECTORS.get(self.sector)
        return names.get(self.language, names["es"]) if names else self.sector.strip()

    def panel_kind(self) -> Literal["demo", "generate"]:
        known = self.sector == DEMO_SECTOR or self.sector_text().lower().startswith(
            ("comida a domicilio", "food delivery")
        )
        return "demo" if known else "generate"

    def form(self) -> BriefForm:
        questions, batches = PLANS[self.plan]
        return BriefForm(
            brand=self.brand.strip(),
            competitors=", ".join(c.strip() for c in self.competitors if c.strip()),
            sector=self.sector_text(),
            city=self.city.strip(),
            country=self.country.strip(),
            language=self.language,
            panel=self.panel_kind(),
            questions=questions,
            batches=batches,
        )


class StartRequest(BaseModel):
    brief: BriefIn
    questions: list[str] | None = Field(default=None, max_length=30)


class DecisionRequest(BaseModel):
    approved: bool
    comment: str | None = Field(default=None, max_length=2000)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    lang: Literal["es", "en"] = "es"


class _RateLimiter:
    """A sliding window per client address. Small and in memory: one local process."""

    def __init__(self) -> None:
        self._hits: dict[tuple[str, str], deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, client: str, bucket: str, limit: tuple[int, float]) -> bool:
        count, window = limit
        now = time.monotonic()
        with self._lock:
            hits = self._hits[(client, bucket)]
            while hits and now - hits[0] > window:
                hits.popleft()
            if len(hits) >= count:
                return False
            hits.append(now)
            return True


def _check_run_id(run_id: str) -> None:
    try:
        StatsToolInput(run_id=run_id)
    except ValidationError as error:
        raise HTTPException(400, "invalid run id") from error


def _form_errors(error: ValidationError) -> list[str]:
    return [f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in error.errors()]


def create_app(
    settings: Settings,
    controller: RunController,
    *,
    static_dir: Path | None = None,
    gate: AccessGate | None = None,
) -> FastAPI:
    app = FastAPI(
        title="AI Share of Voice API", docs_url="/api/docs", openapi_url="/api/openapi.json"
    )
    runs_dir = controller.runs_dir
    log = EventLog(runs_dir)
    simulated = settings.mode is LlmMode.SIMULATED
    gate = gate or gate_from_env(os.environ, settings.mode)
    limiter = _RateLimiter()
    langs: dict[str, str] = {}

    @app.middleware("http")
    async def rate_limit(request: Request, call_next: Any) -> Any:
        if request.url.path.startswith("/api/"):
            client = request.client.host if request.client else "?"
            bucket = "files" if "/files/" in request.url.path else "api"
            limit = DOWNLOAD_LIMIT if bucket == "files" else RATE_LIMIT
            if not limiter.allow(client, bucket, limit):
                return JSONResponse({"detail": "too many requests"}, status_code=429)
        return await call_next(request)

    def require_access(password: str | None) -> None:
        """The shared demo password before anything that spends budget.

        ``gate_from_env`` only closes the gate in live mode, so simulated runs stay open unless
        a gate is passed in on purpose (tests, rehearsals).
        """
        if not gate.enabled:
            return
        result = gate.check(password)
        if result is Access.LOCKED:
            raise HTTPException(423, "locked: too many wrong passwords, wait a minute")
        if result is not Access.OK:
            raise HTTPException(401, "demo password required")

    def agent_client() -> Any:
        demo = Brief(
            brand=BrandSpec(name="Demo"),
            competitors=(BrandSpec(name="A"), BrandSpec(name="B")),
            sector="demo",
            city="demo",
            spend_cap_usd=settings.spend_cap_usd,
        )
        responders, text = demo_responders(demo)
        clients = build_clients(
            settings,
            usage_log=runs_dir / "chat_usage.jsonl",
            responders=responders,
            text_responder=text,
        )
        return clients.agent

    def ensure_run(run_id: str) -> None:
        _check_run_id(run_id)
        try:
            controller.open_run(run_id)
        except KeyError as error:
            raise HTTPException(404, "no such measurement") from error

    def graph_values(run_id: str) -> dict[str, Any]:
        # RunController keeps no public accessor for the live state; reading it is safe.
        run = controller._runs.get(run_id)  # noqa: SLF001
        if run is None or run.graph is None:
            return {}
        try:
            return dict(run.graph.get_state(run.config).values)
        except Exception:
            return {}

    # --- meta -------------------------------------------------------------------------------

    @app.get("/api/meta")
    def meta() -> dict[str, Any]:
        seconds = SIMULATED_SECONDS_PER_ANSWER if simulated else LIVE_SECONDS_PER_ANSWER
        return {
            "mode": "simulated" if simulated else "live",
            "subject_model": settings.subject_model,
            "cap_usd": settings.spend_cap_usd,
            "password_required": gate.enabled,
            "seconds_per_answer": seconds,
            "plans": {
                k: {
                    "questions": q,
                    "rounds": b,
                    "answers": q * b,
                    "seconds": round(q * b * seconds),
                }
                for k, (q, b) in PLANS.items()
            },
            "stages": [{"node": s, "es": AGENTS[s][0], "en": AGENTS[s][1]} for s in STAGES],
            "sectors": [
                {"key": key, "es": names["es"], "en": names["en"], "demo_panel": key == DEMO_SECTOR}
                for key, names in catalog.SECTORS.items()
                if key != catalog.OTHER_SECTOR
            ],
            "stop_rules": {
                name: field.default
                for name, field in StopConfig.model_fields.items()
                if not field.is_required()
            },
        }

    @app.post("/api/access")
    def check_access(x_demo_password: str | None = Header(default=None)) -> dict[str, bool]:
        require_access(x_demo_password)
        return {"ok": True}

    # --- brief ------------------------------------------------------------------------------

    @app.post("/api/brief/parse")
    def parse_brief(
        request: ParseRequest, x_demo_password: str | None = Header(default=None)
    ) -> dict[str, Any]:
        draft = concierge.read_brief(request.text, request.lang)
        if draft.missing and not simulated:
            # Only a sentence the rules could not read costs a model call, behind the gate.
            require_access(x_demo_password)
            try:
                draft = concierge.llm_brief(agent_client(), request.text, request.lang)
            except Exception:
                pass  # the rule-based reading stands; the card lets the person fill gaps
        result = draft.model_dump()
        result["sector_key"] = _sector_key(draft.sector)
        return result

    @app.post("/api/brief/questions")
    def draft_questions(
        brief: BriefIn, x_demo_password: str | None = Header(default=None)
    ) -> dict[str, Any]:
        try:
            form = brief.form()
        except ValidationError as error:
            raise HTTPException(422, _form_errors(error)) from error
        if form.panel == "generate":
            require_access(x_demo_password)
        try:
            panel = controller.draft_panel(form)
        except Exception as error:
            raise HTTPException(502, f"could not prepare the questions: {error}") from error
        return {
            "questions": [q.text for q in panel.questions],
            "generated": form.panel == "generate",
        }

    # --- runs -------------------------------------------------------------------------------

    @app.post("/api/runs")
    def start(
        request: StartRequest, x_demo_password: str | None = Header(default=None)
    ) -> dict[str, str]:
        require_access(x_demo_password)
        try:
            form = request.brief.form()
        except ValidationError as error:
            raise HTTPException(422, _form_errors(error)) from error
        panel = None
        if request.questions:
            rows = [(text[:500], "recommendation") for text in request.questions]
            panel, problems = check_user_panel(rows, form.to_brief(settings.spend_cap_usd))
            if problems or panel is None:
                raise HTTPException(
                    422, [{"row": p.row, "kind": p.kind} for p in problems] or "no questions"
                )
        run_id = controller.start(form, panel)
        langs[run_id] = form.language
        return {"run_id": run_id}

    @app.get("/api/runs")
    def history() -> list[dict[str, Any]]:
        return [
            {
                "run_id": item.run_id,
                "date": item.date,
                "time": item.time,
                "brand": item.brand,
                "sector": item.sector,
                "city": item.city,
                "mention": item.share,
                "ci_low": item.ci_low,
                "ci_high": item.ci_high,
                "n_answers": item.n_answers,
                "n_questions": item.n_questions,
            }
            for item in controller.history()
        ]

    @app.get("/api/runs/{run_id}")
    def run_view(run_id: str, lang: Literal["es", "en"] | None = None) -> dict[str, Any]:
        ensure_run(run_id)
        lang = lang or langs.get(run_id, "es")
        view = controller.view(run_id)
        values = graph_values(run_id)
        events = []
        sampling_rounds = 0
        for event in log.read(run_id):
            if event.node == "sampling" and event.status.value == "started":
                sampling_rounds += 1
            names = AGENTS.get(event.node, (event.node, event.node))
            events.append(
                {
                    "time": event.timestamp.isoformat(),
                    "node": event.node,
                    "agent": names[0] if lang == "es" else names[1],
                    "tool": event.tool,
                    "status": event.status.value,
                    "detail": event_line(
                        lang, event.node, event.status.value, max(sampling_rounds, 1)
                    ),
                    "spend": event.cumulative_spend_usd,
                }
            )
        rounds_started = sum(
            1 for e in events if e["node"] == "sampling" and e["status"] == "started"
        )
        current = next((e for e in reversed(events) if e["status"] == "started"), None)
        brief = values.get("brief")
        stop = values.get("stop_decision")
        return {
            "run_id": run_id,
            "phase": view.phase,
            "events": events,
            "spend_usd": view.spend_usd,
            "cap_usd": view.cap_usd,
            "rounds_done": view.rounds_done,
            "max_rounds": view.max_rounds,
            "draft": view.draft,
            "objections": list(view.objections),
            "objections_intro": objections_intro(lang, len(view.objections)),
            "now": stage_line(lang, current["node"], max(rounds_started, 1)) if current else "",
            "error": view.error,
            "failure_kind": view.failure_kind,
            "brief": {
                "brand": brief.brand.name,
                "competitors": [c.name for c in brief.competitors],
                "sector": brief.sector,
                "city": brief.city,
            }
            if brief
            else None,
            "target_half_width": brief.target_half_width if brief else None,
            "funnel": [
                {
                    "round": item.batch_index + 1,
                    "estimate": item.brand_share,
                    "low": item.ci_low,
                    "high": item.ci_high,
                    "half_width": (item.ci_high - item.ci_low) / 2,
                    "answers": item.n_answers,
                    "questions": item.n_questions,
                    "spend": item.cumulative_spend_usd,
                }
                for item in values.get("history", [])
            ],
            "stop": {
                "stop": stop.stop,
                "reason": stop.reason.value,
                "reason_text": STOP_TEXT[lang].get(stop.reason.value, ""),
                "explanation": stop.explanation,
            }
            if stop
            else None,
        }

    @app.post("/api/runs/{run_id}/decision")
    def decide(run_id: str, request: DecisionRequest) -> dict[str, str]:
        ensure_run(run_id)
        try:
            controller.decide(run_id, approved=request.approved, comment=request.comment)
        except ValueError as error:
            raise HTTPException(409, str(error)) from error
        return {"status": "ok"}

    def facts(run_id: str, lang: str) -> dict[str, Any]:
        data = _facts(controller, runs_dir, run_id, lang)
        stop = graph_values(run_id).get("stop_decision")
        if stop is not None:
            data["stop"] = {
                "reason": stop.reason.value,
                "reason_text": STOP_TEXT.get(lang, STOP_TEXT["es"]).get(stop.reason.value, ""),
            }
        return data

    @app.get("/api/runs/{run_id}/results")
    def results(run_id: str, lang: Literal["es", "en"] = "es") -> dict[str, Any]:
        ensure_run(run_id)
        return facts(run_id, lang)

    @app.get("/api/runs/{run_id}/answers")
    def answers(run_id: str) -> dict[str, Any]:
        ensure_run(run_id)
        return {"answers": _answers(runs_dir / run_id)}

    @app.post("/api/runs/{run_id}/ask")
    def ask(run_id: str, request: AskRequest) -> dict[str, str]:
        """Answers built from the computed figures only: no model call, nothing invented."""
        ensure_run(run_id)
        data = facts(run_id, request.lang)
        return {"reply": concierge.answer_question(request.question, data, request.lang)}

    @app.get("/api/runs/{run_id}/files/{name}")
    def download(run_id: str, name: str) -> FileResponse:
        _check_run_id(run_id)
        run_dir = (runs_dir / run_id).resolve()
        if name in DOWNLOADS:
            path, media = run_dir / name, DOWNLOADS[name]
        elif CHART_NAME.match(name):
            path, media = run_dir / "charts" / name, "image/png"
        else:
            raise HTTPException(404, "not available")
        resolved = path.resolve()
        if not resolved.is_file() or run_dir not in resolved.parents:
            raise HTTPException(404, "not available")
        return FileResponse(resolved, media_type=media, filename=name)

    if static_dir is not None and static_dir.is_dir():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="web")
    return app


def _sector_key(text: str) -> str:
    lowered = text.strip().lower()
    for key, names in catalog.SECTORS.items():
        if lowered and lowered in (names["es"].lower(), names["en"].lower()):
            return key
    return ""


def _metric(runs_dir: Path, run_id: str, metric: str) -> tuple[dict[str, dict[str, Any]], int]:
    try:
        result = run_stats(StatsToolInput(run_id=run_id, metric=metric, chart=False), runs_dir)
    except StatsToolError:
        return {}, 0
    rows = {
        row.brand: {
            "estimate": row.estimate,
            "ci_low": row.ci_low,
            "ci_high": row.ci_high,
            "method": row.ci_method,
        }
        for row in result.rows
    }
    return rows, result.n_questions


def _answers(run_dir: Path) -> list[dict[str, Any]]:
    panel_path = run_dir / PANEL_FILE_NAME
    questions = (
        {q.id: q.text for q in read_panel_file(panel_path).questions}
        if panel_path.is_file()
        else {}
    )
    analyzed = {item.response_id: item for item in load_analyzed(run_dir)}
    rows = []
    for path in sorted((run_dir / RAW_FOLDER).glob("batch_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            raw = RawAnswer.model_validate_json(line)
            analysis = analyzed.get(raw.response_id)
            rows.append(
                {
                    "id": raw.response_id,
                    "question_id": raw.question_id,
                    "question": questions.get(raw.question_id, raw.question_id),
                    "round": raw.batch_index + 1,
                    "model": raw.model,
                    "text": raw.text,
                    "mentions": [m.model_dump(mode="json") for m in analysis.mentions]
                    if analysis
                    else [],
                }
            )
    return rows


def _facts(controller: RunController, runs_dir: Path, run_id: str, lang: str) -> dict[str, Any]:
    bundle = controller.results(run_id)
    run_dir = runs_dir / run_id
    mention, n_questions = _metric(runs_dir, run_id, "mention_rate")
    recommend, _ = _metric(runs_dir, run_id, "recommendation_rate")
    view = bundle.view
    n_answers = view.headline.n_answers if view and view.headline else 0
    brands = [
        {
            "name": name,
            "is_client": name == bundle.brand,
            "mention": mention.get(name),
            "recommend": recommend.get(name),
        }
        for name in mention
    ]
    brands.sort(key=lambda b: (not b["is_client"], -((b["mention"] or {}).get("estimate") or 0)))
    client = next((b for b in brands if b["is_client"]), None)
    for brand in brands:
        if client and not brand["is_client"]:
            # "clear" only when the per-question intervals do not overlap.
            brand["mention_vs_client"] = _compare(brand["mention"], client["mention"])
            brand["recommend_vs_client"] = _compare(brand["recommend"], client["recommend"])
    checks = {c.get("name"): c for c in (bundle.research or {}).get("checks", [])}
    discovered = []
    for row in bundle.discovered:
        check = checks.get(row.get("name"), {})
        discovered.append(
            {
                "name": row.get("name"),
                "kind": row.get("kind"),
                "rate": float(row.get("rate") or 0),
                "ci_low": float(row.get("ci_low") or 0),
                "ci_high": float(row.get("ci_high") or 0),
                "n_questions_with": int(row.get("n_questions_with") or 0),
                "suggested": row.get("suggested") == "True",
                # Only what a read source supports: found with a URL, or "not checked".
                "found_online": bool(check.get("found_online")) and bool(check.get("evidence_url")),
                "evidence_url": check.get("evidence_url"),
            }
        )
    report = run_dir / CLIENT_REPORT_FILE_NAME
    charts_dir = run_dir / "charts"
    return {
        "run_id": run_id,
        "brand": bundle.brand,
        "sector": bundle.sector,
        "city": bundle.city,
        "date": bundle.date,
        "model": bundle.subject_model,
        "spend_usd": bundle.spend_usd,
        "n_answers": n_answers,
        "n_questions": n_questions,
        "few_questions": 0 < n_questions < 10,
        "brands": brands,
        "client_gap_clear": overlaps((client or {}).get("mention"), (client or {}).get("recommend"))
        is False
        if client
        else None,
        "headline": headline(
            lang,
            bundle.brand,
            (client or {}).get("mention"),
            (client or {}).get("recommend"),
            n_answers,
        )
        if client
        else "",
        "discovered": discovered,
        "research": bundle.research,
        "notes": list(view.notes) if view else bundle.notes,
        "report_md": report.read_text(encoding="utf-8") if report.is_file() else "",
        "downloads": [name for name in DOWNLOADS if (run_dir / name).is_file()],
        "charts": sorted(p.name for p in charts_dir.glob("*.png")) if charts_dir.is_dir() else [],
        "approved": report.is_file(),
        "trend": [
            {
                "run_id": item.run_id,
                "date": item.date,
                "time": item.time,
                "mention": item.share,
                "ci_low": item.ci_low,
                "ci_high": item.ci_high,
                "n_questions": item.n_questions,
            }
            for item in controller.trend(run_id)
        ],
    }


def _compare(a: dict[str, Any] | None, b: dict[str, Any] | None) -> str:
    """ "higher", "lower" or "not_conclusive" (overlapping intervals), "unknown" if missing."""
    o = overlaps(a, b)
    if o is None:
        return "unknown"
    if o:
        return "not_conclusive"
    return "higher" if (a or {}).get("estimate", 0) > (b or {}).get("estimate", 0) else "lower"
