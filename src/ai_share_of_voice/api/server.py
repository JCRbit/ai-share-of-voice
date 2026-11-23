"""HTTP API for the chat frontend (``web/``): the same runs as the Gradio app, as JSON.

Run it with ``python serve.py`` (add ``--simulated`` for the free demo). Everything goes through
``RunController``, so the graph, the spending caps, the human approval and the files written are
exactly those of the Gradio app; this module only translates them for the browser.

The server listens on this computer only. Run ids are checked with the stats tool's pattern, and
downloads are limited to a fixed list of file names inside the run's folder.
"""

import csv
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError

from ai_share_of_voice.api import concierge
from ai_share_of_voice.config import LlmMode, Settings
from ai_share_of_voice.event_log import EventLog
from ai_share_of_voice.llm_client import build_clients
from ai_share_of_voice.nodes_real import PANEL_FILE_NAME, RAW_FOLDER, load_analyzed
from ai_share_of_voice.panel import check_user_panel, read_panel_file
from ai_share_of_voice.schemas.analysis import RawAnswer
from ai_share_of_voice.schemas.questions import Question, QuestionPanel
from ai_share_of_voice.tools.stats_tool import StatsToolError, StatsToolInput, run_stats
from ai_share_of_voice.ui.demo_responders import demo_responders
from ai_share_of_voice.ui.run_controller import BriefForm, RunController
from ai_share_of_voice.voice import headline, objections_intro, stage_line
from ai_share_of_voice.writing import CLIENT_REPORT_FILE_NAME

logger = logging.getLogger(__name__)

DOWNLOADS = {
    "client_report.md": "text/markdown",
    "report.md": "text/markdown",
    "answers.csv": "text/csv",
    "tracking_row.csv": "text/csv",
    "discovered_businesses.csv": "text/csv",
    "questions.csv": "text/csv",
    "events.jsonl": "application/json",
}
CHART_SUFFIXES = (".png",)
PLANS = {"quick": (5, 2), "standard": (10, 3), "deep": (20, 5)}
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
    "finalize": ("Publicación", "Publishing"),
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


class ParseRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    lang: Literal["es", "en"] = "es"


class BriefIn(BaseModel):
    brand: str
    competitors: list[str]
    sector: str
    city: str
    country: str = ""
    language: Literal["es", "en"] = "es"
    plan: Literal["quick", "standard", "deep"] = "standard"
    panel: Literal["demo", "generate"] = "demo"

    def form(self) -> BriefForm:
        questions, batches = PLANS[self.plan]
        return BriefForm(
            brand=self.brand.strip(),
            competitors=", ".join(c.strip() for c in self.competitors if c.strip()),
            sector=self.sector.strip(),
            city=self.city.strip(),
            country=self.country.strip(),
            language=self.language,
            panel=self.panel,
            questions=questions,
            batches=batches,
        )


class StartRequest(BaseModel):
    brief: BriefIn
    questions: list[str] | None = None


class DecisionRequest(BaseModel):
    approved: bool
    comment: str | None = Field(default=None, max_length=2000)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    lang: Literal["es", "en"] = "es"


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
) -> FastAPI:
    app = FastAPI(title="AI Share of Voice API", docs_url="/api/docs", openapi_url="/api/openapi.json")
    runs_dir = controller.runs_dir
    log = EventLog(runs_dir)
    simulated = settings.mode is LlmMode.SIMULATED
    langs: dict[str, str] = {}

    def agent_client(brief_text: str = "") -> Any:
        from ai_share_of_voice.schemas.brief import BrandSpec, Brief

        demo = Brief(
            brand=BrandSpec(name="Demo"),
            competitors=(BrandSpec(name="A"), BrandSpec(name="B")),
            sector="demo",
            city="demo",
            spend_cap_usd=settings.spend_cap_usd,
        )
        responders, text = demo_responders(demo)
        return build_clients(
            settings, usage_log=runs_dir / "chat_usage.jsonl", responders=responders, text_responder=text
        ).agent

    def ensure_run(run_id: str) -> None:
        _check_run_id(run_id)
        try:
            controller.open_run(run_id)
        except KeyError as error:
            raise HTTPException(404, "no such measurement") from error

    # --- meta -------------------------------------------------------------------------------

    @app.get("/api/meta")
    def meta() -> dict[str, Any]:
        return {
            "mode": "simulated" if simulated else "live",
            "subject_model": settings.subject_model,
            "cap_usd": settings.spend_cap_usd,
            "plans": {k: {"questions": q, "rounds": b, "answers": q * b} for k, (q, b) in PLANS.items()},
            "stages": [{"node": s, "es": AGENTS[s][0], "en": AGENTS[s][1]} for s in STAGES],
        }

    # --- brief ------------------------------------------------------------------------------

    @app.post("/api/brief/parse")
    def parse_brief(request: ParseRequest) -> dict[str, Any]:
        if not simulated:
            try:
                return concierge.llm_brief(agent_client(), request.text, request.lang).model_dump()
            except Exception:  # the rule-based reader is always there
                logger.exception("brief model call failed; using the rule-based reader")
        return concierge.read_brief(request.text, request.lang).model_dump()

    @app.post("/api/brief/questions")
    def draft_questions(brief: BriefIn) -> dict[str, Any]:
        try:
            form = brief.form()
        except ValidationError as error:
            raise HTTPException(422, _form_errors(error)) from error
        panel = controller.draft_panel(form)
        return {"questions": [q.text for q in panel.questions]}

    # --- runs -------------------------------------------------------------------------------

    @app.post("/api/runs")
    def start(request: StartRequest) -> dict[str, str]:
        try:
            form = request.brief.form()
        except ValidationError as error:
            raise HTTPException(422, _form_errors(error)) from error
        panel = None
        if request.questions:
            panel = QuestionPanel(
                questions=tuple(
                    Question(id=f"u{i + 1:02d}", text=text.strip(), intent="recommendation")
                    for i, text in enumerate(request.questions)
                    if text.strip()
                )
            )
            problems = check_user_panel(panel, form.to_brief(settings.spend_cap_usd))
            if problems:
                raise HTTPException(422, list(problems))
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
                "n_answers": item.n_answers,
            }
            for item in controller.history()
        ]

    @app.get("/api/runs/{run_id}")
    def run_view(run_id: str) -> dict[str, Any]:
        ensure_run(run_id)
        lang = langs.get(run_id, "es")
        view = controller.view(run_id)
        events = []
        for event in log.read(run_id):
            names = AGENTS.get(event.node, (event.node, event.node))
            events.append(
                {
                    "time": event.timestamp.isoformat(),
                    "node": event.node,
                    "agent": names[0] if lang == "es" else names[1],
                    "agent_id": event.agent,
                    "tool": event.tool,
                    "status": event.status.value,
                    "detail": event.result or event.reason,
                    "spend": event.cumulative_spend_usd,
                }
            )
        sampling_rounds = sum(1 for e in events if e["node"] == "sampling" and e["status"] == "started")
        current = next((e for e in reversed(events) if e["status"] == "started"), None)
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
            "now": stage_line(lang, current["node"], max(sampling_rounds, 1)) if current else "",
            "error": view.error,
            "failure_kind": view.failure_kind,
            "convergence": _convergence(events),
        }

    @app.post("/api/runs/{run_id}/decision")
    def decide(run_id: str, request: DecisionRequest) -> dict[str, str]:
        ensure_run(run_id)
        try:
            controller.decide(run_id, approved=request.approved, comment=request.comment)
        except ValueError as error:
            raise HTTPException(409, str(error)) from error
        return {"status": "ok"}

    @app.get("/api/runs/{run_id}/results")
    def results(run_id: str, lang: Literal["es", "en"] = "es") -> dict[str, Any]:
        ensure_run(run_id)
        return _facts(controller, runs_dir, run_id, lang)

    @app.get("/api/runs/{run_id}/answers")
    def answers(run_id: str) -> dict[str, Any]:
        ensure_run(run_id)
        return {"answers": _answers(runs_dir / run_id)}

    @app.post("/api/runs/{run_id}/ask")
    def ask(run_id: str, request: AskRequest) -> dict[str, str]:
        ensure_run(run_id)
        facts = _facts(controller, runs_dir, run_id, request.lang)
        facts.pop("report_md", None)
        if not simulated:
            try:
                return {"reply": concierge.llm_answer(agent_client(), request.question, facts, request.lang)}
            except Exception:
                logger.exception("chat model call failed; answering from the figures")
        return {"reply": concierge.answer_question(request.question, facts, request.lang)}

    @app.get("/api/runs/{run_id}/files/{name}")
    def download(run_id: str, name: str) -> FileResponse:
        _check_run_id(run_id)
        run_dir = (runs_dir / run_id).resolve()
        if name in DOWNLOADS:
            path = run_dir / name
            media = DOWNLOADS[name]
        elif name.endswith(CHART_SUFFIXES) and "/" not in name and ".." not in name:
            path = run_dir / "charts" / name
            media = "image/png"
        else:
            raise HTTPException(404, "not available")
        if not path.is_file() or run_dir not in path.resolve().parents:
            raise HTTPException(404, "not available")
        return FileResponse(path, media_type=media, filename=name)

    if static_dir is not None and static_dir.is_dir():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="web")
    return app


def _convergence(events: list[dict[str, Any]]) -> list[dict[str, float]]:
    """The client's interval after each round, read from the analyst's event lines."""
    import re

    points = []
    for event in events:
        if event["node"] == "metrics" and event["status"] == "succeeded" and event["detail"]:
            match = re.search(r"(\d+)% \[(\d+)%, (\d+)%\], (\d+) answers", event["detail"])
            if match:
                est, lo, hi, n = (int(x) for x in match.groups())
                points.append({"estimate": est / 100, "low": lo / 100, "high": hi / 100, "answers": n})
    return points


def _metric(runs_dir: Path, run_id: str, metric: str) -> dict[str, dict[str, Any]]:
    try:
        result = run_stats(StatsToolInput(run_id=run_id, metric=metric, chart=False), runs_dir)
    except StatsToolError:
        return {}
    return {
        row.brand: {"estimate": row.estimate, "ci_low": row.ci_low, "ci_high": row.ci_high}
        for row in result.rows
    }


def _answers(run_dir: Path) -> list[dict[str, Any]]:
    panel_path = run_dir / PANEL_FILE_NAME
    questions = (
        {q.id: q.text for q in read_panel_file(panel_path).questions} if panel_path.is_file() else {}
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
    mention = _metric(runs_dir, run_id, "mention_rate")
    recommend = _metric(runs_dir, run_id, "recommendation_rate")
    view = bundle.view
    n_answers = view.headline.n_answers if view and view.headline else 0
    n_questions = view.headline.n_questions if view and view.headline else 0
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
    answers = _answers(run_dir)
    wins: dict[str, Counter[str]] = defaultdict(Counter)
    for answer in answers:
        for m in answer["mentions"]:
            if m.get("recommended"):
                wins[m["brand"]][answer["question"]] += 1
    best_questions = {brand: counts.most_common(1)[0][0] for brand, counts in wins.items() if counts}
    client = next((b for b in brands if b["is_client"]), None)
    report = run_dir / CLIENT_REPORT_FILE_NAME
    charts = sorted(p.name for p in (run_dir / "charts").glob("*.png")) if (run_dir / "charts").is_dir() else []
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
        "brands": brands,
        "headline": headline(
            lang,
            bundle.brand,
            (client or {}).get("mention", {}).get("estimate") if client and client["mention"] else None,
            (client or {}).get("recommend", {}).get("estimate") if client and client["recommend"] else None,
            n_answers,
        )
        if client
        else "",
        "best_questions": best_questions,
        "discovered": bundle.discovered,
        "research": bundle.research,
        "notes": list(view.notes) if view else bundle.notes,
        "report_md": report.read_text(encoding="utf-8") if report.is_file() else "",
        "downloads": [name for name in DOWNLOADS if (run_dir / name).is_file()],
        "charts": charts,
        "approved": report.is_file(),
    }


def tracking_rows(runs_dir: Path) -> list[dict[str, str]]:
    path = runs_dir / "tracking.csv"
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


__all__ = ["create_app", "json"]
