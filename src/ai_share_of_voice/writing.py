"""Report writer and Skeptic with the full run record (T7, local integration, E).

The writer works in two steps, so the model never has anything to invent from:

1. ``template_draft`` writes the report in code from ``ReportFacts``: every figure comes from
   the vetted stats tool, every source is numbered [S#], and the limits are always stated.
   It passes every Skeptic check by construction (a test checks it).
2. The agent model rewrites that draft in plain, friendly language under strict rules: no
   new figure, source or claim. A rewrite that adds an unrecorded figure or source is
   discarded and the code draft is used; so is a failed call. There is always a report.

On a revision round the model gets the Skeptic's objections and fixes only those claims.

The Skeptic reuses J's checks and model review (``agents/skeptic*.py``) with one change: the
record holds every figure the run computed (all brands, mention and recommendation,
discovered businesses), not only the client's headline. Upstream, J's node would take the
record as a parameter; until then this wrapper keeps his file untouched.
"""

import csv
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ai_share_of_voice.agents.skeptic import (
    MAX_OBJECTIONS,
    SkepticReview,
    accept_model_objections,
    record_from_state,
)
from ai_share_of_voice.agents.skeptic import (
    build_prompt as build_skeptic_prompt,
)
from ai_share_of_voice.agents.skeptic_checks import ObjectionKind, RunRecord, run_checks
from ai_share_of_voice.errors import ToolFailure
from ai_share_of_voice.event_log import EventLog
from ai_share_of_voice.nodes import NodeFn
from ai_share_of_voice.schemas.events import EventStatus, GraphEvent
from ai_share_of_voice.schemas.llm import ModelClient
from ai_share_of_voice.schemas.stopping import StopReason
from ai_share_of_voice.state import GraphState
from ai_share_of_voice.tools.stats_tool import StatsToolError, StatsToolInput, run_stats

DISCOVERED_FILE_NAME = "discovered_businesses.csv"  # written by the metrics node
RESEARCH_FILE_NAME = "research.json"  # written by the research node
MAX_DISCOVERED = 5
CLIENT_REPORT_FILE_NAME = "client_report.md"
Rate = tuple[float, float, float]  # estimate, interval low, interval high
_BLOCKING = {ObjectionKind.UNRECORDED_FIGURE, ObjectionKind.UNKNOWN_SOURCE}


# --- facts ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class BrandFacts:
    brand: str
    mention: Rate
    recommendation: Rate | None


@dataclass(frozen=True)
class DiscoveredFacts:
    name: str
    rate: float
    n_questions_with: int
    suggested: bool
    evidence_url: str | None = None


@dataclass(frozen=True)
class ReportFacts:
    language: str
    brand: str
    sector: str
    city: str
    date: str
    n_answers: int
    n_questions: int
    few_questions: bool
    confidence: float
    stop_reason: StopReason
    brands: tuple[BrandFacts, ...]
    discovered: tuple[DiscoveredFacts, ...]
    sources: tuple[str, ...]  # [S1] is sources[0]


def _rate(row: Any) -> Rate | None:
    if row is None or row.estimate is None:
        return None
    return (row.estimate, row.ci_low or 0.0, row.ci_high or 0.0)


def collect_facts(state: GraphState, run_dir: Path) -> ReportFacts:
    """Everything the report may say, from the state and the run folder."""
    run_dir = Path(run_dir)
    brief = state["brief"]
    last = state["history"][-1]
    runs_dir, run_id = run_dir.parent, run_dir.name
    mention = run_stats(StatsToolInput(run_id=run_id, metric="mention_rate", chart=False), runs_dir)
    recommendation = run_stats(
        StatsToolInput(run_id=run_id, metric="recommendation_rate", chart=False), runs_dir
    )
    recommended = {row.brand: row for row in recommendation.rows}
    brands = tuple(
        BrandFacts(row.brand, _rate(row) or (0.0, 0.0, 0.0), _rate(recommended.get(row.brand)))
        for row in mention.rows
    )

    research: dict[str, Any] = {}
    if (run_dir / RESEARCH_FILE_NAME).is_file():
        research = json.loads((run_dir / RESEARCH_FILE_NAME).read_text(encoding="utf-8"))
    evidence = {
        check["name"]: check["evidence_url"]
        for check in research.get("checks", [])
        if check.get("found_online") and check.get("evidence_url")
    }
    sources = list(state.get("sources", []))
    for url in evidence.values():
        if url not in sources:
            sources.append(url)

    discovered: list[DiscoveredFacts] = []
    if (run_dir / DISCOVERED_FILE_NAME).is_file():
        with (run_dir / DISCOVERED_FILE_NAME).open(encoding="utf-8", newline="") as handle:
            rows = sorted(csv.DictReader(handle), key=lambda row: -float(row["rate"]))
        discovered = [
            DiscoveredFacts(
                name=row["name"],
                rate=float(row["rate"]),
                n_questions_with=int(row["n_questions_with"]),
                suggested=row["suggested"] == "True",
                evidence_url=evidence.get(row["name"]),
            )
            for row in rows[:MAX_DISCOVERED]
        ]

    return ReportFacts(
        language=brief.language,
        brand=brief.brand.name,
        sector=brief.sector,
        city=brief.city,
        date=f"{datetime.now(UTC):%Y-%m-%d}",
        n_answers=last.n_answers,
        n_questions=last.n_questions or mention.n_questions,
        few_questions=mention.few_questions,
        confidence=last.confidence,
        stop_reason=state["stop_decision"].reason,
        brands=brands,
        discovered=tuple(discovered),
        sources=tuple(sources),
    )


def facts_record(facts: ReportFacts, state: GraphState | None = None) -> RunRecord:
    """Every figure and source the draft may use: what the run actually computed."""
    base = record_from_state(state) if state is not None else RunRecord()
    values = [*base.percentages, round(facts.confidence * 100, 2)]
    for row in facts.brands:
        for rate in (row.mention, row.recommendation):
            if rate is not None:
                values += [round(value * 100, 2) for value in rate]
    values += [round(item.rate * 100, 2) for item in facts.discovered]
    return RunRecord(
        sources=facts.sources,
        percentages=tuple(sorted(set(values))),
        answer_counts=tuple(sorted({*base.answer_counts, facts.n_answers})),
    )


# --- code draft -----------------------------------------------------------------------------

_TEXT = {
    "es": {
        "title": "# Visibilidad de {brand} en las respuestas de la IA",
        "meta": "_{sector} · {city} · {date} · {answers} respuestas a {questions} preguntas_",
        "summary": "## Resumen",
        "headline": "La IA nombra a {brand} en el {m} de las respuestas (margen de error: entre "
        "{lo} y {hi}) y la recomienda en el {r}.",
        "leader": "La marca más nombrada es {leader}, en el {rate} de las respuestas.",
        "you_lead": "{brand} es la marca más nombrada de la comparativa.",
        "tie": "{names} empatan como las marcas más nombradas, en el {rate} de las respuestas.",
        "nobody": "La IA no nombra a ninguna de las marcas de la comparativa.",
        "approved": "_Aprobado por una persona el {date}._",
        "snapshot": "Es una muestra fechada de lo que responde la IA; no es una cuota de mercado.",
        "few": "Solo se usaron {questions} preguntas: toma el resultado como orientativo.",
        "comparison": "## Comparativa",
        "brand_line": "- **{brand}**: la nombra en el {m} de las respuestas (entre {lo} y {hi}); "
        "la recomienda en el {r}.",
        "others": "## Otros negocios que nombra la IA",
        "other_line": "- **{name}**: aparece en el {rate} de las respuestas, en {n} preguntas.",
        "suggested": " Conviene vigilarlo como competidor.",
        "evidence": " Aparece en internet en esta zona [S{n}].",
        "sources": "## Fuentes consultadas",
        "no_sources": "No se consultaron fuentes externas en esta medición.",
        "method": "## Cómo se midió y sus límites",
        "stop": "- La medición se detuvo cuando {reason}.",
        "named": "- Que la IA nombre una marca no significa que la recomiende: por eso se miden "
        "por separado.",
        "cause": "- Las fuentes muestran en qué información puede apoyarse la IA; no demuestran "
        "por qué responde así.",
        "na": "sin datos",
        "reasons": {
            StopReason.PRECISION_REACHED: "se alcanzó la precisión buscada",
            StopReason.MARGINAL_RETURN: "el margen de error dejó de mejorar",
            StopReason.BUDGET_EXHAUSTED: "se llegó al presupuesto disponible",
            StopReason.MAX_BATCHES: "se llegó al número máximo de rondas",
            StopReason.CONTINUE: "terminó la última ronda prevista",
        },
    },
    "en": {
        "title": "# Visibility of {brand} in AI answers",
        "meta": "_{sector} · {city} · {date} · {answers} answers to {questions} questions_",
        "summary": "## Summary",
        "headline": "The AI names {brand} in {m} of its answers (margin of error: between {lo} "
        "and {hi}) and recommends it in {r}.",
        "leader": "The most named brand is {leader}, in {rate} of the answers.",
        "you_lead": "{brand} is the most named brand of the comparison.",
        "tie": "{names} tie as the most named brands, in {rate} of the answers.",
        "nobody": "The AI names none of the brands of the comparison.",
        "approved": "_Approved by a person on {date}._",
        "snapshot": "This is a dated sample of what the AI answers; it is not a market share.",
        "few": "Only {questions} questions were used: take the result as indicative.",
        "comparison": "## Comparison",
        "brand_line": "- **{brand}**: named in {m} of the answers (between {lo} and {hi}); "
        "recommended in {r}.",
        "others": "## Other businesses the AI names",
        "other_line": "- **{name}**: appears in {rate} of the answers, in {n} questions.",
        "suggested": " Worth tracking as a competitor.",
        "evidence": " It appears online in this area [S{n}].",
        "sources": "## Sources consulted",
        "no_sources": "No external sources were consulted in this measurement.",
        "method": "## How it was measured and its limits",
        "stop": "- The measurement stopped when {reason}.",
        "named": "- Being named by the AI does not mean being recommended: they are measured "
        "separately.",
        "cause": "- The sources show what information the AI may rely on; they do not show why "
        "it answers as it does.",
        "na": "no data",
        "reasons": {
            StopReason.PRECISION_REACHED: "the target precision was reached",
            StopReason.MARGINAL_RETURN: "the margin of error stopped improving",
            StopReason.BUDGET_EXHAUSTED: "the available budget was reached",
            StopReason.MAX_BATCHES: "the maximum number of rounds was reached",
            StopReason.CONTINUE: "the last planned round ended",
        },
    },
}


def _pct(value: float | None, na: str) -> str:
    return na if value is None else f"{round(value * 100)}%"


def _leader_sentence(facts: ReportFacts, text: dict[str, Any]) -> str:
    """Who leads, said honestly: nobody when no brand is named, a tie when there is one."""
    top = max((row.mention[0] for row in facts.brands), default=0.0)
    if top <= 0:
        return text["nobody"]
    leaders = [row.brand for row in facts.brands if abs(row.mention[0] - top) < 1e-9]
    rate = _pct(top, text["na"])
    if len(leaders) > 1:
        joiner = " y " if facts.language == "es" else " and "
        names = ", ".join(leaders[:-1]) + joiner + leaders[-1]
        return text["tie"].format(names=names, rate=rate)
    if leaders[0] == facts.brand:
        return text["you_lead"].format(brand=facts.brand)
    return text["leader"].format(leader=leaders[0], rate=rate)


def template_draft(facts: ReportFacts) -> str:
    """The report written in code: verified figures, numbered sources, stated limits."""
    text = _TEXT.get(facts.language, _TEXT["en"])
    na = text["na"]
    mine = next((row for row in facts.brands if row.brand == facts.brand), None)
    lines = [
        text["title"].format(brand=facts.brand),
        text["meta"].format(
            sector=facts.sector,
            city=facts.city,
            date=facts.date,
            answers=facts.n_answers,
            questions=facts.n_questions,
        ),
        "",
        text["summary"],
    ]
    if mine is not None:
        recommended = mine.recommendation[0] if mine.recommendation else None
        lines.append(
            text["headline"].format(
                brand=facts.brand,
                m=_pct(mine.mention[0], na),
                lo=_pct(mine.mention[1], na),
                hi=_pct(mine.mention[2], na),
                r=_pct(recommended, na),
            )
        )
    lines.append(_leader_sentence(facts, text))
    lines.append(text["snapshot"])
    if facts.few_questions:
        lines.append(text["few"].format(questions=facts.n_questions))

    lines += ["", text["comparison"]]
    for row in sorted(facts.brands, key=lambda item: -item.mention[0]):
        lines.append(
            text["brand_line"].format(
                brand=row.brand,
                m=_pct(row.mention[0], na),
                lo=_pct(row.mention[1], na),
                hi=_pct(row.mention[2], na),
                r=_pct(row.recommendation[0] if row.recommendation else None, na),
            )
        )

    if facts.discovered:
        lines += ["", text["others"]]
        for item in facts.discovered:
            line = text["other_line"].format(
                name=item.name, rate=_pct(item.rate, na), n=item.n_questions_with
            )
            if item.suggested:
                line += text["suggested"]
            if item.evidence_url in facts.sources:
                line += text["evidence"].format(n=facts.sources.index(item.evidence_url) + 1)
            lines.append(line)

    lines += ["", text["sources"]]
    if facts.sources:
        lines += [f"- [S{index}] {url}" for index, url in enumerate(facts.sources, start=1)]
    else:
        lines.append(text["no_sources"])

    lines += [
        "",
        text["method"],
        text["stop"].format(reason=text["reasons"][facts.stop_reason]),
        text["named"],
        text["cause"],
    ]
    return "\n".join(lines) + "\n"


# --- writer node ----------------------------------------------------------------------------


class DraftReport(BaseModel):
    """Schema the model answers with."""

    markdown: str = Field(min_length=20)


LANGUAGE_NAMES = {"es": "Spanish", "en": "English"}


def writer_prompt(template: str, language: str, previous: str = "", objections=()) -> str:
    rules = (
        "Rules:\n"
        "- Never add a figure, percentage, count, brand, business, source or claim that is not "
        "in the verified draft. Keep every figure exactly as written.\n"
        "- Keep the source markers such as [S1] and their links exactly.\n"
        "- Keep 'named' and 'recommended' separate: a named brand is not a recommended one.\n"
        "- Never say that a source or anything else causes the AI's answers, and never call "
        "the results a market share.\n"
        "- Keep the sections and the limits section.\n"
        '- Return a JSON object with one key, "markdown".\n'
    )
    task = (
        f"You edit a report for a small business owner. Rewrite the verified draft below in "
        f"clear, friendly {LANGUAGE_NAMES.get(language, language)} with short sentences.\n\n"
    )
    if objections:
        listed = "\n".join(f"- {item}" for item in objections)
        task = (
            "A reviewer raised objections about the current draft. Fix only those claims, "
            "using the verified draft as the source of truth, and keep everything else.\n\n"
            f"Objections:\n{listed}\n\n<current_draft>\n{previous}\n</current_draft>\n\n"
        )
    return f"{task}{rules}\n<verified_draft>\n{template}\n</verified_draft>\n"


def _event(state: GraphState, node: str, agent: str, status: EventStatus, **fields) -> GraphEvent:
    return GraphEvent(
        run_id=state["run_id"],
        node=node,
        agent=agent,
        status=status,
        cumulative_spend_usd=fields.pop("spend", state.get("spend_usd")),
        **fields,
    )


def make_writer_node(event_log: EventLog, client: ModelClient, runs_dir: Path) -> NodeFn:
    def write_draft(state: GraphState) -> dict[str, Any]:
        events = [_event(state, "write_draft", "writer", EventStatus.STARTED)]
        event_log.append(events[0])
        objections = list(state.get("objections", []))
        version = state.get("draft_version", 0) + 1
        review_round = state.get("review_round", 0) + (1 if objections else 0)

        facts = collect_facts(state, Path(runs_dir) / state["run_id"])
        template = template_draft(facts)
        record = facts_record(facts, state)
        prompt = writer_prompt(template, facts.language, state.get("draft", ""), objections)
        draft, how, cost = template, "code template", 0.0
        try:
            rewrite, usage = client.complete(prompt, DraftReport)
            cost = usage.cost_usd
        except (ValueError, ToolFailure):
            how = "code template (model rewrite unavailable)"
        else:
            unverified = [
                item for item in run_checks(rewrite.markdown, record) if item.kind in _BLOCKING
            ]
            if unverified:
                how = f"code template (model rewrite discarded: {len(unverified)} unverified)"
            else:
                draft, how = rewrite.markdown.strip() + "\n", "model rewrite of the code draft"

        spend = state.get("spend_usd", 0.0) + cost
        done = _event(
            state,
            "write_draft",
            "writer",
            EventStatus.SUCCEEDED,
            result=f"draft v{version}: {how}",
            spend=spend,
        )
        event_log.append(done)
        update: dict[str, Any] = {
            "draft": draft,
            "draft_version": version,
            "review_round": review_round,
            "objections": [],
            "events": [*events, done],
        }
        if cost:
            update["spend_usd"] = spend
        return update

    return write_draft


# --- skeptic with the full record -----------------------------------------------------------


def _with_run_facts(prompt: str, state: GraphState) -> str:
    """Add what J's prompt does not carry: the run facts and a rule for method statements.

    Without them, a live review called "only 5 questions were used" unverifiable, and a
    sentence describing the method an unsourced claim.
    """
    history = state.get("history", [])
    decision = state.get("stop_decision")
    last = history[-1] if history else None
    facts = (
        "Recorded run facts: "
        f"questions: {last.n_questions if last else 'unknown'}; "
        f"answers: {last.n_answers if last else 'unknown'}; "
        f"rounds: {len(history)}; "
        f"stop reason: {decision.reason.value if decision else 'unknown'}.\n"
        "Sentences that describe the method or its limits (how mentions and "
        "recommendations are measured, what the sample is not) are not claims about "
        "brands and need no source: do not object to them.\n\n"
    )
    return prompt.replace("<draft>", facts + "<draft>", 1)


def make_checked_skeptic_node(
    event_log: EventLog,
    client: ModelClient | None,
    runs_dir: Path,
    *,
    max_objections: int = MAX_OBJECTIONS,
) -> NodeFn:
    """J's Skeptic (code checks, then a model review) over the full run record."""

    def skeptic(state: GraphState) -> dict[str, Any]:
        started = _event(state, "skeptic", "skeptic", EventStatus.STARTED, tool="read_results")
        event_log.append(started)
        draft = state.get("draft", "")
        try:
            record = facts_record(collect_facts(state, Path(runs_dir) / state["run_id"]), state)
        except (StatsToolError, KeyError, IndexError):
            record = record_from_state(state)  # not enough data: the headline record only

        found = run_checks(draft, record)
        from_code, from_model, cost, skipped = len(found), 0, 0.0, False
        if client is not None and draft.strip():
            try:
                prompt = _with_run_facts(build_skeptic_prompt(draft, record), state)
                review, usage = client.complete(prompt, SkepticReview)
            except ValueError:
                skipped = True  # an invalid answer must not discard the code findings
            else:
                cost = usage.cost_usd
                raised, _dropped = accept_model_objections(review, draft)
                known = {(item.kind, item.claim) for item in found}
                extra = [item for item in raised if (item.kind, item.claim) not in known]
                from_model = len(extra)
                found += extra

        kept = found[:max_objections]
        spend = state.get("spend_usd", 0.0) + cost
        result = f"{len(kept)} objections kept ({from_code} from code, {from_model} from the model)"
        if skipped:
            result += "; model review skipped: invalid answer"
        if kept:  # what was objected, for the audit trail (the full list goes to the state)
            result += ": " + " | ".join(item.render()[:160] for item in kept[:5])
        done = _event(
            state,
            "skeptic",
            "skeptic",
            EventStatus.SUCCEEDED,
            tool="read_results",
            result=result,
            spend=spend,
        )
        event_log.append(done)
        update: dict[str, Any] = {
            "objections": [item.render() for item in kept],
            "events": [started, done],
        }
        if cost:
            update["spend_usd"] = spend
        return update

    return skeptic


# --- client report --------------------------------------------------------------------------


def make_client_report_finalize(finalize: NodeFn, runs_dir: Path) -> NodeFn:
    """Run J's finalize (audit report and tracking CSV), then write the client report.

    ``report.md`` is the audit report in English. ``client_report.md`` is the approved draft
    in the language of the brief, the file a non-technical client reads. It is written only
    after the approved finalize succeeded, so a failure never leaves a half-published run.
    """

    def finalize_with_client_report(state: GraphState) -> dict[str, Any]:
        update = finalize(state)
        text = _TEXT.get(state["brief"].language, _TEXT["en"])
        stamp = text["approved"].format(date=f"{datetime.now(UTC):%Y-%m-%d}")
        path = Path(runs_dir) / state["run_id"] / CLIENT_REPORT_FILE_NAME
        body = state.get("draft", "").rstrip()
        path.write_text(f"{body}\n\n---\n{stamp}\n", encoding="utf-8")
        return update

    return finalize_with_client_report
