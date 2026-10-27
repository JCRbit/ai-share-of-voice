"""Client report in Markdown (owner: J).

Every figure carries the tool that produced it, sources are listed with their URL, visual
findings are labelled as hypotheses, and the run log is attached, so a third party can
trace each claim. The report always states that it is a dated sample and not market share.
"""

import re
from datetime import UTC, datetime

from pydantic import AwareDatetime, Field

from ai_share_of_voice.schemas.base import Contract
from ai_share_of_voice.schemas.events import GraphEvent
from ai_share_of_voice.schemas.stopping import BatchSummary, StopDecision
from ai_share_of_voice.state import GraphState

REPORT_FILE_NAME = "report.md"
MAX_CELL_CHARS = 120

_SAFE_URL = re.compile(r"https?://[^\s<>()\[\]|`]+")


class ReportError(Exception):
    """Raised when the state does not hold what a report needs."""


class ReportData(Contract):
    """Everything the report shows, taken from the final graph state."""

    run_id: str = Field(min_length=1)
    generated_at: AwareDatetime
    brand: str = Field(min_length=1)
    competitors: tuple[str, ...]
    sector: str = Field(min_length=1)
    city: str = Field(min_length=1)
    history: tuple[BatchSummary, ...] = Field(min_length=1)
    stop: StopDecision
    spend_usd: float = Field(ge=0)
    sources: tuple[str, ...] = ()
    findings: tuple[str, ...] = ()
    draft: str = ""
    objections: tuple[str, ...] = ()
    events: tuple[GraphEvent, ...] = ()

    @property
    def summary(self) -> BatchSummary:
        """Cumulative figures after the last batch."""
        return self.history[-1]


def build_report_data(state: GraphState, generated_at: datetime) -> ReportData:
    """Collect the report data from the state; fail clearly if something essential is missing."""
    missing = [key for key in ("run_id", "brief", "history", "stop_decision") if not state.get(key)]
    if missing:
        raise ReportError(f"cannot write a report without: {', '.join(missing)}")
    brief = state["brief"]
    return ReportData(
        run_id=state["run_id"],
        generated_at=generated_at,
        brand=brief.brand.name,
        competitors=tuple(competitor.name for competitor in brief.competitors),
        sector=brief.sector,
        city=brief.city,
        history=tuple(state["history"]),
        stop=state["stop_decision"],
        spend_usd=state.get("spend_usd", 0.0),
        sources=tuple(state.get("sources", [])),
        findings=tuple(state.get("findings", [])),
        draft=state.get("draft", ""),
        objections=tuple(state.get("objections", [])),
        events=tuple(state.get("events", [])),
    )


def _pct(value: float) -> str:
    return f"{value:.0%}"


def _line(text: str) -> str:
    """Collapse whitespace so the text stays on one Markdown line."""
    return " ".join(text.split())


def _cell(text: str | None) -> str:
    """Make text safe for a Markdown table cell: one line, no pipes, bounded length."""
    clean = _line(text or "").replace("|", "\\|")
    if len(clean) <= MAX_CELL_CHARS:
        return clean
    return clean[: MAX_CELL_CHARS - 3] + "..."


def _source_line(index: int, url: str) -> str:
    if _SAFE_URL.fullmatch(url):
        return f"{index}. <{url}>"
    return f"{index}. (invalid URL omitted)"


def render_report(data: ReportData) -> str:
    """Render the report as Markdown."""
    summary = data.summary
    generated = data.generated_at.astimezone(UTC)
    competitors = ", ".join(data.competitors) or "its competitors"
    batch = f"batch {summary.batch_index + 1}"
    brand_cell = _cell(data.brand)

    lines = [
        f"# AI Share of Voice report: {data.brand}",
        "",
        f"Run `{data.run_id}` | generated {generated:%Y-%m-%d %H:%M} UTC | "
        f"{data.sector} in {data.city}",
        "",
        "> **Dated sample, not market share.** This report measures how often AI assistant "
        f"answers mention {data.brand} compared with {competitors}. It does not measure sales "
        "or market share, and it does not show why an assistant answers the way it does.",
        "",
        "## Key figures",
        "",
        "| Metric | Value | Source |",
        "| --- | --- | --- |",
        f"| Answers analyzed | {summary.n_answers} | sampling, {len(data.history)} batches |",
        f"| Mention share of {brand_cell} | {_pct(summary.brand_share)} | `stats_tool`, {batch} |",
        f"| Interval | {_pct(summary.ci_low)} to {_pct(summary.ci_high)} | `stats_tool`, {batch} |",
        f"| Sampling cost | {data.spend_usd:.2f} USD | model client log |",
        "",
        "## Sampling and stopping",
        "",
        f"Sampling stopped because of **{data.stop.reason.value}**: {_line(data.stop.explanation)}",
        "",
        "| Batch | Answers | Share | Interval | Cost (USD) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in data.history:
        lines.append(
            f"| {item.batch_index + 1} | {item.n_answers} | {_pct(item.brand_share)} | "
            f"{_pct(item.ci_low)} to {_pct(item.ci_high)} | {item.cumulative_spend_usd:.2f} |"
        )

    lines += ["", "## Evidence", "", "### Sources", ""]
    if data.sources:
        lines += [_source_line(index, url) for index, url in enumerate(data.sources, start=1)]
    else:
        lines.append("No sources were recorded.")

    lines += ["", "### Visual findings (hypotheses, not causes)", ""]
    if data.findings:
        lines += [f"- {_line(finding)}" for finding in data.findings]
    else:
        lines.append("No visual findings were recorded.")

    lines += ["", "## Analysis", "", data.draft.strip() or "No narrative was recorded."]

    if data.objections:
        lines += ["", "## Open objections from the Skeptic", ""]
        lines += [f"- {_line(objection)}" for objection in data.objections]

    lines += [
        "",
        "## Limits",
        "",
        f"- This is a sample taken on {generated:%Y-%m-%d}. Answers vary between runs and models.",
        "- A mention is not a recommendation, and neither is market share.",
        "- Sources show what an assistant may rely on. This report does not claim that any "
        "source caused a recommendation.",
        "- Visual findings are hypotheses.",
        "",
        "## Run log (until approval)",
        "",
        "| Time (UTC) | Node | Agent | Tool | Status | Detail |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for event in data.events:
        stamp = event.timestamp.astimezone(UTC).strftime("%H:%M:%S")
        detail = _cell(event.result or event.reason)
        lines.append(
            f"| {stamp} | {_cell(event.node)} | {_cell(event.agent) or '-'} | "
            f"{_cell(event.tool) or '-'} | {event.status.value} | {detail} |"
        )
    return "\n".join(lines) + "\n"
