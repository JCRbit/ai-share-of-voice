"""Vision on our own chart, checked against the data (T6, local integration, E).

This is the guaranteed vision path of the plan: the vision model looks at the brand chart
the run drew and says which brand has the highest bar and where the client stands. Code then
checks that reading against the computed figures:

* a reading consistent with the data is kept as a finding, always labelled a hypothesis;
* a reading that contradicts the data is discarded and logged, so vision never adds a wrong
  claim to the report.

Vision is secondary: a failed call or an exhausted quota (Gemini free tier) skips the step
and the run goes on. ``BudgetExceeded`` still stops the run, like everywhere else.
"""

import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ai_share_of_voice.errors import ToolFailure
from ai_share_of_voice.event_log import EventLog
from ai_share_of_voice.nodes import NodeFn
from ai_share_of_voice.schemas.brief import normalize_brand_name
from ai_share_of_voice.schemas.events import EventStatus, GraphEvent
from ai_share_of_voice.schemas.vision import Confidence, VisionFinding
from ai_share_of_voice.state import GraphState
from ai_share_of_voice.stats.charts import BrandRates, RatesChartText, rates_chart
from ai_share_of_voice.tools.stats_tool import CHARTS_FOLDER
from ai_share_of_voice.writing import collect_facts

CHART_FILE_NAME = "vision_rates.png"
CHART_TEXT = RatesChartText(
    title="Which brands does the AI name and recommend?",
    mention="Named",
    recommendation="Recommended",
    axis="% of AI answers",
    note="Thin lines: margin of error.",
)


class ChartReading(BaseModel):
    """Schema the vision model answers with."""

    highest_brand: str = Field(min_length=1)
    client_rank: int = Field(ge=1)  # 1 = the client has the highest "Named" bar
    observation: str = Field(min_length=5, max_length=400)
    confidence: Confidence


def vision_prompt(client: str, brands: list[str]) -> str:
    return (
        "The image is a bar chart of how often an AI assistant names and recommends brands. "
        "Dark bars are 'Named', light bars are 'Recommended'.\n"
        f"Brands: {', '.join(brands)}. The client brand is {client}.\n\n"
        "Answer from the image only:\n"
        "- highest_brand: the brand with the highest 'Named' bar.\n"
        f"- client_rank: the position of {client} by 'Named' bar height, 1 = highest.\n"
        "- observation: one sentence describing what the chart shows.\n"
        "- confidence: low, medium or high.\n"
        "Never explain why the bars differ and never guess causes: describe only what is "
        "drawn."
    )


def _ranks(rows: list[BrandRates]) -> dict[str, int]:
    """Competition ranking by mention rate: ties share a rank (1, 1, 3)."""
    return {
        normalize_brand_name(row.brand): 1
        + sum(other.mention[0] > row.mention[0] for other in rows)
        for row in rows
    }


_NAME_SEPARATORS = re.compile(r"\s*(?:,|;|/|&|\band\b|\by\b)\s*", re.IGNORECASE)


def reading_agrees(answer: ChartReading, rows: list[BrandRates], client: str) -> bool:
    """Whether the reading matches the figures. A tie may be read as several leaders
    ("Glovo, Just Eat"): every name read must be a leader, and the client's rank right."""
    ranks = _ranks(rows)
    named = [
        normalize_brand_name(part)
        for part in _NAME_SEPARATORS.split(answer.highest_brand)
        if part.strip()
    ]
    leaders_ok = bool(named) and all(ranks.get(name) == 1 for name in named)
    return leaders_ok and ranks.get(normalize_brand_name(client)) == answer.client_rank


def make_vision_node(event_log: EventLog, client: Any, runs_dir: Path) -> NodeFn:
    def screenshots(state: GraphState) -> dict[str, Any]:
        def event(status: EventStatus, result: str | None = None) -> GraphEvent:
            item = GraphEvent(
                run_id=state["run_id"],
                node="screenshots",
                agent="visual_analyst",
                tool="analyze_image",
                status=status,
                result=result,
                cumulative_spend_usd=state.get("spend_usd"),
            )
            event_log.append(item)
            return item

        events = [event(EventStatus.STARTED)]
        run_dir = Path(runs_dir) / state["run_id"]
        facts = collect_facts(state, run_dir)
        rows = [BrandRates(row.brand, row.mention, row.recommendation) for row in facts.brands]
        chart = rates_chart(
            rows, facts.brand, run_dir / CHARTS_FOLDER / CHART_FILE_NAME, CHART_TEXT
        )
        question = vision_prompt(facts.brand, [row.brand for row in rows])

        try:
            answer, usage = client.complete_with_image(question, chart, ChartReading)
        except (ToolFailure, ValueError) as error:  # QuotaExhausted is a ToolFailure
            reason = getattr(error, "reason", None) or str(error)
            events.append(event(EventStatus.SUCCEEDED, f"vision skipped: {reason[:200]}"))
            return {"findings": [], "events": events}

        spend = state.get("spend_usd", 0.0) + usage.cost_usd
        if not reading_agrees(answer, rows, facts.brand):
            result = (
                f"reading discarded: says {answer.highest_brand} leads and the client is "
                f"#{answer.client_rank}, which the computed figures do not support"
            )
            events.append(event(EventStatus.SUCCEEDED, result))
            return {"findings": [], "events": events, "spend_usd": spend}

        finding = VisionFinding(
            image_path=chart,
            question="Which brand has the highest bar and where is the client?",
            observation=answer.observation,
            confidence=answer.confidence,
        )
        text = (
            f"Hypothesis from the chart (checked against the figures, "
            f"confidence {finding.confidence.value}): {finding.observation}"
        )
        events.append(event(EventStatus.SUCCEEDED, "1 finding, consistent with the figures"))
        return {"findings": [text], "events": events, "spend_usd": spend}

    return screenshots
