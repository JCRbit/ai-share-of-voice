"""Real graph nodes, swapped in one stage at a time (owner: An, task C2).

``make_real_nodes`` starts from the fake nodes of ``nodes.py`` and replaces the ones of each
stage with a real version that has the same signature, so ``graph.py`` does not change:

    client  questions, sampling, metrics   the measured model, the mention analysis, the real
                                           statistics (D1) and the real stopping rule (D2)
    mcp     research                       the URLs the answers cite, checked with the sources
                                           service of the MCP server (I4)
    vision  screenshots                    our own chart and the saved screenshots, read by the
                                           vision model (I5)

Stages are added in this order and none can be skipped, because each one needs the data of the
one before: the sources are the ones cited in the analyzed answers, and the chart is drawn from
the metrics. The writer is still simulated, so the nodes after the three stages stay fake on
purpose: the real Skeptic (``make_skeptic_node``) and finalize (``make_finalize_node``) exist, but
swapping them in now would review and publish a draft that nobody wrote.

What the nodes keep, and where:

* Heavy data lives in the run folder (``run_store.py``): the brief (saved by the first node),
  raw answers, analyzed answers, and the ``answers.csv`` and ``metrics.csv`` tables. The state
  only holds counters, the stopping history, the verified sources and the findings.
* A node can be run again after an interruption without paying twice: an answer or an analysis
  already stored is reused, and the counters are read from the store, not added up.
* Every model call goes through the one ``ModelClient`` of the run, so the run cap and the person
  cap apply to the answers, the analysis and the images together.

If something fails after money was spent, the node closes the run in a partial state itself and
reports the spend up to that point; the generic guard would only know the spend of the last step.
"""

import csv
import logging
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

from langgraph.errors import GraphBubbleUp
from pydantic import ValidationError

from ai_share_of_voice.automation.files import write_text_atomic
from ai_share_of_voice.errors import classify
from ai_share_of_voice.exceptions import ConfigurationError
from ai_share_of_voice.mcp_server.sources import SourceError
from ai_share_of_voice.nodes import FakeBehavior, NodeDeps, NodeFn, emit_event, make_fake_nodes
from ai_share_of_voice.run_store import RunStore
from ai_share_of_voice.sampling import ask, load_panel
from ai_share_of_voice.schemas.analysis import AnalyzedResponse, RawAnswer
from ai_share_of_voice.schemas.brief import Brief
from ai_share_of_voice.schemas.events import EventStatus, GraphEvent
from ai_share_of_voice.schemas.llm import ModelClient, VisionModel
from ai_share_of_voice.schemas.sources import UrlCheck
from ai_share_of_voice.schemas.stopping import BatchSummary, StopConfig
from ai_share_of_voice.schemas.vision import VisionRequest
from ai_share_of_voice.state import GraphState
from ai_share_of_voice.stats.charts import brand_comparison_chart
from ai_share_of_voice.stats.metrics import brand_metrics, responses_to_frame
from ai_share_of_voice.stats.stopping import should_stop
from ai_share_of_voice.tools.mention_tool import extract_stored_mentions
from ai_share_of_voice.vision import analyze_image

logger = logging.getLogger(__name__)

STAGES = ("client", "mcp", "vision")
ANSWERS_TABLE = "answers.csv"
METRICS_TABLE = "metrics.csv"
CHART_NAME = "charts/mention_rate.png"
CHART_QUESTION = (
    "Which brand has the highest bar and which the lowest? Do the intervals of the top two overlap?"
)
SCREENSHOT_QUESTION = (
    "This is a captured page of {brand}. Describe what a customer sees first: offers, ordering "
    "and labels."
)


class SourceChecker(Protocol):
    """What the research node needs from the sources service of the MCP server."""

    def check_url(self, url: str) -> UrlCheck:
        """Does ``url`` answer with a success status, and may we read it?"""
        ...


@dataclass(frozen=True)
class RealDeps:
    """What the real nodes need besides the event log and the stopping rule of ``NodeDeps``."""

    client: ModelClient
    store: RunStore
    runs_dir: Path = Path("runs")
    data_dir: Path = Path("data")
    sources: SourceChecker | None = None
    vision: VisionModel | None = None
    max_sources: int = 5
    max_screenshots: int = 3


def check_stages(stages: Sequence[str]) -> tuple[str, ...]:
    """The stages as a tuple, refusing unknown names, repeats and any gap in the order."""
    chosen = tuple(stages)
    unknown = [stage for stage in chosen if stage not in STAGES]
    if unknown:
        raise ConfigurationError(f"unknown stages {unknown}; the stages are {list(STAGES)}")
    if chosen != STAGES[: len(chosen)]:
        raise ConfigurationError(
            f"stages must be added in order, without gaps: {list(STAGES)}; got {list(chosen)}"
        )
    return chosen


def brand_names(brief: Brief) -> list[str]:
    """The tracked brands: the client first, then the competitors."""
    return [brief.brand.name, *(item.name for item in brief.competitors)]


def stop_config(brief: Brief) -> StopConfig:
    """Limits of the sampling loop for ``brief``."""
    return StopConfig(
        target_half_width=brief.target_half_width,
        min_batches=min(2, brief.max_batches),
        max_batches=brief.max_batches,
        max_spend_usd=brief.spend_cap_usd,
    )


def _mentions_brand(analysis: AnalyzedResponse, brand: str) -> bool:
    return any(m.mentioned and m.brand.casefold() == brand.casefold() for m in analysis.mentions)


def _metrics_table(store: RunStore, run_id: str, brief: Brief) -> tuple[Any, Any]:
    """The answers table and the metrics of every brand, computed from the stored analyses."""
    frame = responses_to_frame(store.load_analyses(run_id), brand_names(brief))
    return frame, brand_metrics(frame)


def _client_spend(client: object) -> float | None:
    """Read a client's cumulative spend when it exposes a valid counter."""
    try:
        value = getattr(client, "spent_usd", None)
    except Exception:
        return None

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        return None
    return float(value)


def _reconciled_spend(
    state_spend: float,
    local_spend: float,
    client_start: float | None,
    client: object,
) -> float:
    """Recover billed calls that failed after the client recorded their cost."""
    current = _client_spend(client)
    if client_start is None or current is None:
        return local_spend

    client_delta = max(0.0, current - client_start)
    return max(local_spend, state_spend + client_delta)


def _failure(
    deps: NodeDeps,
    state: GraphState,
    node: str,
    error: Exception,
    spend: float,
    events: list[GraphEvent],
    *,
    tool: str,
) -> dict[str, Any]:
    """Close the run in a partial state, reporting the spend up to the failure."""
    failure = classify(error)
    logger.warning("node %s failed (%s): %s", node, failure.kind.value, failure.reason)
    event = emit_event(
        deps,
        state,
        node,
        EventStatus.FAILED,
        tool=tool,
        result=failure.kind.value,
        reason=failure.reason,
        spend=spend,
    )
    return {
        "partial_reason": failure.reason,
        "failure_kind": failure.kind.value,
        "spend_usd": spend,
        "events": [*events, event],
    }


def _client_nodes(deps: NodeDeps, real: RealDeps) -> dict[str, NodeFn]:
    store, client = real.store, real.client

    def questions(state: GraphState) -> dict[str, Any]:
        tool = "question_panel"
        events = [emit_event(deps, state, "questions", EventStatus.STARTED, tool=tool)]
        brief = state["brief"]
        if stop_config(brief).sampling_budget_usd <= 0:
            raise ConfigurationError(
                f"the spend cap of {brief.spend_cap_usd:.2f} USD leaves no budget for sampling "
                "after the amount reserved for the agents that run later"
            )
        store.save_brief(state["run_id"], brief)  # the mention extraction reads it from here
        panel = load_panel(brief.language, real.data_dir)
        result = f"{len(panel.questions)} questions from the fixed {brief.language} panel"
        events.append(
            emit_event(deps, state, "questions", EventStatus.SUCCEEDED, tool=tool, result=result)
        )
        return {"panel": panel, "events": events}

    def sampling(state: GraphState) -> dict[str, Any]:
        tool = "llm_client"
        events = [emit_event(deps, state, "sampling", EventStatus.STARTED, tool=tool)]
        run_id, brief = state["run_id"], state["brief"]
        batch = len(state.get("history", []))
        state_spend = state.get("spend_usd", 0.0)
        spend = state_spend
        client_start = _client_spend(client)
        reused = 0
        try:
            asked_before = set(store.answer_ids(run_id))
            analyzed_before = {item.response_id for item in store.load_analyses(run_id)}
            for question in state["panel"].questions:
                response_id = f"{question.id}-b{batch}"
                if response_id in asked_before:
                    reused += 1
                else:
                    text, usage = ask(client, question)
                    spend += usage.cost_usd
                    answer = RawAnswer(
                        response_id=response_id,
                        question_id=question.id,
                        batch_index=batch,
                        model=usage.model,
                        text=text,
                    )
                    store.append_answer(run_id, answer)
                if response_id not in analyzed_before:
                    extraction = extract_stored_mentions(run_id, response_id, store, client)
                    if extraction.usage is not None:
                        spend += extraction.usage.cost_usd
                    store.append_analysis(run_id, extraction.analysis)
            analyses = store.load_analyses(run_id)
        except GraphBubbleUp:
            raise
        except Exception as error:
            spend = _reconciled_spend(state_spend, spend, client_start, client)
            return _failure(deps, state, "sampling", error, spend, events, tool=tool)
        with_brand = sum(1 for item in analyses if _mentions_brand(item, brief.brand.name))
        count = len(state["panel"].questions)
        note = f" ({reused} already stored)" if reused else ""
        result = f"batch {batch}: {count} answers asked and analyzed{note}"
        events.append(
            emit_event(
                deps,
                state,
                "sampling",
                EventStatus.SUCCEEDED,
                tool=tool,
                result=result,
                spend=spend,
            )
        )
        return {
            "n_answers": len(analyses),
            "n_brand_mentions": with_brand,
            "spend_usd": spend,
            "events": events,
        }

    def metrics(state: GraphState) -> dict[str, Any]:
        agent, tool = "analyst", "stats_tool"
        events = [emit_event(deps, state, "metrics", EventStatus.STARTED, agent=agent, tool=tool)]
        run_id, brief = state["run_id"], state["brief"]
        frame, table = _metrics_table(store, run_id, brief)
        folder = store.run_dir(run_id)
        write_text_atomic(folder / ANSWERS_TABLE, frame.to_csv(index=False))
        write_text_atomic(folder / METRICS_TABLE, table.to_csv(index=False))
        row = table[(table["brand"] == brief.brand.name) & (table["metric"] == "mention_rate")]
        if row.empty or row["estimate"].isna().any():
            raise ConfigurationError(f"no mention rate could be computed for {brief.brand.name}")
        head = row.iloc[0]
        summary = BatchSummary(
            batch_index=len(state.get("history", [])),
            n_answers=int(frame["response_id"].nunique()),
            brand_share=float(head["estimate"]),
            ci_low=float(head["ci_low"]),
            ci_high=float(head["ci_high"]),
            cumulative_spend_usd=state.get("spend_usd", 0.0),
            n_questions=int(head["n_questions"]),
        )
        history = [*state.get("history", []), summary]
        decision = deps.should_stop(history, stop_config(brief))
        result = (
            f"mention rate {summary.brand_share:.0%} "
            f"[{summary.ci_low:.0%}, {summary.ci_high:.0%}], half-width {summary.half_width:.3f}"
        )
        events.append(
            emit_event(
                deps,
                state,
                "metrics",
                EventStatus.SUCCEEDED,
                agent=agent,
                tool=tool,
                result=result,
                reason=decision.explanation,
            )
        )
        return {"history": [summary], "stop_decision": decision, "events": events}

    return {"questions": questions, "sampling": sampling, "metrics": metrics}


def _research_node(deps: NodeDeps, real: RealDeps) -> NodeFn:
    sources, store = real.sources, real.store

    def research(state: GraphState) -> dict[str, Any]:
        agent, tool = "source_researcher", "mcp_sources"
        events = [emit_event(deps, state, "research", EventStatus.STARTED, agent=agent, tool=tool)]
        cited = Counter(
            str(url) for item in store.load_analyses(state["run_id"]) for url in item.sources
        )
        chosen = [url for url, _ in cited.most_common(real.max_sources)]
        verified: list[str] = []
        missing = unchecked = 0
        for url in chosen:
            try:
                check = sources.check_url(url)
            except (SourceError, ValidationError):
                unchecked += 1
                continue
            if check.exists:
                verified.append(url)
            elif not check.allowed_by_robots:
                unchecked += 1
            else:
                missing += 1
        if not cited:
            result = "the answers cite no sources"
        else:
            result = (
                f"{len(verified)} of the {len(chosen)} most cited sources verified "
                f"({len(cited)} cited in all); {missing} not found, {unchecked} not checked"
            )
        events.append(
            emit_event(
                deps,
                state,
                "research",
                EventStatus.SUCCEEDED,
                agent=agent,
                tool=tool,
                result=result,
            )
        )
        return {"sources": verified, "events": events}

    return research


def _screenshots(data_dir: Path, brief: Brief, limit: int) -> list[tuple[Path, str]]:
    """Saved screenshots of tracked brands that exist on disk: (file, brand), client first."""
    folder = data_dir / "screenshots"
    index = folder / "index.csv"
    if not index.is_file() or limit <= 0:
        return []
    with index.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    order = {name.casefold(): position for position, name in enumerate(brand_names(brief))}
    tracked = []
    for row in rows:
        brand, file = row.get("competitor", "").strip(), row.get("file", "").strip()
        if file and brand.casefold() in order:
            tracked.append((order[brand.casefold()], file, brand))
    found = [(folder / file, brand) for _, file, brand in sorted(tracked)]
    return [(path, brand) for path, brand in found if path.is_file()][:limit]


def _vision_node(deps: NodeDeps, real: RealDeps) -> NodeFn:
    store, model = real.store, real.vision

    def screenshots(state: GraphState) -> dict[str, Any]:
        agent, tool = "visual_analyst", "analyze_image"
        events = [
            emit_event(deps, state, "screenshots", EventStatus.STARTED, agent=agent, tool=tool)
        ]
        run_id, brief = state["run_id"], state["brief"]
        state_spend = state.get("spend_usd", 0.0)
        spend = state_spend
        client_start = _client_spend(model)
        findings: list[str] = []
        try:
            _, table = _metrics_table(store, run_id, brief)
            chart = brand_comparison_chart(
                table, "mention_rate", store.run_dir(run_id) / CHART_NAME
            )
            roots = [real.runs_dir, real.data_dir / "screenshots"]
            jobs = [("our chart of the mention rate", chart, CHART_QUESTION)]
            jobs += [
                (f"{brand} page {path.name}", path, SCREENSHOT_QUESTION.format(brand=brand))
                for path, brand in _screenshots(real.data_dir, brief, real.max_screenshots)
            ]
            for label, path, question in jobs:
                result = analyze_image(
                    VisionRequest(path=path, question=question), model, allowed_roots=roots
                )
                spend += result.usage.cost_usd
                finding = result.finding
                findings.append(
                    f"{label}: {finding.observation} "
                    f"(confidence {finding.confidence.value}; a hypothesis, not a cause)"
                )
        except GraphBubbleUp:
            raise
        except Exception as error:
            spend = _reconciled_spend(state_spend, spend, client_start, model)
            return _failure(deps, state, "screenshots", error, spend, events, tool=tool)
        result_text = f"{len(findings)} findings: our chart and {len(findings) - 1} screenshots"
        events.append(
            emit_event(
                deps,
                state,
                "screenshots",
                EventStatus.SUCCEEDED,
                agent=agent,
                tool=tool,
                result=result_text,
                spend=spend,
            )
        )
        return {"findings": findings, "spend_usd": spend, "events": events}

    return screenshots


def make_real_nodes(
    deps: NodeDeps,
    real: RealDeps,
    stages: Sequence[str],
    behavior: FakeBehavior | None = None,
) -> dict[str, NodeFn]:
    """Fake nodes with the real ones of ``stages`` swapped in (see the module docstring)."""
    chosen = check_stages(stages)
    if "mcp" in chosen and real.sources is None:
        raise ConfigurationError("the mcp stage needs a sources service (RealDeps.sources)")
    if "vision" in chosen and real.vision is None:
        raise ConfigurationError(
            "the vision stage needs a vision model: set VISION_MODEL and its prices, or pass one"
        )
    nodes = make_fake_nodes(deps, behavior)
    if "client" in chosen:
        client_deps = replace(deps, should_stop=should_stop)
        nodes.update(_client_nodes(client_deps, real))
    if "mcp" in chosen:
        nodes["research"] = _research_node(deps, real)
    if "vision" in chosen:
        nodes["screenshots"] = _vision_node(deps, real)
    return nodes
