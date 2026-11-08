from datetime import UTC, datetime
from pathlib import Path

import pytest
from langgraph.types import Command
from pydantic import BaseModel

from ai_share_of_voice.checkpoint import memory_checkpointer
from ai_share_of_voice.event_log import EventLog
from ai_share_of_voice.exceptions import BudgetExceeded, ConfigurationError
from ai_share_of_voice.graph import build_graph, initial_state, new_run_id, run_config
from ai_share_of_voice.guard import guard_nodes
from ai_share_of_voice.llm_client import Pricing, SimulatedClient
from ai_share_of_voice.mcp_server.sources import UnsafeUrlError
from ai_share_of_voice.nodes import NodeDeps, fake_should_stop
from ai_share_of_voice.real_nodes import RealDeps, check_stages, make_real_nodes
from ai_share_of_voice.run_store import RunStore
from ai_share_of_voice.schemas.llm import Usage
from ai_share_of_voice.schemas.sources import UrlCheck
from ai_share_of_voice.schemas.stopping import StopReason
from ai_share_of_voice.stats.metrics import load_answers
from ai_share_of_voice.vision import SimulatedVisionModel
from helpers_real_run import (
    DEAD_LINK,
    LIVE_LINK,
    PNG,
    QUESTIONS,
    FakeSources,
    make_brief,
    respond,
    write_data,
)

# ---- fakes -----------------------------------------------------------------------------------


def make_client(**options) -> SimulatedClient:
    options.setdefault("cap_usd", 5.0)
    return SimulatedClient(responder=respond, pricing=Pricing(1.0, 1.0), **options)


class FailingClient:
    """Delegates to ``inner`` and raises a spending-cap error on the call number ``fail_on``."""

    def __init__(self, inner: SimulatedClient, fail_on: int) -> None:
        self.inner, self.fail_on, self.count = inner, fail_on, 0

    def complete(self, prompt: str, schema: type[BaseModel]) -> tuple[BaseModel, Usage]:
        self.count += 1
        if self.count == self.fail_on:
            raise BudgetExceeded(spent_usd=self.inner.spent_usd, cap_usd=5.0)
        return self.inner.complete(prompt, schema)


class CostlyVision(SimulatedVisionModel):
    """A vision model that costs money and can run out of budget on the nth image."""

    def __init__(self, cost: float = 0.01, fail_on: int | None = None) -> None:
        super().__init__()
        self.cost, self.fail_on, self.count = cost, fail_on, 0

    def complete_with_image(self, prompt, image, schema):
        self.count += 1
        if self.count == self.fail_on:
            raise BudgetExceeded(spent_usd=0.0, cap_usd=1.0)
        answer, usage = super().complete_with_image(prompt, image, schema)
        return answer, usage.model_copy(update={"cost_usd": self.cost})


# ---- helpers ---------------------------------------------------------------------------------


class Rig:
    """A run in a temporary folder: the graph with the real nodes of some stages."""

    def __init__(self, tmp_path: Path, stages, *, client=None, brief=None, **real_options) -> None:
        self.runs, self.data = tmp_path / "runs", tmp_path / "data"
        write_data(self.data, screenshots="vision" in stages)
        self.client = client or make_client()
        self.store = RunStore(self.runs)
        self.log = EventLog(self.runs)
        self.brief = brief or make_brief()
        self.run_id = new_run_id()
        real = RealDeps(
            client=self.client,
            store=self.store,
            runs_dir=self.runs,
            data_dir=self.data,
            **real_options,
        )
        deps = NodeDeps(event_log=self.log, should_stop=fake_should_stop(2))
        self.nodes = make_real_nodes(deps, real, stages)
        guarded = guard_nodes(self.nodes, self.log)
        self.graph = build_graph(guarded, checkpointer=memory_checkpointer())
        self.config = run_config(self.run_id, self.brief)

    def run(self) -> dict:
        self.graph.invoke(initial_state(self.brief, self.run_id), self.config)
        return self.state()

    def state(self) -> dict:
        return self.graph.get_state(self.config).values

    def approve(self) -> dict:
        self.graph.invoke(Command(resume={"approved": True, "comment": "ok"}), self.config)
        return self.state()


# ---- the stages ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "stages", [(), ("client",), ("client", "mcp"), ("client", "mcp", "vision")]
)
def test_stages_are_accepted_in_order(stages: tuple[str, ...]) -> None:
    assert check_stages(stages) == stages


@pytest.mark.parametrize(
    "stages", [("mcp",), ("vision",), ("client", "vision"), ("client", "client"), ("bogus",)]
)
def test_a_gap_a_repeat_or_an_unknown_stage_is_refused(stages: tuple[str, ...]) -> None:
    with pytest.raises(ConfigurationError, match="stages"):
        check_stages(stages)


def test_a_stage_needs_what_it_uses(tmp_path: Path) -> None:
    deps = NodeDeps(event_log=EventLog(tmp_path), should_stop=fake_should_stop(2))
    real = RealDeps(client=make_client(), store=RunStore(tmp_path))
    with pytest.raises(ConfigurationError, match="sources service"):
        make_real_nodes(deps, real, ("client", "mcp"))
    with_sources = RealDeps(client=make_client(), store=RunStore(tmp_path), sources=FakeSources())
    with pytest.raises(ConfigurationError, match="vision model"):
        make_real_nodes(deps, with_sources, ("client", "mcp", "vision"))


# ---- client: questions, sampling, metrics ----------------------------------------------------


def test_the_client_stage_runs_the_graph_to_the_approval_with_real_data(tmp_path: Path) -> None:
    rig = Rig(tmp_path, ("client",))
    state = rig.run()
    assert rig.graph.get_state(rig.config).next == ("approval_gate",)
    assert not state.get("partial_reason")

    store, folder = rig.store, rig.runs / rig.run_id
    assert store.load_brief(rig.run_id) == rig.brief
    assert len(store.answer_ids(rig.run_id)) == 2 * QUESTIONS
    assert len(store.load_analyses(rig.run_id)) == 2 * QUESTIONS
    answers = load_answers(folder / "answers.csv")
    assert len(answers) == 2 * QUESTIONS * 3 and (folder / "metrics.csv").is_file()

    assert state["n_answers"] == 2 * QUESTIONS and state["n_brand_mentions"] == QUESTIONS
    history = state["history"]
    assert [h.batch_index for h in history] == [0, 1]
    assert [h.n_answers for h in history] == [QUESTIONS, 2 * QUESTIONS]
    last = history[-1]
    assert last.brand_share == pytest.approx(0.5) and last.ci_low < 0.5 < last.ci_high
    assert last.n_questions == QUESTIONS
    assert state["stop_decision"].stop and state["stop_decision"].reason is StopReason.MAX_BATCHES


def test_the_spend_in_the_state_is_what_the_client_counted(tmp_path: Path) -> None:
    rig = Rig(tmp_path, ("client",))
    state = rig.run()
    assert state["spend_usd"] > 0
    assert state["spend_usd"] == pytest.approx(rig.client.spent_usd)
    spends = [e.cumulative_spend_usd for e in state["events"] if e.node == "sampling"]
    spends = [value for value in spends if value]
    assert spends == sorted(spends) and len(spends) >= 2
    assert state["history"][-1].cumulative_spend_usd == pytest.approx(state["spend_usd"])


def test_the_events_in_the_state_are_the_events_in_the_log(tmp_path: Path) -> None:
    rig = Rig(tmp_path, ("client",))
    state = rig.run()
    assert [(e.node, e.status) for e in rig.log.read(rig.run_id)] == [
        (e.node, e.status) for e in state["events"]
    ]
    nodes = [e.node for e in state["events"]]
    assert nodes.count("sampling") == nodes.count("metrics") == 4  # started and succeeded, twice


def test_running_a_batch_again_does_not_pay_for_it_twice(tmp_path: Path) -> None:
    rig = Rig(tmp_path, ("client",))
    state = {"run_id": rig.run_id, "brief": rig.brief}
    state |= rig.nodes["questions"](state)
    first = rig.nodes["sampling"](state)
    calls = len(rig.client.calls)
    again = rig.nodes["sampling"](state)
    assert len(rig.client.calls) == calls
    assert again["n_answers"] == first["n_answers"] == QUESTIONS
    assert again["spend_usd"] == state.get("spend_usd", 0.0)
    assert "already stored" in again["events"][-1].result


def test_a_cap_that_runs_out_mid_batch_closes_the_run_with_the_real_spend(tmp_path: Path) -> None:
    inner = make_client()
    rig = Rig(tmp_path, ("client",), client=FailingClient(inner, fail_on=6))
    state = rig.run()
    assert state["status"] == "partial" and state["failure_kind"] == "budget_exhausted"
    failed = [e for e in state["events"] if e.status.value == "failed" and e.node == "sampling"]
    assert len(failed) == 1
    assert inner.spent_usd > 0
    assert failed[0].cumulative_spend_usd == pytest.approx(inner.spent_usd)
    assert state["spend_usd"] == pytest.approx(inner.spent_usd)
    assert not [e for e in state["events"] if e.node == "finalize"]
    assert 0 < len(rig.store.answer_ids(rig.run_id)) < QUESTIONS


def test_sampling_failure_keeps_spend_recorded_before_log_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig = Rig(tmp_path, ("client",), client=make_client(log_path=tmp_path / "calls.jsonl"))
    original_open = Path.open

    def fail_call_log(path: Path, *args, **kwargs):
        if path.name == "calls.jsonl":
            raise OSError("simulated disk failure")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_call_log)

    state = rig.run()

    assert state["status"] == "partial"
    assert state["spend_usd"] == pytest.approx(rig.client.spent_usd)
    assert state["spend_usd"] > 0
    failed = [e for e in state["events"] if e.status.value == "failed"]
    assert len(failed) == 2
    assert failed[0].cumulative_spend_usd == pytest.approx(rig.client.spent_usd)


def test_a_spend_cap_that_leaves_nothing_for_sampling_is_refused_up_front(tmp_path: Path) -> None:
    rig = Rig(tmp_path, ("client",), brief=make_brief(cap=0.1))
    state = rig.run()
    assert state["status"] == "partial" and "no budget" in state["partial_reason"]
    assert not rig.client.calls and rig.store.answer_ids(rig.run_id) == []


# ---- mcp: research ---------------------------------------------------------------------------


def test_research_verifies_the_most_cited_sources(tmp_path: Path) -> None:
    sources = FakeSources()
    rig = Rig(tmp_path, ("client", "mcp"), sources=sources)
    state = rig.run()
    assert sources.checked == [LIVE_LINK, DEAD_LINK]
    assert state["sources"] == [LIVE_LINK]
    research = [e for e in state["events"] if e.node == "research"][-1]
    assert "1 of the 2 most cited sources verified" in research.result
    assert "1 not found, 0 not checked" in research.result


def test_a_source_that_robots_txt_forbids_is_not_checked_and_never_counts_as_verified(
    tmp_path: Path,
) -> None:
    class Forbidden(FakeSources):
        def check_url(self, url: str) -> UrlCheck:
            if url != DEAD_LINK:
                return super().check_url(url)
            return UrlCheck(
                url=url, exists=False, allowed_by_robots=False, checked_at=datetime.now(UTC)
            )

    rig = Rig(tmp_path, ("client", "mcp"), sources=Forbidden())
    state = rig.run()
    assert state["sources"] == [LIVE_LINK]
    result = [e for e in state["events"] if e.node == "research"][-1].result
    assert "0 not found, 1 not checked" in result


def test_research_checks_no_more_than_the_limit_and_survives_unsafe_urls(tmp_path: Path) -> None:
    sources = FakeSources()
    rig = Rig(tmp_path, ("client", "mcp"), sources=sources, max_sources=1)
    assert rig.run()["sources"] == [LIVE_LINK] and sources.checked == [LIVE_LINK]

    class Unsafe(FakeSources):
        def check_url(self, url: str) -> UrlCheck:
            raise UnsafeUrlError("private address")

    other = Rig(tmp_path / "again", ("client", "mcp"), sources=Unsafe())
    state = other.run()
    assert state["sources"] == [] and not state.get("partial_reason")
    assert "2 not checked" in [e for e in state["events"] if e.node == "research"][-1].result


# ---- vision: screenshots ---------------------------------------------------------------------


def test_vision_reads_our_chart_and_the_screenshots_of_tracked_brands(tmp_path: Path) -> None:
    rig = Rig(tmp_path, ("client", "mcp", "vision"), sources=FakeSources(), vision=CostlyVision())
    state = rig.run()
    findings = state["findings"]
    assert len(findings) == 4  # the chart, and three of the four tracked screenshots that exist
    assert findings[0].startswith("our chart of the mention rate:")
    assert [f.split(" page ")[0] for f in findings[1:]] == ["Acme Food", "Beta Eats", "Gamma Bites"]
    assert all("a hypothesis, not a cause" in f for f in findings)
    assert (rig.runs / rig.run_id / "charts" / "mention_rate.png").read_bytes().startswith(PNG[:8])
    assert not any("Some Other App" in f or "gone.png" in f for f in findings)


def test_the_screenshot_limit_and_the_vision_spend(tmp_path: Path) -> None:
    sources = FakeSources()
    rig = Rig(
        tmp_path,
        ("client", "mcp", "vision"),
        sources=sources,
        vision=CostlyVision(0.01),
        max_screenshots=1,
    )
    state = rig.run()
    assert len(state["findings"]) == 2 and "Acme Food page" in state["findings"][1]
    assert state["spend_usd"] == pytest.approx(rig.client.spent_usd + 2 * 0.01)


def test_a_vision_failure_closes_the_run_and_keeps_the_spend_so_far(tmp_path: Path) -> None:
    rig = Rig(
        tmp_path,
        ("client", "mcp", "vision"),
        sources=FakeSources(),
        vision=CostlyVision(0.01, fail_on=2),
    )
    state = rig.run()
    assert state["status"] == "partial" and state["failure_kind"] == "budget_exhausted"
    failed = [e for e in state["events"] if e.status.value == "failed"][0]
    assert failed.node == "screenshots"
    assert failed.cumulative_spend_usd == pytest.approx(rig.client.spent_usd + 0.01)
    assert state["spend_usd"] == pytest.approx(rig.client.spent_usd + 0.01)


def test_an_unusable_image_is_a_tool_failure_not_a_crash(tmp_path: Path) -> None:
    rig = Rig(tmp_path, ("client", "mcp", "vision"), sources=FakeSources(), vision=CostlyVision())
    (rig.data / "screenshots" / "acme-home.png").write_bytes(b"not an image at all")
    state = rig.run()
    assert state["status"] == "partial"
    assert "not a JPEG, PNG or WebP" in state["partial_reason"]


# ---- all three stages, to the approval and past it -------------------------------------------


def test_all_three_stages_reach_the_approval_and_complete_once_approved(tmp_path: Path) -> None:
    rig = Rig(tmp_path, ("client", "mcp", "vision"), sources=FakeSources(), vision=CostlyVision())
    state = rig.run()
    assert rig.graph.get_state(rig.config).next == ("approval_gate",)
    order = [e.node for e in state["events"]]
    steps = ("questions", "sampling", "metrics", "research", "screenshots")
    first = [order.index(n) for n in steps]
    assert first == sorted(first)
    assert state["sources"] == [LIVE_LINK] and len(state["findings"]) == 4
    assert not [e for e in state["events"] if e.node == "finalize"]
    done = rig.approve()
    assert done["status"] == "completed" and done["events"][-1].node == "finalize"
    assert done["spend_usd"] == pytest.approx(rig.client.spent_usd + 4 * 0.01)
