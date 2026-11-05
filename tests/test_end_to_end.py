"""C0: the whole simulated system, from the brief to the approval, as CI runs it.

The graph nodes are still fakes (task C2 swaps them one by one for real ones), so these tests
guard the skeleton: the order of the steps, the pause before the final action, the event log,
the spend under the cap, and the promise that none of it needs a key or the internet.
"""

import importlib.util
import json
import re
import socket
from pathlib import Path
from types import ModuleType

import pytest
from langgraph.types import Command
from pydantic import BaseModel

from ai_share_of_voice.checkpoint import memory_checkpointer
from ai_share_of_voice.event_log import EventLog
from ai_share_of_voice.graph import build_graph, initial_state, new_run_id, run_config
from ai_share_of_voice.llm_client import SimulatedClient, client_from_env
from ai_share_of_voice.nodes import NodeDeps, fake_should_stop, make_fake_nodes
from ai_share_of_voice.schemas.events import EventStatus
from ai_share_of_voice.ui.mock_run import build_demo_brief, simulate_run

ROOT = Path(__file__).parents[1]
PIPELINE = (
    "questions",
    "sampling",
    "metrics",
    "research",
    "screenshots",
    "write_draft",
    "skeptic",
    "approval_request",
)


class Pong(BaseModel):
    reply: str = "pong"


def load_script() -> ModuleType:
    """The smoke script at the repository root, which is not an importable package."""
    spec = importlib.util.spec_from_file_location("mock_run_script", ROOT / "mock_run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> tuple[dict[str, str], list[tuple[int, dict]]]:
    """Run ``mock_run.py`` inside ``tmp_path`` (it logs into ``./runs``) and parse its output.

    Returns the ``key=value`` summary lines and every JSON line with its position in the output.
    """
    monkeypatch.chdir(tmp_path)
    load_script().main()
    lines = capsys.readouterr().out.splitlines()
    keys = re.compile(r"(run_id|status|report_path)=")
    pairs = [line.split("=", 1) for line in lines if keys.match(line)]
    printed = [(i, json.loads(line)) for i, line in enumerate(lines) if line.startswith("{")]
    return dict(pairs), printed


def event_lines(printed: list[tuple[int, dict]]) -> list[dict]:
    """The graph events among the JSON lines (the approval line has no ``node``)."""
    return [item for _, item in printed if "node" in item]


# ---- CI never needs a key --------------------------------------------------------------------


def test_ci_forces_the_simulated_mode_and_runs_the_tests_and_the_smoke_script() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert re.search(r"^\s+LLM_MODE:\s*simulated\b", workflow, re.MULTILINE)
    assert re.search(r"run:\s*pytest\b", workflow)
    assert re.search(r"run:\s*python mock_run\.py\b", workflow)


@pytest.mark.parametrize(
    "env",
    [
        {},
        {"LLM_MODE": "simulated"},
        {"LLM_MODE": " Simulated "},
        {"LLM_MODE": "simulated", "LLM_PROVIDER": "groq", "LLM_API_KEY": "a-real-looking-key"},
    ],
)
def test_simulated_is_the_default_and_a_key_does_not_turn_it_live(env: dict[str, str]) -> None:
    assert isinstance(client_from_env(env), SimulatedClient)


def test_every_test_runs_in_simulated_mode_whatever_the_developer_has() -> None:
    assert isinstance(client_from_env(), SimulatedClient)


def test_the_offline_guard_blocks_the_internet_but_not_this_computer(
    no_network: list[str],
) -> None:
    with socket.socket() as probe, pytest.raises(AssertionError, match="network access"):
        probe.connect(("93.184.216.34", 80))
    with pytest.raises(AssertionError, match="name lookup attempted"):
        socket.getaddrinfo("example.com", 80)
    assert len(no_network) == 2
    no_network.clear()  # these attempts were the point of this test

    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        with socket.create_connection(server.getsockname(), timeout=2):
            pass
    assert not no_network


# ---- the whole system, brief to approval -----------------------------------------------------


def test_the_whole_system_runs_from_the_brief_to_the_approval_offline(
    tmp_path: Path, no_network: list[str]
) -> None:
    brief, run_id = build_demo_brief(), new_run_id()
    client = client_from_env({"LLM_MODE": "simulated"}, run_id=run_id)
    reply, usage = client.complete("ping", Pong)
    assert reply == Pong() and usage.cost_usd == 0 and len(client.calls) == 1

    log = EventLog(tmp_path / "runs")
    deps = NodeDeps(event_log=log, should_stop=fake_should_stop(2))
    graph = build_graph(make_fake_nodes(deps), checkpointer=memory_checkpointer())
    config = run_config(run_id, brief)
    graph.invoke(initial_state(brief, run_id), config)

    paused = graph.get_state(config)
    state = paused.values
    events = state["events"]
    assert paused.next == ("approval_gate",)
    assert events[-1].status is EventStatus.WAITING_APPROVAL
    assert "status" not in state and not [e for e in events if e.node == "finalize"]
    nodes = [e.node for e in events]
    first_seen = [nodes.index(node) for node in PIPELINE]
    assert first_seen == sorted(first_seen)
    assert not [e for e in events if e.status is EventStatus.FAILED]
    assert state["stop_decision"].stop and state["history"]
    assert 0 < state["spend_usd"] <= brief.spend_cap_usd
    assert all(e.run_id == run_id for e in events)
    assert (tmp_path / "runs" / run_id / "events.jsonl").is_file()
    assert [(e.node, e.status) for e in log.read(run_id)] == [(e.node, e.status) for e in events]

    graph.invoke(Command(resume={"approved": True, "comment": "CI approval"}), config)
    done = graph.get_state(config)
    assert done.next == () and done.values["status"] == "completed"
    assert done.values["report_path"].endswith("report.md")
    assert done.values["events"][-1].node == "finalize"
    assert client.spent_usd == 0


def test_the_ui_runner_stops_at_the_approval_and_finishes_once_it_is_given(tmp_path: Path) -> None:
    waiting = list(simulate_run(brief=build_demo_brief(), log_dir=tmp_path / "a"))
    assert waiting[-1].status is EventStatus.WAITING_APPROVAL
    assert not [e for e in waiting if e.node == "finalize"]

    approved = list(
        simulate_run(brief=build_demo_brief(), log_dir=tmp_path / "b", auto_approve=True)
    )
    assert (approved[-1].node, approved[-1].status) == ("finalize", EventStatus.SUCCEEDED)
    assert [(e.node, e.status) for e in approved[: len(waiting)]] == [
        (e.node, e.status) for e in waiting
    ]


# ---- mock_run.py ---------------------------------------------------------------------------


def test_mock_run_goes_from_the_brief_to_a_completed_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
    no_network: list[str],
) -> None:
    summary, printed = run_script(tmp_path, monkeypatch, capsys)
    events = event_lines(printed)
    assert summary["status"] == "completed" and summary["report_path"].endswith("report.md")
    assert events and {e["run_id"] for e in events} == {summary["run_id"]}
    approvals = [i for i, item in printed if item.get("type") == "approval_requested"]
    waiting = [i for i, item in printed if item.get("status") == "waiting_approval"]
    finalized = [i for i, item in printed if item.get("node") == "finalize"]
    assert len(approvals) == 1 and waiting and finalized
    assert max(waiting) < approvals[0] < min(finalized)
    logged = EventLog(tmp_path / "runs").read(summary["run_id"])
    assert [(e.node, e.status.value) for e in logged] == [(e["node"], e["status"]) for e in events]


def test_mock_run_and_the_ui_runner_follow_the_same_flow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    _, printed = run_script(tmp_path, monkeypatch, capsys)
    events = event_lines(printed)
    ui_events = simulate_run(brief=build_demo_brief(), log_dir=tmp_path / "ui", auto_approve=True)
    assert [(e["node"], e["status"]) for e in events] == [
        (e.node, e.status.value) for e in ui_events
    ]
