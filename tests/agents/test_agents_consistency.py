from pathlib import Path

import pytest

from ai_share_of_voice.agents.specs import AGENT_SPECS, KNOWN_TOOLS
from ai_share_of_voice.schemas.brief import BrandSpec, Brief


def test_the_mention_extractor_uses_the_tool_built_in_a3() -> None:
    mention_tool = pytest.importorskip("ai_share_of_voice.tools.mention_tool")
    assert mention_tool.MentionTool.name in KNOWN_TOOLS
    assert AGENT_SPECS["mention_extractor"].tools == (mention_tool.MentionTool.name,)


def test_agents_named_in_graph_events_exist(tmp_path: Path) -> None:
    """Every agent that appears in the graph's event log must be a defined crew agent."""
    nodes = pytest.importorskip("ai_share_of_voice.nodes")
    graph_module = pytest.importorskip("ai_share_of_voice.graph")
    checkpoint = pytest.importorskip("ai_share_of_voice.checkpoint")
    event_log = pytest.importorskip("ai_share_of_voice.event_log")

    log = event_log.EventLog(tmp_path)
    deps = nodes.NodeDeps(event_log=log, should_stop=nodes.fake_should_stop(2))
    graph = graph_module.build_graph(
        nodes.make_fake_nodes(deps), checkpointer=checkpoint.memory_checkpointer()
    )
    brief = Brief(
        brand=BrandSpec(name="Acme"),
        competitors=(BrandSpec(name="Beta"), BrandSpec(name="Gamma")),
        sector="food delivery",
        city="Madrid",
        spend_cap_usd=2.0,
    )
    config = graph_module.run_config("run-agents", brief)
    graph.invoke(graph_module.initial_state(brief, "run-agents"), config)

    agents = {event.agent for event in log.read("run-agents") if event.agent}
    assert agents
    assert agents <= set(AGENT_SPECS)
