import sys

import pytest
from pydantic import BaseModel

from ai_share_of_voice.agents.factory import (
    MissingToolError,
    _crewai_agent,
    build_agent,
    build_agents,
)
from ai_share_of_voice.agents.specs import AGENT_SPECS, KNOWN_TOOLS


class RecordingFactory:
    """Stands in for ``crewai.Agent``: records the arguments and returns them."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(kwargs)
        return kwargs


ALL_TOOLS = {name: f"tool:{name}" for name in KNOWN_TOOLS}
LLM = "fake-llm"


def test_build_agent_passes_the_spec_to_the_agent_class() -> None:
    spec = AGENT_SPECS["analyst"]
    factory = RecordingFactory()
    agent = build_agent(spec, llm=LLM, tools=ALL_TOOLS, agent_factory=factory)
    assert agent == {
        "role": spec.role,
        "goal": spec.goal,
        "backstory": spec.backstory,
        "tools": ["tool:stats_tool"],
        "llm": LLM,
        "allow_delegation": False,
        "max_iter": spec.max_iter,
        "verbose": False,
    }


def test_an_agent_only_receives_the_tools_of_its_spec() -> None:
    factory = RecordingFactory()
    agent = build_agent(AGENT_SPECS["skeptic"], llm=LLM, tools=ALL_TOOLS, agent_factory=factory)
    assert agent["tools"] == ["tool:read_results"]


def test_tools_keep_the_order_of_the_spec() -> None:
    agent = build_agent(
        AGENT_SPECS["source_researcher"],
        llm=LLM,
        tools=ALL_TOOLS,
        agent_factory=RecordingFactory(),
    )
    assert agent["tools"] == ["tool:search_sources", "tool:read_source", "tool:check_url"]


def test_a_missing_tool_is_reported_by_name() -> None:
    tools = {name: tool for name, tool in ALL_TOOLS.items() if name != "check_url"}
    with pytest.raises(MissingToolError, match="check_url"):
        build_agent(
            AGENT_SPECS["source_researcher"],
            llm=LLM,
            tools=tools,
            agent_factory=RecordingFactory(),
        )


def test_an_agent_without_tools_needs_none() -> None:
    agent = build_agent(
        AGENT_SPECS["question_generator"], llm=LLM, tools={}, agent_factory=RecordingFactory()
    )
    assert agent["tools"] == []


def test_build_agents_builds_the_whole_crew_by_default() -> None:
    factory = RecordingFactory()
    crew = build_agents(llm=LLM, tools=ALL_TOOLS, agent_factory=factory)
    assert list(crew) == list(AGENT_SPECS)
    assert len(factory.calls) == 7


def test_build_agents_can_build_a_subset_while_other_tools_do_not_exist_yet() -> None:
    factory = RecordingFactory()
    crew = build_agents(
        llm=LLM,
        tools={},
        keys=["question_generator"],
        agent_factory=factory,
    )
    assert list(crew) == ["question_generator"]


def test_unknown_agent_keys_are_rejected() -> None:
    with pytest.raises(KeyError, match="ghost"):
        build_agents(llm=LLM, tools=ALL_TOOLS, keys=["ghost"], agent_factory=RecordingFactory())


def test_llm_has_no_default() -> None:
    with pytest.raises(TypeError):
        build_agents(tools=ALL_TOOLS, agent_factory=RecordingFactory())  # type: ignore[call-arg]


def test_a_clear_error_when_crewai_is_not_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "crewai", None)
    with pytest.raises(RuntimeError, match="Python 3.12 or 3.13"):
        _crewai_agent()


# ---- integration with the real CrewAI (skipped when it is not installed) ----------------


def make_crewai_tool(tool_name: str) -> object:
    from crewai.tools import BaseTool

    class StubInput(BaseModel):
        query: str = ""

    class StubTool(BaseTool):
        name: str = tool_name
        description: str = f"Stub of {tool_name} used to build agents in tests."
        args_schema: type[BaseModel] = StubInput

        def _run(self, query: str = "") -> str:
            return "ok"

    return StubTool()


def test_crewai_accepts_the_agent_specs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Spike check: the real ``crewai.Agent`` builds from every spec without network calls."""
    pytest.importorskip("crewai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-used")
    tools = {name: make_crewai_tool(name) for name in KNOWN_TOOLS}
    crew = build_agents(llm="openai/gpt-4o-mini", tools=tools)
    assert list(crew) == list(AGENT_SPECS)
    for key, agent in crew.items():
        spec = AGENT_SPECS[key]
        assert agent.role == spec.role
        assert agent.allow_delegation is False
        assert agent.max_iter == spec.max_iter
        assert len(agent.tools) == len(spec.tools)
