import json

import pytest
from pydantic import BaseModel

from ai_share_of_voice.agents.factory import build_budgeted_agents
from ai_share_of_voice.agents.llm_adapter import AgentReply
from ai_share_of_voice.agents.specs import AGENT_SPECS, KNOWN_TOOLS
from ai_share_of_voice.exceptions import BudgetExceeded
from ai_share_of_voice.llm_client import Pricing, SimulatedClient

crewai = pytest.importorskip("crewai")

from ai_share_of_voice.agents.crewai_bridge import (  # noqa: E402
    ModelClientLLM,
    as_crewai_tool,
    bridge_tools,
)

MESSAGES = [
    {"role": "system", "content": "You are a careful analyst."},
    {"role": "user", "content": "Say hello."},
]


class Greeting(BaseModel):
    greeting: str = "hola"


def reply_with(text: str, **options: object) -> SimulatedClient:
    def respond(prompt: str, schema: type[BaseModel]) -> BaseModel:
        return schema(text=text) if schema is AgentReply else schema()

    return SimulatedClient(responder=respond, pricing=Pricing(1.0, 1.0), **options)


# ---- the LLM ---------------------------------------------------------------------------------


def test_it_is_a_crewai_llm_that_declares_what_it_cannot_do() -> None:
    llm = ModelClientLLM(reply_with("hola"), context_window=64_000)
    assert isinstance(llm, crewai.BaseLLM)
    assert llm.supports_function_calling() is False
    assert llm.supports_stop_words() is False
    assert llm.get_context_window_size() == 64_000


def test_a_call_runs_through_the_client_and_returns_text() -> None:
    client = reply_with("Final Answer: hola")
    llm = ModelClientLLM(client)
    assert llm.call(MESSAGES) == "Final Answer: hola"
    assert llm.call("Say hello.") == "Final Answer: hola"
    assert len(client.calls) == 2 and client.spent_usd > 0


def test_the_extra_arguments_of_the_executor_are_accepted() -> None:
    llm = ModelClientLLM(reply_with("ok"))
    result = llm.call(
        MESSAGES,
        tools=[{"type": "function"}],
        callbacks=[],
        available_functions={},
        from_task=object(),
        from_agent=object(),
        something_new=True,
    )
    assert result == "ok"


def test_the_reply_is_cut_at_the_stop_words_crewai_sets() -> None:
    llm = ModelClientLLM(reply_with("Thought: x\nAction: t\nObservation: invented"))
    llm.stop = ["\nObservation:"]
    assert llm.call(MESSAGES) == "Thought: x\nAction: t"


def test_a_structured_call_returns_json_text() -> None:
    llm = ModelClientLLM(reply_with("unused"))
    assert json.loads(llm.call(MESSAGES, response_model=Greeting)) == {"greeting": "hola"}


def test_the_spending_cap_error_reaches_crewai() -> None:
    client = reply_with("ok", cap_usd=0.0000001)
    with pytest.raises(BudgetExceeded):
        ModelClientLLM(client).call(MESSAGES)
    assert client.spent_usd == 0


# ---- the tools -------------------------------------------------------------------------------


def test_a_project_tool_becomes_a_crewai_tool_with_the_same_name_and_arguments() -> None:
    bridged = as_crewai_tool(_Stub("example_tool"))
    assert bridged.name == "example_tool"
    assert set(bridged.args_schema.model_fields) == {"query"}


def test_a_tool_result_is_rendered_as_compact_text() -> None:
    bridged = as_crewai_tool(_Stub("example_tool"))
    assert bridged.run(query="hello") == "ok"


def test_tools_are_bridged_by_key() -> None:
    tools = bridge_tools({"example_tool": _Stub("example_tool")})
    assert list(tools) == ["example_tool"]


# ---- agents built with it --------------------------------------------------------------------


def test_budgeted_agents_use_the_client_and_need_no_provider_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GROQ_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    client = reply_with("Final Answer: done")
    tools = {name: as_crewai_tool(_Stub(name)) for name in KNOWN_TOOLS}
    agents = build_budgeted_agents(client=client, tools=tools)
    assert list(agents) == list(AGENT_SPECS)
    assert all(isinstance(agent.llm, ModelClientLLM) for agent in agents.values())


def test_a_real_crewai_task_runs_through_the_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The point of A2b: CrewAI runs the agent and every model call is the client's."""
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GROQ_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("CREWAI_DISABLE_TELEMETRY", "true")
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    client = reply_with("Thought: I now can give a great answer\nFinal Answer: hola")
    agent = build_budgeted_agents(client=client, tools={}, keys=["question_generator"])[
        "question_generator"
    ]
    task = crewai.Task(description="Say hello", expected_output="A greeting", agent=agent)
    result = crewai.Crew(agents=[agent], tasks=[task]).kickoff()
    assert "hola" in result.raw.lower()
    assert len(client.calls) >= 1 and client.spent_usd > 0


class _StubInput(BaseModel):
    query: str = ""


class _Stub:
    """Minimal project tool used only to build agents."""

    args_schema = _StubInput

    def __init__(self, name: str) -> None:
        self.name = name
        self.description = f"Stub of {name} used to build agents in tests."

    def run(self, **arguments: object) -> str:
        return "ok"
