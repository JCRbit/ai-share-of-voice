"""Builds CrewAI agents from the specs (owner: J).

``crewai`` is imported only when an agent is built, so the rest of the package works
without it. The language model and the tools are always passed in explicitly:

* ``llm`` is required. There is no default, so an agent can never silently use a provider
  key from the environment and spend money outside the spending cap.
* An agent only receives the tools named in its spec, even if more are available.
"""

from collections.abc import Callable, Iterable, Mapping
from typing import Any

from ai_share_of_voice.agents.specs import AGENT_SPECS, AgentSpec

AgentFactory = Callable[..., Any]


class MissingToolError(Exception):
    """Raised when an agent needs a tool that was not provided."""


def _crewai_agent() -> AgentFactory:
    """Return ``crewai.Agent``, with a clear message if CrewAI is not installed."""
    try:
        from crewai import Agent
    except ImportError as error:
        raise RuntimeError(
            "crewai is not installed. CrewAI supports Python 3.10 to 3.13: use Python 3.12 "
            "or 3.13 and install it there."
        ) from error
    return Agent


def build_agent(
    spec: AgentSpec,
    *,
    llm: Any,
    tools: Mapping[str, Any],
    agent_factory: AgentFactory | None = None,
) -> Any:
    """Build one agent with exactly the tools of its spec."""
    missing = [name for name in spec.tools if name not in tools]
    if missing:
        raise MissingToolError(
            f"agent '{spec.key}' needs tools that were not provided: {', '.join(missing)}"
        )
    factory = agent_factory or _crewai_agent()
    return factory(
        role=spec.role,
        goal=spec.goal,
        backstory=spec.backstory,
        tools=[tools[name] for name in spec.tools],
        llm=llm,
        allow_delegation=spec.allow_delegation,
        max_iter=spec.max_iter,
        verbose=False,
    )


def build_agents(
    *,
    llm: Any,
    tools: Mapping[str, Any],
    keys: Iterable[str] | None = None,
    agent_factory: AgentFactory | None = None,
) -> dict[str, Any]:
    """Build the agents named in ``keys`` (all seven by default), keyed by agent key."""
    selected = list(keys) if keys is not None else list(AGENT_SPECS)
    unknown = [key for key in selected if key not in AGENT_SPECS]
    if unknown:
        raise KeyError(f"unknown agents: {', '.join(unknown)}")
    return {
        key: build_agent(AGENT_SPECS[key], llm=llm, tools=tools, agent_factory=agent_factory)
        for key in selected
    }
