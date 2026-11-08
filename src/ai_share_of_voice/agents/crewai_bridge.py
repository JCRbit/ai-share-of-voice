"""The CrewAI side of the budgeted agents (owner: An, tasks A2b and A3b).

This module imports ``crewai``, so import it only where agents are built; ``agents/factory.py``
does it lazily. The logic that does not need CrewAI is in ``agents/llm_adapter.py``.

* ``ModelClientLLM`` is a ``BaseLLM`` whose every call goes through ``ModelClient``: the run and
  person spending caps, the token and cost log. Give it to ``crewai.Agent`` as ``llm``; never
  pass a model name string, because CrewAI would then call the provider with its own client.
* ``as_crewai_tool`` wraps a project tool (``name``, ``description``, ``args_schema``, ``run``)
  as the ``crewai.tools.BaseTool`` that CrewAI accepts, and shows the agent a compact text.

Written against the custom LLM guide of CrewAI 1.15 (``BaseLLM(model, temperature)``, ``call``,
``supports_function_calling``, ``supports_stop_words``, ``get_context_window_size``). The
executor also passes ``from_task``, ``from_agent`` and ``response_model``, so ``call`` takes them.
"""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from crewai import BaseLLM
from pydantic import BaseModel

from ai_share_of_voice.agents.llm_adapter import (
    DEFAULT_CONTEXT_WINDOW,
    BudgetedCaller,
    Message,
)
from ai_share_of_voice.schemas.llm import ModelClient

Renderer = Callable[[Any], str]


class ModelClientLLM(BaseLLM):
    """A CrewAI LLM that runs every call through a ``ModelClient``."""

    def __init__(
        self,
        client: ModelClient,
        *,
        model: str | None = None,
        temperature: float | None = None,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
    ) -> None:
        name = model or getattr(client, "model", "model-client")
        super().__init__(model=name, temperature=temperature)
        self._caller = BudgetedCaller(client)
        self._context_window = context_window

    def call(
        self,
        messages: str | Sequence[Message],
        tools: list[dict] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Any = None,
        from_agent: Any = None,
        response_model: type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> str:
        """Run one agent turn. ``tools`` is ignored: the model does not call functions."""
        if response_model is not None:
            return self._caller.structured(messages, response_model).model_dump_json()
        return self._caller.text(messages, getattr(self, "stop", None))

    def supports_function_calling(self) -> bool:
        """False, so CrewAI uses its text format and runs the tools itself."""
        return False

    def supports_stop_words(self) -> bool:
        """False: the reply is cut at the stop words here, because the client takes none."""
        return False

    def get_context_window_size(self) -> int:
        """Context window used to decide when CrewAI must shorten the conversation."""
        return self._context_window


def render_result(result: Any) -> str:
    """Compact text for an agent: the analysis of an extraction, else the result as JSON."""
    inner = getattr(result, "analysis", result)
    if isinstance(inner, BaseModel):
        return inner.model_dump_json()
    return str(inner)


def as_crewai_tool(tool: Any, *, render: Renderer = render_result) -> Any:
    """Wrap a project tool as a ``crewai.tools.BaseTool``.

    A failure of the tool is not caught here: ``BudgetExceeded`` and ``ToolFailure`` reach
    CrewAI, which decides how to report them to the agent.
    """
    from crewai.tools import BaseTool

    class BridgedTool(BaseTool):
        name: str = tool.name
        description: str = tool.description
        args_schema: type[BaseModel] = tool.args_schema

        def _run(self, **arguments: Any) -> str:
            return render(tool.run(**arguments))

    return BridgedTool()


def bridge_tools(tools: Mapping[str, Any], *, render: Renderer = render_result) -> dict[str, Any]:
    """Wrap every tool of ``tools`` for CrewAI, keeping the keys."""
    return {key: as_crewai_tool(tool, render=render) for key, tool in tools.items()}
