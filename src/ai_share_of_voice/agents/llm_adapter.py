"""Sends the model turns of a CrewAI agent through ``ModelClient`` (owner: An, task A2b).

A CrewAI agent talks to its own LLM client, which would bypass the spending cap and the token
and cost log of ``ModelClient``. This module is the part of the fix that does not need CrewAI:
it turns the agent's messages into one prompt and runs it with ``client.complete``, so each
agent turn is checked against the run and person caps and written to the call log like any
other model call. ``agents/crewai_bridge.py`` wraps it in the ``BaseLLM`` CrewAI expects.

``ModelClient.complete`` answers with a structured schema, so the agent's reply travels in a
one-field schema (``AgentReply``). Three consequences, all handled here:

* No tool calling at the model level: agents use CrewAI's text format (Thought, Action,
  Action Input) and the tools run in CrewAI, not in the model.
* No stop sequences: CrewAI relies on a stop word such as ``\\nObservation:`` to end a turn, so
  the reply is cut at the first stop word instead.
* Images are not supported in agent messages; images go through ``analyze_image``.
"""

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from pydantic import BaseModel, Field

from ai_share_of_voice.errors import ToolFailure
from ai_share_of_voice.schemas.base import Contract
from ai_share_of_voice.schemas.llm import ModelClient

TOOL_NAME = "agent_llm"
DEFAULT_CONTEXT_WINDOW = 131_072
MAX_REPLY_CHARS = 60_000
_INSTRUCTION = "Write the assistant's next message in the text field, exactly as asked above."

Message = Mapping[str, Any]


class AgentReply(Contract):
    """Schema of one agent turn: the whole message, as the agent should have written it."""

    text: str = Field(min_length=1, max_length=MAX_REPLY_CHARS)


def _fail(reason: str) -> ToolFailure:
    return ToolFailure(TOOL_NAME, reason)


def _content_text(content: Any) -> str:
    """Text of one message: a string, nothing, or a list of text parts. Anything else fails."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence):
        parts: list[str] = []
        for part in content:
            if isinstance(part, Mapping) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
            else:
                raise _fail("agent messages can only contain text; images are not supported")
        return "\n".join(parts)
    raise _fail(f"unsupported message content of type {type(content).__name__}")


def messages_to_prompt(messages: str | Sequence[Message]) -> str:
    """One prompt from the agent's messages, each under its role, ending with an instruction."""
    if isinstance(messages, str):
        body = messages.strip()
    else:
        blocks = []
        for message in messages:
            text = _content_text(message.get("content")).strip()
            if text:
                blocks.append(f"[{message.get('role', 'user')}]\n{text}")
        body = "\n\n".join(blocks)
    if not body:
        raise _fail("the agent sent no text to the model")
    return f"{body}\n\n{_INSTRUCTION}"


def cut_at_stop_words(text: str, stop: Iterable[str] | None) -> str:
    """Cut ``text`` at the earliest stop word.

    If that would leave nothing, the text is returned whole: CrewAI then sees a badly formed
    turn and asks for a correction, which is better than an empty reply.
    """
    cuts = [text.find(word) for word in (stop or ()) if word and word in text]
    if not cuts:
        return text
    kept = text[: min(cuts)]
    return kept if kept.strip() else text


class BudgetedCaller:
    """Runs agent turns with a ``ModelClient``; ``BudgetExceeded`` and failures propagate."""

    def __init__(self, client: ModelClient) -> None:
        self._client = client

    def text(self, messages: str | Sequence[Message], stop: Iterable[str] | None = None) -> str:
        """The agent's next message as text, cut at the stop words."""
        reply, _ = self._client.complete(messages_to_prompt(messages), AgentReply)
        return cut_at_stop_words(reply.text, stop)

    def structured[T: BaseModel](self, messages: str | Sequence[Message], schema: type[T]) -> T:
        """The agent's next message parsed as ``schema`` (CrewAI asks for this for task output)."""
        answer, _ = self._client.complete(messages_to_prompt(messages), schema)
        return answer
