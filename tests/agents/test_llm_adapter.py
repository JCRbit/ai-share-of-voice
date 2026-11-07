import json
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from ai_share_of_voice.agents.llm_adapter import (
    MAX_REPLY_CHARS,
    AgentReply,
    BudgetedCaller,
    cut_at_stop_words,
    messages_to_prompt,
)
from ai_share_of_voice.errors import ToolFailure
from ai_share_of_voice.exceptions import BudgetExceeded
from ai_share_of_voice.llm_client import Pricing, SimulatedClient

CONVERSATION = [
    {"role": "system", "content": "You are a careful analyst."},
    {"role": "user", "content": "Which brand is mentioned first?"},
    {"role": "assistant", "content": "Thought: I need the data."},
]
REACT_TURN = (
    "Thought: I should look it up.\nAction: stats_tool\nAction Input: {}\nObservation: invented"
)


class Verdict(BaseModel):
    brand: str = "Acme"
    score: int = 7


def scripted(text: str = "Final Answer: done", prompts: list[str] | None = None) -> SimulatedClient:
    """A simulated client that answers every agent turn with ``text`` and records the prompts."""

    def respond(prompt: str, schema: type[BaseModel]) -> BaseModel:
        if prompts is not None:
            prompts.append(prompt)
        return schema(text=text) if schema is AgentReply else schema()

    return SimulatedClient(responder=respond, pricing=Pricing(1.0, 1.0))


# ---- messages to one prompt ------------------------------------------------------------------


def test_a_string_is_the_prompt_and_a_conversation_keeps_its_roles() -> None:
    assert messages_to_prompt("Hola").startswith("Hola\n\n")
    prompt = messages_to_prompt(CONVERSATION)
    assert prompt.index("[system]") < prompt.index("[user]") < prompt.index("[assistant]")
    assert "Which brand is mentioned first?" in prompt
    assert prompt.rstrip().endswith("exactly as asked above.")


def test_empty_messages_and_text_parts_are_handled() -> None:
    messages = [
        {"role": "assistant", "content": None},
        {
            "role": "user",
            "content": [{"type": "text", "text": "uno"}, {"type": "text", "text": "dos"}],
        },
        {"role": "tool", "content": "   "},
    ]
    prompt = messages_to_prompt(messages)
    assert "[user]\nuno\ndos" in prompt and "[assistant]" not in prompt and "[tool]" not in prompt


@pytest.mark.parametrize("messages", ["", "   ", [], [{"role": "user", "content": None}]])
def test_nothing_to_send_is_a_tool_failure(messages: object) -> None:
    with pytest.raises(ToolFailure, match="no text"):
        messages_to_prompt(messages)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "content",
    [[{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}], 42],
)
def test_images_and_odd_content_are_refused(content: object) -> None:
    with pytest.raises(ToolFailure, match="images are not supported|unsupported"):
        messages_to_prompt([{"role": "user", "content": content}])


# ---- stop words ------------------------------------------------------------------------------


def test_the_reply_is_cut_at_the_earliest_stop_word() -> None:
    assert cut_at_stop_words(REACT_TURN, ["\nObservation:"]).endswith("Action Input: {}")
    both = cut_at_stop_words("a STOP2 b STOP1 c", ["STOP1", "STOP2"])
    assert both == "a "


@pytest.mark.parametrize("stop", [None, [], [""], ["\nObservation:"]])
def test_without_a_matching_stop_word_the_text_is_untouched(stop: list[str] | None) -> None:
    assert cut_at_stop_words("Final Answer: done", stop) == "Final Answer: done"


def test_a_reply_that_would_become_empty_is_kept_whole() -> None:
    text = "\nObservation: only this"
    assert cut_at_stop_words(text, ["\nObservation:"]) == text


# ---- the reply schema ------------------------------------------------------------------------


def test_the_reply_schema_needs_text_of_a_sane_size() -> None:
    assert AgentReply(text="  hola ").text == "hola"
    for bad in ("", "   ", "x" * (MAX_REPLY_CHARS + 1)):
        with pytest.raises(ValidationError):
            AgentReply(text=bad)


# ---- every turn goes through the client ------------------------------------------------------


def test_a_turn_is_one_client_call_with_the_whole_conversation() -> None:
    prompts: list[str] = []
    client = scripted(REACT_TURN, prompts)
    text = BudgetedCaller(client).text(CONVERSATION, stop=["\nObservation:"])
    assert text.endswith("Action Input: {}") and "invented" not in text
    assert len(prompts) == 1 and len(client.calls) == 1
    assert "You are a careful analyst." in prompts[0] and "Which brand" in prompts[0]


def test_each_turn_is_counted_against_the_cap_and_written_to_the_call_log(tmp_path: Path) -> None:
    log = tmp_path / "calls.jsonl"
    client = SimulatedClient(
        responder=lambda prompt, schema: schema(text="Final Answer: ok"),
        pricing=Pricing(1.0, 1.0),
        run_id="run-test",
        log_path=log,
    )
    caller = BudgetedCaller(client)
    for _ in range(3):
        caller.text(CONVERSATION)
    entries = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert len(entries) == 3 and len(client.calls) == 3
    assert all(e["run_id"] == "run-test" and e["outcome"] == "ok" for e in entries)
    assert all(e["input_tokens"] > 0 and e["cost_usd"] > 0 for e in entries)
    assert client.spent_usd == pytest.approx(sum(e["cost_usd"] for e in entries))


def test_the_spending_cap_stops_agent_turns_before_they_are_paid() -> None:
    client = SimulatedClient(
        responder=lambda prompt, schema: schema(text="Final Answer: ok"),
        pricing=Pricing(1_000.0, 1_000.0),
        cap_usd=0.01,
    )
    with pytest.raises(BudgetExceeded):
        BudgetedCaller(client).text(CONVERSATION)
    assert client.spent_usd == 0 and not client.calls


def test_a_structured_turn_returns_the_schema_the_caller_asked_for() -> None:
    client = SimulatedClient(responder=lambda prompt, schema: Verdict(brand="Beta", score=3))
    verdict = BudgetedCaller(client).structured(CONVERSATION, Verdict)
    assert verdict == Verdict(brand="Beta", score=3)
