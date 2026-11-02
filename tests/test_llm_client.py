import json
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from ai_share_of_voice.config import load_settings
from ai_share_of_voice.errors import ToolFailure
from ai_share_of_voice.exceptions import BudgetExceeded
from ai_share_of_voice.llm_client import (
    Budget,
    InvalidModelResponse,
    LiveClient,
    SimulatedClient,
    build_clients,
    price_of,
)


class Verdict(BaseModel):
    recommends: bool
    reason: str


class FakeCompletions:
    """Stands in for ``openai.OpenAI().chat.completions``: scripted replies, recorded calls."""

    def __init__(self, contents, input_tokens=100, output_tokens=50, error=None):
        self.contents = list(contents)
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.contents.pop(0)))],
            usage=SimpleNamespace(
                prompt_tokens=self.input_tokens, completion_tokens=self.output_tokens
            ),
        )


def fake_sdk(completions):
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


MODEL = "llama-3.3-70b-versatile"


def live(completions, cap=1.0, **kwargs):
    return LiveClient(MODEL, Budget(cap), fake_sdk(completions), **kwargs)


# --- budget ---------------------------------------------------------------------------------


def test_budget_reserves_then_settles_the_actual_cost():
    budget = Budget(1.0)

    reservation = budget.reserve(0.3)
    assert budget.committed_usd == pytest.approx(0.3)
    budget.settle(reservation, actual_usd=0.1)

    assert budget.spent_usd == pytest.approx(0.1)
    assert budget.committed_usd == pytest.approx(0.1)


def test_budget_refuses_a_reservation_that_would_pass_the_cap():
    budget = Budget(0.5)
    budget.settle(budget.reserve(0.4), actual_usd=0.4)

    with pytest.raises(BudgetExceeded) as error:
        budget.reserve(0.2)
    assert error.value.cap_usd == 0.5
    assert budget.spent_usd == pytest.approx(0.4)  # a refused call costs nothing


def test_concurrent_reservations_count_against_the_cap():
    # Two calls in flight: the second sees the first reservation, not only settled spend.
    budget = Budget(0.5)
    budget.reserve(0.3)

    with pytest.raises(BudgetExceeded):
        budget.reserve(0.3)


# --- live client ----------------------------------------------------------------------------


def test_complete_parses_the_schema_and_reports_usage_and_cost():
    completions = FakeCompletions([json.dumps({"recommends": True, "reason": "fast"})])
    client = live(completions)

    verdict, usage = client.complete("Does the answer recommend Glovo?", Verdict)

    assert verdict == Verdict(recommends=True, reason="fast")
    assert (usage.model, usage.input_tokens, usage.output_tokens) == (MODEL, 100, 50)
    assert usage.cost_usd == pytest.approx(price_of(MODEL).cost(100, 50))
    assert client.budget.spent_usd == pytest.approx(usage.cost_usd)


def test_complete_asks_for_json_and_sends_the_schema():
    completions = FakeCompletions([json.dumps({"recommends": False, "reason": "none"})])

    live(completions).complete("prompt", Verdict)

    call = completions.calls[0]
    assert call["model"] == MODEL
    assert call["response_format"] == {"type": "json_object"}
    assert "recommends" in call["messages"][0]["content"]  # schema in the system message
    assert call["messages"][-1] == {"role": "user", "content": "prompt"}


def test_a_fenced_json_answer_is_accepted():
    completions = FakeCompletions(['```json\n{"recommends": true, "reason": "ok"}\n```'])

    verdict, _ = live(completions).complete("prompt", Verdict)

    assert verdict.recommends is True


def test_an_answer_that_breaks_the_schema_is_a_value_error_and_still_costs():
    # ValueError is what the nodes catch for invalid answers (see agents/skeptic.py).
    client = live(FakeCompletions(['{"recommends": "maybe"}']))

    with pytest.raises(InvalidModelResponse) as error:
        client.complete("prompt", Verdict)
    assert isinstance(error.value, ValueError)
    assert client.budget.spent_usd == pytest.approx(price_of(MODEL).cost(100, 50))


def test_an_api_error_is_a_tool_failure_without_the_key():
    client = live(FakeCompletions([], error=RuntimeError("connection reset")))

    with pytest.raises(ToolFailure) as error:
        client.complete("prompt", Verdict)
    assert error.value.tool == "llm_client"
    assert client.budget.spent_usd > 0  # charged the input estimate: the cap stays honest


def test_no_call_is_made_when_the_budget_cannot_cover_it():
    completions = FakeCompletions([json.dumps({"recommends": True, "reason": "x"})])
    client = live(completions, cap=0.000001)

    with pytest.raises(BudgetExceeded):
        client.complete("prompt", Verdict)
    assert completions.calls == []


def test_answer_returns_free_text_without_json_mode():
    # The measured model answers like a normal user would see it: no JSON, no schema.
    completions = FakeCompletions(["Te recomiendo Glovo o Just Eat."])

    text, usage = live(completions).answer("¿Qué app de comida a domicilio me recomiendas?")

    assert text == "Te recomiendo Glovo o Just Eat."
    assert "response_format" not in completions.calls[0]
    assert completions.calls[0]["messages"] == [
        {"role": "user", "content": "¿Qué app de comida a domicilio me recomiendas?"}
    ]
    assert usage.output_tokens == 50


def test_every_call_is_appended_to_the_usage_log(tmp_path):
    log = tmp_path / "usage.jsonl"
    completions = FakeCompletions(["a", "b"])
    client = live(completions, usage_log=log, label="subject")

    client.answer("one")
    client.answer("two")

    lines = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert [line["label"] for line in lines] == ["subject", "subject"]
    assert all(line["model"] == MODEL and line["cost_usd"] > 0 for line in lines)
    assert "one" not in log.read_text(encoding="utf-8")  # prompts are not logged


def test_a_model_without_a_price_is_refused():
    with pytest.raises(ValueError, match="price"):
        LiveClient("unknown-model", Budget(1.0), fake_sdk(FakeCompletions([])))


# --- simulated client and factory -----------------------------------------------------------


def test_simulated_client_uses_responders_and_charges_the_shared_budget():
    budget = Budget(1.0)
    client = SimulatedClient(
        budget,
        responders={Verdict: lambda prompt: Verdict(recommends=True, reason=prompt)},
        text_responder=lambda prompt: f"echo: {prompt}",
        cost_per_call_usd=0.01,
    )

    verdict, _ = client.complete("hi", Verdict)
    text, _ = client.answer("hello")

    assert verdict.reason == "hi"
    assert text == "echo: hello"
    assert budget.spent_usd == pytest.approx(0.02)


def test_simulated_client_without_a_responder_is_a_clear_error():
    with pytest.raises(InvalidModelResponse, match="Verdict"):
        SimulatedClient(Budget(1.0)).complete("hi", Verdict)


def test_build_clients_in_simulated_mode_needs_no_key_and_shares_one_budget():
    clients = build_clients(load_settings({"SPEND_CAP_USD": "0.5"}))

    assert all(
        isinstance(client, SimulatedClient)
        for client in (clients.subject, clients.agent, clients.vision)
    )
    assert clients.subject.budget is clients.agent.budget is clients.budget
    assert clients.budget.cap_usd == 0.5


def test_build_clients_in_live_mode_uses_one_model_per_role():
    settings = load_settings({"LLM_MODE": "live", "LLM_API_KEY": "gsk_fake"})

    clients = build_clients(settings, sdk_factory=lambda url, key: fake_sdk(FakeCompletions([])))

    assert clients.subject.model == settings.subject_model
    assert clients.agent.model == settings.agent_model
    assert clients.vision.model == settings.vision_model
    assert clients.subject.budget is clients.vision.budget


def test_build_clients_puts_the_subject_on_its_own_provider():
    settings = load_settings(
        {
            "LLM_MODE": "live",
            "LLM_API_KEY": "gsk_fake",
            "SUBJECT_PROVIDER": "gemini",
            "SUBJECT_API_KEY": "AIza_fake",
        }
    )
    made = []

    def sdk_factory(base_url, api_key):
        made.append((base_url, api_key))
        return fake_sdk(FakeCompletions([]))

    clients = build_clients(settings, sdk_factory=sdk_factory)

    assert clients.subject.model == settings.subject_model
    assert sorted(made) == sorted(
        [(settings.base_url, "gsk_fake"), (settings.subject_base_url, "AIza_fake")]
    )


def test_build_clients_opens_one_sdk_per_endpoint():
    settings = load_settings(
        {
            "LLM_MODE": "live",
            "LLM_API_KEY": "gsk_fake",
            "SUBJECT_PROVIDER": "gemini",
            "SUBJECT_API_KEY": "AIza_fake",
            "VISION_PROVIDER": "gemini",
        }
    )
    made = []

    def sdk_factory(base_url, api_key):
        made.append((base_url, api_key))
        return fake_sdk(FakeCompletions([]))

    clients = build_clients(settings, sdk_factory=sdk_factory)

    assert len(made) == 2  # Groq for the agents, Gemini shared by subject and vision
    assert clients.vision._sdk is clients.subject._sdk
    assert clients.agent._sdk is not clients.subject._sdk
    assert clients.vision.model == settings.vision_model
