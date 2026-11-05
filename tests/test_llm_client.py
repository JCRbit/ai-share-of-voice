import json
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from ai_share_of_voice.errors import FailureKind, ToolFailure, classify
from ai_share_of_voice.exceptions import BudgetExceeded, ConfigurationError
from ai_share_of_voice.llm_client import (
    DEFAULT_MODEL,
    GroqClient,
    Pricing,
    SimulatedClient,
    TransportError,
    _parse_retry_after,
    client_from_env,
    read_logged_spend,
)
from ai_share_of_voice.schemas.llm import ModelClient


class Answer(BaseModel):
    text: str = "ok"


class Strict(BaseModel):
    value: int


def groq_body(content: str, prompt_tokens: int = 100, completion_tokens: int = 50) -> dict:
    return {
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }


class ScriptedTransport:
    """Replays a list of bodies or errors and records the payloads it received."""

    def __init__(self, *steps: dict | Exception) -> None:
        self.steps = list(steps)
        self.payloads: list[dict[str, Any]] = []

    def __call__(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.payloads.append(payload)
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


def make_groq(transport: ScriptedTransport, **kwargs: Any) -> GroqClient:
    kwargs.setdefault("sleep", lambda _seconds: None)
    return GroqClient("key", transport=transport, **kwargs)


def read_log(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# -- contract -----------------------------------------------------------------------------


def test_both_clients_satisfy_the_model_client_protocol() -> None:
    simulated: ModelClient = SimulatedClient()
    live: ModelClient = make_groq(ScriptedTransport())
    assert simulated is not None and live is not None


def test_simulated_client_returns_object_and_free_usage() -> None:
    client = SimulatedClient()
    answer, usage = client.complete("hello", Answer)
    assert answer == Answer()
    assert usage.cost_usd == 0.0
    assert usage.total_tokens > 0
    assert client.spent_usd == 0.0


def test_simulated_client_uses_the_responder() -> None:
    client = SimulatedClient(responder=lambda prompt, schema: Strict(value=len(prompt)))
    answer, _ = client.complete("abcd", Strict)
    assert answer.value == 4


def test_simulated_client_without_defaults_fails_clearly() -> None:
    with pytest.raises(ToolFailure, match="responder"):
        SimulatedClient().complete("hi", Strict)


# -- cost and usage -----------------------------------------------------------------------


def test_groq_cost_uses_provider_tokens_and_the_price() -> None:
    transport = ScriptedTransport(groq_body('{"text":"hi"}', 1_000_000, 1_000_000))
    client = make_groq(transport, pricing=Pricing(0.15, 0.60), cap_usd=5.0, max_output_tokens=10)
    # 1M tokens in and out cost 0.75 USD, far above the worst-case guess for a tiny prompt.
    answer, usage = client.complete("hi", Answer)
    assert answer.text == "hi"
    assert usage.model == DEFAULT_MODEL
    assert (usage.input_tokens, usage.output_tokens) == (1_000_000, 1_000_000)
    assert usage.cost_usd == pytest.approx(0.75)
    assert client.spent_usd == pytest.approx(0.75)


def test_default_price_of_the_decided_model() -> None:
    client = make_groq(ScriptedTransport(groq_body('{"text":"hi"}', 2_000, 1_000)))
    _, usage = client.complete("hi", Answer)
    assert usage.cost_usd == pytest.approx(2_000 * 0.15e-6 + 1_000 * 0.60e-6)


def test_request_asks_for_a_json_schema_and_bounds_the_output() -> None:
    transport = ScriptedTransport(groq_body('{"text":"hi"}'))
    client = make_groq(transport, max_output_tokens=321, reasoning_effort="low")
    client.complete("question", Answer)
    payload = transport.payloads[0]
    assert payload["model"] == DEFAULT_MODEL
    assert payload["messages"] == [{"role": "user", "content": "question"}]
    assert payload["max_completion_tokens"] == 321
    assert payload["reasoning_effort"] == "low"
    assert payload["response_format"]["json_schema"]["name"] == "Answer"
    assert "text" in payload["response_format"]["json_schema"]["schema"]["properties"]


def test_json_in_a_code_fence_is_accepted() -> None:
    client = make_groq(ScriptedTransport(groq_body('```json\n{"text": "fenced"}\n```')))
    answer, _ = client.complete("hi", Answer)
    assert answer.text == "fenced"


def test_missing_usage_is_charged_at_the_worst_case() -> None:
    body = {"choices": [{"message": {"content": '{"text":"hi"}'}}]}
    client = make_groq(ScriptedTransport(body), max_output_tokens=1000)
    _, usage = client.complete("hi", Answer)
    assert usage.output_tokens == 1000
    assert usage.cost_usd > 0


@pytest.mark.parametrize("content", [[], {}, 123, False])
def test_invalid_provider_content_uses_safe_fallback(content: Any) -> None:
    body = groq_body(content)
    client = make_groq(
        ScriptedTransport(body),
        max_output_tokens=77,
    )

    generation = client._generate("hi", Answer)

    assert generation.text == ""
    assert generation.input_tokens > 0
    assert generation.output_tokens == 77


@pytest.mark.parametrize(
    ("prompt_tokens", "completion_tokens"),
    [
        (100.5, 50),
        (True, 50),
        (-1, 50),
        (100, 50.5),
        (100, False),
        (100, -1),
    ],
)
def test_invalid_provider_token_counts_use_worst_case(
    prompt_tokens: Any, completion_tokens: Any
) -> None:
    body = groq_body(
        '{"text":"hi"}',
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    client = make_groq(
        ScriptedTransport(body),
        max_output_tokens=77,
    )

    generation = client._generate("hi", Answer)

    assert generation.input_tokens > 0
    assert generation.output_tokens == 77


# -- spending cap -------------------------------------------------------------------------


def test_cap_blocks_the_call_before_it_is_made() -> None:
    transport = ScriptedTransport(groq_body('{"text":"hi"}'))
    # Worst case: 4096 output tokens at 1000 USD per million is 4.1 USD, over the 1 USD cap.
    client = make_groq(transport, pricing=Pricing(1000, 1000), cap_usd=1.0)
    with pytest.raises(BudgetExceeded) as caught:
        client.complete("hi", Answer)
    assert caught.value.cap_usd == pytest.approx(1.0)
    assert transport.payloads == []
    assert client.spent_usd == 0.0


def test_cap_stops_later_calls_once_spend_reaches_it() -> None:
    replies = iter([Answer(text="a"), Answer(text="b"), Answer(text="c")])
    client = SimulatedClient(
        responder=lambda prompt, schema: next(replies),
        pricing=Pricing(0, 1_000_000),  # output only: 1 USD per token
        cap_usd=9,
        max_output_tokens=5,  # worst case per call: 5 USD
    )
    # Each answer is 4 tokens (12 characters of JSON), so it costs 4 USD.
    client.complete("hi", Answer)  # 0 + 5 <= 9, spends 4
    client.complete("hi", Answer)  # 4 + 5 <= 9, spends 4
    with pytest.raises(BudgetExceeded) as caught:
        client.complete("hi", Answer)  # 8 + 5 > 9
    assert caught.value.spent_usd == pytest.approx(8)
    assert client.spent_usd == pytest.approx(8)


def test_budget_exceeded_is_classified_as_budget_exhausted() -> None:
    client = SimulatedClient(pricing=Pricing(1000, 1000), cap_usd=0.001)
    with pytest.raises(BudgetExceeded) as caught:
        client.complete("hi", Answer)
    assert classify(caught.value).kind is FailureKind.BUDGET_EXHAUSTED


def test_person_cap_adds_up_spend_from_earlier_runs(tmp_path: Path) -> None:
    log = tmp_path / "calls.jsonl"
    log.write_text(
        json.dumps({"cost_usd": 4.99}) + "\n\nnot json\n" + json.dumps({"cost_usd": 0.005}) + "\n"
    )
    assert read_logged_spend(log) == pytest.approx(4.995)
    client = SimulatedClient(
        pricing=Pricing(100, 100),
        cap_usd=2.0,
        person_cap_usd=5.0,
        log_path=log,
        max_output_tokens=100,
    )
    # Worst case is about 0.01 USD, and only 0.005 USD is left for this person.
    with pytest.raises(BudgetExceeded) as caught:
        client.complete("hi", Answer)
    assert caught.value.cap_usd == pytest.approx(5.0)
    assert caught.value.spent_usd == pytest.approx(4.995)


@pytest.mark.parametrize(
    "invalid_cost",
    [-1, float("nan"), float("inf"), float("-inf"), True, "1.25"],
)
def test_read_logged_spend_ignores_invalid_costs(tmp_path: Path, invalid_cost: Any) -> None:
    log = tmp_path / "calls.jsonl"
    entries = [
        json.dumps({"cost_usd": 1.5}),
        json.dumps({"cost_usd": invalid_cost}),
        json.dumps({"cost_usd": 2.0}),
    ]
    log.write_text("\n".join(entries) + "\n", encoding="utf-8")

    assert read_logged_spend(log) == pytest.approx(3.5)


def test_read_logged_spend_ignores_non_object_entries(tmp_path: Path) -> None:
    log = tmp_path / "calls.jsonl"
    log.write_text(
        json.dumps([{"cost_usd": 100}]) + "\n" + json.dumps({"cost_usd": 2.0}) + "\n",
        encoding="utf-8",
    )

    assert read_logged_spend(log) == pytest.approx(2.0)


def test_read_logged_spend_ignores_overflowing_total(tmp_path: Path) -> None:
    log = tmp_path / "calls.jsonl"
    log.write_text(
        json.dumps({"cost_usd": 1e308}) + "\n" + json.dumps({"cost_usd": 1e308}) + "\n",
        encoding="utf-8",
    )

    assert read_logged_spend(log) == pytest.approx(1e308)


def test_person_cap_needs_a_log_file() -> None:
    with pytest.raises(ConfigurationError):
        SimulatedClient(person_cap_usd=5.0)


@pytest.mark.parametrize("cap", [0, -1])
def test_cap_must_be_positive(cap: float) -> None:
    with pytest.raises(ConfigurationError):
        SimulatedClient(cap_usd=cap)


# -- call log -----------------------------------------------------------------------------


def test_every_call_is_logged_with_tokens_and_cost(tmp_path: Path) -> None:
    log = tmp_path / "runs" / "calls.jsonl"
    transport = ScriptedTransport(
        groq_body('{"text":"a"}', 10, 5), groq_body('{"text":"b"}', 20, 8)
    )
    client = make_groq(transport, log_path=log, run_id="run-1")
    client.complete("one", Answer)
    client.complete("two", Answer)
    first, second = read_log(log)
    assert first["run_id"] == "run-1"
    assert first["model"] == DEFAULT_MODEL
    assert (first["input_tokens"], first["output_tokens"]) == (10, 5)
    assert first["outcome"] == "ok"
    assert first["cost_usd"] == pytest.approx(10 * 0.15e-6 + 5 * 0.60e-6)
    assert second["run_spent_usd"] == pytest.approx(first["cost_usd"] + second["cost_usd"])
    assert [call.input_tokens for call in client.calls] == [10, 20]


def test_api_key_never_reaches_the_log(tmp_path: Path) -> None:
    log = tmp_path / "calls.jsonl"
    client = GroqClient(
        "gsk_secret", transport=ScriptedTransport(groq_body('{"text":"a"}')), log_path=log
    )
    client.complete("hi", Answer)
    assert "gsk_secret" not in log.read_text()


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("max_retries", -1),
        ("max_retries", True),
        ("max_retries", 1.5),
        ("backoff_s", -1),
        ("backoff_s", True),
        ("backoff_s", float("nan")),
        ("backoff_s", float("inf")),
        ("backoff_s", float("-inf")),
        ("timeout_s", 0),
        ("timeout_s", -1),
        ("timeout_s", True),
        ("timeout_s", float("nan")),
        ("timeout_s", float("inf")),
        ("timeout_s", float("-inf")),
    ],
)
def test_groq_rejects_invalid_retry_and_timeout_settings(parameter: str, value: Any) -> None:
    with pytest.raises(ConfigurationError):
        make_groq(ScriptedTransport(), **{parameter: value})


# -- invalid output and provider errors ---------------------------------------------------


def test_invalid_answer_is_retried_and_both_attempts_are_paid(tmp_path: Path) -> None:
    log = tmp_path / "calls.jsonl"
    transport = ScriptedTransport(
        groq_body('{"value": "not a number"}', 10, 5), groq_body('{"value": 7}', 12, 6)
    )
    client = make_groq(transport, log_path=log)
    answer, usage = client.complete("give a number", Strict)
    assert answer.value == 7
    assert (usage.input_tokens, usage.output_tokens) == (22, 11)
    assert [row["outcome"] for row in read_log(log)] == ["invalid_output", "ok"]
    assert "rejected" in transport.payloads[1]["messages"][0]["content"]
    assert client.spent_usd == pytest.approx(usage.cost_usd)


def test_answer_that_never_matches_raises_tool_failure() -> None:
    transport = ScriptedTransport(groq_body("nonsense"), groq_body(""))
    client = make_groq(transport)
    with pytest.raises(ToolFailure) as caught:
        client.complete("hi", Strict)
    assert caught.value.tool == "llm_client"
    assert classify(caught.value).kind is FailureKind.TOOL_FAILURE
    assert client.spent_usd > 0  # the failed attempts were still paid


def test_transient_errors_are_retried_with_backoff() -> None:
    delays: list[float] = []
    transport = ScriptedTransport(
        TransportError("busy", status=429, retry_after=2),
        TransportError("down", status=503),
        groq_body('{"text":"hi"}'),
    )
    client = make_groq(transport, sleep=delays.append, backoff_s=1.0)
    answer, _ = client.complete("hi", Answer)
    assert answer.text == "hi"
    assert delays == [2, 2.0]  # Retry-After first, then 1.0 * 2**1


def test_parse_retry_after_http_date() -> None:
    retry_at = datetime.now(UTC) + timedelta(seconds=30)
    value = format_datetime(retry_at, usegmt=True)

    delay = _parse_retry_after(value)

    assert delay is not None
    assert 0 <= delay <= 30


def test_parse_retry_after_past_http_date_returns_zero() -> None:
    retry_at = datetime.now(UTC) - timedelta(seconds=30)
    value = format_datetime(retry_at, usegmt=True)

    assert _parse_retry_after(value) == 0.0


@pytest.mark.parametrize("value", [None, "", "invalid-date", "not-a-number"])
def test_parse_retry_after_invalid_value_returns_none(value: str | None) -> None:
    assert _parse_retry_after(value) is None


def test_errors_that_will_not_heal_are_not_retried() -> None:
    transport = ScriptedTransport(TransportError("HTTP 401: bad key", status=401))
    client = make_groq(transport)
    with pytest.raises(ToolFailure, match="401"):
        client.complete("hi", Answer)
    assert len(transport.payloads) == 1
    assert client.spent_usd == 0.0


def test_gives_up_after_the_retry_limit() -> None:
    transport = ScriptedTransport(*[TransportError("down", status=500) for _ in range(3)])
    client = make_groq(transport, max_retries=2)
    with pytest.raises(ToolFailure, match="provider call failed"):
        client.complete("hi", Answer)
    assert len(transport.payloads) == 3


def test_unknown_model_needs_a_price() -> None:
    with pytest.raises(ConfigurationError, match="price"):
        GroqClient("key", model="some/other-model", transport=ScriptedTransport())


# -- configuration ------------------------------------------------------------------------


def test_simulated_is_the_default_mode() -> None:
    client = client_from_env({})
    assert isinstance(client, SimulatedClient)
    assert client.cap_usd == pytest.approx(2.0)


def test_simulated_mode_ignores_person_cap_without_a_log_file() -> None:
    client = client_from_env({"LLM_MODE": "simulated", "PERSON_CAP_USD": "5.00"})

    assert isinstance(client, SimulatedClient)
    assert client.person_cap_usd is None


def test_env_sets_cap_and_run_cap_override() -> None:
    assert client_from_env({"SPEND_CAP_USD": "1.5"}).cap_usd == pytest.approx(1.5)
    assert client_from_env({"SPEND_CAP_USD": "1.5"}, cap_usd=0.5).cap_usd == pytest.approx(0.5)


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "Infinity", "-Infinity"])
def test_env_rejects_non_finite_spending_caps(value: str) -> None:
    with pytest.raises(ConfigurationError, match="finite"):
        client_from_env({"SPEND_CAP_USD": value})


def test_live_mode_builds_the_groq_client(tmp_path: Path) -> None:
    env = {
        "LLM_MODE": "live",
        "LLM_PROVIDER": "Groq",
        "LLM_API_KEY": "key",
        "LLM_MODEL": "",
        "SPEND_CAP_USD": "2.00",
        "PERSON_CAP_USD": "5.00",
    }
    client = client_from_env(env, log_path=tmp_path / "calls.jsonl", run_id="r1")
    assert isinstance(client, GroqClient)
    assert client.model == DEFAULT_MODEL
    assert client.person_cap_usd == pytest.approx(5.0)
    assert client.run_id == "r1"


def test_live_mode_requires_a_log_file_for_person_cap() -> None:
    env = {
        "LLM_MODE": "live",
        "LLM_PROVIDER": "groq",
        "LLM_API_KEY": "key",
        "PERSON_CAP_USD": "5.00",
    }

    with pytest.raises(ConfigurationError, match="needs a log file"):
        client_from_env(env)


def test_live_mode_accepts_price_overrides_for_other_models() -> None:
    env = {
        "LLM_MODE": "live",
        "LLM_PROVIDER": "groq",
        "LLM_API_KEY": "key",
        "LLM_MODEL": "other/model",
        "LLM_PRICE_INPUT_PER_M": "1",
        "LLM_PRICE_OUTPUT_PER_M": "2",
    }
    client = client_from_env(env)
    assert client.pricing == Pricing(1.0, 2.0)


@pytest.mark.parametrize(
    "env",
    [
        {"LLM_MODE": "nope"},
        {"LLM_MODE": "live", "LLM_PROVIDER": "openai", "LLM_API_KEY": "k"},
        {"LLM_MODE": "live", "LLM_PROVIDER": "groq", "LLM_API_KEY": ""},
        {"SPEND_CAP_USD": "abc"},
        {
            "LLM_MODE": "live",
            "LLM_PROVIDER": "groq",
            "LLM_API_KEY": "k",
            "LLM_PRICE_INPUT_PER_M": "1",
        },
    ],
)
def test_bad_configuration_fails_with_a_clear_error(env: dict[str, str]) -> None:
    with pytest.raises(ConfigurationError):
        client_from_env(env)


def test_concurrent_calls_on_one_client_are_serialized() -> None:
    entered = threading.Event()
    release = threading.Event()
    active_calls = 0
    max_active_calls = 0
    state_lock = threading.Lock()

    def transport(payload: dict[str, Any]) -> dict:
        nonlocal active_calls, max_active_calls

        with state_lock:
            active_calls += 1
            max_active_calls = max(max_active_calls, active_calls)
            entered.set()

        if not release.wait(timeout=5):
            raise TimeoutError("Test did not release the first call")

        with state_lock:
            active_calls -= 1

        return groq_body('{"text": "ok"}')

    client = GroqClient("key", transport=transport, sleep=lambda _: None)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(client.complete, "first", Answer)

        try:
            assert entered.wait(timeout=2), "First call did not reach the transport"
            second = executor.submit(client.complete, "second", Answer)

            # Give the second worker a chance to run while the first is blocked.
            assert not second.done()
        finally:
            release.set()

        first.result(timeout=5)
        second.result(timeout=5)

    assert max_active_calls == 1
    assert len(client.calls) == 2


def test_log_write_failure_propagates_after_recording_spend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = tmp_path / "calls.jsonl"
    client = make_groq(
        ScriptedTransport(groq_body('{"text": "ok"}', 100, 50)),
        log_path=log,
    )

    def fail_open(*args: Any, **kwargs: Any) -> None:
        raise OSError("simulated disk failure")

    monkeypatch.setattr(Path, "open", fail_open)

    with pytest.raises(OSError, match="simulated disk failure"):
        client.complete("hello", Answer)

    assert client.spent_usd > 0
    assert len(client.calls) == 1
    assert not log.exists()

def _shared_budget_process_worker(log_path: str, barrier: Any, results: Any) -> None:
    client = SimulatedClient(
        pricing=Pricing(0, 1_000_000),
        cap_usd=10.0,
        person_cap_usd=5.1,
        log_path=log_path,
        max_output_tokens=5,
    )
    barrier.wait(timeout=10)
    try:
        client.complete("shared budget test", Answer)
        results.put("completed")
    except BudgetExceeded:
        results.put("blocked")


def test_person_cap_is_shared_between_processes(tmp_path: Path) -> None:
    import multiprocessing

    context = multiprocessing.get_context("spawn")
    log = tmp_path / "shared-calls.jsonl"
    barrier = context.Barrier(2)
    results = context.Queue()
    processes = [
        context.Process(
            target=_shared_budget_process_worker,
            args=(str(log), barrier, results),
        )
        for _ in range(2)
    ]

    for process in processes:
        process.start()

    for process in processes:
        process.join(timeout=20)

    assert all(not process.is_alive() for process in processes)
    assert all(process.exitcode == 0 for process in processes)

    outcomes = [results.get(timeout=2) for _ in processes]
    assert outcomes.count("completed") == 1
    assert outcomes.count("blocked") == 1
    assert read_logged_spend(log) == pytest.approx(5.0)


@pytest.mark.parametrize(
    ("price_name", "price_value"),
    [
        ("LLM_PRICE_INPUT_PER_M", "-0.01"),
        ("LLM_PRICE_OUTPUT_PER_M", "-0.01"),
    ],
)
def test_environment_rejects_negative_text_prices(
    price_name: str, price_value: str
) -> None:
    env = {
        "LLM_MODE": "live",
        "LLM_PROVIDER": "groq",
        "LLM_API_KEY": "key",
        "LLM_PRICE_INPUT_PER_M": "0.15",
        "LLM_PRICE_OUTPUT_PER_M": "0.60",
        price_name: price_value,
    }
    with pytest.raises(ConfigurationError):
        client_from_env(env)


@pytest.mark.parametrize(
    ("price_name", "price_value"),
    [
        ("VISION_PRICE_INPUT_PER_M", "-0.01"),
        ("VISION_PRICE_OUTPUT_PER_M", "-0.01"),
    ],
)
def test_environment_rejects_negative_vision_prices(
    price_name: str, price_value: str
) -> None:
    env = {
        "LLM_MODE": "live",
        "LLM_PROVIDER": "groq",
        "LLM_API_KEY": "key",
        "VISION_MODEL": "vision-model",
        "VISION_PRICE_INPUT_PER_M": "0.80",
        "VISION_PRICE_OUTPUT_PER_M": "4.00",
        price_name: price_value,
    }
    with pytest.raises(ConfigurationError):
        client_from_env(env)
