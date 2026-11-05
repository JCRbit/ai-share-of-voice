import base64
import json
from pathlib import Path

import pytest
from ai_share_of_voice.llm_client import (
    GroqClient,
    Pricing,
    SimulatedClient,
    client_from_env,
)

from ai_share_of_voice.errors import ToolFailure
from ai_share_of_voice.exceptions import BudgetExceeded, ConfigurationError
from ai_share_of_voice.schemas.llm import ImagePart
from ai_share_of_voice.schemas.vision import Confidence, VisionRequest
from ai_share_of_voice.vision import VisionJudgement, VisionTool, analyze_image

SCREENSHOT = Path(__file__).parents[1] / "data" / "screenshots" / "glovo-madrid-home-2026-10-27.jpg"
TEXT_PRICE = Pricing(0.15, 0.60)
VISION_PRICE = Pricing(2.0, 8.0)
ANSWER = json.dumps({"observation": "A food-delivery home page.", "confidence": "medium"})
REQUEST = VisionRequest(path=SCREENSHOT, question="What does the page show?")


class Transport:
    """Fake HTTP layer: records payloads and answers like the provider would."""

    def __init__(self, content: str = ANSWER) -> None:
        self.content = content
        self.payloads: list[dict] = []

    def __call__(self, payload: dict) -> dict:
        self.payloads.append(payload)
        usage = {"prompt_tokens": 3_000, "completion_tokens": 50}
        return {"choices": [{"message": {"content": self.content}}], "usage": usage}


def groq(transport: Transport, **options) -> GroqClient:
    options.setdefault("vision_model", "vision-model")
    options.setdefault("vision_pricing", VISION_PRICE)
    return GroqClient(
        "key", pricing=TEXT_PRICE, transport=transport, sleep=lambda _: None, **options
    )


def run(client):
    return analyze_image(REQUEST, client, allowed_roots=[SCREENSHOT.parent])


def test_a_finding_through_the_simulated_client_counts_against_the_cap() -> None:
    answer = VisionJudgement(observation="A home page.", confidence=Confidence.HIGH)
    client = SimulatedClient(responder=lambda prompt, schema: answer, pricing=VISION_PRICE)
    result = run(client)
    assert result.finding.observation == "A home page."
    assert result.usage.model == "simulated-vision"
    assert client.spent_usd == pytest.approx(result.usage.cost_usd) and client.spent_usd > 0
    assert len(client.calls) == 1


def test_the_image_estimate_can_stop_a_call_before_it_is_paid() -> None:
    client = SimulatedClient(pricing=Pricing(1_000.0, 1_000.0), cap_usd=0.01)
    with pytest.raises(BudgetExceeded):
        run(client)
    assert client.spent_usd == 0 and not client.calls


def test_the_live_client_sends_the_image_to_the_vision_model_and_pays_its_price(
    tmp_path: Path,
) -> None:
    transport = Transport()
    client = groq(transport, reasoning_effort="low", log_path=tmp_path / "calls.jsonl")
    result = run(client)
    payload = transport.payloads[0]
    text_part, image_part = payload["messages"][0]["content"]
    assert payload["model"] == "vision-model" and "reasoning_effort" not in payload
    assert text_part["type"] == "text" and "What does the page show?" in text_part["text"]
    prefix, encoded = image_part["image_url"]["url"].split(",", 1)
    assert prefix == "data:image/jpeg;base64"
    assert base64.b64decode(encoded) == SCREENSHOT.read_bytes()
    expected = VISION_PRICE.cost(3_000, 50)
    assert result.usage.model == "vision-model"
    assert result.usage.cost_usd == pytest.approx(expected)
    assert client.spent_usd == pytest.approx(expected)
    entry = json.loads((tmp_path / "calls.jsonl").read_text().splitlines()[0])
    assert entry["model"] == "vision-model" and entry["outcome"] == "ok"


def test_a_text_call_on_the_same_client_is_unchanged() -> None:
    transport = Transport('{"observation": "x", "confidence": "low"}')
    client = groq(transport)
    client.complete("hello", VisionJudgement)
    payload = transport.payloads[0]
    assert payload["messages"][0]["content"] == "hello"
    assert payload["model"] == client.model != "vision-model"


def test_a_client_without_a_vision_model_refuses_images_without_calling_the_provider() -> None:
    transport = Transport()
    client = groq(transport, vision_model=None, vision_pricing=None)
    with pytest.raises(ToolFailure, match="no vision model"):
        run(client)
    assert not transport.payloads and client.spent_usd == 0


def test_the_vision_model_and_its_prices_go_together() -> None:
    with pytest.raises(ConfigurationError, match="together"):
        groq(Transport(), vision_pricing=None)


def test_the_environment_needs_prices_for_the_vision_model() -> None:
    env = {"LLM_MODE": "live", "LLM_PROVIDER": "groq", "LLM_API_KEY": "key"}
    with pytest.raises(ConfigurationError, match="VISION_PRICE"):
        client_from_env({**env, "VISION_MODEL": "vision-model"})
    priced = {**env, "VISION_MODEL": "vision-model"}
    priced |= {"VISION_PRICE_INPUT_PER_M": "2", "VISION_PRICE_OUTPUT_PER_M": "8"}
    client = client_from_env(priced, transport=Transport())
    assert client.vision_model == "vision-model" and client.vision_pricing == VISION_PRICE
    assert client_from_env(env, transport=Transport()).vision_model is None


def test_the_tool_works_end_to_end_with_the_live_client() -> None:
    transport = Transport()
    tool = VisionTool(groq(transport), allowed_roots=[SCREENSHOT.parent])
    result = tool.run(path=SCREENSHOT, question="What does the page show?")
    assert result.finding.confidence is Confidence.MEDIUM
    assert result.finding.is_hypothesis is True
    assert len(transport.payloads) == 1


def test_the_image_part_prints_without_its_bytes() -> None:
    assert "JFIF" not in repr(ImagePart("image/jpeg", b"\xff\xd8\xff\xe0JFIF"))


@pytest.mark.parametrize(
    "prices",
    [
        {"VISION_PRICE_INPUT_PER_M": "0.80"},
        {"VISION_PRICE_OUTPUT_PER_M": "4.00"},
        {
            "VISION_PRICE_INPUT_PER_M": "0.80",
            "VISION_PRICE_OUTPUT_PER_M": "4.00",
        },
    ],
)
def test_vision_prices_without_a_model_are_rejected(prices: dict[str, str]) -> None:
    env = {"LLM_MODE": "live", "LLM_PROVIDER": "groq", "LLM_API_KEY": "key"}
    with pytest.raises(ConfigurationError, match="VISION_MODEL"):
        client_from_env({**env, **prices}, transport=Transport())
