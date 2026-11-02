"""Model client with a shared spending cap and a usage log (I3, local integration).

It fits ``schemas.llm.ModelClient``: ``complete(prompt, schema)`` returns the parsed object
and its ``Usage``. Three rules hold for every call, live or simulated:

* **Reserve before calling.** The worst-case cost (estimated input plus the output limit) is
  reserved in the shared ``Budget`` first; if it does not fit, ``BudgetExceeded`` is raised
  and nothing is sent. Afterwards the reservation is replaced by the real cost.
* **Invalid answers are ``ValueError``.** Nodes already treat them as "skip the model part";
  API errors are ``ToolFailure`` so the graph closes the run in a partial state.
* **Prompts are not logged.** The usage log only has model, tokens and cost.

``answer(prompt)`` is the free-text call for the measured (subject) model: no JSON mode and no
system message, so the answer looks like what a real user would read.
"""

import base64
import json
import re
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, SecretStr, ValidationError

from ai_share_of_voice.config import LlmMode, Settings
from ai_share_of_voice.errors import ToolFailure
from ai_share_of_voice.exceptions import BudgetExceeded
from ai_share_of_voice.schemas.llm import Usage

TOOL_NAME = "llm_client"
DEFAULT_MAX_OUTPUT_TOKENS = 2048
CHARS_PER_TOKEN_ESTIMATE = 3  # deliberately low: overestimating input keeps the cap safe
SDK_MAX_RETRIES = 3  # the SDK retries 429 and 5xx with backoff, honouring Retry-After
SDK_TIMEOUT_SECONDS = 60.0
SIMULATED_COST_PER_CALL_USD = 0.001
IMAGE_TOKEN_ESTIMATE = 1500  # generous: Gemini bills a chart image at a few hundred tokens


@dataclass(frozen=True)
class Price:
    """US dollars per million tokens."""

    input_per_million: float
    output_per_million: float

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.input_per_million + output_tokens * self.output_per_million
        ) / 1_000_000


# Paid-tier prices, used by the cap even on free tiers. Check before a live run:
# https://groq.com/pricing · https://ai.google.dev/gemini-api/docs/pricing
PRICES: dict[str, Price] = {
    "llama-3.3-70b-versatile": Price(0.59, 0.79),
    "llama-3.1-8b-instant": Price(0.05, 0.08),
    "openai/gpt-oss-120b": Price(0.15, 0.75),
    "openai/gpt-oss-20b": Price(0.075, 0.30),
    "meta-llama/llama-4-scout-17b-16e-instruct": Price(0.11, 0.34),
    # Gemini thinks before answering: thinking tokens are billed as output.
    # 3.6/3.8 Flash prices double on 2027-01-01 (1.50 / 7.50); checked 2026-11-02.
    "gemini-3.8-flash": Price(0.75, 3.75),
    "gemini-3.6-flash": Price(0.75, 3.75),
    "gemini-3.5-flash-lite": Price(0.30, 2.50),
    # Deprecated: from 2026-12-01 its requests are routed to gemini-3.5-flash-lite.
    "gemini-2.5-flash": Price(0.30, 2.50),
    "gemini-2.5-flash-lite": Price(0.10, 0.40),
}


def price_of(model: str) -> Price:
    try:
        return PRICES[model]
    except KeyError:
        raise ValueError(
            f"no price for model '{model}': add it to PRICES in llm_client.py"
        ) from None


class InvalidModelResponse(ValueError):
    """The model answered, but not in the requested shape."""


class QuotaExhausted(ToolFailure):
    """The provider refuses calls until its quota resets (a daily or plan limit).

    Retrying or asking the next question cannot help, so callers stop at once.
    """


_QUOTA_HINTS = ("per day", "quota")  # not "billing": Groq links it on per-minute limits


def _is_rejected_json(error: Exception) -> bool:
    """Groq answers 400 json_validate_failed when the model's JSON is invalid."""
    return getattr(error, "status_code", None) == 400 and "json_validate_failed" in str(error)


def _is_quota_error(error: Exception) -> bool:
    text = str(error).casefold()
    return getattr(error, "status_code", None) == 429 and any(hint in text for hint in _QUOTA_HINTS)


class Budget:
    """Spending cap shared by every client of one run. Thread-safe."""

    def __init__(self, cap_usd: float) -> None:
        self.cap_usd = cap_usd
        self._spent = 0.0
        self._reserved = 0.0
        self._lock = threading.Lock()

    @property
    def spent_usd(self) -> float:
        return self._spent

    @property
    def committed_usd(self) -> float:
        """Settled spend plus the reservations of calls still in flight."""
        return self._spent + self._reserved

    def reserve(self, amount_usd: float) -> float:
        with self._lock:
            if self._spent + self._reserved + amount_usd > self.cap_usd:
                raise BudgetExceeded(spent_usd=self._spent, cap_usd=self.cap_usd)
            self._reserved += amount_usd
        return amount_usd

    def settle(self, reservation_usd: float, actual_usd: float) -> None:
        with self._lock:
            self._reserved -= reservation_usd
            self._spent += actual_usd


class UsageLog:
    """Appends one JSON line per call. No prompts or answers, only usage."""

    def __init__(self, path: Path | None, label: str) -> None:
        self.path = path
        self.label = label
        self._lock = threading.Lock()

    def write(self, usage: Usage, ok: bool, error: Exception | None = None) -> None:
        if self.path is None:
            return
        record: dict[str, Any] = {"label": self.label, "ok": ok, **usage.model_dump()}
        if error is not None:
            record["error"] = type(error).__name__
            record["status_code"] = getattr(error, "status_code", None)
        line = json.dumps(record)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")


_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def _parse[T: BaseModel](content: str | None, schema: type[T]) -> T:
    text = (content or "").strip()
    fenced = _FENCE.match(text)
    if fenced:
        text = fenced.group(1)
    try:
        return schema.model_validate_json(text)
    except ValidationError as error:
        raise InvalidModelResponse(
            f"answer does not match {schema.__name__}: {error.error_count()} errors"
        ) from None


def _text_of(content: Any) -> str:
    """The text of a message: a string, or the text parts of a multimodal message."""
    if isinstance(content, list):
        return " ".join(part.get("text", "") for part in content if part.get("type") == "text")
    return str(content)


def _json_instructions(schema: type[BaseModel]) -> str:
    return (
        "Reply with one JSON object only, no prose and no code fences. "
        f"It must validate against this JSON schema:\n{json.dumps(schema.model_json_schema())}"
    )


class LiveClient:
    """Calls an OpenAI-compatible chat API (Groq) for one model."""

    def __init__(
        self,
        model: str,
        budget: Budget,
        sdk: Any,
        *,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        temperature: float | None = None,
        usage_log: Path | None = None,
        label: str = "",
    ) -> None:
        self.model = model
        self.price = price_of(model)
        self.budget = budget
        self._sdk = sdk
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self._log = UsageLog(usage_log, label or model)

    def complete[T: BaseModel](self, prompt: str, schema: type[T]) -> tuple[T, Usage]:
        messages = [
            {"role": "system", "content": _json_instructions(schema)},
            {"role": "user", "content": prompt},
        ]
        content, usage = self._call(messages, response_format={"type": "json_object"})
        return _parse(content, schema), usage

    def answer(self, prompt: str) -> tuple[str, Usage]:
        content, usage = self._call([{"role": "user", "content": prompt}])
        return (content or "").strip(), usage

    def complete_with_image[T: BaseModel](
        self, prompt: str, image: Path, schema: type[T]
    ) -> tuple[T, Usage]:
        """Ask about a PNG image (the vision model). The image goes as a data URL."""
        encoded = base64.b64encode(Path(image).read_bytes()).decode("ascii")
        messages = [
            {"role": "system", "content": _json_instructions(schema)},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
                ],
            },
        ]
        content, usage = self._call(messages, images=1, response_format={"type": "json_object"})
        return _parse(content, schema), usage

    def _call(
        self, messages: list[dict[str, Any]], *, images: int = 0, **extra: Any
    ) -> tuple[str | None, Usage]:
        # Count the text only: base64 image data would inflate the estimate ~20 times.
        chars = sum(len(_text_of(message["content"])) for message in messages)
        estimated_input = chars // CHARS_PER_TOKEN_ESTIMATE + 1 + images * IMAGE_TOKEN_ESTIMATE
        reservation = self.budget.reserve(self.price.cost(estimated_input, self.max_output_tokens))
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": self.max_output_tokens,
            **extra,
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        try:
            response = self._sdk.chat.completions.create(**kwargs)
        except Exception as error:
            # The request may have been billed: charge the input estimate, never nothing.
            usage = self._usage(estimated_input, 0)
            self.budget.settle(reservation, usage.cost_usd)
            self._log.write(usage, ok=False, error=error)
            if _is_rejected_json(error):
                # The provider refused the model's output, not the request: an invalid answer.
                raise InvalidModelResponse(f"provider rejected the JSON answer: {error}") from None
            failure = QuotaExhausted if _is_quota_error(error) else ToolFailure
            raise failure(TOOL_NAME, f"{type(error).__name__}: {error}") from None

        usage = self._usage(response.usage.prompt_tokens, response.usage.completion_tokens)
        self.budget.settle(reservation, usage.cost_usd)
        self._log.write(usage, ok=True)
        return response.choices[0].message.content, usage

    def _usage(self, input_tokens: int, output_tokens: int) -> Usage:
        return Usage(
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=self.price.cost(input_tokens, output_tokens),
        )


Responder = Callable[[str], BaseModel]


class SimulatedClient:
    """No network. Answers come from responders; every call still goes through the budget."""

    model = "simulated"

    def __init__(
        self,
        budget: Budget,
        responders: Mapping[type[BaseModel], Responder] | None = None,
        text_responder: Callable[[str], str] | None = None,
        cost_per_call_usd: float = SIMULATED_COST_PER_CALL_USD,
    ) -> None:
        self.budget = budget
        self.responders = dict(responders or {})
        self.text_responder = text_responder or (lambda prompt: "Simulated answer.")
        self.cost_per_call_usd = cost_per_call_usd

    def complete[T: BaseModel](self, prompt: str, schema: type[T]) -> tuple[T, Usage]:
        responder = self.responders.get(schema)
        if responder is None:
            raise InvalidModelResponse(f"no simulated responder for {schema.__name__}")
        return schema.model_validate(responder(prompt).model_dump()), self._charge()

    def answer(self, prompt: str) -> tuple[str, Usage]:
        return self.text_responder(prompt), self._charge()

    def complete_with_image[T: BaseModel](
        self, prompt: str, image: Path, schema: type[T]
    ) -> tuple[T, Usage]:
        return self.complete(prompt, schema)

    def _charge(self) -> Usage:
        reservation = self.budget.reserve(self.cost_per_call_usd)
        self.budget.settle(reservation, self.cost_per_call_usd)
        return Usage(
            model=self.model, input_tokens=0, output_tokens=0, cost_usd=self.cost_per_call_usd
        )


@dataclass(frozen=True)
class ModelClients:
    """One client per role, all sharing one budget."""

    subject: LiveClient | SimulatedClient
    agent: LiveClient | SimulatedClient
    vision: LiveClient | SimulatedClient
    budget: Budget


SdkFactory = Callable[[str, str], Any]


def _openai_sdk(base_url: str, api_key: str) -> Any:
    from openai import OpenAI

    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        max_retries=SDK_MAX_RETRIES,
        timeout=SDK_TIMEOUT_SECONDS,
    )


def build_clients(
    settings: Settings,
    *,
    usage_log: Path | None = None,
    sdk_factory: SdkFactory = _openai_sdk,
    responders: Mapping[type[BaseModel], Responder] | None = None,
    text_responder: Callable[[str], str] | None = None,
) -> ModelClients:
    """Build the subject, agent and vision clients for one run, all on one budget."""
    budget = Budget(settings.spend_cap_usd)
    if settings.mode is LlmMode.SIMULATED:
        simulated = SimulatedClient(budget, responders, text_responder)
        return ModelClients(subject=simulated, agent=simulated, vision=simulated, budget=budget)

    sdks: dict[tuple[str, str], Any] = {}

    def client(
        model: str, base_url: str, key: SecretStr | None, label: str, temperature: float | None
    ) -> LiveClient:
        endpoint = (base_url, key.get_secret_value() if key else "")
        if endpoint not in sdks:  # one SDK (and connection pool) per endpoint
            sdks[endpoint] = sdk_factory(*endpoint)
        return LiveClient(
            model, budget, sdks[endpoint], temperature=temperature, usage_log=usage_log, label=label
        )

    return ModelClients(
        # Subject keeps the provider's default temperature: we measure what users get.
        subject=client(
            settings.subject_model,
            settings.subject_base_url,
            settings.subject_api_key,
            "subject",
            None,
        ),
        agent=client(settings.agent_model, settings.base_url, settings.api_key, "agent", 0.0),
        vision=client(
            settings.vision_model, settings.vision_base_url, settings.vision_api_key, "vision", 0.0
        ),
        budget=budget,
    )
