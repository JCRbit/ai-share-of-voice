"""Model client with a spending cap and a call log (owner: An, task I3).

Every model call in the project goes through ``ModelClient.complete(prompt, schema)``
(see ``schemas/llm.py``). Two implementations share the same budget and logging rules:

- ``GroqClient``: the live client (Groq's OpenAI-compatible API, decision 3 in
  ``docs/decisions.md``). Used when ``LLM_MODE=live``.
- ``SimulatedClient``: makes no external call and, by default, costs nothing. CI and the
  mock run use it (``LLM_MODE=simulated``).

Rules enforced by both:

- Before each call the client estimates its worst-case cost (input estimate plus the
  maximum output tokens). If that cost could pass the cap, it raises ``BudgetExceeded``
  *before* spending anything. The graph turns that error into a partial close.
- Every paid call is appended to a JSONL log with tokens, cost and the running total. A
  call whose answer does not match the schema is still paid, so it is logged too.
- The optional per-person cap (decision 3) adds up the cost already in the log file, so it
  holds across runs.
- Provider failures and answers that never match the schema raise ``ToolFailure``.

Images go through the same rules: ``complete_with_image(prompt, image, schema)`` uses the
vision model and prices set with ``VISION_MODEL`` (it is a separate model from the text one,
so there is no default) and counts against the same cap and log. Without a vision model it
raises ``ToolFailure``.

Build the right client from the environment with ``client_from_env``.
"""

import json
import logging
import math
import os
import re
import threading
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

from filelock import FileLock
from pydantic import BaseModel, ValidationError

from ai_share_of_voice.errors import ToolFailure
from ai_share_of_voice.exceptions import BudgetExceeded, ConfigurationError
from ai_share_of_voice.schemas.llm import ImagePart, Usage

logger = logging.getLogger(__name__)

TOOL_NAME = "llm_client"
DEFAULT_MODEL = "openai/gpt-oss-120b"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_SPEND_CAP_USD = 2.00
DEFAULT_MAX_OUTPUT_TOKENS = 4096
CHARS_PER_TOKEN = 3  # deliberately low, so the input estimate errs on the expensive side
RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})
MAX_BACKOFF_S = 30.0
# Assumed cost of one image in input tokens, only for the pre-call budget check; the
# provider's own count is what gets charged and logged.
IMAGE_TOKEN_ESTIMATE = 2_048


@dataclass(frozen=True)
class Pricing:
    """Price of a model in USD per million tokens."""

    input_per_million_usd: float
    output_per_million_usd: float

    def __post_init__(self) -> None:
        prices = (self.input_per_million_usd, self.output_per_million_usd)
        if any(
            isinstance(price, bool)
            or not isinstance(price, (int, float))
            or not math.isfinite(price)
            or price < 0
            for price in prices
        ):
            raise ValueError("Prices must be finite and non-negative")

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        """Cost in USD of a call with these token counts."""
        tokens = (input_tokens, output_tokens)
        if any(type(count) is not int or count < 0 for count in tokens):
            raise ValueError("Token counts must be non-negative integers")

        result = (
            input_tokens * self.input_per_million_usd + output_tokens * self.output_per_million_usd
        ) / 1_000_000

        if not math.isfinite(result):
            raise ValueError("Calculated cost must be finite")

        return result


FREE = Pricing(0.0, 0.0)

# Groq list prices at the time of writing; override with LLM_PRICE_*_PER_M if they change.
KNOWN_PRICING: dict[str, Pricing] = {DEFAULT_MODEL: Pricing(0.15, 0.60)}


@dataclass(frozen=True)
class Generation:
    """What one round trip to a provider returns: the raw text and the tokens it used."""

    text: str
    input_tokens: int
    output_tokens: int

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise ValueError("Generation text must be a string")

        tokens = (self.input_tokens, self.output_tokens)
        if any(type(count) is not int or count < 0 for count in tokens):
            raise ValueError("Generation token counts must be non-negative integers")


class BudgetedClient(ABC):
    """Common behaviour: budget check, retry on invalid output, usage and call log.

    Subclasses only implement ``_generate``: one provider round trip.
    """

    def __init__(
        self,
        *,
        model: str,
        pricing: Pricing,
        cap_usd: float = DEFAULT_SPEND_CAP_USD,
        person_cap_usd: float | None = None,
        log_path: Path | str | None = None,
        run_id: str | None = None,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        max_attempts: int = 2,
        vision_model: str | None = None,
        vision_pricing: Pricing | None = None,
    ) -> None:

        if (
            isinstance(cap_usd, bool)
            or not isinstance(cap_usd, (int, float))
            or not math.isfinite(cap_usd)
            or cap_usd <= 0
        ):
            raise ConfigurationError("The spending cap must be finite and greater than zero")

        if person_cap_usd is not None and (
            isinstance(person_cap_usd, bool)
            or not isinstance(person_cap_usd, (int, float))
            or not math.isfinite(person_cap_usd)
            or person_cap_usd <= 0
        ):
            raise ConfigurationError("The per-person cap must be finite and greater than zero")

        if person_cap_usd is not None and log_path is None:
            raise ConfigurationError("The per-person cap needs a log file to add up past spend")

        if (
            type(max_output_tokens) is not int
            or max_output_tokens < 1
            or type(max_attempts) is not int
            or max_attempts < 1
        ):
            raise ConfigurationError("max_output_tokens and max_attempts must be positive integers")

        if (vision_model is None) != (vision_pricing is None):
            raise ConfigurationError("vision_model and vision_pricing must be set together")

        self.model = model
        self.pricing = pricing
        self.vision_model = vision_model
        self.vision_pricing = vision_pricing
        self.cap_usd = cap_usd
        self.person_cap_usd = person_cap_usd
        self.run_id = run_id
        self.max_output_tokens = max_output_tokens
        self.max_attempts = max_attempts
        self._log_path = Path(log_path) if log_path is not None else None
        self._lock = threading.Lock()
        self._call_lock = threading.Lock()
        self._spent_usd = 0.0
        self._calls: list[Usage] = []
        self._prior_person_usd = (
            read_logged_spend(self._log_path) if person_cap_usd is not None else 0.0
        )

    # -- public state ---------------------------------------------------------------------

    @property
    def spent_usd(self) -> float:
        """Spend of this client (this run) so far."""
        return self._spent_usd

    @property
    def remaining_usd(self) -> float:
        """What is left under the run cap."""
        return max(0.0, self.cap_usd - self._spent_usd)

    @property
    def calls(self) -> tuple[Usage, ...]:
        """Usage of every paid call made by this client, in order."""
        return tuple(self._calls)

    # -- the contract ---------------------------------------------------------------------

    def _complete[T: BaseModel](
        self, prompt: str, schema: type[T], image: ImagePart | None = None
    ) -> tuple[T, Usage]:
        """Run ``prompt`` (with ``image``, if given) and return the answer as ``schema``.

        The usage adds up all attempts when the first answer did not match the schema.
        Raises ``BudgetExceeded`` before spending if the cap could be passed, and
        ``ToolFailure`` if the provider fails or no attempt matches the schema.
        """
        model, pricing, image_tokens = self.model, self.pricing, 0
        if image is not None:
            if self.vision_model is None or self.vision_pricing is None:
                raise ToolFailure(
                    TOOL_NAME, "no vision model is configured; set VISION_MODEL and its prices"
                )
            model, pricing, image_tokens = (
                self.vision_model,
                self.vision_pricing,
                IMAGE_TOKEN_ESTIMATE,
            )
        schema_text = json.dumps(schema.model_json_schema(), separators=(",", ":"))
        current_prompt = prompt
        input_total = output_total = 0
        cost_total = 0.0
        last_error = "no attempt was made"

        for attempt in range(1, self.max_attempts + 1):
            self._check_budget(
                self._worst_case_cost(current_prompt, schema_text, pricing, image_tokens)
            )
            started = time.perf_counter()
            if image is None:
                generation = self._generate(current_prompt, schema)
            else:
                generation = self._generate_with_image(current_prompt, image, schema)
            cost = pricing.cost(generation.input_tokens, generation.output_tokens)
            input_total += generation.input_tokens
            output_total += generation.output_tokens
            cost_total += cost
            try:
                answer = _parse(generation.text, schema)
            except ValueError as error:
                last_error = str(error)
                self._record(generation, cost, attempt, "invalid_output", started, model)
                current_prompt = _with_feedback(prompt, last_error)
                continue
            self._record(generation, cost, attempt, "ok", started, model)
            usage = Usage(
                model=model,
                input_tokens=input_total,
                output_tokens=output_total,
                cost_usd=cost_total,
            )
            return answer, usage

        raise ToolFailure(
            TOOL_NAME,
            f"the answer did not match {schema.__name__} after {self.max_attempts} attempts "
            f"({last_error})",
        )

    def _run_complete[T: BaseModel](
        self, prompt: str, schema: type[T], image: ImagePart | None = None
    ) -> tuple[T, Usage]:
        """Serialize calls and enforce a shared per-person budget when configured."""
        with self._call_lock:
            if self.person_cap_usd is None:
                return self._complete(prompt, schema, image)

            assert self._log_path is not None
            self._log_path.parent.mkdir(parents=True, exist_ok=True)

            with FileLock(f"{self._log_path}.lock"):
                logged_spend = read_logged_spend(self._log_path)
                self._prior_person_usd = max(0.0, logged_spend - self._spent_usd)
                return self._complete(prompt, schema, image)

    def complete[T: BaseModel](self, prompt: str, schema: type[T]) -> tuple[T, Usage]:
        """Run a budget-checked completion."""
        return self._run_complete(prompt, schema)

    def complete_with_image[T: BaseModel](
        self, prompt: str, image: ImagePart, schema: type[T]
    ) -> tuple[T, Usage]:
        """Run a budget-checked completion with an image."""
        return self._run_complete(prompt, schema, image)

    # -- to implement ---------------------------------------------------------------------

    @abstractmethod
    def _generate(self, prompt: str, schema: type[BaseModel]) -> Generation:
        """One provider round trip. Raise ``ToolFailure`` if the provider cannot answer."""

    def _generate_with_image(
        self, prompt: str, image: ImagePart, schema: type[BaseModel]
    ) -> Generation:
        """One provider round trip with an image; clients that can see images override it."""
        raise ToolFailure(TOOL_NAME, f"{type(self).__name__} cannot analyze images")

    # -- budget and log -------------------------------------------------------------------

    def _worst_case_cost(
        self,
        prompt: str,
        schema_text: str,
        pricing: Pricing | None = None,
        image_tokens: int = 0,
    ) -> float:
        estimated_input = math.ceil((len(prompt) + len(schema_text)) / CHARS_PER_TOKEN)
        input_tokens = estimated_input + image_tokens
        return (pricing or self.pricing).cost(input_tokens, self.max_output_tokens)

    def _check_budget(self, worst_case_usd: float) -> None:
        with self._lock:
            if self._spent_usd + worst_case_usd > self.cap_usd:
                raise BudgetExceeded(self._spent_usd, self.cap_usd)
            if self.person_cap_usd is not None:
                person_spent = self._prior_person_usd + self._spent_usd
                if person_spent + worst_case_usd > self.person_cap_usd:
                    raise BudgetExceeded(person_spent, self.person_cap_usd)

    def _record(
        self,
        generation: Generation,
        cost: float,
        attempt: int,
        outcome: str,
        started: float,
        model: str | None = None,
    ) -> None:
        model = model or self.model
        usage = Usage(
            model=model,
            input_tokens=generation.input_tokens,
            output_tokens=generation.output_tokens,
            cost_usd=cost,
        )
        with self._lock:
            self._spent_usd += cost
            self._calls.append(usage)
            entry = {
                "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
                "run_id": self.run_id,
                "model": model,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cost_usd": round(cost, 8),
                "run_spent_usd": round(self._spent_usd, 8),
                "attempt": attempt,
                "outcome": outcome,
                "latency_s": round(time.perf_counter() - started, 3),
            }
            logger.info(
                "llm call %s: %d in, %d out, %.6f USD (run total %.6f of %.2f)",
                outcome,
                usage.input_tokens,
                usage.output_tokens,
                cost,
                self._spent_usd,
                self.cap_usd,
            )
            if self._log_path is not None:
                self._log_path.parent.mkdir(parents=True, exist_ok=True)
                with self._log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def read_logged_spend(log_path: Path) -> float:
    """Add up ``cost_usd`` over a call log; a missing file counts as zero spend."""
    if not log_path.exists():
        return 0.0

    total = 0.0
    with log_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue

            try:
                entry = json.loads(line)
                if not isinstance(entry, dict):
                    raise ValueError("Log entry must be a JSON object")

                raw_cost = entry.get("cost_usd", 0.0)
                if isinstance(raw_cost, bool) or not isinstance(raw_cost, (int, float)):
                    raise ValueError("Logged cost must be numeric")

                cost = float(raw_cost)
                if not math.isfinite(cost) or cost < 0:
                    raise ValueError("Logged cost must be finite and non-negative")

                new_total = total + cost
                if not math.isfinite(new_total):
                    raise ValueError("Accumulated spend must be finite")

                total = new_total

            except (ValueError, TypeError, AttributeError, OverflowError):
                logger.warning("ignoring a malformed line in %s", log_path)

    return total


_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)


def _parse[T: BaseModel](text: str, schema: type[T]) -> T:
    """Parse ``text`` as ``schema``; raise ``ValueError`` with a short reason if it fails."""
    clean = text.strip()
    fenced = _FENCE.match(clean)
    if fenced:
        clean = fenced.group(1)
    if not clean:
        raise ValueError("the model returned an empty answer")
    try:
        return schema.model_validate_json(clean)
    except ValidationError as error:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in item['loc']) or 'root'}: {item['msg']}"
            for item in error.errors()[:5]
        )
        raise ValueError(f"invalid JSON for {schema.__name__}: {problems}") from error


def _with_feedback(prompt: str, problem: str) -> str:
    return (
        f"{prompt}\n\nYour previous answer was rejected ({problem}). "
        "Answer again with only a JSON object that matches the schema."
    )


# -- simulated client ---------------------------------------------------------------------


Responder = Callable[[str, type[BaseModel]], BaseModel | str]


def _default_responder(prompt: str, schema: type[BaseModel]) -> BaseModel:
    try:
        return schema()
    except ValidationError as error:
        raise ToolFailure(
            TOOL_NAME,
            f"SimulatedClient cannot build {schema.__name__} from defaults; "
            "pass a responder that returns it",
        ) from error


class SimulatedClient(BudgetedClient):
    """Client with no external calls, for CI, the mock run and tests.

    ``responder(prompt, schema)`` returns the answer as a model or as raw JSON text; the
    default builds ``schema()`` from its defaults. Tokens are estimated from the text and
    priced with ``pricing`` (free by default), so tests can exercise the cap with a price.
    """

    def __init__(
        self,
        *,
        responder: Responder | None = None,
        pricing: Pricing = FREE,
        model: str = "simulated",
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("vision_model", "simulated-vision")
        kwargs.setdefault("vision_pricing", pricing)
        super().__init__(model=model, pricing=pricing, **kwargs)
        self._responder = responder or _default_responder

    def _generate(self, prompt: str, schema: type[BaseModel]) -> Generation:
        answer = self._responder(prompt, schema)
        text = answer if isinstance(answer, str) else answer.model_dump_json()
        return Generation(
            text=text,
            input_tokens=math.ceil(len(prompt) / CHARS_PER_TOKEN),
            output_tokens=math.ceil(len(text) / CHARS_PER_TOKEN),
        )

    def _generate_with_image(
        self, prompt: str, image: ImagePart, schema: type[BaseModel]
    ) -> Generation:
        text_only = self._generate(prompt, schema)
        return Generation(
            text=text_only.text,
            input_tokens=text_only.input_tokens + IMAGE_TOKEN_ESTIMATE,
            output_tokens=text_only.output_tokens,
        )


# -- live client (Groq) -------------------------------------------------------------------


class TransportError(Exception):
    """An HTTP exchange failed. ``status`` is ``None`` for network errors and timeouts."""

    def __init__(
        self, message: str, *, status: int | None = None, retry_after: float | None = None
    ) -> None:
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after

    @property
    def retryable(self) -> bool:
        """Network errors, timeouts, rate limits and server errors are worth another try."""
        return self.status is None or self.status in RETRYABLE_STATUS


Transport = Callable[[dict[str, Any]], dict[str, Any]]


def _parse_retry_after(value: str | None) -> float | None:
    """Parse Retry-After as seconds or an HTTP date."""
    if not value:
        return None

    try:
        delay = float(value)
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
            if retry_at.tzinfo is None:
                return None
            delay = (retry_at - datetime.now(UTC)).total_seconds()
        except (TypeError, ValueError, OverflowError):
            return None

    return max(0.0, delay) if math.isfinite(delay) else None


def _urllib_transport(api_key: str, url: str, timeout_s: float) -> Transport:
    def send(payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "ai-share-of-voice/0.1",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            retry_after = error.headers.get("Retry-After") if error.headers else None
            detail = error.read().decode("utf-8", errors="replace")[:300]
            raise TransportError(
                f"HTTP {error.code}: {detail}",
                status=error.code,
                retry_after=_parse_retry_after(retry_after),
            ) from error
        except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as e:
            raise TransportError(f"{type(e).__name__}: {e}") from e

    return send


def _safe_text(body: Any) -> str:
    """Return provider content only when it is a string."""
    try:
        content = body["choices"][0]["message"].get("content")
        return content if isinstance(content, str) else ""
    except (KeyError, IndexError, TypeError, AttributeError):
        return ""


class GroqClient(BudgetedClient):
    """Live client for Groq's OpenAI-compatible chat completions API.

    The API key is only used in the Authorization header and is never logged. Pass
    ``transport`` to replace the HTTP layer in tests.
    """

    def __init__(
        self,
        api_key: str,
        *,
        model: str = DEFAULT_MODEL,
        pricing: Pricing | None = None,
        transport: Transport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        max_retries: int = 3,
        backoff_s: float = 1.0,
        timeout_s: float = 60.0,
        reasoning_effort: str | None = None,
        **kwargs: Any,
    ) -> None:
        if not api_key and transport is None:
            raise ConfigurationError("LLM_API_KEY is empty")

        if type(max_retries) is not int or max_retries < 0:
            raise ConfigurationError("max_retries must be a non-negative integer")

        if (
            isinstance(backoff_s, bool)
            or not isinstance(backoff_s, (int, float))
            or not math.isfinite(backoff_s)
            or backoff_s < 0
        ):
            raise ConfigurationError("backoff_s must be finite and non-negative")

        if (
            isinstance(timeout_s, bool)
            or not isinstance(timeout_s, (int, float))
            or not math.isfinite(timeout_s)
            or timeout_s <= 0
        ):
            raise ConfigurationError("timeout_s must be finite and greater than zero")

        resolved = pricing or KNOWN_PRICING.get(model)
        if resolved is None:
            raise ConfigurationError(
                f"No price known for model '{model}'; set LLM_PRICE_INPUT_PER_M and "
                "LLM_PRICE_OUTPUT_PER_M"
            )
        super().__init__(model=model, pricing=resolved, **kwargs)
        self._transport = transport or _urllib_transport(api_key, GROQ_URL, timeout_s)
        self._sleep = sleep
        self._max_retries = max_retries
        self._backoff_s = backoff_s
        self._reasoning_effort = reasoning_effort

    def _payload(
        self, prompt: str, schema: type[BaseModel], image: ImagePart | None = None
    ) -> dict[str, Any]:
        content: Any = prompt
        model = self.model
        if image is not None:
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image.data_uri()}},
            ]
            model = self.vision_model
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "max_completion_tokens": self.max_output_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": re.sub(r"[^A-Za-z0-9_-]", "_", schema.__name__)[:64],
                    "schema": schema.model_json_schema(),
                    "strict": False,
                },
            },
        }
        if self._reasoning_effort and image is None:
            payload["reasoning_effort"] = self._reasoning_effort
        return payload

    def _generate(self, prompt: str, schema: type[BaseModel]) -> Generation:
        body = self._send_with_retries(self._payload(prompt, schema))
        return self._read(body, prompt)

    def _generate_with_image(
        self, prompt: str, image: ImagePart, schema: type[BaseModel]
    ) -> Generation:
        body = self._send_with_retries(self._payload(prompt, schema, image))
        return self._read(body, prompt, IMAGE_TOKEN_ESTIMATE)

    def _read(self, body: Any, prompt: str, extra_input_tokens: int = 0) -> Generation:
        try:
            content = body["choices"][0]["message"].get("content")

            if content is None:
                text = ""
            elif isinstance(content, str):
                text = content
            else:
                raise ValueError("Provider response content must be a string")
            if not isinstance(text, str):
                raise ValueError("Provider response content must be a string")
            usage = body.get("usage") or {}

            input_tokens = usage["prompt_tokens"]
            output_tokens = usage["completion_tokens"]

            if (
                type(input_tokens) is not int
                or input_tokens < 0
                or type(output_tokens) is not int
                or output_tokens < 0
            ):
                raise ValueError("Provider token counts must be non-negative integers")
        except (KeyError, IndexError, TypeError, ValueError, AttributeError):
            # No usable usage block: charge the worst case rather than nothing.
            logger.warning("provider response without usage; charging the estimate")
            input_tokens = math.ceil(len(prompt) / CHARS_PER_TOKEN) + extra_input_tokens
            output_tokens = self.max_output_tokens
            text = _safe_text(body)
        return Generation(text=text, input_tokens=input_tokens, output_tokens=output_tokens)

    def _send_with_retries(self, payload: dict[str, Any]) -> dict[str, Any]:
        for attempt in range(self._max_retries + 1):
            try:
                return self._transport(payload)
            except TransportError as error:
                if not error.retryable or attempt == self._max_retries:
                    raise ToolFailure(TOOL_NAME, f"provider call failed: {error}") from error
                delay = error.retry_after
                if delay is None:
                    delay = self._backoff_s * 2**attempt
                delay = min(delay, MAX_BACKOFF_S)
                logger.warning("provider error (%s); retrying in %.1fs", error, delay)
                self._sleep(delay)
        raise AssertionError("unreachable")  # pragma: no cover


# -- configuration ------------------------------------------------------------------------


def _env_float(env: Mapping[str, str], name: str, default: float | None) -> float | None:
    raw = (env.get(name) or "").strip()
    if not raw:
        return default

    try:
        value = float(raw)
    except ValueError as error:
        raise ConfigurationError(f"{name} must be a number, got '{raw}'") from error

    if not math.isfinite(value):
        raise ConfigurationError(f"{name} must be finite, got '{raw}'")

    return value


def client_from_env(
    env: Mapping[str, str] | None = None,
    *,
    run_id: str | None = None,
    log_path: Path | str | None = None,
    cap_usd: float | None = None,
    transport: Transport | None = None,
) -> BudgetedClient:
    """Build the client chosen by ``LLM_MODE`` (``simulated`` by default).

    Variables: ``LLM_MODE``, ``LLM_PROVIDER`` (only ``groq``), ``LLM_API_KEY``,
    ``LLM_MODEL``, ``SPEND_CAP_USD`` (per run), ``PERSON_CAP_USD`` (needs ``log_path``) and
    the optional price overrides ``LLM_PRICE_INPUT_PER_M`` / ``LLM_PRICE_OUTPUT_PER_M``.
    Image analysis needs ``VISION_MODEL`` with ``VISION_PRICE_INPUT_PER_M`` and
    ``VISION_PRICE_OUTPUT_PER_M``; the simulated client always has a free vision model.
    ``cap_usd`` overrides ``SPEND_CAP_USD``, for example with the Brief's cap.
    """
    if env is None:
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:  # pragma: no cover - python-dotenv is a requirement
            pass
        env = os.environ

    mode = (env.get("LLM_MODE") or "simulated").strip().lower()
    run_cap = cap_usd if cap_usd is not None else _env_float(env, "SPEND_CAP_USD", None)
    common: dict[str, Any] = {
        "cap_usd": run_cap if run_cap is not None else DEFAULT_SPEND_CAP_USD,
        "log_path": log_path,
        "run_id": run_id,
    }

    if mode == "simulated":
        return SimulatedClient(**common)
    if mode != "live":
        raise ConfigurationError(f"LLM_MODE must be 'simulated' or 'live', got '{mode}'")
    common["person_cap_usd"] = _env_float(env, "PERSON_CAP_USD", None)

    provider = (env.get("LLM_PROVIDER") or "").strip().lower()
    if provider != "groq":
        raise ConfigurationError(f"LLM_PROVIDER must be 'groq' in live mode, got '{provider}'")
    api_key = (env.get("LLM_API_KEY") or "").strip()
    if not api_key:
        raise ConfigurationError("LLM_API_KEY is required in live mode")
    model = (env.get("LLM_MODEL") or "").strip() or DEFAULT_MODEL
    price_in = _env_float(env, "LLM_PRICE_INPUT_PER_M", None)
    price_out = _env_float(env, "LLM_PRICE_OUTPUT_PER_M", None)
    pricing = None
    if price_in is not None and price_out is not None:
        try:
            pricing = Pricing(price_in, price_out)
        except ValueError as error:
            raise ConfigurationError(f"Invalid text pricing: {error}") from error
    elif price_in is not None or price_out is not None:
        raise ConfigurationError("Set both LLM_PRICE_INPUT_PER_M and LLM_PRICE_OUTPUT_PER_M")
    vision_model = (env.get("VISION_MODEL") or "").strip() or None
    vision_in = _env_float(env, "VISION_PRICE_INPUT_PER_M", None)
    vision_out = _env_float(env, "VISION_PRICE_OUTPUT_PER_M", None)
    vision_pricing = None

    if vision_model:
        if vision_in is None or vision_out is None:
            raise ConfigurationError(
                "VISION_MODEL needs VISION_PRICE_INPUT_PER_M and VISION_PRICE_OUTPUT_PER_M"
            )
        try:
            vision_pricing = Pricing(vision_in, vision_out)
        except ValueError as error:
            raise ConfigurationError(f"Invalid vision pricing: {error}") from error
    elif vision_in is not None or vision_out is not None:
        raise ConfigurationError("VISION_PRICE settings require VISION_MODEL")
    return GroqClient(
        api_key,
        model=model,
        pricing=pricing,
        transport=transport,
        vision_model=vision_model,
        vision_pricing=vision_pricing,
        **common,
    )
