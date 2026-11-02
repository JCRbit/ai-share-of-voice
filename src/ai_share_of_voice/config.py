"""Runtime settings read from the environment (``.env`` in development).

One place decides which provider, models, keys and cap a run uses. The measured (subject)
model and the vision model may live on another provider than the agents. A role on another
provider reuses a key already given for that provider, or needs its own. Two rules:

* ``simulated`` is the default, so tests and CI never need a key or spend money.
* Keys are ``SecretStr``: they never show up in ``repr``, logs or error messages.
"""

import os
from collections.abc import Mapping
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEFAULT_SPEND_CAP_USD = 2.0


class ConfigError(Exception):
    """Raised when the environment does not describe a valid configuration."""


class LlmMode(StrEnum):
    SIMULATED = "simulated"
    LIVE = "live"


class Provider(StrEnum):
    GROQ = "groq"
    GEMINI = "gemini"


# Both expose an OpenAI-compatible chat API, so one client serves them.
BASE_URLS = {Provider.GROQ: GROQ_BASE_URL, Provider.GEMINI: GEMINI_BASE_URL}

# Model lists change often; check them before a live run:
# https://console.groq.com/docs/models · https://ai.google.dev/gemini-api/docs/models
DEFAULT_MODELS = {
    Provider.GROQ: {
        # Groq retired the Llama 3.3 and Llama 4 models in 2026 (checked 2026-11-02).
        "subject": "openai/gpt-oss-20b",
        "agent": "openai/gpt-oss-120b",
        # No vision model on Groq at that date: use VISION_PROVIDER=gemini.
        "vision": "meta-llama/llama-4-scout-17b-16e-instruct",
    },
    # Pinned version, never an alias such as gemini-flash-latest: the measured model must
    # not change between runs without us knowing.
    Provider.GEMINI: {
        "subject": "gemini-3.8-flash",
        "agent": "gemini-3.8-flash",
        "vision": "gemini-3.8-flash",
    },
}


class SearchProvider(StrEnum):
    NONE = "none"  # fixture sources only, no network
    DUCKDUCKGO = "duckduckgo"  # no key needed; unofficial, may rate-limit
    TAVILY = "tavily"


class Settings(BaseModel):
    """Validated configuration of one process. Built only by ``load_settings``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: LlmMode
    provider: Provider
    api_key: SecretStr | None
    base_url: str
    subject_provider: Provider  # the model we measure, possibly on another provider
    subject_api_key: SecretStr | None
    subject_base_url: str
    subject_model: str = Field(min_length=1)
    agent_model: str = Field(min_length=1)  # the model that runs the agents
    vision_provider: Provider
    vision_api_key: SecretStr | None
    vision_base_url: str
    vision_model: str = Field(min_length=1)
    spend_cap_usd: float = Field(gt=0)
    search_provider: SearchProvider
    search_api_key: SecretStr | None


def _read(env: Mapping[str, str], name: str) -> str | None:
    """Return the stripped value, or ``None`` when it is missing or blank."""
    value = env.get(name, "").strip()
    return value or None


def _choice[E: StrEnum](env: Mapping[str, str], name: str, enum: type[E], default: E) -> E:
    raw = _read(env, name)
    if raw is None:
        return default
    try:
        return enum(raw.lower())
    except ValueError:
        allowed = ", ".join(member.value for member in enum)
        raise ConfigError(f"{name} must be one of: {allowed}") from None


def _spend_cap(env: Mapping[str, str]) -> float:
    raw = _read(env, "SPEND_CAP_USD")
    if raw is None:
        return DEFAULT_SPEND_CAP_USD
    try:
        cap = float(raw)
    except ValueError:
        raise ConfigError("SPEND_CAP_USD must be a number of US dollars") from None
    if cap <= 0:
        raise ConfigError("SPEND_CAP_USD must be greater than zero")
    return cap


def _secret(env: Mapping[str, str], name: str) -> SecretStr | None:
    raw = _read(env, name)
    return SecretStr(raw) if raw else None


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    """Build the settings from ``env``, or from ``.env`` plus the process environment."""
    if env is None:
        from dotenv import load_dotenv

        load_dotenv()
        env = os.environ

    mode = _choice(env, "LLM_MODE", LlmMode, LlmMode.SIMULATED)
    provider = _choice(env, "LLM_PROVIDER", Provider, Provider.GROQ)
    api_key = _secret(env, "LLM_API_KEY")
    if mode is LlmMode.LIVE and api_key is None:
        raise ConfigError("LLM_MODE=live needs LLM_API_KEY in .env")

    keys: dict[Provider, SecretStr] = {provider: api_key} if api_key else {}

    def role_key(role: str, role_provider: Provider) -> SecretStr | None:
        """The role's own key, else a key already given for the same provider."""
        key = _secret(env, f"{role}_API_KEY") or keys.get(role_provider)
        if key is None and mode is LlmMode.LIVE:
            raise ConfigError(
                f"{role}_PROVIDER={role_provider} has no key: set {role}_API_KEY in .env"
            )
        if key is not None:
            keys.setdefault(role_provider, key)
        return key

    subject_provider = _choice(env, "SUBJECT_PROVIDER", Provider, provider)
    subject_api_key = role_key("SUBJECT", subject_provider)
    vision_provider = _choice(env, "VISION_PROVIDER", Provider, provider)
    vision_api_key = role_key("VISION", vision_provider)

    search_provider = _choice(env, "SEARCH_PROVIDER", SearchProvider, SearchProvider.NONE)
    search_api_key = _secret(env, "SEARCH_API_KEY")
    if search_provider is SearchProvider.TAVILY and search_api_key is None:
        raise ConfigError("SEARCH_PROVIDER=tavily needs SEARCH_API_KEY in .env")

    try:
        return Settings(
            mode=mode,
            provider=provider,
            api_key=api_key,
            base_url=_read(env, "LLM_BASE_URL") or BASE_URLS[provider],
            subject_provider=subject_provider,
            subject_api_key=subject_api_key,
            subject_base_url=_read(env, "SUBJECT_BASE_URL") or BASE_URLS[subject_provider],
            subject_model=_read(env, "SUBJECT_MODEL")
            or DEFAULT_MODELS[subject_provider]["subject"],
            agent_model=_read(env, "LLM_MODEL") or DEFAULT_MODELS[provider]["agent"],
            vision_provider=vision_provider,
            vision_api_key=vision_api_key,
            vision_base_url=_read(env, "VISION_BASE_URL") or BASE_URLS[vision_provider],
            vision_model=_read(env, "VISION_MODEL") or DEFAULT_MODELS[vision_provider]["vision"],
            spend_cap_usd=_spend_cap(env),
            search_provider=search_provider,
            search_api_key=search_api_key,
        )
    except ValidationError as error:
        # Report only the field names: a validation message could echo a value.
        fields = ", ".join(str(item["loc"][0]) for item in error.errors())
        raise ConfigError(f"invalid settings: {fields}") from None
