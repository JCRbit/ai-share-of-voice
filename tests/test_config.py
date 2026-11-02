import pytest

from ai_share_of_voice.config import (
    GROQ_BASE_URL,
    ConfigError,
    LlmMode,
    SearchProvider,
    load_settings,
)

FAKE_KEY = "gsk_test_not_a_real_key"


def live_env(**overrides: str) -> dict[str, str]:
    env = {"LLM_MODE": "live", "LLM_PROVIDER": "groq", "LLM_API_KEY": FAKE_KEY}
    env.update(overrides)
    return env


def test_an_empty_environment_runs_simulated_without_a_key():
    settings = load_settings({})

    assert settings.mode is LlmMode.SIMULATED
    assert settings.api_key is None
    assert settings.spend_cap_usd == 2.0
    assert settings.search_provider is SearchProvider.NONE


def test_live_mode_without_a_key_is_a_clear_error():
    with pytest.raises(ConfigError, match="LLM_API_KEY"):
        load_settings({"LLM_MODE": "live"})


def test_groq_gets_its_base_url_and_default_models():
    settings = load_settings(live_env())

    assert settings.base_url == GROQ_BASE_URL
    assert settings.subject_model
    assert settings.agent_model
    assert settings.vision_model


def test_explicit_models_override_the_defaults():
    settings = load_settings(
        live_env(SUBJECT_MODEL="subject-x", LLM_MODEL="agent-y", VISION_MODEL="vision-z")
    )

    assert (settings.subject_model, settings.agent_model, settings.vision_model) == (
        "subject-x",
        "agent-y",
        "vision-z",
    )


def test_blank_values_count_as_missing():
    # .env.example ships with empty values; they must not become empty model names.
    settings = load_settings({"LLM_MODE": "", "LLM_MODEL": "  ", "SPEND_CAP_USD": ""})

    assert settings.mode is LlmMode.SIMULATED
    assert settings.agent_model
    assert settings.spend_cap_usd == 2.0


@pytest.mark.parametrize("value", ["0", "-1", "abc"])
def test_an_invalid_spend_cap_is_rejected(value):
    with pytest.raises(ConfigError, match="SPEND_CAP_USD"):
        load_settings({"SPEND_CAP_USD": value})


def test_an_unknown_mode_is_rejected():
    with pytest.raises(ConfigError, match="LLM_MODE"):
        load_settings({"LLM_MODE": "production"})


def test_tavily_search_needs_its_own_key():
    with pytest.raises(ConfigError, match="SEARCH_API_KEY"):
        load_settings({"SEARCH_PROVIDER": "tavily"})


def test_the_key_never_appears_in_repr_or_errors():
    settings = load_settings(live_env())

    assert FAKE_KEY not in repr(settings)
    assert FAKE_KEY not in str(settings)
    assert settings.api_key.get_secret_value() == FAKE_KEY
    with pytest.raises(ConfigError) as error:
        load_settings(live_env(SPEND_CAP_USD="abc"))
    assert FAKE_KEY not in str(error.value)
