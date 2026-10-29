import pytest
from unittest.mock import patch, MagicMock
from pydantic import BaseModel
from ai_share_of_voice.llm_client import GroqClient, BudgetExceededError

class MockSchema(BaseModel):
    message: str
    confidence: float

@pytest.fixture
def mock_groq():
    with patch("ai_share_of_voice.llm_client.Groq") as mock:
        yield mock

def test_unknown_model_fails_fast(mock_groq):
    with patch("os.getenv", return_value="unknown-hacker-model"):
        with pytest.raises(ValueError, match="Unknown model configured: 'unknown-hacker-model'"):
            GroqClient(spend_cap=1.0)

def test_budget_rejection(mock_groq):
    # Set a tiny budget that can't cover the input prompt
    client = GroqClient(spend_cap=0.000001)
    with pytest.raises(BudgetExceededError, match="Insufficient budget"):
        client.complete("Analyze this text", MockSchema)

def test_schema_injection_and_success(mock_groq):
    mock_instance = mock_groq.return_value
    mock_response = MagicMock()
    mock_response.choices[0].message.content = '{"message": "success", "confidence": 0.99}'
    mock_response.usage.prompt_tokens = 100
    mock_response.usage.completion_tokens = 50
    mock_response.usage.total_tokens = 150
    mock_instance.chat.completions.create.return_value = mock_response

    client = GroqClient(spend_cap=1.0)
    result, usage = client.complete("Test prompt", MockSchema)

    # Verify structured parsing
    assert result.message == "success"
    assert result.confidence == 0.99
    
    # Verify accurate ledger reconciliation
    expected_cost = (100 / 1000.0) * client.rates["input"] + (50 / 1000.0) * client.rates["output"]
    assert usage.cost_usd == expected_cost
    assert client.current_spend == expected_cost

    # Verify schema injection
    call_args = mock_instance.chat.completions.create.call_args[1]
    messages = call_args["messages"]
    assert "schema" in messages[0]["content"]
    assert "confidence" in messages[0]["content"]
    assert call_args["max_tokens"] <= 4096

@pytest.mark.integration
def test_live_api():
    """
    Opt-in live integration test. Run with: pytest -m integration
    Requires valid LLM_API_KEY environment variable.
    """
    client = GroqClient(spend_cap=1.0)
    result, usage = client.complete('{"message": "success", "confidence": 0.5}', MockSchema)
    
    assert isinstance(result.message, str)
    assert isinstance(result.confidence, float)
    assert usage.total_tokens > 0
    assert client.current_spend > 0.0
