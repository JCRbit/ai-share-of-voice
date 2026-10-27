from pydantic import BaseModel

from ai_share_of_voice.llm_client import GroqClient


class MockResponse(BaseModel):
    message: str


def test_groq_completion():
    """Ensure the Groq client successfully connects and validates Pydantic schemas."""
    client = GroqClient(spend_cap=1.0)

    # If the API fails, pytest will naturally catch the exception and fail the test
    result, usage = client.complete(prompt='{"message": "Hello, World!"}', schema=MockResponse)

    assert isinstance(result, MockResponse)
    assert usage.cost_usd > 0
    assert client.current_spend == usage.cost_usd
