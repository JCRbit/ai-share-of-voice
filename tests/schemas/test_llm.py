import pytest
from pydantic import BaseModel, ValidationError

from ai_share_of_voice.exceptions import BudgetExceeded
from ai_share_of_voice.schemas.llm import ModelClient, Usage


class Answer(BaseModel):
    text: str = "ok"


class FakeClient:
    def complete[T: BaseModel](self, prompt: str, schema: type[T]) -> tuple[T, Usage]:
        usage = Usage(model="fake", input_tokens=10, output_tokens=5, cost_usd=0.0)
        return schema(), usage


def test_usage_total_tokens() -> None:
    usage = Usage(model="fake", input_tokens=10, output_tokens=5, cost_usd=0.001)
    assert usage.total_tokens == 15


def test_usage_rejects_negative_values() -> None:
    with pytest.raises(ValidationError):
        Usage(model="fake", input_tokens=-1, output_tokens=0, cost_usd=0.0)
    with pytest.raises(ValidationError):
        Usage(model="fake", input_tokens=0, output_tokens=0, cost_usd=-0.1)


def test_a_fake_client_satisfies_the_model_client_protocol() -> None:
    client: ModelClient = FakeClient()
    answer, usage = client.complete("hello", Answer)
    assert answer.text == "ok"
    assert usage.total_tokens == 15


def test_budget_exceeded_exposes_spent_and_cap() -> None:
    error = BudgetExceeded(spent_usd=2.5, cap_usd=2.0)
    assert error.spent_usd == pytest.approx(2.5)
    assert error.cap_usd == pytest.approx(2.0)
    assert "cap" in str(error)
