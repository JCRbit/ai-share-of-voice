"""Model client contract (owner: An).

``complete(prompt, schema)`` returns the parsed object and the usage of the call.
It raises ``ai_share_of_voice.exceptions.BudgetExceeded`` when the spending cap
would be exceeded.
"""

from typing import Protocol

from pydantic import BaseModel, Field

from ai_share_of_voice.schemas.base import Contract


class Usage(Contract):
    """Tokens and cost of one model call."""

    model: str = Field(min_length=1)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cost_usd: float = Field(ge=0)

    @property
    def total_tokens(self) -> int:
        """Input plus output tokens."""
        return self.input_tokens + self.output_tokens


class ModelClient(Protocol):
    """Signature of the model client; the real client and the simulated one both fit."""

    def complete[T: BaseModel](self, prompt: str, schema: type[T]) -> tuple[T, Usage]:
        """Run ``prompt`` and return the answer parsed as ``schema`` with its usage."""
        ...
