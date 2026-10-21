"""Stopping-rule contracts (owner: E).

``should_stop(history, config)`` receives the cumulative summary after each batch
and returns a ``StopDecision`` that always carries its reason.
"""

from collections.abc import Sequence
from enum import StrEnum
from typing import Protocol, Self

from pydantic import Field, model_validator

from ai_share_of_voice.schemas.base import Contract


class StopReason(StrEnum):
    """Why the sampling loop continues or stops."""

    CONTINUE = "continue"
    PRECISION_REACHED = "precision_reached"
    MARGINAL_RETURN = "marginal_return"
    BUDGET_EXHAUSTED = "budget_exhausted"
    MAX_BATCHES = "max_batches"


class StopConfig(Contract):
    """Limits of the sampling loop.

    ``marginal_return`` stops the loop when the confidence-interval half-width
    shrinks by less than ``marginal_gain_threshold`` for ``patience`` batches in a row.
    """

    target_half_width: float = Field(gt=0, le=0.5)
    min_batches: int = Field(default=2, ge=1)
    max_batches: int = Field(ge=1)
    max_spend_usd: float = Field(gt=0)
    marginal_gain_threshold: float = Field(default=0.005, ge=0)
    patience: int = Field(default=2, ge=1)

    @model_validator(mode="after")
    def min_not_above_max(self) -> Self:
        if self.min_batches > self.max_batches:
            raise ValueError("min_batches cannot be greater than max_batches")
        return self


class BatchSummary(Contract):
    """Cumulative metrics for the focus brand after a batch (all batches so far)."""

    batch_index: int = Field(ge=0)
    n_answers: int = Field(ge=1)
    brand_share: float = Field(ge=0, le=1)
    ci_low: float = Field(ge=0, le=1)
    ci_high: float = Field(ge=0, le=1)
    cumulative_spend_usd: float = Field(ge=0)

    @property
    def half_width(self) -> float:
        """Half of the confidence-interval width."""
        return (self.ci_high - self.ci_low) / 2

    @model_validator(mode="after")
    def interval_contains_share(self) -> Self:
        tolerance = 1e-9
        if not (self.ci_low - tolerance <= self.brand_share <= self.ci_high + tolerance):
            raise ValueError("brand_share must lie inside [ci_low, ci_high]")
        return self


class StopDecision(Contract):
    """Outcome of the stopping rule: continue or stop, always with a reason."""

    stop: bool
    reason: StopReason
    explanation: str = Field(min_length=1)

    @model_validator(mode="after")
    def reason_matches_decision(self) -> Self:
        if self.stop == (self.reason is StopReason.CONTINUE):
            raise ValueError("reason must be 'continue' if and only if stop is False")
        return self


class StopRule(Protocol):
    """Signature of ``should_stop``; lets the graph accept a fake in tests."""

    def __call__(self, history: Sequence[BatchSummary], config: StopConfig) -> StopDecision:
        """Decide whether to run another batch."""
        ...
