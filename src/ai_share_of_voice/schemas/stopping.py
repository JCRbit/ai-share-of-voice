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

    ``max_spend_usd`` is the cap of the whole run. Sampling may only use part of it, so the
    agents that run after it (research, Skeptic, writer) still have money: at most
    ``sampling_budget_share`` of the cap, and always leaving ``agent_reserve_usd``.
    """

    target_half_width: float = Field(gt=0, le=0.5)
    min_batches: int = Field(default=2, ge=1)
    max_batches: int = Field(ge=1)
    max_spend_usd: float = Field(gt=0)
    marginal_gain_threshold: float = Field(default=0.005, ge=0)
    patience: int = Field(default=2, ge=1)
    sampling_budget_share: float = Field(default=0.7, gt=0, le=1)
    agent_reserve_usd: float = Field(default=0.2, ge=0)

    @property
    def sampling_budget_usd(self) -> float:
        """Spend available to sampling: the stricter of the share and the minimum reserve."""
        by_share = self.max_spend_usd * self.sampling_budget_share
        by_reserve = self.max_spend_usd - self.agent_reserve_usd
        return max(0.0, min(by_share, by_reserve))

    @model_validator(mode="after")
    def min_not_above_max(self) -> Self:
        if self.min_batches > self.max_batches:
            raise ValueError("min_batches cannot be greater than max_batches")
        return self


class SummaryMetric(StrEnum):
    """Metrics of ``stats/metrics.py`` that are shares in [0, 1].

    ``net_tone`` is left out: it ranges from -1 to 1, so it is not a share.
    """

    MENTION_RATE = "mention_rate"
    RECOMMENDATION_RATE = "recommendation_rate"
    TOP1_RATE = "top1_rate"
    SHARE_OF_VOICE = "share_of_voice"
    RECOMMENDATION_CONVERSION = "recommendation_conversion"


class IntervalMethod(StrEnum):
    """How ``ci_low``/``ci_high`` were computed; mirrors ``ci_method`` in ``stats/metrics.py``."""

    WILSON_BOOTSTRAP_HULL = "wilson_bootstrap_hull"
    CLUSTER_BOOTSTRAP = "cluster_bootstrap"


# Share of voice is not a count of successes among answers, so Wilson does not apply to it.
_BOOTSTRAP_ONLY_METRICS = frozenset({SummaryMetric.SHARE_OF_VOICE})


class BatchSummary(Contract):
    """Cumulative metrics for the focus brand after a batch (all batches so far).

    ``brand_share`` is the estimate of ``metric``. The defaults describe the headline
    figure (mention rate with the Wilson and bootstrap hull at 95%), so existing callers
    keep working. ``n_questions`` is the number of question clusters behind the interval;
    ``None`` means the producer did not report it.
    """

    batch_index: int = Field(ge=0)
    n_answers: int = Field(ge=1)
    brand_share: float = Field(ge=0, le=1)
    ci_low: float = Field(ge=0, le=1)
    ci_high: float = Field(ge=0, le=1)
    cumulative_spend_usd: float = Field(ge=0)
    metric: SummaryMetric = SummaryMetric.MENTION_RATE
    interval_method: IntervalMethod = IntervalMethod.WILSON_BOOTSTRAP_HULL
    confidence: float = Field(default=0.95, gt=0, lt=1)
    n_questions: int | None = Field(default=None, ge=1)

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

    @model_validator(mode="after")
    def interval_method_matches_metric(self) -> Self:
        expected = (
            IntervalMethod.CLUSTER_BOOTSTRAP
            if self.metric in _BOOTSTRAP_ONLY_METRICS
            else IntervalMethod.WILSON_BOOTSTRAP_HULL
        )
        if self.interval_method is not expected:
            raise ValueError(f"{self.metric.value} is reported with the {expected.value} interval")
        return self

    @model_validator(mode="after")
    def questions_do_not_outnumber_answers(self) -> Self:
        if self.n_questions is not None and self.n_questions > self.n_answers:
            raise ValueError("n_questions cannot be greater than n_answers")
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
