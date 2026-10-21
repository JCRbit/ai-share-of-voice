from collections.abc import Sequence

import pytest
from pydantic import ValidationError

from ai_share_of_voice.schemas.stopping import (
    BatchSummary,
    StopConfig,
    StopDecision,
    StopReason,
    StopRule,
)


def make_summary(**overrides: object) -> BatchSummary:
    data: dict[str, object] = {
        "batch_index": 0,
        "n_answers": 30,
        "brand_share": 0.2,
        "ci_low": 0.1,
        "ci_high": 0.3,
        "cumulative_spend_usd": 0.1,
    }
    data.update(overrides)
    return BatchSummary(**data)


def test_config_defaults() -> None:
    config = StopConfig(target_half_width=0.05, max_batches=10, max_spend_usd=2.0)
    assert config.min_batches == 2
    assert config.patience == 2


def test_config_rejects_min_above_max() -> None:
    with pytest.raises(ValidationError):
        StopConfig(target_half_width=0.05, min_batches=5, max_batches=3, max_spend_usd=2.0)


def test_half_width() -> None:
    assert make_summary().half_width == pytest.approx(0.1)


def test_share_must_lie_inside_interval() -> None:
    with pytest.raises(ValidationError):
        make_summary(brand_share=0.5)


def test_continue_decision_must_not_stop() -> None:
    with pytest.raises(ValidationError):
        StopDecision(stop=True, reason=StopReason.CONTINUE, explanation="x")


def test_stop_decision_needs_a_stop_reason() -> None:
    with pytest.raises(ValidationError):
        StopDecision(stop=False, reason=StopReason.MAX_BATCHES, explanation="x")


def test_decision_requires_an_explanation() -> None:
    with pytest.raises(ValidationError):
        StopDecision(stop=True, reason=StopReason.BUDGET_EXHAUSTED, explanation="")


def test_a_plain_function_satisfies_the_stop_rule_protocol() -> None:
    def always_continue(history: Sequence[BatchSummary], config: StopConfig) -> StopDecision:
        return StopDecision(stop=False, reason=StopReason.CONTINUE, explanation="not enough data")

    rule: StopRule = always_continue
    config = StopConfig(target_half_width=0.05, max_batches=10, max_spend_usd=2.0)
    decision = rule([make_summary()], config)
    assert decision.stop is False
