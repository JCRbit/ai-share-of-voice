from collections.abc import Sequence

import pytest
from pydantic import ValidationError

from ai_share_of_voice.schemas.stopping import (
    BatchSummary,
    IntervalMethod,
    StopConfig,
    StopDecision,
    StopReason,
    StopRule,
    SummaryMetric,
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


def test_config_reserves_part_of_the_cap_for_the_agents_by_default() -> None:
    config = StopConfig(target_half_width=0.05, max_batches=10, max_spend_usd=2.0)
    assert config.sampling_budget_share == pytest.approx(0.7)
    assert config.agent_reserve_usd == pytest.approx(0.2)
    assert config.sampling_budget_usd == pytest.approx(1.4)


def test_the_minimum_reserve_wins_when_the_cap_is_small() -> None:
    # 70% of 0.50 would leave 0.15 for the agents, below the 0.20 minimum.
    config = StopConfig(target_half_width=0.05, max_batches=10, max_spend_usd=0.5)
    assert config.sampling_budget_usd == pytest.approx(0.3)


def test_the_sampling_budget_is_never_negative() -> None:
    config = StopConfig(target_half_width=0.05, max_batches=10, max_spend_usd=0.1)
    assert config.sampling_budget_usd == 0.0


def test_the_whole_cap_can_go_to_sampling_when_asked() -> None:
    config = StopConfig(
        target_half_width=0.05,
        max_batches=10,
        max_spend_usd=2.0,
        sampling_budget_share=1.0,
        agent_reserve_usd=0.0,
    )
    assert config.sampling_budget_usd == pytest.approx(2.0)


@pytest.mark.parametrize(
    "overrides",
    [
        {"sampling_budget_share": 0.0},
        {"sampling_budget_share": 1.2},
        {"agent_reserve_usd": -0.1},
    ],
)
def test_invalid_budget_split_is_rejected(overrides: dict[str, float]) -> None:
    with pytest.raises(ValidationError):
        StopConfig(target_half_width=0.05, max_batches=10, max_spend_usd=2.0, **overrides)


def test_config_rejects_min_above_max() -> None:
    with pytest.raises(ValidationError):
        StopConfig(target_half_width=0.05, min_batches=5, max_batches=3, max_spend_usd=2.0)


def test_half_width() -> None:
    assert make_summary().half_width == pytest.approx(0.1)


def test_share_must_lie_inside_interval() -> None:
    with pytest.raises(ValidationError):
        make_summary(brand_share=0.5)


def test_summary_defaults_describe_the_headline_mention_rate() -> None:
    summary = make_summary()
    assert summary.metric is SummaryMetric.MENTION_RATE
    assert summary.interval_method is IntervalMethod.WILSON_BOOTSTRAP_HULL
    assert summary.confidence == pytest.approx(0.95)
    assert summary.n_questions is None


def test_summary_accepts_the_metric_names_of_the_stats_module() -> None:
    summary = make_summary(
        metric="share_of_voice", interval_method="cluster_bootstrap", n_questions=12
    )
    assert summary.metric is SummaryMetric.SHARE_OF_VOICE
    assert summary.n_questions == 12


def test_net_tone_cannot_be_summarised_as_a_share() -> None:
    with pytest.raises(ValidationError):
        make_summary(metric="net_tone", interval_method="cluster_bootstrap")


@pytest.mark.parametrize(
    ("metric", "method"),
    [
        ("mention_rate", "cluster_bootstrap"),
        ("share_of_voice", "wilson_bootstrap_hull"),
    ],
)
def test_interval_method_must_match_the_metric(metric: str, method: str) -> None:
    with pytest.raises(ValidationError, match="interval"):
        make_summary(metric=metric, interval_method=method)


@pytest.mark.parametrize("confidence", [0.0, 1.0, 1.5])
def test_confidence_must_be_a_probability_strictly_between_zero_and_one(
    confidence: float,
) -> None:
    with pytest.raises(ValidationError):
        make_summary(confidence=confidence)


def test_questions_cannot_outnumber_answers() -> None:
    with pytest.raises(ValidationError, match="n_questions"):
        make_summary(n_answers=10, n_questions=11)


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
