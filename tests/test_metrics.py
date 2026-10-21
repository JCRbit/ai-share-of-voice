from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ai_share_of_voice.schemas import AnalyzedResponse, BrandMention, Tone
from ai_share_of_voice.stats.metrics import (
    METRICS,
    brand_metrics,
    cluster_bootstrap_interval,
    load_answers,
    responses_to_frame,
    wilson_interval,
)


@pytest.mark.parametrize(
    ("successes", "n", "expected"),
    [
        (5, 10, (0.2366, 0.7634)),
        (0, 10, (0.0, 0.2775)),
        (10, 10, (0.7225, 1.0)),
        (3, 6, (0.1876, 0.8124)),
    ],
)
def test_wilson_matches_known_values(successes: int, n: int, expected: tuple[float, float]) -> None:
    low, high = wilson_interval(successes, n)
    assert low == pytest.approx(expected[0], abs=1e-4)
    assert high == pytest.approx(expected[1], abs=1e-4)


def test_wilson_is_wider_with_higher_confidence() -> None:
    low_95, high_95 = wilson_interval(5, 10)
    low_99, high_99 = wilson_interval(5, 10, confidence=0.99)
    assert low_99 < low_95 and high_99 > high_95


@pytest.mark.parametrize(("successes", "n"), [(0, 0), (-1, 10), (11, 10)])
def test_wilson_rejects_invalid_counts(successes: int, n: int) -> None:
    with pytest.raises(ValueError):
        wilson_interval(successes, n)


@pytest.mark.parametrize("confidence", [0.0, 1.0, 1.5])
def test_wilson_rejects_invalid_confidence(confidence: float) -> None:
    with pytest.raises(ValueError):
        wilson_interval(5, 10, confidence=confidence)


def test_bootstrap_is_reproducible_with_the_same_seed() -> None:
    numerators = np.array([3, 0, 2, 1, 4])
    denominators = np.array([4, 4, 4, 4, 4])
    first = cluster_bootstrap_interval(numerators, denominators, seed=7)
    second = cluster_bootstrap_interval(numerators, denominators, seed=7)
    assert first == second


def test_bootstrap_contains_the_estimate() -> None:
    numerators = np.array([3, 0, 2, 1, 4, 2])
    denominators = np.array([4, 4, 4, 4, 4, 4])
    low, high = cluster_bootstrap_interval(numerators, denominators)
    assert low <= numerators.sum() / denominators.sum() <= high


def test_bootstrap_collapses_when_every_question_gives_the_same_ratio() -> None:
    low, high = cluster_bootstrap_interval(np.array([1, 1, 1]), np.array([2, 2, 2]))
    assert low == pytest.approx(0.5)
    assert high == pytest.approx(0.5)


def test_bootstrap_is_wider_than_wilson_when_answers_repeat_within_a_question() -> None:
    # 10 questions x 10 answers: half the questions always mention the brand, half never.
    numerators = np.array([10] * 5 + [0] * 5)
    denominators = np.full(10, 10)
    boot_low, boot_high = cluster_bootstrap_interval(numerators, denominators)
    wilson_low, wilson_high = wilson_interval(50, 100)
    assert boot_high - boot_low > 2 * (wilson_high - wilson_low)


def test_bootstrap_needs_at_least_two_questions() -> None:
    with pytest.raises(ValueError):
        cluster_bootstrap_interval(np.array([3]), np.array([4]))


def test_bootstrap_rejects_mismatched_inputs() -> None:
    with pytest.raises(ValueError):
        cluster_bootstrap_interval(np.array([1, 2]), np.array([2, 2, 2]))


def test_bootstrap_returns_nan_when_the_denominator_is_always_zero() -> None:
    low, high = cluster_bootstrap_interval(np.array([0, 0]), np.array([0, 0]))
    assert np.isnan(low) and np.isnan(high)


FIXTURE = Path(__file__).parent / "fixtures" / "sample_answers.csv"
BRANDS = ("Just Eat", "Glovo", "Uber Eats")


@pytest.fixture
def metrics() -> pd.DataFrame:
    return brand_metrics(load_answers(FIXTURE)).set_index(["brand", "metric"])


@pytest.mark.parametrize(
    ("brand", "metric", "expected"),
    [
        ("Just Eat", "mention_rate", 3 / 6),
        ("Just Eat", "recommendation_rate", 2 / 6),
        ("Just Eat", "top1_rate", 2 / 6),
        ("Just Eat", "share_of_voice", 3 / 11),
        ("Just Eat", "recommendation_conversion", 2 / 3),
        ("Just Eat", "net_tone", 2 / 3),
        ("Glovo", "mention_rate", 5 / 6),
        ("Glovo", "recommendation_conversion", 3 / 5),
        ("Glovo", "net_tone", 3 / 5),
        ("Uber Eats", "top1_rate", 1 / 6),
        ("Uber Eats", "recommendation_conversion", 0.0),
        ("Uber Eats", "net_tone", -1 / 3),
    ],
)
def test_metrics_match_hand_computed_values(
    metrics: pd.DataFrame, brand: str, metric: str, expected: float
) -> None:
    assert metrics.loc[(brand, metric), "estimate"] == pytest.approx(expected)


def test_every_brand_has_all_six_metrics(metrics: pd.DataFrame) -> None:
    assert len(metrics) == len(BRANDS) * len(METRICS)


def test_share_of_voice_adds_up_to_one(metrics: pd.DataFrame) -> None:
    shares = metrics.xs("share_of_voice", level="metric")["estimate"]
    assert shares.sum() == pytest.approx(1.0)


def test_wilson_is_only_reported_for_binomial_metrics(metrics: pd.DataFrame) -> None:
    for metric in ("share_of_voice", "net_tone"):
        assert metrics.xs(metric, level="metric")["wilson_low"].isna().all()
    assert metrics.xs("mention_rate", level="metric")["wilson_low"].notna().all()


def test_bootstrap_reflects_mentions_concentrated_in_one_question(metrics: pd.DataFrame) -> None:
    # Every Just Eat mention comes from q2: without q2 in a resample its rate is 0.
    row = metrics.loc[("Just Eat", "mention_rate")]
    assert row["bootstrap_low"] == pytest.approx(0.0)
    assert row["wilson_low"] > 0.18


def test_responses_to_frame_matches_the_csv() -> None:
    responses = [
        AnalyzedResponse(
            response_id="r1",
            question_id="q1",
            batch_index=0,
            mentions=(
                BrandMention(brand="Glovo", mentioned=True, recommended=True, position=1),
                BrandMention(brand="Just Eat", mentioned=True, position=2, tone=Tone.NEUTRAL),
            ),
        )
    ]
    frame = responses_to_frame(responses, BRANDS)
    assert list(frame["brand"]) == list(BRANDS)
    assert list(frame["mentioned"]) == [True, True, False]
    assert frame.loc[2, "tone"] is None or pd.isna(frame.loc[2, "tone"])


def test_responses_to_frame_rejects_unknown_brands() -> None:
    response = AnalyzedResponse(
        response_id="r1",
        question_id="q1",
        batch_index=0,
        mentions=(BrandMention(brand="Deliveroo", mentioned=True),),
    )
    with pytest.raises(ValueError, match="unknown"):
        responses_to_frame([response], BRANDS)


def test_every_response_must_list_every_brand() -> None:
    answers = load_answers(FIXTURE).iloc[1:]
    with pytest.raises(ValueError, match="every brand"):
        brand_metrics(answers)


def test_recommendation_without_mention_is_rejected() -> None:
    answers = load_answers(FIXTURE)
    answers.loc[2, "recommended"] = True
    with pytest.raises(ValueError, match="mentioned"):
        brand_metrics(answers)


def test_missing_columns_are_rejected() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        brand_metrics(load_answers(FIXTURE).drop(columns="tone"))


def test_duplicated_brand_in_a_response_is_rejected() -> None:
    answers = load_answers(FIXTURE)
    with pytest.raises(ValueError, match="only once"):
        brand_metrics(pd.concat([answers, answers.iloc[[0]]]))
