from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ai_share_of_voice.schemas import AnalyzedResponse, BrandMention, Tone
from ai_share_of_voice.stats.metrics import (
    METRICS,
    MIN_RELIABLE_QUESTIONS,
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


@pytest.mark.parametrize("second", ["glovo", " Glovo "])
def test_responses_to_frame_rejects_brands_that_collide_after_normalisation(second: str) -> None:
    # model_construct skips the schema check, as a producer that bypasses it would.
    response = AnalyzedResponse.model_construct(
        response_id="r1",
        question_id="q1",
        batch_index=0,
        mentions=(
            BrandMention(brand="Glovo", mentioned=True, recommended=True, position=1),
            BrandMention.model_construct(
                brand=second, mentioned=False, recommended=False, position=None, tone=None
            ),
        ),
    )
    with pytest.raises(ValueError, match="only once"):
        responses_to_frame([response], BRANDS)


def test_responses_to_frame_rejects_duplicated_tracked_brands() -> None:
    with pytest.raises(ValueError, match="tracked brands"):
        responses_to_frame([], ("Glovo", "glovo"))


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


def make_answers(mentions: dict[str, list[bool]]) -> pd.DataFrame:
    """One brand "A" and a never-mentioned brand "B"; keys are question ids."""
    rows = []
    for question_id, flags in mentions.items():
        for index, flag in enumerate(flags):
            response_id = f"{question_id}-{index}"
            for brand, mentioned in (("A", flag), ("B", False)):
                rows.append(
                    {
                        "response_id": response_id,
                        "question_id": question_id,
                        "batch_index": 0,
                        "brand": brand,
                        "mentioned": mentioned,
                        "recommended": False,
                        "position": 1 if mentioned else None,
                        "tone": None,
                    }
                )
    return pd.DataFrame(rows)


def mention_row(answers: pd.DataFrame, brand: str = "A") -> pd.Series:
    result = brand_metrics(answers).set_index(["brand", "metric"])
    return result.loc[(brand, "mention_rate")]


def test_every_answer_weighs_the_same() -> None:
    # q1 has 4 answers that all mention A, q2 has 2 that do not: 4 of 6 answers, not 1/2.
    row = mention_row(make_answers({"q1": [True] * 4, "q2": [False] * 2}))
    assert row["estimate"] == pytest.approx(4 / 6)


def test_never_mentioned_brand_keeps_a_non_degenerate_interval() -> None:
    answers = make_answers({f"q{i}": [True, False, False] for i in range(12)})
    row = mention_row(answers, brand="B")
    assert row["bootstrap_high"] == pytest.approx(0.0)
    assert row["ci_low"] == pytest.approx(0.0)
    assert row["ci_high"] == pytest.approx(row["wilson_high"]) and row["ci_high"] > 0


def test_headline_interval_covers_wilson_and_bootstrap(metrics: pd.DataFrame) -> None:
    binomial = metrics[metrics["ci_method"] == "wilson_bootstrap_hull"]
    assert (binomial["ci_low"] <= binomial[["wilson_low", "bootstrap_low"]].min(axis=1)).all()
    assert (binomial["ci_high"] >= binomial[["wilson_high", "bootstrap_high"]].max(axis=1)).all()
    other = metrics[metrics["ci_method"] == "cluster_bootstrap"]
    assert (other["ci_low"] == other["bootstrap_low"]).all()


def test_few_questions_are_flagged(metrics: pd.DataFrame) -> None:
    assert metrics["few_questions"].all()
    answers = make_answers({f"q{i}": [True, False] for i in range(MIN_RELIABLE_QUESTIONS)})
    assert not mention_row(answers)["few_questions"]


def test_brands_are_resampled_with_the_same_questions() -> None:
    # B mirrors A, so with shared resamples their intervals are mirror images.
    answers = make_answers({f"q{i}": [i % 3 == 0, True] for i in range(12)})
    answers.loc[answers["brand"] == "B", "mentioned"] = ~answers.loc[
        answers["brand"] == "A", "mentioned"
    ].to_numpy()
    answers.loc[answers["brand"] == "B", "position"] = None
    result = brand_metrics(answers).set_index(["brand", "metric"])
    a, b = result.loc[("A", "mention_rate")], result.loc[("B", "mention_rate")]
    assert a["bootstrap_low"] == pytest.approx(1 - b["bootstrap_high"])


def test_brand_names_are_matched_without_case() -> None:
    answers = load_answers(FIXTURE)
    answers.loc[answers["response_id"] == "r2", "brand"] = answers.loc[
        answers["response_id"] == "r2", "brand"
    ].str.lower()
    result = brand_metrics(answers)
    assert set(result["brand"]) == set(BRANDS)


def test_non_boolean_flags_are_rejected() -> None:
    answers = load_answers(FIXTURE).astype({"mentioned": str})
    with pytest.raises(ValueError, match="True/False"):
        brand_metrics(answers)


def test_unknown_tone_is_rejected() -> None:
    answers = load_answers(FIXTURE)
    answers.loc[0, "tone"] = "great"
    with pytest.raises(ValueError, match="unknown tone"):
        brand_metrics(answers)


def test_tone_is_normalised() -> None:
    answers = load_answers(FIXTURE)
    answers.loc[0, "tone"] = " Neutral "
    result = brand_metrics(answers).set_index(["brand", "metric"])
    assert result.loc[("Just Eat", "net_tone"), "estimate"] == pytest.approx(2 / 3)


def test_tone_without_mention_is_rejected() -> None:
    answers = load_answers(FIXTURE)
    answers.loc[2, "tone"] = "positive"
    with pytest.raises(ValueError, match="require the brand to be mentioned"):
        brand_metrics(answers)


def test_response_in_two_questions_is_rejected() -> None:
    answers = load_answers(FIXTURE)
    answers.loc[0, "question_id"] = "q9"
    with pytest.raises(ValueError, match="single question"):
        brand_metrics(answers)


def test_empty_answers_are_rejected() -> None:
    with pytest.raises(ValueError, match="no answers"):
        brand_metrics(load_answers(FIXTURE).iloc[0:0])


def test_invalid_resample_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="n_resamples"):
        cluster_bootstrap_interval(np.array([1, 2]), np.array([2, 2]), n_resamples=0)
