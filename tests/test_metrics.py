import numpy as np
import pytest

from ai_share_of_voice.stats.metrics import cluster_bootstrap_interval, wilson_interval


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
