import pytest

from ai_share_of_voice.stats.metrics import wilson_interval


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
