"""Brand metrics and their confidence intervals (owner: E).

Every result is a dated sample of one model's answers, never a market share.
"""

import math
from statistics import NormalDist

import numpy as np


def _check_confidence(confidence: float) -> None:
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a proportion.

    Assumes the ``n`` trials are independent. Answers to the same question are not,
    so this is the reference interval; the headline one is the cluster bootstrap.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= successes <= n:
        raise ValueError("successes must be between 0 and n")
    _check_confidence(confidence)

    z = NormalDist().inv_cdf(1 - (1 - confidence) / 2)
    p = successes / n
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def cluster_bootstrap_interval(
    numerators: np.ndarray,
    denominators: np.ndarray,
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap interval of ``sum(numerators) / sum(denominators)``.

    Each position is one question (a cluster): its numerator and denominator are the
    totals over all its answers. Resampling whole questions keeps the correlation
    between answers to the same question, which Wilson ignores.
    """
    numerators = np.asarray(numerators, dtype=float)
    denominators = np.asarray(denominators, dtype=float)
    if numerators.shape != denominators.shape or numerators.ndim != 1:
        raise ValueError("numerators and denominators must be 1-D arrays of the same length")
    if len(numerators) < 2:
        raise ValueError("the cluster bootstrap needs at least two questions")
    _check_confidence(confidence)

    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(numerators), size=(n_resamples, len(numerators)))
    resampled_den = denominators[picks].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = numerators[picks].sum(axis=1) / resampled_den
    ratios = ratios[resampled_den > 0]
    if ratios.size == 0:
        return math.nan, math.nan

    tail = (1 - confidence) / 2 * 100
    low, high = np.percentile(ratios, [tail, 100 - tail])
    return float(low), float(high)
