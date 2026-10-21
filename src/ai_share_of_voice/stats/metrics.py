"""Brand metrics and their confidence intervals (owner: E).

Every result is a dated sample of one model's answers, never a market share.
"""

import math
from statistics import NormalDist


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
