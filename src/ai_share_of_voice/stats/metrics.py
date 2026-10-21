"""Brand metrics and their confidence intervals (owner: E).

Every result is a dated sample of one model's answers, never a market share.
"""

import math
from collections.abc import Iterable, Sequence
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from ai_share_of_voice.schemas import AnalyzedResponse, Tone

ANSWER_COLUMNS = (
    "response_id",
    "question_id",
    "batch_index",
    "brand",
    "mentioned",
    "recommended",
    "position",
    "tone",
)
METRICS = (
    "mention_rate",
    "recommendation_rate",
    "top1_rate",
    "share_of_voice",
    "recommendation_conversion",
    "net_tone",
)
# Metrics whose numerator counts successes among independent 0/1 trials.
BINOMIAL_METRICS = frozenset(METRICS) - {"share_of_voice", "net_tone"}
TONE_SCORES = {Tone.POSITIVE.value: 1, Tone.NEUTRAL.value: 0, Tone.NEGATIVE.value: -1}


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


def responses_to_frame(
    responses: Iterable[AnalyzedResponse], brands: Sequence[str]
) -> pd.DataFrame:
    """One row per answer and brand; a brand missing from an answer was not mentioned."""
    known = {brand.casefold() for brand in brands}
    rows = []
    for response in responses:
        by_brand = {mention.brand.casefold(): mention for mention in response.mentions}
        if unknown := set(by_brand) - known:
            raise ValueError(f"unknown brands in {response.response_id}: {sorted(unknown)}")
        for brand in brands:
            mention = by_brand.get(brand.casefold())
            rows.append(
                {
                    "response_id": response.response_id,
                    "question_id": response.question_id,
                    "batch_index": response.batch_index,
                    "brand": brand,
                    "mentioned": bool(mention and mention.mentioned),
                    "recommended": bool(mention and mention.recommended),
                    "position": mention.position if mention else None,
                    "tone": mention.tone.value if mention and mention.tone else None,
                }
            )
    return _validate(pd.DataFrame(rows, columns=ANSWER_COLUMNS))


def load_answers(path: Path | str) -> pd.DataFrame:
    """Read analyzed answers saved with the columns of ``ANSWER_COLUMNS``."""
    return _validate(pd.read_csv(path, dtype={"response_id": str, "question_id": str}))


def _validate(answers: pd.DataFrame) -> pd.DataFrame:
    if missing := set(ANSWER_COLUMNS) - set(answers.columns):
        raise ValueError(f"missing columns: {sorted(missing)}")
    answers = answers.astype({"mentioned": bool, "recommended": bool, "position": "Int64"})
    if answers.duplicated(["response_id", "brand"]).any():
        raise ValueError("each brand can appear only once per response")
    if answers.groupby("response_id")["brand"].nunique().lt(answers["brand"].nunique()).any():
        raise ValueError("every response must list every brand")
    if (answers["recommended"] & ~answers["mentioned"]).any():
        raise ValueError("a brand cannot be recommended without being mentioned")
    return answers


def brand_metrics(
    answers: pd.DataFrame,
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """Estimate and intervals of every metric in ``METRICS`` for every brand.

    Each metric is a ratio of sums over answers, so the bootstrap resamples questions
    and recomputes the ratio. Wilson is only reported for binomial metrics.
    """
    answers = _validate(answers)
    mentioned = answers["mentioned"].astype(int)
    recommended = answers["recommended"].astype(int)
    tone = answers["tone"].map(TONE_SCORES)
    parts = {
        "mention_rate": (mentioned, 1),
        "recommendation_rate": (recommended, 1),
        "top1_rate": (answers["position"].eq(1).fillna(False).astype(int), 1),
        "share_of_voice": (mentioned, mentioned.groupby(answers["response_id"]).transform("sum")),
        "recommendation_conversion": (recommended, mentioned),
        "net_tone": (tone.fillna(0), tone.notna().astype(int)),
    }

    rows = []
    for metric, (numerator, denominator) in parts.items():
        per_question = (
            pd.DataFrame(
                {
                    "brand": answers["brand"],
                    "question_id": answers["question_id"],
                    "num": numerator,
                    "den": denominator,
                }
            )
            .groupby(["brand", "question_id"], sort=False)[["num", "den"]]
            .sum()
        )
        for brand, group in per_question.groupby(level="brand", sort=False):
            num, den = group["num"].sum(), group["den"].sum()
            wilson = (
                wilson_interval(int(num), int(den), confidence)
                if metric in BINOMIAL_METRICS and den > 0
                else (math.nan, math.nan)
            )
            bootstrap = cluster_bootstrap_interval(
                group["num"].to_numpy(), group["den"].to_numpy(), confidence, n_resamples, seed
            )
            rows.append(
                {
                    "brand": brand,
                    "metric": metric,
                    "estimate": num / den if den > 0 else math.nan,
                    "numerator": float(num),
                    "denominator": float(den),
                    "n_questions": len(group),
                    "wilson_low": wilson[0],
                    "wilson_high": wilson[1],
                    "bootstrap_low": bootstrap[0],
                    "bootstrap_high": bootstrap[1],
                }
            )
    return pd.DataFrame(rows)
