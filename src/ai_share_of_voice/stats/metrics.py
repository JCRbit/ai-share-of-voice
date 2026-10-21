"""Brand metrics and their confidence intervals (owner: E).

Every result is a dated sample of one model's answers, never a market share.

Each metric is a ratio of sums over answers (a ratio estimator): every answer weighs the
same. With the balanced sampling plan (same number of answers per question in a batch)
this equals the average over questions. Questions are the resampling unit, because
answers to the same question are correlated.
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
# Metrics whose numerator counts successes among 0/1 trials, where Wilson applies.
BINOMIAL_METRICS = frozenset(METRICS) - {"share_of_voice", "net_tone"}
TONE_SCORES = {Tone.POSITIVE.value: 1, Tone.NEUTRAL.value: 0, Tone.NEGATIVE.value: -1}
# Below this many questions the cluster bootstrap tends to be too narrow.
MIN_RELIABLE_QUESTIONS = 10


def _check_confidence(confidence: float) -> None:
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a proportion.

    Assumes the ``n`` trials are independent. Answers to the same question are not,
    so on its own this interval is too narrow; ``brand_metrics`` combines it with the
    cluster bootstrap.
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


def _resample_questions(n_questions: int, n_resamples: int, seed: int) -> np.ndarray:
    """Question indices drawn with replacement: one row per bootstrap resample."""
    if n_questions < 2:
        raise ValueError("the cluster bootstrap needs at least two questions")
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_questions, size=(n_resamples, n_questions))


def _percentile_interval(
    numerators: np.ndarray, denominators: np.ndarray, picks: np.ndarray, confidence: float
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap interval of each column's ratio of sums; rows are questions.

    Resamples with a zero denominator (e.g. a brand never mentioned, for the
    conversion) have no ratio and are left out. A column without any ratio gives NaN.
    """
    resampled_num = numerators[picks].sum(axis=1)
    resampled_den = denominators[picks].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(resampled_den > 0, resampled_num / resampled_den, np.nan)

    tail = (1 - confidence) / 2 * 100
    low = np.full(ratios.shape[1], np.nan)
    high = np.full(ratios.shape[1], np.nan)
    defined = ~np.isnan(ratios).all(axis=0)
    if defined.any():
        low[defined], high[defined] = np.nanpercentile(
            ratios[:, defined], [tail, 100 - tail], axis=0
        )
    return low, high


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
    _check_confidence(confidence)

    picks = _resample_questions(len(numerators), n_resamples, seed)
    low, high = _percentile_interval(numerators[:, None], denominators[:, None], picks, confidence)
    return float(low[0]), float(high[0])


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
    """Read analyzed answers saved with the columns of ``ANSWER_COLUMNS``.

    ``mentioned`` and ``recommended`` must be written as True/False.
    """
    return _validate(pd.read_csv(path, dtype={"response_id": str, "question_id": str}))


def _validate(answers: pd.DataFrame) -> pd.DataFrame:
    """Check the answer table and return a normalised copy.

    Brand names are stripped and matched without case ("glovo" and "Glovo" are one
    brand, spelled as first seen); tones are stripped and lower-cased.
    """
    if missing := set(ANSWER_COLUMNS) - set(answers.columns):
        raise ValueError(f"missing columns: {sorted(missing)}")
    if answers.empty:
        raise ValueError("there are no answers")
    for column in ("mentioned", "recommended"):
        if not pd.api.types.is_bool_dtype(answers[column]):
            raise ValueError(f"{column} must contain only True/False values")

    answers = answers.astype({"position": "Int64"})
    brand = answers["brand"].astype(str).str.strip()
    answers["brand"] = brand.groupby(brand.str.casefold()).transform("first")
    answers["tone"] = answers["tone"].astype("string").str.strip().str.lower()

    if answers.duplicated(["response_id", "brand"]).any():
        raise ValueError("each brand can appear only once per response")
    if answers.groupby("response_id")["question_id"].nunique().gt(1).any():
        raise ValueError("each response must belong to a single question")
    if answers.groupby("response_id")["brand"].nunique().lt(answers["brand"].nunique()).any():
        raise ValueError("every response must list every brand")
    if (answers["recommended"] & ~answers["mentioned"]).any():
        raise ValueError("a brand cannot be recommended without being mentioned")
    not_mentioned = ~answers["mentioned"]
    if (not_mentioned & (answers["position"].notna() | answers["tone"].notna())).any():
        raise ValueError("position and tone require the brand to be mentioned")
    if answers["position"].lt(1).any():
        raise ValueError("position starts at 1")
    if unknown := set(answers["tone"].dropna()) - set(TONE_SCORES):
        raise ValueError(f"unknown tone values: {sorted(unknown)}")
    return answers


def _metric_parts(answers: pd.DataFrame) -> dict[str, tuple[pd.Series, pd.Series]]:
    """Numerator and denominator of every metric, per answer and brand."""
    one = pd.Series(1, index=answers.index)
    mentioned = answers["mentioned"].astype(int)
    recommended = answers["recommended"].astype(int)
    tone = answers["tone"].map(TONE_SCORES).astype(float)
    return {
        "mention_rate": (mentioned, one),
        "recommendation_rate": (recommended, one),
        "top1_rate": (answers["position"].eq(1).fillna(False).astype(int), one),
        "share_of_voice": (mentioned, mentioned.groupby(answers["response_id"]).transform("sum")),
        "recommendation_conversion": (recommended, mentioned),
        "net_tone": (tone.fillna(0), tone.notna().astype(int)),
    }


def brand_metrics(
    answers: pd.DataFrame,
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """Estimate and intervals of every metric in ``METRICS`` for every brand.

    ``ci_low``/``ci_high`` is the headline interval:

    - binomial metrics: the hull of Wilson and the cluster bootstrap, so it is never
      narrower than either (the bootstrap alone collapses to a point when a brand is
      never or always mentioned, and Wilson alone ignores the clustering);
    - share of voice and net tone: the cluster bootstrap.

    Every brand and metric uses the same question resamples, so brands are compared on
    the same simulated panels. ``few_questions`` flags runs with fewer than
    ``MIN_RELIABLE_QUESTIONS`` questions, where the intervals are too optimistic.
    """
    _check_confidence(confidence)
    answers = _validate(answers)
    questions = pd.unique(answers["question_id"])
    brands = pd.unique(answers["brand"])
    picks = _resample_questions(len(questions), n_resamples, seed)
    keys = [answers["question_id"], answers["brand"]]

    rows = []
    for metric, (numerator, denominator) in _metric_parts(answers).items():
        num, den = (
            values.groupby(keys)
            .sum()
            .unstack("brand")
            .reindex(index=questions, columns=brands)
            .to_numpy(dtype=float)
            for values in (numerator, denominator)
        )
        boot_low, boot_high = _percentile_interval(num, den, picks, confidence)
        for column, brand in enumerate(brands):
            total_num, total_den = num[:, column].sum(), den[:, column].sum()
            wilson = (
                wilson_interval(int(total_num), int(total_den), confidence)
                if metric in BINOMIAL_METRICS and total_den > 0
                else (math.nan, math.nan)
            )
            rows.append(
                {
                    "brand": brand,
                    "metric": metric,
                    "estimate": total_num / total_den if total_den > 0 else math.nan,
                    "ci_low": float(np.fmin(wilson[0], boot_low[column])),
                    "ci_high": float(np.fmax(wilson[1], boot_high[column])),
                    "ci_method": (
                        "wilson_bootstrap_hull"
                        if metric in BINOMIAL_METRICS
                        else "cluster_bootstrap"
                    ),
                    "numerator": total_num,
                    "denominator": total_den,
                    "n_questions": len(questions),
                    "few_questions": len(questions) < MIN_RELIABLE_QUESTIONS,
                    "wilson_low": wilson[0],
                    "wilson_high": wilson[1],
                    "bootstrap_low": float(boot_low[column]),
                    "bootstrap_high": float(boot_high[column]),
                }
            )
    return pd.DataFrame(rows)
