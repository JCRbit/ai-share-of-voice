"""Vetted charts for the report and the vision agent (owner: E).

Plain matplotlib functions with validated inputs: no model writes plotting code at run
time. They render off-screen (``Agg``) and save without variable metadata, so the same
input always gives the same PNG and a chart can be reproduced from the run folder.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

from ai_share_of_voice.schemas.stopping import BatchSummary  # noqa: E402
from ai_share_of_voice.stats.metrics import METRICS  # noqa: E402

REQUIRED_COLUMNS = ("brand", "metric", "estimate", "ci_low", "ci_high")
METRIC_LABELS = {
    "mention_rate": "Mention rate (share of answers that mention the brand)",
    "recommendation_rate": "Recommendation rate (share of answers that recommend it)",
    "top1_rate": "Top-1 rate (share of answers that name it first)",
    "share_of_voice": "Share of voice (share of all tracked mentions)",
    "recommendation_conversion": "Recommendation conversion (recommended when mentioned)",
    "net_tone": "Net tone (positive minus negative, -1 to 1)",
}
# Fixed size, resolution and metadata keep the PNG bytes identical for the same input.
_FIGSIZE = (7.0, 4.0)
_DPI = 120
_METADATA = {"Software": None}


def _check_png(path: Path) -> Path:
    path = Path(path)
    if path.suffix.lower() != ".png":
        raise ValueError(f"charts are written as .png files, got {path.name!r}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _save(figure: plt.Figure, path: Path) -> Path:
    try:
        figure.savefig(path, dpi=_DPI, metadata=_METADATA)
    finally:
        plt.close(figure)
    return path


def brand_comparison_chart(metrics: pd.DataFrame, metric: str, path: Path) -> Path:
    """Bar chart of one metric per brand, with its interval as an error bar.

    ``metrics`` is the table returned by ``brand_metrics``. A brand whose metric cannot be
    computed (``NaN``, e.g. recommendation conversion of a brand never mentioned) gets no
    bar and is marked "n/a", so it is not mistaken for a real 0.
    """
    path = _check_png(path)
    return _save(_brand_figure(metrics, metric), path)


def _brand_figure(metrics: pd.DataFrame, metric: str) -> plt.Figure:
    """Build the brand comparison figure; separate from saving so tests can inspect it."""
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}; expected one of {', '.join(METRICS)}")
    if missing := set(REQUIRED_COLUMNS) - set(metrics.columns):
        raise ValueError(f"missing columns: {sorted(missing)}")
    rows = metrics[metrics["metric"] == metric].reset_index(drop=True)
    if rows.empty:
        raise ValueError(f"the table has no rows for {metric!r}")

    defined = rows[rows["estimate"].notna()]
    undefined = rows[rows["estimate"].isna()]
    estimate = defined["estimate"]
    errors = [
        (estimate - defined["ci_low"].fillna(estimate)).clip(lower=0),
        (defined["ci_high"].fillna(estimate) - estimate).clip(lower=0),
    ]
    figure, axes = plt.subplots(figsize=_FIGSIZE)
    if not defined.empty:
        axes.bar(defined.index, estimate, yerr=errors, capsize=6, color="#4C72B0")
    for position in undefined.index:
        axes.text(position, 0.02, "n/a", ha="center", va="bottom", color="#555555")
    axes.set_xticks(rows.index, rows["brand"])
    axes.set_xlim(-0.6, len(rows) - 0.4)
    if metric == "net_tone":
        axes.set_ylim(-1.0, 1.0)
        axes.axhline(0.0, color="black", linewidth=0.8)
    else:
        axes.set_ylim(0.0, 1.0)
        axes.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    name, _, definition = METRIC_LABELS[metric].partition(" (")
    axes.set_ylabel(name)
    axes.set_title(f"{name} by brand (dated sample)")
    axes.grid(axis="y", alpha=0.3)

    notes = [(f"{name}: {definition.rstrip(')')}. Error bars: confidence interval.", "#555555")]
    if not undefined.empty:
        notes.append(
            (
                "n/a: the metric cannot be computed for that brand (e.g. it was never mentioned).",
                "#555555",
            )
        )
    if "few_questions" in rows and rows["few_questions"].any():
        n_questions = int(rows["n_questions"].iloc[0]) if "n_questions" in rows else None
        count = f"Only {n_questions} questions" if n_questions else "Few questions"
        notes.append(
            (f"{count}: intervals are not reliable, read the bars as indicative only.", "#C44E52")
        )
    for line, (text, color) in enumerate(reversed(notes)):
        figure.text(0.5, 0.01 + 0.04 * line, text, ha="center", fontsize=8, color=color)
    figure.tight_layout(rect=(0, 0.04 * len(notes) + 0.01, 1, 1))
    return figure


def convergence_chart(
    history: Sequence[BatchSummary], target_half_width: float, path: Path
) -> Path:
    """Interval half-width after each batch against the target: why sampling stopped."""
    if not history:
        raise ValueError("the convergence chart needs at least one batch")
    if not 0 < target_half_width <= 0.5:
        raise ValueError("target_half_width must be in (0, 0.5]")
    path = _check_png(path)

    batches = [summary.batch_index + 1 for summary in history]
    widths = [summary.half_width for summary in history]
    figure, axes = plt.subplots(figsize=_FIGSIZE)
    axes.plot(batches, widths, marker="o", color="#4C72B0", label="Interval half-width")
    axes.axhline(target_half_width, color="#C44E52", linestyle="--", label="Target")
    axes.set_xlabel("Batch")
    axes.set_ylabel("Half-width of the interval")
    axes.set_xticks(batches)
    axes.set_ylim(0.0, max([*widths, target_half_width]) * 1.15)
    axes.set_title("Precision after each batch")
    axes.legend()
    axes.grid(alpha=0.3)
    figure.tight_layout()
    return _save(figure, path)


# --- charts of the app (texts are passed in, so the UI can translate them) -----------------

Rate = tuple[float, float, float]  # estimate, interval low, interval high


@dataclass(frozen=True)
class BrandRates:
    """Mention and recommendation rates of one brand, each with its interval."""

    brand: str
    mention: Rate
    recommendation: Rate | None = None


@dataclass(frozen=True)
class RatesChartText:
    title: str
    mention: str
    recommendation: str
    axis: str
    note: str


@dataclass(frozen=True)
class PrecisionChartText:
    title: str
    rounds: str
    margin: str
    target: str
    note: str


_CLIENT = ("#4f46e5", "#a5b4fc")  # mention, recommendation
_OTHERS = ("#64748b", "#cbd5e1")


def _errors(rates: Sequence[Rate]) -> list[list[float]]:
    return [
        [max(0.0, estimate - low) for estimate, low, _ in rates],
        [max(0.0, high - estimate) for estimate, _, high in rates],
    ]


def _rates_figure(rows: Sequence[BrandRates], client: str, text: RatesChartText) -> plt.Figure:
    if not rows:
        raise ValueError("the chart needs at least one brand")
    ordered = sorted(rows, key=lambda row: -row.mention[0])
    positions = list(range(len(ordered)))
    width = 0.38
    figure, axes = plt.subplots(figsize=(7.5, 4.2))
    series_list = (
        (-width / 2, [row.mention for row in ordered], text.mention),
        (
            width / 2,
            [row.recommendation or (0.0, 0.0, 0.0) for row in ordered],
            text.recommendation,
        ),
    )
    for tone, (offset, series, label) in enumerate(series_list):
        colors = [(_CLIENT if row.brand == client else _OTHERS)[tone] for row in ordered]
        bars = axes.bar(
            [position + offset for position in positions],
            [rate[0] for rate in series],
            width,
            yerr=_errors(series),
            capsize=4,
            color=colors,
            error_kw={"elinewidth": 1, "ecolor": "#334155"},
            label=label,
        )
        for bar, rate in zip(bars, series, strict=True):
            axes.text(
                bar.get_x() + bar.get_width() / 2,
                0.015,
                f"{rate[0]:.0%}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="white" if rate[0] > 0.12 else "#334155",
            )
    axes.set_xticks(positions, [row.brand for row in ordered])
    for tick, row in zip(axes.get_xticklabels(), ordered, strict=True):
        tick.set_fontweight("bold" if row.brand == client else "normal")
    axes.set_ylim(0.0, 1.05)
    axes.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    axes.set_ylabel(text.axis)
    axes.set_title(text.title, loc="left", fontweight="bold", pad=26)
    axes.grid(axis="y", alpha=0.25)
    axes.spines[["top", "right"]].set_visible(False)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=_CLIENT[0]),
        plt.Rectangle((0, 0), 1, 1, color=_CLIENT[1]),
    ]
    # Between the title and the plot, so it never hides a bar or an error bar.
    axes.legend(
        handles,
        [text.mention, text.recommendation],
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.0),
        ncol=2,
    )
    figure.text(0.5, 0.01, text.note, ha="center", fontsize=8, color="#555555")
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    return figure


def rates_chart(rows: Sequence[BrandRates], client: str, path: Path, text: RatesChartText) -> Path:
    """Mention next to recommendation for every brand, with margins of error.

    The client brand is drawn in colour and the others in grey, so the reader finds it at
    once; brands are ordered by mention rate. A missing recommendation is drawn as 0.
    """
    path = _check_png(path)
    return _save(_rates_figure(rows, client, text), path)


def precision_chart(
    history: Sequence[BatchSummary], target_half_width: float, path: Path, text: PrecisionChartText
) -> Path:
    """Margin of error (interval half-width) after each round, against the target."""
    if not history:
        raise ValueError("the precision chart needs at least one batch")
    if not 0 < target_half_width <= 0.5:
        raise ValueError("target_half_width must be in (0, 0.5]")
    path = _check_png(path)
    rounds = [summary.batch_index + 1 for summary in history]
    margins = [summary.half_width for summary in history]
    figure, axes = plt.subplots(figsize=(7.5, 4.2))
    axes.fill_between(rounds, margins, color=_CLIENT[1], alpha=0.35)
    axes.plot(rounds, margins, marker="o", color=_CLIENT[0], linewidth=2, label=text.margin)
    axes.axhline(target_half_width, color="#15803d", linestyle="--", label=text.target)
    for x, y in zip(rounds, margins, strict=True):
        axes.annotate(
            f"±{y:.0%}", (x, y), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8
        )
    axes.set_xticks(rounds)
    axes.set_xlabel(text.rounds)
    axes.set_ylabel(text.margin)
    axes.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    axes.set_ylim(0.0, max([*margins, target_half_width]) * 1.25)
    axes.set_title(text.title, loc="left", fontweight="bold", pad=26)
    axes.grid(alpha=0.25)
    axes.spines[["top", "right"]].set_visible(False)
    axes.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2)
    figure.text(0.5, 0.01, text.note, ha="center", fontsize=8, color="#555555")
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    return _save(figure, path)


@dataclass(frozen=True)
class TrendPoint:
    """One approved measurement of the client brand: a short label and its mention rate."""

    label: str
    rate: Rate


@dataclass(frozen=True)
class TrendChartText:
    title: str
    axis: str
    note: str


def _trend_figure(points: Sequence[TrendPoint], text: TrendChartText) -> plt.Figure:
    if len(points) < 2:
        raise ValueError("a trend needs at least two measurements")
    positions = list(range(len(points)))
    estimates = [point.rate[0] for point in points]
    figure, axes = plt.subplots(figsize=(7.5, 4.2))
    axes.fill_between(
        positions,
        [point.rate[1] for point in points],
        [point.rate[2] for point in points],
        color=_CLIENT[1],
        alpha=0.35,
    )
    axes.plot(positions, estimates, marker="o", color=_CLIENT[0], linewidth=2)
    for x, y in zip(positions, estimates, strict=True):
        axes.annotate(
            f"{y:.0%}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8
        )
    axes.set_xticks(positions, [point.label for point in points])
    axes.set_ylim(0.0, 1.05)
    axes.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    axes.set_ylabel(text.axis)
    axes.set_title(text.title, loc="left", fontweight="bold", pad=12)
    axes.grid(alpha=0.25)
    axes.spines[["top", "right"]].set_visible(False)
    figure.text(0.5, 0.01, text.note, ha="center", fontsize=8, color="#555555")
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    return figure


def trend_chart(points: Sequence[TrendPoint], path: Path, text: TrendChartText) -> Path:
    """Mention rate of the client brand across measurements, oldest first, with its band."""
    path = _check_png(path)
    return _save(_trend_figure(points, text), path)
