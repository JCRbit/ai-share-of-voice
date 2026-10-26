"""Vetted charts for the report and the vision agent (owner: E).

Plain matplotlib functions with validated inputs: no model writes plotting code at run
time. They render off-screen (``Agg``) and save without variable metadata, so the same
input always gives the same PNG and a chart can be reproduced from the run folder.
"""

from collections.abc import Sequence
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

    ``metrics`` is the table returned by ``brand_metrics``.
    """
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}; expected one of {', '.join(METRICS)}")
    if missing := set(REQUIRED_COLUMNS) - set(metrics.columns):
        raise ValueError(f"missing columns: {sorted(missing)}")
    path = _check_png(path)
    rows = metrics[metrics["metric"] == metric]
    if rows.empty:
        raise ValueError(f"the table has no rows for {metric!r}")

    estimate = rows["estimate"].fillna(0.0)
    errors = [
        (estimate - rows["ci_low"].fillna(estimate)).clip(lower=0),
        (rows["ci_high"].fillna(estimate) - estimate).clip(lower=0),
    ]
    figure, axes = plt.subplots(figsize=_FIGSIZE)
    axes.bar(rows["brand"], estimate, yerr=errors, capsize=6, color="#4C72B0")
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
    if "few_questions" in rows and rows["few_questions"].any():
        n_questions = int(rows["n_questions"].iloc[0]) if "n_questions" in rows else None
        count = f"Only {n_questions} questions" if n_questions else "Few questions"
        notes.append(
            (f"{count}: intervals are not reliable, read the bars as indicative only.", "#C44E52")
        )
    for line, (text, color) in enumerate(reversed(notes)):
        figure.text(0.5, 0.01 + 0.04 * line, text, ha="center", fontsize=8, color=color)
    figure.tight_layout(rect=(0, 0.04 * len(notes) + 0.01, 1, 1))
    return _save(figure, path)


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
