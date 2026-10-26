"""Statistics tool of the analyst agent (owner: E).

This is the "code execution" of the project: the agent never writes or sends code. It
picks a run and a metric from a validated menu, and the tool runs vetted functions
(``brand_metrics`` and ``brand_comparison_chart``) on the answers stored in the run folder
``runs/<run_id>/answers.csv``. Every figure the agent reports comes from here.
"""

from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from ai_share_of_voice.schemas.base import Contract
from ai_share_of_voice.stats.charts import brand_comparison_chart
from ai_share_of_voice.stats.metrics import brand_metrics, load_answers

ANSWERS_FILE_NAME = "answers.csv"
CHARTS_FOLDER = "charts"

MetricName = Literal[
    "mention_rate",
    "recommendation_rate",
    "top1_rate",
    "share_of_voice",
    "recommendation_conversion",
    "net_tone",
]


class StatsToolError(Exception):
    """Raised when the run has no usable answers."""


class StatsToolInput(Contract):
    """What the agent may ask for: a run, a metric and whether to draw the chart."""

    # A plain folder name: letters, digits, dashes and underscores, so it cannot leave runs/.
    run_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$", max_length=100)
    metric: MetricName = "mention_rate"
    chart: bool = True


class MetricRow(Contract):
    """One brand's estimate of the requested metric and its headline interval."""

    brand: str
    estimate: float | None
    ci_low: float | None
    ci_high: float | None
    ci_method: str


class StatsToolResult(Contract):
    """Figures and chart paths for one metric of one run."""

    run_id: str
    metric: MetricName
    rows: tuple[MetricRow, ...]
    n_questions: int
    few_questions: bool
    chart_paths: tuple[str, ...] = ()


def _number(value: float) -> float | None:
    """NaN (e.g. conversion of a brand never mentioned) becomes ``None``."""
    return None if value != value else float(value)


def run_stats(data: StatsToolInput, runs_dir: Path | str) -> StatsToolResult:
    """Compute the metric for every brand of the run and optionally chart it."""
    run_dir = Path(runs_dir) / data.run_id
    answers_path = run_dir / ANSWERS_FILE_NAME
    if not answers_path.is_file():
        raise StatsToolError(f"run {data.run_id!r} has no analyzed answers ({ANSWERS_FILE_NAME})")
    try:
        metrics = brand_metrics(load_answers(answers_path))
    except ValueError as error:
        raise StatsToolError(f"cannot compute metrics for {data.run_id!r}: {error}") from error

    selected = metrics[metrics["metric"] == data.metric]
    rows = tuple(
        MetricRow(
            brand=row.brand,
            estimate=_number(row.estimate),
            ci_low=_number(row.ci_low),
            ci_high=_number(row.ci_high),
            ci_method=row.ci_method,
        )
        for row in selected.itertuples()
    )
    chart_paths: tuple[str, ...] = ()
    if data.chart:
        chart_path = run_dir / CHARTS_FOLDER / f"{data.metric}_by_brand.png"
        chart_paths = (str(brand_comparison_chart(metrics, data.metric, chart_path)),)

    first = selected.iloc[0]
    return StatsToolResult(
        run_id=data.run_id,
        metric=data.metric,
        rows=rows,
        n_questions=int(first["n_questions"]),
        few_questions=bool(first["few_questions"]),
        chart_paths=chart_paths,
    )


class StatsTool:
    """Agent-facing wrapper: validates the arguments and runs the vetted statistics.

    A2 exposes it to the analyst agent using ``name``, ``description`` and ``args_schema``.
    """

    name = "stats_tool"
    description = (
        "Compute one brand metric (mention rate, recommendation rate, top-1 rate, share of "
        "voice, recommendation conversion or net tone) with its confidence interval for every "
        "brand of a run, and draw the comparison chart. Returns figures and chart paths."
    )
    args_schema = StatsToolInput

    def __init__(self, runs_dir: Path | str = "runs") -> None:
        self._runs_dir = Path(runs_dir)

    def run(self, **arguments: Any) -> StatsToolResult:
        """Validate ``arguments`` against ``args_schema`` and run the statistics."""
        return run_stats(self.args_schema(**arguments), self._runs_dir)
