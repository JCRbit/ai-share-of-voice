"""Results tab: shares with intervals, charts and downloads (owner: E).

The view is built in three layers that degrade on their own, so the tab shows something
useful for any run, simulated or real:

* A. the headline figure and the convergence chart come from the run state (``ReportData``);
* B. the per-brand table and chart come from the vetted statistics over ``answers.csv``;
* C. the downloads exist only after the human approval wrote the report.

Nothing is recomputed here: the figures come from ``stats_tool`` and the run state. The view
is plain data (``ResultsView``) so it is tested without a UI; the UI only draws it.
"""

from pathlib import Path

from pydantic import Field, ValidationError

from ai_share_of_voice.automation.report import REPORT_FILE_NAME, ReportData
from ai_share_of_voice.automation.tracking_csv import build_row, upsert_row
from ai_share_of_voice.schemas.base import Contract
from ai_share_of_voice.stats.charts import convergence_chart
from ai_share_of_voice.tools.stats_tool import (
    CHARTS_FOLDER,
    MetricName,
    MetricRow,
    StatsToolError,
    StatsToolInput,
    run_stats,
)

TRACKING_ROW_FILE_NAME = "tracking_row.csv"
TABLE_HEADERS = ("Brand", "Estimate (95% interval)", "Interval method")
# The headline (from the state) and the table (from the answers) should tell the same story.
HEADLINE_TOLERANCE = 0.005


class HeadlineRow(Contract):
    """The client brand's cumulative figure after the last batch."""

    metric: str
    estimate: float = Field(ge=0, le=1)
    ci_low: float = Field(ge=0, le=1)
    ci_high: float = Field(ge=0, le=1)
    interval_method: str
    n_answers: int = Field(ge=1)
    n_questions: int | None = None


class ResultsView(Contract):
    """Everything the tab draws. Local to the UI: it is not a shared contract."""

    run_id: str
    metric: MetricName
    sample_label: str
    headline: HeadlineRow | None = None
    table: tuple[MetricRow, ...] = ()
    chart_paths: tuple[str, ...] = ()
    downloads: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


def _check_run_id(run_id: str) -> None:
    """Reuse the stats tool's folder-name pattern so a run id cannot leave ``runs/``."""
    try:
        StatsToolInput(run_id=run_id)
    except ValidationError as error:
        raise ValueError(f"unsafe run_id {run_id!r}: use letters, digits, - and _") from error


def build_results_view(
    data: ReportData,
    runs_dir: Path | str,
    *,
    metric: MetricName = "mention_rate",
    target_half_width: float | None = None,
) -> ResultsView:
    """Build the view of one run. ``target_half_width`` enables the convergence chart."""
    _check_run_id(data.run_id)
    run_dir = Path(runs_dir) / data.run_id
    summary = data.summary
    notes: list[str] = []
    chart_paths: list[str] = []

    headline = HeadlineRow(
        metric=summary.metric.value,
        estimate=summary.brand_share,
        ci_low=summary.ci_low,
        ci_high=summary.ci_high,
        interval_method=summary.interval_method.value,
        n_answers=summary.n_answers,
        n_questions=summary.n_questions,
    )
    if target_half_width is not None:
        path = run_dir / CHARTS_FOLDER / "convergence.png"
        chart_paths.append(str(convergence_chart(data.history, target_half_width, path)))

    table: tuple[MetricRow, ...] = ()
    try:
        result = run_stats(StatsToolInput(run_id=data.run_id, metric=metric), runs_dir)
    except StatsToolError:
        notes.append("No analyzed answers for this run yet: the per-brand table is not available.")
    else:
        table = result.rows
        chart_paths.extend(result.chart_paths)
        if result.few_questions:
            notes.append(
                f"Few questions ({result.n_questions}): the intervals are not reliable, "
                "read them as indicative only."
            )
        if metric == summary.metric.value:
            notes.extend(_headline_notes(data, table))

    downloads: tuple[str, ...] = ()
    report_path = run_dir / REPORT_FILE_NAME
    if report_path.is_file():
        row_path = run_dir / TRACKING_ROW_FILE_NAME
        upsert_row(row_path, build_row(data, report_path))
        downloads = (str(report_path), str(row_path))
    else:
        notes.append("Downloads appear after the report is approved.")

    return ResultsView(
        run_id=data.run_id,
        metric=metric,
        sample_label=_sample_label(data),
        headline=headline,
        table=table,
        chart_paths=tuple(chart_paths),
        downloads=downloads,
        notes=tuple(notes),
    )


def _sample_label(data: ReportData) -> str:
    return (
        f"This is a dated sample, not market share: {data.summary.n_answers} answers about "
        f"{data.brand} in {data.sector}, {data.city}, collected on "
        f"{data.generated_at:%Y-%m-%d}."
    )


def _headline_notes(data: ReportData, table: tuple[MetricRow, ...]) -> list[str]:
    row = next((candidate for candidate in table if candidate.brand == data.brand), None)
    if row is None:
        return [f"{data.brand} is not in the per-brand table: check the brand names."]
    if row.estimate is None:
        return []
    if abs(row.estimate - data.summary.brand_share) > HEADLINE_TOLERANCE:
        return [
            "The headline and the table differ for the client brand "
            f"({data.summary.brand_share:.1%} vs {row.estimate:.1%}): the run state and the "
            "analyzed answers do not come from the same data (expected in a simulated run)."
        ]
    return []


def _cell(view: ResultsView, row: MetricRow) -> str:
    if row.estimate is None or row.ci_low is None or row.ci_high is None:
        return "n/a"
    if view.metric == "net_tone":
        return f"{row.estimate:+.2f} ({row.ci_low:+.2f} to {row.ci_high:+.2f})"
    return f"{row.estimate:.0%} ({row.ci_low:.0%}–{row.ci_high:.0%})"


def table_rows(view: ResultsView) -> list[list[str]]:
    """The table as plain strings, ready for a dataframe (never HTML)."""
    return [[row.brand, _cell(view, row), row.ci_method] for row in view.table]
