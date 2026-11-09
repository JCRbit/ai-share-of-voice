"""Results and downloads tab (owner: E, task D4).

The tab shows what a run found, read from its folder ``<runs_dir>/<run_id>/``:

* a table with every metric of every brand, its estimate and its interval, built by the vetted
  functions of ``stats/`` from ``answers.csv`` (nothing is computed by a model);
* one chart per metric, the same ``<metric>_by_brand.png`` files the statistics tool writes;
* the files to download: the report and the tracking CSV when the run was approved, the answers
  and the metrics tables, the charts and the event log.

Everything that does not need Gradio is plain Python and tested without it; ``build_results_tab``
creates the components, with Gradio imported there, like ``brief_tab.py``.

What is shown and offered is chosen with care:

* A run id comes from a menu but is checked again: a plain folder name that cannot leave the runs
  folder. Files are taken from fixed names, never from a path typed by anyone, and a file that
  resolves outside the runs folder (a link) is not offered.
* The brand names come from the brief and can be anything, even ``=HYPERLINK(...)``. The CSV
  files are offered as copies in ``downloads/`` whose cells that start a formula are neutralized,
  the same rule as the tracking CSV. The files the code reads back are left untouched.
* Every figure is a dated sample of one model's answers, never a market share, and the interval
  is the conservative rule of ``brand_metrics``; both are stated next to the numbers.
* The table is plain text; nothing from a model is rendered as Markdown or HTML here.
"""

import math
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import ValidationError

from ai_share_of_voice.automation.files import write_text_atomic
from ai_share_of_voice.automation.report import REPORT_FILE_NAME
from ai_share_of_voice.automation.tracking_csv import TRACKING_FILE_NAME, neutralize_formula
from ai_share_of_voice.event_log import EventLog
from ai_share_of_voice.stats.charts import METRIC_LABELS, brand_comparison_chart
from ai_share_of_voice.stats.metrics import METRICS, brand_metrics, load_answers
from ai_share_of_voice.tools.stats_tool import ANSWERS_FILE_NAME, CHARTS_FOLDER
from ai_share_of_voice.ui.render import STATUS_LABELS, results_text
from ai_share_of_voice.ui.run_manager import RunSnapshot, status_from_events

# Same rule as ``StatsToolInput.run_id``: a plain folder name.
RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
DOWNLOADS_FOLDER = "downloads"
METRICS_FILE_NAME = "metrics.csv"
EVENTS_FILE_NAME = "events.jsonl"
TABLE_HEADERS = ["Metric", "Brand", "Estimate", "95% interval", "Note"]
SAMPLE_NOTE = (
    "Every figure is a dated sample of one model's answers to a fixed set of questions. "
    "It is not a market share."
)
INTERVAL_NOTE = (
    "The interval is at least as wide as the Wilson and the question-level bootstrap intervals "
    "(for share of voice and net tone, the bootstrap). Its coverage has not been validated."
)
CACHE_SIZE = 8

_CACHE: OrderedDict[tuple[str, int, int], pd.DataFrame] = OrderedDict()
_CACHE_LOCK = threading.Lock()


class ResultsError(Exception):
    """The results of a run cannot be shown; the message is safe to display."""


@dataclass(frozen=True)
class RunResults:
    """Everything the tab shows for one run."""

    run_id: str
    rows: list[list[str]]
    charts: list[tuple[str, str]]
    downloads: list[str]
    n_answers: int
    n_questions: int
    n_batches: int
    few_questions: bool
    has_report: bool


@dataclass(frozen=True)
class ResultsView:
    """The four outputs of the tab: summary, table, charts and files."""

    summary: str
    rows: list[list[str]]
    charts: list[tuple[str, str]]
    files: list[str]

    def as_outputs(self) -> tuple[str, list[list[str]], list[tuple[str, str]], list[str]]:
        """The values in the order of ``ResultsPanel.outputs``."""
        return self.summary, self.rows, self.charts, self.files


@dataclass(frozen=True)
class ResultsPanel:
    """The components of the tab; ``outputs`` follow ``ResultsView.as_outputs``."""

    run: Any
    refresh: Any
    summary: Any
    table: Any
    charts: Any
    files: Any

    @property
    def outputs(self) -> list[Any]:
        """The components a handler fills with a ``ResultsView``."""
        return [self.summary, self.table, self.charts, self.files]


# ---- finding runs ----------------------------------------------------------------------------


def list_runs(runs_dir: Path | str) -> list[str]:
    """Runs that have analyzed answers, newest first (run ids start with their date)."""
    folder = Path(runs_dir)
    if not folder.is_dir():
        return []
    found = [
        item.name
        for item in folder.iterdir()
        if RUN_ID.match(item.name) and (item / ANSWERS_FILE_NAME).is_file()
    ]
    return sorted(found, reverse=True)


def _within(directory: Path, path: Path) -> bool:
    """True if the resolved path stays inside the resolved directory."""
    try:
        return path.resolve().is_relative_to(directory.resolve())
    except (OSError, RuntimeError):
        return False


def _inside(runs_dir: Path, path: Path) -> bool:
    """True if ``path`` exists and really lives under ``runs_dir``."""
    return path.is_file() and _within(runs_dir, path)


# ---- formatting ------------------------------------------------------------------------------


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not math.isnan(value)


def format_estimate(metric: str, value: Any) -> str:
    """A rate as a percentage; net tone as a signed score between -1 and 1."""
    if not _is_number(value):
        return "n/a"
    return f"{value:+.2f}" if metric == "net_tone" else f"{value:.1%}"


def format_interval(metric: str, low: Any, high: Any) -> str:
    """The interval with the same unit as the estimate."""
    if not (_is_number(low) and _is_number(high)):
        return "n/a"
    return f"{format_estimate(metric, low)} to {format_estimate(metric, high)}"


def metric_name(metric: str) -> str:
    """Short name of a metric: its label without the explanation in brackets."""
    return METRIC_LABELS.get(metric, metric).split(" (")[0]


def table_rows(metrics: pd.DataFrame) -> list[list[str]]:
    """One row per metric and brand, metrics in their usual order, brands as in the answers."""
    rows: list[list[str]] = []
    for metric in METRICS:
        for item in metrics[metrics["metric"] == metric].itertuples():
            defined = _is_number(item.estimate)
            rows.append(
                [
                    metric_name(metric),
                    str(item.brand),
                    format_estimate(metric, item.estimate),
                    format_interval(metric, item.ci_low, item.ci_high),
                    "" if defined else "not defined for this brand (nothing to divide)",
                ]
            )
    return rows


# ---- loading and computing -------------------------------------------------------------------


def _metrics_of(path: Path, answers: pd.DataFrame) -> pd.DataFrame:
    """``brand_metrics`` of the answers, remembered until the file changes."""
    stat = path.stat()
    key = (str(path.resolve()), stat.st_mtime_ns, stat.st_size)
    with _CACHE_LOCK:
        if key in _CACHE:
            _CACHE.move_to_end(key)
            return _CACHE[key]
    try:
        computed = brand_metrics(answers)
    except ValueError as error:
        raise ResultsError(f"not enough data to compute the metrics: {error}") from error
    with _CACHE_LOCK:
        _CACHE[key] = computed
        while len(_CACHE) > CACHE_SIZE:
            _CACHE.popitem(last=False)
    return computed


def safe_csv_text(frame: pd.DataFrame) -> str:
    """CSV text of ``frame`` whose text cells cannot start a formula in a spreadsheet."""
    safe = frame.copy()
    for column in safe.columns:
        safe[column] = safe[column].map(_neutralize_cell)
    return safe.to_csv(index=False, lineterminator="\n")


def _neutralize_cell(value: Any) -> Any:
    return neutralize_formula(value) if isinstance(value, str) else value


def _write_if_changed(path: Path, text: str) -> None:
    """Write ``text`` unless the file already has it, so looking at a run changes nothing."""
    if path.is_file() and path.read_text(encoding="utf-8") == text:
        return
    write_text_atomic(path, text)


def _charts(run_dir: Path, metrics: pd.DataFrame, answers_path: Path) -> list[tuple[str, str]]:
    """The chart of every metric, drawn again only if missing or older than the answers."""
    charts = []
    newest = answers_path.stat().st_mtime
    for metric in METRICS:
        path = run_dir / CHARTS_FOLDER / f"{metric}_by_brand.png"
        if not _within(run_dir, path):
            continue
        if not path.is_file() or path.stat().st_mtime < newest:
            brand_comparison_chart(metrics, metric, path)
        charts.append((str(path), metric_name(metric)))
    return charts


def _downloads(
    runs_dir: Path, run_dir: Path, answers: pd.DataFrame, metrics: pd.DataFrame, charts: list[str]
) -> list[str]:
    """Files to offer, in a fixed order: report, tracking, tables, charts, events."""
    folder = run_dir / DOWNLOADS_FOLDER
    candidates = [
        run_dir / REPORT_FILE_NAME,
        runs_dir / TRACKING_FILE_NAME,
        folder / ANSWERS_FILE_NAME,
        folder / METRICS_FILE_NAME,
        *(Path(path) for path in charts),
        run_dir / EVENTS_FILE_NAME,
    ]

    if _within(run_dir, folder):
        _write_if_changed(folder / ANSWERS_FILE_NAME, safe_csv_text(answers))
        _write_if_changed(folder / METRICS_FILE_NAME, safe_csv_text(metrics))

    return [str(path) for path in candidates if _inside(runs_dir, path)]


def load_results(runs_dir: Path | str, run_id: str) -> RunResults:
    """Compute and collect everything the tab shows for ``run_id``."""
    runs = Path(runs_dir)
    if not RUN_ID.match(run_id or ""):
        raise ResultsError("choose a run from the list")
    run_dir = runs / run_id
    answers_path = run_dir / ANSWERS_FILE_NAME
    if not _inside(runs, answers_path):
        raise ResultsError(f"run {run_id} has no analyzed answers, so there is nothing to show")
    try:
        answers = load_answers(answers_path)
    except (OSError, ValueError) as error:
        reason = _short(error)
        raise ResultsError(f"the answers of run {run_id} cannot be read: {reason}") from error
    metrics = _metrics_of(answers_path, answers)
    charts = _charts(run_dir, metrics, answers_path)
    files = _downloads(runs, run_dir, answers, metrics, [path for path, _ in charts])
    first = metrics.iloc[0]
    return RunResults(
        run_id=run_id,
        rows=table_rows(metrics),
        charts=charts,
        downloads=files,
        n_answers=int(answers["response_id"].nunique()),
        n_questions=int(first["n_questions"]),
        n_batches=int(answers["batch_index"].nunique()),
        few_questions=bool(first["few_questions"]),
        has_report=_inside(runs, run_dir / REPORT_FILE_NAME),
    )


def _short(error: Exception, limit: int = 160) -> str:
    text = " ".join(str(error).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


# ---- text ------------------------------------------------------------------------------------


def _how_it_ended(runs_dir: Path, run_id: str) -> str | None:
    """Final status and spend from the event log, if the run has one."""
    try:
        events = EventLog(runs_dir).read(run_id)
    except (OSError, ValueError, ValidationError):
        return None
    status = status_from_events(events)
    spends = [e.cumulative_spend_usd for e in events if e.cumulative_spend_usd is not None]
    if status is None and not spends:
        return None
    parts = [STATUS_LABELS.get(status, "no closing event") if status else "no closing event"]
    if spends:
        parts.append(f"spend ${spends[-1]:.2f}")
    return ", ".join(parts)


def summary_text(runs_dir: Path | str, results: RunResults) -> str:
    """The text above the table, as Markdown built only from numbers and fixed sentences."""
    lines = [
        f"**Run `{results.run_id}`:** {results.n_answers} answers to {results.n_questions} "
        f"questions, in {results.n_batches} batches.",
        SAMPLE_NOTE,
    ]
    ended = _how_it_ended(Path(runs_dir), results.run_id)
    if ended:
        lines.append(f"How it ended: {ended}.")
    if results.few_questions:
        lines.append(
            f"**Only {results.n_questions} questions:** with so few, the intervals are too "
            "optimistic. Read them as a lower bound of the uncertainty."
        )
    lines.append(INTERVAL_NOTE)
    lines.append(
        "The approved report is in the downloads."
        if results.has_report
        else "No report was written for this run: it is written only after the approval."
    )
    return "\n\n".join(lines)


def show_results(runs_dir: Path | str, run_id: str | None) -> ResultsView:
    """The view of ``run_id``; a problem becomes a message, never an exception."""
    if not run_id:
        return ResultsView("Choose a run to see its results.", [], [], [])
    try:
        results = load_results(runs_dir, run_id)
    except ResultsError as error:
        return ResultsView(str(error), [], [], [])
    return ResultsView(
        summary_text(runs_dir, results), results.rows, results.charts, results.downloads
    )


def view_for_ended_run(
    runs_dir: Path | str, snapshot: RunSnapshot
) -> tuple[str | None, ResultsView] | None:
    """What to show when the background run ends; ``None`` while there is nothing to change.

    Returns the run to select (``None`` if it saved no analyzed answers) and its view.
    """
    if snapshot.running or snapshot.run_id is None:
        return None
    if snapshot.run_id in list_runs(runs_dir):
        return snapshot.run_id, show_results(runs_dir, snapshot.run_id)
    return None, ResultsView(results_text(snapshot), [], [], [])


# ---- Gradio ----------------------------------------------------------------------------------


def build_results_tab(runs_dir: Path | str, initial_summary: str = "") -> ResultsPanel:
    """Create the components of the tab inside the current Gradio ``Blocks``."""
    import gradio as gr

    runs = list_runs(runs_dir)
    run = gr.Dropdown(label="Run", choices=runs, value=runs[0] if runs else None)
    refresh = gr.Button("Refresh the list of runs")
    summary = gr.Markdown(initial_summary or "Choose a run to see its results.")
    table = gr.Dataframe(headers=TABLE_HEADERS, datatype="str", interactive=False, wrap=True)
    charts = gr.Gallery(label="Charts", columns=2, interactive=False)
    files = gr.File(label="Downloads", file_count="multiple", interactive=False)
    return ResultsPanel(
        run=run, refresh=refresh, summary=summary, table=table, charts=charts, files=files
    )
