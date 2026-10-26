"""Tracking CSV: one row per run, safe to repeat (owner: J)."""

import csv
import io
import threading
from collections.abc import Mapping
from pathlib import Path

from ai_share_of_voice.automation.files import write_text_atomic
from ai_share_of_voice.automation.report import ReportData

TRACKING_FILE_NAME = "tracking.csv"

FIELDS = (
    "run_id",
    "generated_at",
    "brand",
    "sector",
    "city",
    "n_answers",
    "brand_share",
    "ci_low",
    "ci_high",
    "batches",
    "spend_usd",
    "stop_reason",
    "sources",
    "open_objections",
    "report_path",
)

_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")
_LOCK = threading.Lock()


class TrackingError(Exception):
    """Raised when the tracking file cannot be updated safely."""


def _safe(value: str) -> str:
    """Neutralize spreadsheet formulas: a leading quote makes the cell plain text."""
    return f"'{value}" if value.startswith(_FORMULA_PREFIXES) else value


def build_row(data: ReportData, report_path: Path) -> dict[str, str]:
    """Turn the report data into the tracking row of the run."""
    summary = data.summary
    return {
        "run_id": data.run_id,
        "generated_at": data.generated_at.isoformat(timespec="seconds"),
        "brand": data.brand,
        "sector": data.sector,
        "city": data.city,
        "n_answers": str(summary.n_answers),
        "brand_share": f"{summary.brand_share:.4f}",
        "ci_low": f"{summary.ci_low:.4f}",
        "ci_high": f"{summary.ci_high:.4f}",
        "batches": str(len(data.history)),
        "spend_usd": f"{data.spend_usd:.4f}",
        "stop_reason": data.stop.reason.value,
        "sources": str(len(data.sources)),
        "open_objections": str(len(data.objections)),
        "report_path": str(report_path),
    }


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise TrackingError(
                f"{path} does not have the expected columns; move it aside before a new run"
            )
        return list(reader)


def _dump_rows(rows: list[dict[str, str]]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def upsert_row(path: Path, row: Mapping[str, str]) -> bool:
    """Insert the row of a run, or replace it if the run is already there.

    Repeating the same run never duplicates it. Returns ``True`` when a row was added
    and ``False`` when an existing row was replaced.
    """
    if set(row) != set(FIELDS):
        raise TrackingError("the row does not have exactly the tracking columns")
    clean = {field: _safe(str(row[field])) for field in FIELDS}
    with _LOCK:
        rows = _read_rows(path)
        for index, existing in enumerate(rows):
            if existing["run_id"] == clean["run_id"]:
                rows[index] = clean
                inserted = False
                break
        else:
            rows.append(clean)
            inserted = True
        write_text_atomic(path, _dump_rows(rows))
    return inserted
