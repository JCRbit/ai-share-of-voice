"""Append-only JSONL log of graph events (owner: J).

Every event is written to ``<runs_dir>/<run_id>/events.jsonl`` so a third party can audit
which agent did what, with which tool, with what result and why.
"""

import threading
from pathlib import Path

from ai_share_of_voice.schemas.events import GraphEvent


class EventLog:
    """Thread-safe writer and reader of the per-run event log."""

    def __init__(self, runs_dir: Path | str = "runs") -> None:
        self._runs_dir = Path(runs_dir)
        self._lock = threading.Lock()

    def path_for(self, run_id: str) -> Path:
        """Return the log path of a run, rejecting ids that would leave the runs folder."""
        if Path(run_id).name != run_id or run_id in {"", ".", ".."}:
            raise ValueError(f"invalid run id: {run_id!r}")
        return self._runs_dir / run_id / "events.jsonl"

    def append(self, event: GraphEvent) -> None:
        """Append one event as a JSON line."""
        path = self.path_for(event.run_id)
        line = event.model_dump_json()
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def read(self, run_id: str) -> list[GraphEvent]:
        """Return every event of a run, in order; empty if the run has no log yet."""
        path = self.path_for(run_id)
        if not path.exists():
            return []
        with path.open(encoding="utf-8") as handle:
            return [GraphEvent.model_validate_json(line) for line in handle if line.strip()]
