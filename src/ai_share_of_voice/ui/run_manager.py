"""Background run for the interface (owner: An, task I6).

``RunManager`` runs one graph run in a thread and keeps what the pages need: the events so far,
the spend and the status. It knows nothing about Gradio, so it is tested without it. The pages
call ``snapshot()`` on a timer; they never wait for the run.

* One run at a time: ``start`` raises ``RunInProgress`` while another one is running.
* A failure inside the run is classified with ``errors.classify`` and shown as status
  ``failed``; it never reaches the page as a crash.
* The default runner is the simulated graph (``ui/mock_run.py``). C2 swaps it for the real one by
  passing another ``runner``.
"""

import logging
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass

from ai_share_of_voice.errors import classify
from ai_share_of_voice.graph import new_run_id
from ai_share_of_voice.schemas.brief import Brief
from ai_share_of_voice.schemas.events import EventStatus, GraphEvent

logger = logging.getLogger(__name__)

Runner = Callable[[str, Brief], Iterator[GraphEvent]]

IDLE = "idle"
RUNNING = "running"
WAITING_APPROVAL = "waiting_approval"
COMPLETED = "completed"
PARTIAL = "partial"
REJECTED = "rejected"
FAILED = "failed"


class RunInProgress(RuntimeError):
    """A run was requested while another one is still running."""


def simulated_runner(run_id: str, brief: Brief) -> Iterator[GraphEvent]:
    """Run the simulated graph and approve it automatically (approval is task A4)."""
    from ai_share_of_voice.ui.mock_run import simulate_run

    return simulate_run(run_id, brief=brief, auto_approve=True)


@dataclass(frozen=True)
class RunSnapshot:
    """What the pages show at one moment."""

    run_id: str | None
    status: str
    events: tuple[GraphEvent, ...]
    spent_usd: float
    cap_usd: float | None
    error: str | None

    @property
    def running(self) -> bool:
        """True while the background run has not ended."""
        return self.status == RUNNING


def status_from_events(events: tuple[GraphEvent, ...] | list[GraphEvent]) -> str | None:
    """Outcome shown by the closing events of a run, or ``None`` if none of them is there."""
    for event in reversed(events):
        if event.node == "close_partial":
            return PARTIAL
        if event.node == "close_rejected":
            return REJECTED
        if event.node == "finalize" and event.status is EventStatus.SUCCEEDED:
            return COMPLETED
        if event.status is EventStatus.WAITING_APPROVAL:
            return WAITING_APPROVAL
    return None


class RunManager:
    """Starts one run at a time in a background thread and reports its progress."""

    def __init__(self, runner: Runner = simulated_runner) -> None:
        self._runner = runner
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._run_id: str | None = None
        self._cap_usd: float | None = None
        self._events: list[GraphEvent] = []
        self._status = IDLE
        self._error: str | None = None

    def start(self, brief: Brief) -> str:
        """Start a run for ``brief`` and return its id. Raises ``RunInProgress`` if busy."""
        with self._lock:
            if self._status == RUNNING:
                raise RunInProgress("a run is already in progress")
            run_id = new_run_id()
            self._run_id = run_id
            self._cap_usd = brief.spend_cap_usd
            self._events = []
            self._status = RUNNING
            self._error = None
            self._thread = threading.Thread(
                target=self._work, args=(run_id, brief), name=run_id, daemon=True
            )
            thread = self._thread
        thread.start()
        return run_id

    def snapshot(self) -> RunSnapshot:
        """A consistent copy of the current state."""
        with self._lock:
            events = tuple(self._events)
            spends = [e.cumulative_spend_usd for e in events if e.cumulative_spend_usd is not None]
            return RunSnapshot(
                run_id=self._run_id,
                status=self._status,
                events=events,
                spent_usd=spends[-1] if spends else 0.0,
                cap_usd=self._cap_usd,
                error=self._error,
            )

    def wait(self, timeout: float | None = None) -> None:
        """Block until the current run ends; meant for tests and scripts."""
        thread = self._thread
        if thread is not None:
            thread.join(timeout)

    def _work(self, run_id: str, brief: Brief) -> None:
        try:
            for event in self._runner(run_id, brief):
                with self._lock:
                    self._events.append(event)
            with self._lock:
                outcome = status_from_events(self._events)
                if outcome is None:
                    self._status = FAILED
                    self._error = "the run ended without a closing event"
                else:
                    self._status = outcome
        except Exception as error:
            logger.exception("run %s failed", run_id)
            failure = classify(error)
            with self._lock:
                self._status = FAILED
                self._error = f"{failure.kind.value}: {failure.reason}"
