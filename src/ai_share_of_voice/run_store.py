"""Per-run store of the brief and the answers (owner: An, tasks A3b and C2).

A run keeps its data in its own folder, next to the event log:

    <runs_dir>/<run_id>/brief.json             the brief that defines the brands
    <runs_dir>/<run_id>/raw_answers.jsonl      one ``RawAnswer`` per line, in the order sampled
    <runs_dir>/<run_id>/analyzed_answers.jsonl one ``AnalyzedResponse`` per line (mentions, sources)

With them in the folder, a tool only needs ids: it reads the answer and the brief from here
instead of receiving the whole text from an agent, which costs tokens on every call and lets the
agent alter what it passes. Reads fail with ``ToolFailure`` so the graph closes the run cleanly.

Run ids come from an agent as well, so they are checked like the event log checks them: an id
that would leave the runs folder is refused. Answers are found by id inside the file, never by
building a path from them.
"""

import threading
from pathlib import Path

from pydantic import BaseModel, ValidationError

from ai_share_of_voice.automation.files import write_text_atomic
from ai_share_of_voice.errors import ToolFailure
from ai_share_of_voice.schemas.analysis import AnalyzedResponse, RawAnswer
from ai_share_of_voice.schemas.brief import Brief

TOOL_NAME = "run_store"
BRIEF_FILE_NAME = "brief.json"
ANSWERS_FILE_NAME = "raw_answers.jsonl"
ANALYSES_FILE_NAME = "analyzed_answers.jsonl"


def _fail(reason: str) -> ToolFailure:
    return ToolFailure(TOOL_NAME, reason)


class RunStore:
    """Thread-safe reader and writer of the inputs of each run."""

    def __init__(self, runs_dir: Path | str = "runs") -> None:
        self._runs_dir = Path(runs_dir)
        self._lock = threading.Lock()

    def run_dir(self, run_id: str) -> Path:
        """Folder of a run, rejecting ids that would leave the runs folder."""
        if Path(run_id).name != run_id or run_id in {"", ".", ".."} or "\\" in run_id:
            raise _fail(f"invalid run id: {run_id!r}")
        return self._runs_dir / run_id

    def save_brief(self, run_id: str, brief: Brief) -> Path:
        """Write the brief of a run, replacing any earlier one in a single step."""
        path = self.run_dir(run_id) / BRIEF_FILE_NAME
        with self._lock:
            write_text_atomic(path, brief.model_dump_json(indent=2) + "\n")
        return path

    def load_brief(self, run_id: str) -> Brief:
        """Read the brief of a run."""
        path = self.run_dir(run_id) / BRIEF_FILE_NAME
        try:
            return Brief.model_validate_json(path.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise _fail(f"run {run_id!r} has no stored brief") from error
        except (OSError, ValidationError) as error:
            raise _fail(f"the stored brief of run {run_id!r} cannot be read") from error

    def append_answer(self, run_id: str, answer: RawAnswer) -> None:
        """Add one answer. An id that is already stored is refused, so a retry never duplicates."""
        path = self.run_dir(run_id) / ANSWERS_FILE_NAME
        with self._lock:
            if answer.response_id in {item.response_id for item in self._read_all(run_id)}:
                raise _fail(f"answer {answer.response_id!r} is already stored in run {run_id!r}")
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(answer.model_dump_json() + "\n")

    def load_answer(self, run_id: str, response_id: str) -> RawAnswer:
        """Read one answer by its id."""
        for answer in self._read_all(run_id):
            if answer.response_id == response_id:
                return answer
        raise _fail(f"run {run_id!r} has no stored answer {response_id!r}")

    def answer_ids(self, run_id: str) -> list[str]:
        """Ids of the stored answers, in the order they were sampled."""
        return [answer.response_id for answer in self._read_all(run_id)]

    def append_analysis(self, run_id: str, analysis: AnalyzedResponse) -> None:
        """Add the analysis of one answer. An id already analyzed is refused."""
        path = self.run_dir(run_id) / ANALYSES_FILE_NAME
        with self._lock:
            known = {item.response_id for item in self.load_analyses(run_id)}
            if analysis.response_id in known:
                raise _fail(f"answer {analysis.response_id!r} is already analyzed in {run_id!r}")
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(analysis.model_dump_json() + "\n")

    def load_analyses(self, run_id: str) -> list[AnalyzedResponse]:
        """Every analyzed answer of the run, in the order they were analyzed."""
        return self._read(run_id, ANALYSES_FILE_NAME, AnalyzedResponse)

    def _read_all(self, run_id: str) -> list[RawAnswer]:
        return self._read(run_id, ANSWERS_FILE_NAME, RawAnswer)

    def _read[T: BaseModel](self, run_id: str, file_name: str, model: type[T]) -> list[T]:
        path = self.run_dir(run_id) / file_name
        if not path.exists():
            return []
        items: list[T] = []
        try:
            with path.open(encoding="utf-8") as handle:
                for number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        items.append(model.model_validate_json(line))
                    except ValidationError as error:
                        raise _fail(
                            f"the stored data {file_name} of run {run_id!r} is damaged "
                            f"at line {number}"
                        ) from error
        except OSError as error:
            raise _fail(f"the stored data {file_name} of run {run_id!r} cannot be read") from error
        return items
