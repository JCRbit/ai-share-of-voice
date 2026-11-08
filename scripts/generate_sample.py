"""Generate the real answer sample for hand labelling (owner: An, task I7).

It asks the measured model the questions of ``data/questions_es.csv`` (the primary language,
decision 8) and saves each answer to ``data/sample_answers.csv`` with the columns of
``RawAnswer`` (``schemas/analysis.py``), so the file can be read back with that contract.

    python scripts/generate_sample.py                 # 60 answers, run cap 0.50 USD
    python scripts/generate_sample.py --answers 30 --cap 0.20

Rules the script enforces:

* Live only. With ``LLM_MODE=simulated`` it stops, because invented answers must never end up in
  a file called a real sample. It needs the ``.env`` of ``.env.example`` with ``LLM_MODE=live``.
* Every call goes through the budgeted client: the run cap (``--cap``), the person cap
  (``PERSON_CAP_USD``, summed over ``runs/llm_calls.jsonl``) and the call log all apply. When a
  cap would be passed it stops cleanly and keeps what it has.
* Resumable: rerunning skips the answers already in the file. Answer ``i`` is question
  ``i % n`` in repetition ``i // n``, so 60 answers over 30 questions are two complete passes over
  the question panel.
* The prompt is the question and nothing else, so no brand is ever named to the model.
* The text is untrusted and the file will be opened in a spreadsheet. A text that starts with
  ``=``, ``+``, ``-`` or ``@`` gets a leading ``'`` so it is shown, not run, as a formula.
"""

import argparse
import csv
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ai_share_of_voice.errors import ToolFailure
from ai_share_of_voice.exceptions import BudgetExceeded, ConfigurationError
from ai_share_of_voice.llm_client import SimulatedClient, Transport, client_from_env
from ai_share_of_voice.sampling import SubjectAnswer, build_prompt, load_questions
from ai_share_of_voice.schemas.analysis import RawAnswer
from ai_share_of_voice.schemas.llm import ModelClient
from ai_share_of_voice.schemas.questions import Question, QuestionPanel

DEFAULT_QUESTIONS = Path("data/questions_es.csv")
DEFAULT_OUTPUT = Path("data/sample_answers.csv")
DEFAULT_LOG = Path("runs/llm_calls.jsonl")
DEFAULT_ANSWERS = 60
DEFAULT_CAP_USD = 0.50
MAX_CONSECUTIVE_FAILURES = 3
COLUMNS = ("response_id", "question_id", "batch_index", "model", "created_at", "text")
_FORMULA_STARTS = ("=", "+", "-", "@", "\t", "\r")


@dataclass
class Report:
    """What a run did."""

    written: int = 0
    skipped: int = 0
    failed: list[str] = field(default_factory=list)
    stopped: str | None = None


def plan(panel: QuestionPanel, total: int) -> list[tuple[Question, int]]:
    """The ``total`` (question, repetition) pairs, round-robin over the questions."""
    questions = panel.questions
    return [(questions[i % len(questions)], i // len(questions)) for i in range(total)]


def response_id(question: Question, repetition: int) -> str:
    """Stable id, so a rerun recognises the answers it already has."""
    return f"sample-{question.id}-b{repetition}"


def neutralize(text: str) -> str:
    """Make a text safe to open in a spreadsheet (see the module docstring)."""
    return f"'{text}" if text.startswith(_FORMULA_STARTS) else text


def done_ids(path: Path) -> set[str]:
    """Response ids already saved in ``path``; raises if the file is not a sample file."""
    if not path.exists() or path.stat().st_size == 0:
        return set()
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise ValueError(f"{path} does not have the columns {list(COLUMNS)}; move it away")
        return {row["response_id"] for row in reader}


def append_row(path: Path, answer: RawAnswer) -> None:
    """Add one answer, writing the header first if the file is new."""
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if is_new:
            writer.writerow(COLUMNS)
        writer.writerow(
            [
                answer.response_id,
                answer.question_id,
                answer.batch_index,
                answer.model,
                answer.created_at.isoformat(),
                neutralize(answer.text),
            ]
        )


def count_pending(panel: QuestionPanel, output: Path, total: int) -> int:
    """How many of the ``total`` answers are not in ``output`` yet."""
    done = done_ids(output)
    wanted = (response_id(question, repetition) for question, repetition in plan(panel, total))
    return sum(1 for rid in wanted if rid not in done)


def generate(client: ModelClient, panel: QuestionPanel, output: Path, total: int) -> Report:
    """Ask for the missing answers and append them to ``output`` one by one."""
    report = Report()
    done = done_ids(output)
    consecutive_failures = 0
    for question, repetition in plan(panel, total):
        rid = response_id(question, repetition)
        if rid in done:
            report.skipped += 1
            continue
        try:
            reply, usage = client.complete(build_prompt(question), SubjectAnswer)
        except BudgetExceeded as error:
            report.stopped = f"spending cap reached: {error}"
            return report
        except ToolFailure as error:
            report.failed.append(f"{rid}: {error}")
            consecutive_failures += 1
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                report.stopped = f"{consecutive_failures} failures in a row; check the provider"
                return report
            continue
        consecutive_failures = 0
        answer = RawAnswer(
            response_id=rid,
            question_id=question.id,
            batch_index=repetition,
            model=usage.model,
            text=reply.answer,
        )
        append_row(output, answer)
        report.written += 1
    return report


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the real answer sample (task I7).")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG, help="call log, shared by runs")
    parser.add_argument("--answers", type=int, default=DEFAULT_ANSWERS)
    parser.add_argument("--cap", type=float, default=DEFAULT_CAP_USD, help="run cap in USD")
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
    transport: Transport | None = None,
) -> int:
    """Run the script. Returns 0 when done, 1 when stopped early, 2 for a setup problem."""
    args = _parse_args(argv)
    if args.answers < 1 or args.cap <= 0:
        print("error: --answers and --cap must be greater than zero", file=sys.stderr)
        return 2
    try:
        panel = load_questions(args.questions)
        client = client_from_env(
            env, run_id="sample", log_path=args.log, cap_usd=args.cap, transport=transport
        )
        if isinstance(client, SimulatedClient):
            raise ConfigurationError("set LLM_MODE=live: simulated answers are not a real sample")
        pending = count_pending(panel, args.output, args.answers)
    except (OSError, ValueError, ConfigurationError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(f"{pending} answers to ask, run cap ${args.cap:.2f}, output {args.output}")
    report = generate(client, panel, args.output, args.answers)
    print(
        f"written {report.written}, already there {report.skipped}, failed {len(report.failed)}, "
        f"spent ${client.spent_usd:.4f}"
    )
    for line in report.failed:
        print(f"failed: {line}", file=sys.stderr)
    if report.stopped:
        print(f"stopped: {report.stopped}. Run again to continue.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
