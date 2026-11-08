"""What the measured model is asked and how (owner: An, task C2).

One place defines the prompt, the reply schema and the question panel, so the real run
(``real_nodes.py``) and the sample that is labelled by hand (``scripts/generate_sample.py``)
ask the model exactly the same thing.

* The prompt is the question and nothing else: no brand, no sector hint, no instruction that
  could push the answer towards a brand.
* The reply travels in a one-field schema because ``ModelClient.complete`` returns structured
  data; the field holds the whole answer as the model would have written it.
* The question panel is a fixed CSV per language (``id,text,intent``). It is written for food
  delivery in Madrid; another sector needs another panel.
"""

import csv
from pathlib import Path

from pydantic import Field

from ai_share_of_voice.exceptions import ConfigurationError
from ai_share_of_voice.schemas.base import Contract
from ai_share_of_voice.schemas.llm import ModelClient, Usage
from ai_share_of_voice.schemas.questions import Question, QuestionPanel

MAX_ANSWER_CHARS = 20_000
PANEL_FILES = {"es": "questions_es.csv", "en": "questions.csv"}


class SubjectAnswer(Contract):
    """Schema of the measured model's reply: the whole answer in one field."""

    answer: str = Field(min_length=1, max_length=MAX_ANSWER_CHARS)


def build_prompt(question: Question) -> str:
    """The question, as a user would ask it; nothing that could name or hint at a brand."""
    return (
        "Answer the question as a helpful assistant would, in the language of the question. "
        "Write your whole answer in the answer field.\n\n"
        f"Question: {question.text}"
    )


def load_questions(path: Path | str) -> QuestionPanel:
    """Read the question panel from a CSV with the columns ``id,text,intent``."""
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return QuestionPanel(questions=tuple(Question(**row) for row in csv.DictReader(handle)))


def load_panel(language: str, data_dir: Path | str = "data") -> QuestionPanel:
    """The fixed question panel of ``language`` (``es`` or ``en``) from ``data_dir``."""
    if language not in PANEL_FILES:
        known = sorted(PANEL_FILES)
        raise ConfigurationError(f"there is no question panel for {language!r}; use one of {known}")
    return load_questions(Path(data_dir) / PANEL_FILES[language])


def ask(client: ModelClient, question: Question) -> tuple[str, Usage]:
    """Ask the measured model one question; returns its answer and the usage of the call."""
    reply, usage = client.complete(build_prompt(question), SubjectAnswer)
    return reply.answer, usage
