"""Question panel for any sector (R6, local integration, E).

The demo panel is a file written by hand. For another client the agent model drafts the
questions from the sector, city and language, and code decides what is kept:

* a question that names a tracked brand or variant (whole word, ignoring case and accents)
  is rejected: it would measure an answer to our own prompt, not the model's spontaneous
  recall. The prompt does not even list the brands, so it cannot suggest them;
* duplicates (ignoring case and accents), unknown intents and invalid lengths are dropped;
* missing questions are asked for once more; under ``MIN_RELIABLE_QUESTIONS`` the run does
  not start, because the question-level bootstrap would be too narrow.

The panel is saved in the same CSV format as ``data/questions.csv``, so the same questions
can be reused by later runs of the client and the runs stay comparable.
"""

import csv
import re
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from ai_share_of_voice.errors import ToolFailure
from ai_share_of_voice.schemas.brief import Brief, normalize_brand_name
from ai_share_of_voice.schemas.llm import ModelClient
from ai_share_of_voice.schemas.questions import Question, QuestionPanel
from ai_share_of_voice.stats.metrics import MIN_RELIABLE_QUESTIONS

TOOL_NAME = "question_generator"
DEFAULT_PANEL_SIZE = 30
# Same intents as the hand-written demo panel, so generated panels compare with it.
INTENTS = ("recommendation", "comparison", "price", "service_area", "trust_reviews")
LANGUAGE_NAMES = {"es": "Spanish (es)", "en": "English (en)"}


class DraftQuestion(BaseModel):
    text: str
    intent: str


class DraftQuestions(BaseModel):
    """Schema the agent model answers with. Plain model: extra keys are ignored."""

    questions: list[DraftQuestion] = Field(default_factory=list)


def build_prompt(brief: Brief, count: int) -> str:
    language = LANGUAGE_NAMES.get(brief.language, brief.language)
    return (
        "You write questions that ordinary people ask an AI assistant when they look for a "
        "business.\n\n"
        f"Sector: {brief.sector}\nCity: {brief.city}\nLanguage: {language}\n"
        f"Number of questions: {count}\n\n"
        "Rules:\n"
        "- Write natural, varied questions in the requested language, as a real user would.\n"
        "- Never name any brand, company, store, app or website. The goal is to see which "
        "businesses the assistant brings up on its own.\n"
        f"- Spread the questions evenly over these intents: {', '.join(INTENTS)}.\n"
        "- service_area questions ask where to go or what is available nearby; "
        "trust_reviews questions ask what is reliable or well rated.\n"
        "- Each question is one sentence of at most 200 characters.\n"
    )


def _names_a_brand(text: str, brief: Brief) -> bool:
    haystack = normalize_brand_name(text)
    return any(
        re.search(rf"(?<!\w){re.escape(normalize_brand_name(name))}(?!\w)", haystack)
        for brand in (brief.brand, *brief.competitors)
        for name in brand.all_names()
    )


def _accept(
    drafts: DraftQuestions, brief: Brief, seen: set[str], limit: int
) -> list[tuple[str, str]]:
    """Valid (text, intent) pairs, at most ``limit``; ``seen`` collects normalized texts."""
    kept = []
    for draft in drafts.questions:
        text = " ".join(draft.text.split())
        intent = draft.intent.strip().lower()
        key = normalize_brand_name(text)
        if (
            len(kept) == limit
            or intent not in INTENTS
            or not 5 <= len(text) <= 500
            or key in seen
            or _names_a_brand(text, brief)
        ):
            continue
        seen.add(key)
        kept.append((text, intent))
    return kept


def generate_panel(
    brief: Brief, client: ModelClient, size: int = DEFAULT_PANEL_SIZE
) -> QuestionPanel:
    """Draft, check and number a panel of up to ``size`` questions (two calls at most)."""
    seen: set[str] = set()
    accepted: list[tuple[str, str]] = []
    for _attempt in range(2):
        missing = size - len(accepted)
        if missing == 0:
            break
        try:
            drafts, _usage = client.complete(build_prompt(brief, missing), DraftQuestions)
        except (ValueError, ValidationError):
            continue  # an invalid answer: the second call may still work
        accepted += _accept(drafts, brief, seen, missing)

    if len(accepted) < min(size, MIN_RELIABLE_QUESTIONS):
        raise ToolFailure(
            TOOL_NAME,
            f"only {len(accepted)} valid questions; at least "
            f"{min(size, MIN_RELIABLE_QUESTIONS)} are needed",
        )
    return QuestionPanel(
        questions=tuple(
            Question(id=f"q{index:02d}", text=text, intent=intent)
            for index, (text, intent) in enumerate(accepted, start=1)
        )
    )


def write_panel_file(panel: QuestionPanel, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "text", "intent"])
        writer.writeheader()
        writer.writerows(question.model_dump() for question in panel.questions)
    return path


def read_panel_file(path: Path) -> QuestionPanel:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return QuestionPanel(questions=tuple(Question.model_validate(row) for row in rows))
