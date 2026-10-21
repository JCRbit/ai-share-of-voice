import pytest
from pydantic import ValidationError

from ai_share_of_voice.schemas.questions import Question, QuestionPanel


def make_question(question_id: str = "q01") -> Question:
    return Question(
        id=question_id, text="Recommend a language school near my home", intent="recommendation"
    )


def test_question_mirrors_a_csv_row() -> None:
    question = make_question()
    assert question.id == "q01"
    assert question.intent == "recommendation"


def test_question_text_must_not_be_too_short() -> None:
    with pytest.raises(ValidationError):
        Question(id="q01", text="hi", intent="recommendation")


def test_panel_requires_at_least_one_question() -> None:
    with pytest.raises(ValidationError):
        QuestionPanel(questions=())


def test_panel_rejects_duplicate_ids() -> None:
    with pytest.raises(ValidationError, match="unique"):
        QuestionPanel(questions=(make_question("q01"), make_question("q01")))


def test_panel_with_distinct_ids_is_valid() -> None:
    panel = QuestionPanel(questions=(make_question("q01"), make_question("q02")))
    assert len(panel.questions) == 2
