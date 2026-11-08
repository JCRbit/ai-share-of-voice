from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from ai_share_of_voice.exceptions import ConfigurationError
from ai_share_of_voice.llm_client import SimulatedClient
from ai_share_of_voice.sampling import (
    MAX_ANSWER_CHARS,
    SubjectAnswer,
    ask,
    build_prompt,
    load_panel,
    load_questions,
)
from ai_share_of_voice.schemas.questions import Question

ROOT = Path(__file__).parents[1]


def a_question() -> Question:
    text = "¿Qué app de comida a domicilio es mejor?"
    return Question(id="q01", text=text, intent="recommendation")


def test_the_prompt_is_the_question_and_names_no_brand() -> None:
    prompt = build_prompt(a_question())
    assert prompt.endswith("Question: ¿Qué app de comida a domicilio es mejor?")
    assert "answer field" in prompt and prompt.count("\n") == 2


@pytest.mark.parametrize(
    ("language", "file"), [("es", "questions_es.csv"), ("en", "questions.csv")]
)
def test_the_project_panels_load_by_language(language: str, file: str) -> None:
    panel = load_panel(language, ROOT / "data")
    assert len(panel.questions) == 30
    expected = load_questions(ROOT / "data" / file)
    assert [q.id for q in panel.questions] == [q.id for q in expected.questions]


def test_a_language_without_a_panel_is_refused() -> None:
    with pytest.raises(ConfigurationError, match="no question panel for 'fr'"):
        load_panel("fr", ROOT / "data")


def test_the_reply_schema_needs_an_answer_of_a_sane_size() -> None:
    assert SubjectAnswer(answer=" hola ").answer == "hola"
    for bad in ("", "   ", "x" * (MAX_ANSWER_CHARS + 1)):
        with pytest.raises(ValidationError):
            SubjectAnswer(answer=bad)


def test_asking_returns_the_text_and_the_usage_of_the_call() -> None:
    def respond(prompt: str, schema: type[BaseModel]) -> BaseModel:
        return schema(answer="Recomiendo la primera.")

    client = SimulatedClient(responder=respond)
    text, usage = ask(client, a_question())
    assert text == "Recomiendo la primera." and usage.input_tokens > 0
    assert len(client.calls) == 1
