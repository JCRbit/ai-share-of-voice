"""Question contracts: the panel of representative questions (owners: J with Ab)."""

from typing import Self

from pydantic import Field, model_validator

from ai_share_of_voice.schemas.base import Contract


class Question(Contract):
    """One question asked to the measured model; mirrors a row of ``data/questions.csv``."""

    id: str = Field(min_length=1)
    text: str = Field(min_length=5, max_length=500)
    intent: str = Field(min_length=1, max_length=100)


class QuestionPanel(Contract):
    """The set of questions sampled in every batch."""

    questions: tuple[Question, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_ids(self) -> Self:
        """Reject two questions with the same id."""
        ids = [question.id for question in self.questions]
        if len(ids) != len(set(ids)):
            raise ValueError("question ids must be unique")
        return self
