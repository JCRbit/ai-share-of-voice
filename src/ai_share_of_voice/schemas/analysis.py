"""Analysis contracts: raw model answers and their analysis (owner: J)."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Self

from pydantic import AwareDatetime, Field, HttpUrl, model_validator

from ai_share_of_voice.schemas.base import Contract


class Tone(StrEnum):
    """Tone with which an answer talks about a brand."""

    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class RawAnswer(Contract):
    """One answer from the measured model, as returned by the sampling step."""

    response_id: str = Field(min_length=1)
    question_id: str = Field(min_length=1)
    batch_index: int = Field(ge=0)
    model: str = Field(min_length=1)
    text: str = Field(min_length=1)
    created_at: AwareDatetime = Field(default_factory=lambda: datetime.now(UTC))


class BrandMention(Contract):
    """What one answer says about one brand.

    A mention is not a recommendation: ``recommended`` implies ``mentioned``,
    but not the other way round.
    """

    brand: str = Field(min_length=1)
    mentioned: bool
    recommended: bool = False
    position: int | None = Field(default=None, ge=1)
    tone: Tone | None = None

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if self.recommended and not self.mentioned:
            raise ValueError("a brand cannot be recommended without being mentioned")
        if not self.mentioned and (self.position is not None or self.tone is not None):
            raise ValueError("position and tone require the brand to be mentioned")
        return self


class AnalyzedResponse(Contract):
    """Mention, recommendation, position, tone and sources extracted from one answer."""

    response_id: str = Field(min_length=1)
    question_id: str = Field(min_length=1)
    batch_index: int = Field(ge=0)
    mentions: tuple[BrandMention, ...] = Field(min_length=1)
    sources: tuple[HttpUrl, ...] = ()

    @model_validator(mode="after")
    def one_entry_per_brand(self) -> Self:
        brands = [mention.brand.casefold() for mention in self.mentions]
        if len(brands) != len(set(brands)):
            raise ValueError("each brand can appear only once per response")
        return self
