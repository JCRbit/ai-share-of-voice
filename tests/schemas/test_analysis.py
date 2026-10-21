from datetime import datetime

import pytest
from pydantic import ValidationError

from ai_share_of_voice.schemas.analysis import AnalyzedResponse, BrandMention, RawAnswer, Tone


def test_raw_answer_requires_timezone_aware_datetime() -> None:
    with pytest.raises(ValidationError):
        RawAnswer(
            response_id="r1",
            question_id="q1",
            batch_index=0,
            model="m",
            text="answer",
            created_at=datetime(2026, 10, 21),
        )


def test_raw_answer_defaults_to_an_aware_timestamp() -> None:
    answer = RawAnswer(response_id="r1", question_id="q1", batch_index=0, model="m", text="a")
    assert answer.created_at.tzinfo is not None


def test_recommended_brand_must_be_mentioned() -> None:
    with pytest.raises(ValidationError):
        BrandMention(brand="Acme", mentioned=False, recommended=True)


def test_position_and_tone_require_a_mention() -> None:
    with pytest.raises(ValidationError):
        BrandMention(brand="Acme", mentioned=False, position=1)
    with pytest.raises(ValidationError):
        BrandMention(brand="Acme", mentioned=False, tone=Tone.POSITIVE)


def test_mention_without_recommendation_is_valid() -> None:
    mention = BrandMention(brand="Acme", mentioned=True, position=2, tone=Tone.NEUTRAL)
    assert mention.recommended is False


def test_position_starts_at_one() -> None:
    with pytest.raises(ValidationError):
        BrandMention(brand="Acme", mentioned=True, position=0)


def test_analyzed_response_with_sources() -> None:
    response = AnalyzedResponse(
        response_id="r1",
        question_id="q1",
        batch_index=0,
        mentions=(
            BrandMention(brand="Acme", mentioned=True, recommended=True, position=1),
            BrandMention(brand="Beta", mentioned=False),
        ),
        sources=("https://example.com/article",),
    )
    assert len(response.sources) == 1


def test_analyzed_response_rejects_duplicate_brands() -> None:
    with pytest.raises(ValidationError):
        AnalyzedResponse(
            response_id="r1",
            question_id="q1",
            batch_index=0,
            mentions=(
                BrandMention(brand="Acme", mentioned=True),
                BrandMention(brand="acme", mentioned=False),
            ),
        )


def test_analyzed_response_rejects_invalid_source_url() -> None:
    with pytest.raises(ValidationError):
        AnalyzedResponse(
            response_id="r1",
            question_id="q1",
            batch_index=0,
            mentions=(BrandMention(brand="Acme", mentioned=True),),
            sources=("not a url",),
        )
