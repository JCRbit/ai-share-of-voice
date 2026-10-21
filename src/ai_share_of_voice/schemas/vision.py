"""Vision contracts: analyze_image(path, question) returns a finding (owner: An)."""

from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import Field

from ai_share_of_voice.schemas.base import Contract


class Confidence(StrEnum):
    """How sure the vision model is about a finding."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class VisionRequest(Contract):
    """Input of ``analyze_image``."""

    path: Path
    question: str = Field(min_length=1)


class VisionFinding(Contract):
    """What the vision model saw.

    A finding is always a hypothesis, never a cause: ``is_hypothesis`` cannot be false.
    """

    image_path: Path
    question: str = Field(min_length=1)
    observation: str = Field(min_length=1)
    confidence: Confidence
    is_hypothesis: Literal[True] = True
