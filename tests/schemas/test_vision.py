from pathlib import Path

import pytest
from pydantic import ValidationError

from ai_share_of_voice.schemas.vision import Confidence, VisionFinding, VisionRequest


def test_request_coerces_the_path() -> None:
    request = VisionRequest(path="data/screenshots/menu.png", question="Which prices are shown?")
    assert isinstance(request.path, Path)


def test_finding_is_always_a_hypothesis() -> None:
    finding = VisionFinding(
        image_path="chart.png",
        question="What does the chart show?",
        observation="The client's bar is the lowest.",
        confidence=Confidence.MEDIUM,
    )
    assert finding.is_hypothesis is True
    with pytest.raises(ValidationError):
        VisionFinding(
            image_path="chart.png",
            question="q",
            observation="o",
            confidence=Confidence.HIGH,
            is_hypothesis=False,
        )


def test_finding_rejects_unknown_confidence() -> None:
    with pytest.raises(ValidationError):
        VisionFinding(image_path="chart.png", question="q", observation="o", confidence="certain")
