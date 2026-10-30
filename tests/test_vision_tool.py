import pytest
from pathlib import Path
from pydantic import BaseModel

from ai_share_of_voice.schemas.llm import Usage
from ai_share_of_voice.schemas.vision import Confidence, VisionFinding, VisionRequest
from ai_share_of_voice.tools.vision_tool import ModelVisionResponse, VisionTool


class MockVisionClient:
    """Fake model client: answers with a fixed observation and records the prompt."""

    def __init__(self, observation: str, confidence: Confidence) -> None:
        self.observation = observation
        self.confidence = confidence
        self.prompts: list[str] = []

    def complete(self, prompt: str, schema: type[BaseModel]) -> tuple[BaseModel, Usage]:
        self.prompts.append(prompt)
        usage = Usage(model="fake-vision", input_tokens=100, output_tokens=50, cost_usd=0.01)
        return schema(observation=self.observation, confidence=self.confidence), usage


@pytest.fixture
def mock_screenshot(tmp_path: Path) -> Path:
    file = tmp_path / "glovo-madrid-food-listing.jpg"
    file.write_bytes(b"fake_screenshot_data")
    return file


@pytest.fixture
def mock_chart(tmp_path: Path) -> Path:
    file = tmp_path / "share_of_voice_chart.png"
    file.write_bytes(b"fake_chart_data")
    return file


def test_tool_exposes_standard_interface() -> None:
    assert VisionTool.name == "analyze_image"
    assert VisionTool.args_schema is VisionRequest
    assert VisionTool.description


def test_tool_analyzes_screenshot_and_returns_finding(mock_screenshot: Path) -> None:
    client = MockVisionClient("Acme is listed first in the app.", Confidence.HIGH)
    tool = VisionTool(client)

    result = tool.run(path=mock_screenshot, question="Who is listed first?")

    assert isinstance(result, VisionFinding)
    assert result.image_path == mock_screenshot
    assert result.question == "Who is listed first?"
    assert result.observation == "Acme is listed first in the app."
    assert result.confidence == Confidence.HIGH
    assert result.is_hypothesis is True
    assert str(mock_screenshot) in client.prompts[0]


def test_tool_analyzes_chart_and_returns_finding(mock_chart: Path) -> None:
    client = MockVisionClient("Beta Eats has a 32% share of voice.", Confidence.MEDIUM)
    tool = VisionTool(client)

    result = tool.run(path=mock_chart, question="What is the SOV for Beta Eats?")

    assert result.image_path == mock_chart
    assert result.observation == "Beta Eats has a 32% share of voice."
    assert result.confidence == Confidence.MEDIUM


def test_tool_raises_error_if_image_path_does_not_exist(tmp_path: Path) -> None:
    client = MockVisionClient("Will not execute", Confidence.LOW)
    tool = VisionTool(client)
    missing_file = tmp_path / "missing-image.png"

    with pytest.raises(FileNotFoundError, match="Image not found"):
        tool.run(path=missing_file, question="What is this?")
