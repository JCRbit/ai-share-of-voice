import base64
from pathlib import Path
from typing import Any

from pydantic import Field

from ai_share_of_voice.schemas.base import Contract
from ai_share_of_voice.schemas.llm import ModelClient
from ai_share_of_voice.schemas.vision import Confidence, VisionFinding, VisionRequest


class ModelVisionResponse(Contract):
    """Internal contract to strictly type what the LLM is allowed to return."""

    observation: str = Field(min_length=1)
    confidence: Confidence


class VisionTool:
    """Agent-facing wrapper: validates arguments and runs image analysis."""

    name = "analyze_image"
    description = "Analyze an image (chart or screenshot) to answer a specific question."
    args_schema = VisionRequest

    def __init__(self, client: ModelClient) -> None:
        self._client = client

    def run(self, **arguments: Any) -> VisionFinding:
        """Validate arguments against args_schema and extract the finding."""
        data = self.args_schema(**arguments)

        if not data.path.exists():
            raise FileNotFoundError(f"Image not found at path: {data.path}")

        # TODO: When llm_client.py supports vision payloads, inject base64 encoded image here.
        # Currently passes path as text to satisfy the existing ModelClient signature.
        prompt = (
            f"You are a vision analysis AI. Analyze the image located at: {data.path}\n\n"
            f"Question: {data.question}\n\n"
            "Provide a detailed observation and your confidence level."
        )

        # Execute the restricted prompt
        response, _ = self._client.complete(prompt, ModelVisionResponse)

        # Reconstruct the final VisionFinding to guarantee path/question accuracy
        return VisionFinding(
            image_path=data.path,
            question=data.question,
            observation=response.observation,
            confidence=response.confidence,
            is_hypothesis=True,
        )
