"""Shared contracts. Change them only through a reviewed PR and notify the team."""

from ai_share_of_voice.schemas.analysis import AnalyzedResponse, BrandMention, RawAnswer, Tone
from ai_share_of_voice.schemas.base import Contract
from ai_share_of_voice.schemas.brief import BrandSpec, Brief
from ai_share_of_voice.schemas.events import EventStatus, GraphEvent
from ai_share_of_voice.schemas.llm import ModelClient, Usage
from ai_share_of_voice.schemas.questions import Question, QuestionPanel
from ai_share_of_voice.schemas.sources import (
    CheckUrlInput,
    ReadSourceInput,
    SearchSourcesInput,
    SourceDocument,
    SourceHit,
    UrlCheck,
)
from ai_share_of_voice.schemas.stopping import (
    BatchSummary,
    IntervalMethod,
    StopConfig,
    StopDecision,
    StopReason,
    StopRule,
    SummaryMetric,
)
from ai_share_of_voice.schemas.vision import Confidence, VisionFinding, VisionRequest

__all__ = [
    "AnalyzedResponse",
    "BatchSummary",
    "BrandMention",
    "BrandSpec",
    "Brief",
    "CheckUrlInput",
    "Confidence",
    "Contract",
    "EventStatus",
    "GraphEvent",
    "IntervalMethod",
    "ModelClient",
    "Question",
    "QuestionPanel",
    "RawAnswer",
    "ReadSourceInput",
    "SearchSourcesInput",
    "SourceDocument",
    "SourceHit",
    "StopConfig",
    "StopDecision",
    "StopReason",
    "StopRule",
    "SummaryMetric",
    "Tone",
    "UrlCheck",
    "Usage",
    "VisionFinding",
    "VisionRequest",
]
