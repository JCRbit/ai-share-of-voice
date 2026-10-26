"""How the graph classifies failures before closing a run in a partial state (owner: J).

Three situations must end a run cleanly with a logged reason instead of a crash:
a tool that fails, a run with no usable evidence, and an exhausted spending cap.
Anything else unexpected is closed the same way, and its type is kept in the reason.
"""

from enum import StrEnum

from pydantic import Field

from ai_share_of_voice.exceptions import BudgetExceeded
from ai_share_of_voice.schemas.base import Contract

MAX_REASON_CHARS = 500


class FailureKind(StrEnum):
    """Why a run was closed in a partial state."""

    TOOL_FAILURE = "tool_failure"
    NO_EVIDENCE = "no_evidence"
    BUDGET_EXHAUSTED = "budget_exhausted"
    UNEXPECTED = "unexpected"


class ToolFailure(Exception):
    """A tool could not do its job: unreachable server, bad response, timeout."""

    def __init__(self, tool: str, reason: str) -> None:
        super().__init__(f"{tool} failed: {reason}")
        self.tool = tool
        self.reason = reason


class NoEvidence(Exception):
    """The run has nothing usable to report, such as no analyzable answers or no sources."""


class Failure(Contract):
    """A classified failure: what kind it is, why, and which tool was involved."""

    kind: FailureKind
    reason: str = Field(min_length=1, max_length=MAX_REASON_CHARS)
    tool: str | None = None


def _shorten(text: str) -> str:
    clean = " ".join(text.split())
    if len(clean) <= MAX_REASON_CHARS:
        return clean
    return clean[: MAX_REASON_CHARS - 3] + "..."


def classify(error: BaseException) -> Failure:
    """Turn an exception raised by a node into a ``Failure``."""
    if isinstance(error, BudgetExceeded):
        return Failure(kind=FailureKind.BUDGET_EXHAUSTED, reason=_shorten(str(error)))
    if isinstance(error, ToolFailure):
        return Failure(kind=FailureKind.TOOL_FAILURE, reason=_shorten(str(error)), tool=error.tool)
    if isinstance(error, NoEvidence):
        reason = _shorten(str(error)) or "No usable evidence was found"
        return Failure(kind=FailureKind.NO_EVIDENCE, reason=reason)
    name = type(error).__name__
    message = str(error).strip()
    reason = _shorten(f"{name}: {message}" if message else name)
    return Failure(kind=FailureKind.UNEXPECTED, reason=reason)
