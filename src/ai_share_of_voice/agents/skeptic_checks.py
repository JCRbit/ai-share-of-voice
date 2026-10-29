"""Checks the Skeptic runs in code before asking a model (owner: J).

Anything that can be verified against the run record is verified here, because code is
cheap and does not get tricked by the text it reads. A figure or a source that is not in the
record is an objection; so is a causal claim that rests on a source, and any claim of
market share. Objections are tied to the sentence they come from.
"""

import re
from collections.abc import Callable
from enum import StrEnum
from urllib.parse import urlsplit

from pydantic import Field

from ai_share_of_voice.schemas.base import Contract

MAX_CLAIM_CHARS = 160
PERCENT_TOLERANCE = 0.5  # percentage points, to allow for rounding

_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+|\n+")
_PERCENT = re.compile(r"(?<![\w.])(\d{1,3}(?:[.,]\d+)?)\s?%")
_ANSWER_COUNT = re.compile(r"\b(\d[\d.,]*)\s+(?:answers|responses|respuestas)\b", re.IGNORECASE)
_URL = re.compile(r"https?://[^\s<>\"'\[\]]+", re.IGNORECASE)
_MARKER = re.compile(r"\[S(\d+)\]")
_CAUSAL = re.compile(
    r"\b(because|caused?|causes|due to|thanks to|as a result of|leads? to|led to|"
    r"driven by|explains why|porque|debido a|gracias a)\b",
    re.IGNORECASE,
)
_EVIDENCE = re.compile(
    r"\[S\d+\]|https?://|\b(sources?|pages?|websites?|screenshots?|images?|menus?|"
    r"fuentes?)\b",
    re.IGNORECASE,
)
_MARKET_SHARE = re.compile(r"\bmarket[- ]share\b|\bcuota de mercado\b", re.IGNORECASE)
_NEGATION_BEFORE = re.compile(
    r"\b(not|no|never|cannot|can't|does not|doesn't|neither|nor|without|sin)\b"
    r"(?:\W+\w+){0,4}\W*$",
    re.IGNORECASE,
)
_CONTRAST = re.compile(
    r"\b(?:but|however|whereas|yet|pero|sin embargo|mientras que)\b",
    re.IGNORECASE,
)
_URL_TRAILING = ".,;:!?)}"


class ObjectionKind(StrEnum):
    """What is wrong with a claim."""

    UNRECORDED_FIGURE = "unrecorded_figure"
    UNKNOWN_SOURCE = "unknown_source"
    CAUSAL_CLAIM = "causal_claim"
    MARKET_SHARE_CLAIM = "market_share_claim"
    EMPTY_DRAFT = "empty_draft"
    MENTION_VS_RECOMMENDATION = "mention_vs_recommendation"
    CORRELATION_VS_CAUSE = "correlation_vs_cause"
    UNVERIFIABLE_CLAIM = "unverifiable_claim"


class Objection(Contract):
    """One objection, tied to the claim it questions."""

    kind: ObjectionKind
    claim: str = Field(min_length=1, max_length=MAX_CLAIM_CHARS)
    reason: str = Field(min_length=3, max_length=300)

    def render(self) -> str:
        """The one-line form that goes into the graph state."""
        return f'[{self.kind.value}] "{self.claim}": {self.reason}'


class RunRecord(Contract):
    """What the run recorded: the only figures and sources a draft may rely on."""

    sources: tuple[str, ...] = ()
    percentages: tuple[float, ...] = ()
    answer_counts: tuple[int, ...] = ()


def squash(text: str) -> str:
    """Collapse whitespace so quotes and sentences compare reliably."""
    return " ".join(text.split())


def sentences(draft: str) -> list[str]:
    """Split a draft into sentences and lines, dropping empty ones."""
    return [squash(part) for part in _SENTENCE_BREAK.split(draft) if part.strip()]


def _clauses(sentence: str) -> list[str]:
    """Split a sentence at contrastive conjunctions for local checks."""
    return [part.strip() for part in _CONTRAST.split(sentence) if part.strip()]


def _is_negated_before(text: str, position: int) -> bool:
    """Detect a nearby negation before a trigger, not after the claim."""
    return bool(_NEGATION_BEFORE.search(text[:position]))


def _excerpt(sentence: str) -> str:
    text = squash(sentence).lstrip("-*#> ")
    if len(text) <= MAX_CLAIM_CHARS:
        return text
    return text[: MAX_CLAIM_CHARS - 3] + "..."


def _objection(kind: ObjectionKind, sentence: str, reason: str) -> Objection:
    return Objection(kind=kind, claim=_excerpt(sentence), reason=reason)


def _normalize_url(url: str) -> str:
    parts = urlsplit(url.rstrip(_URL_TRAILING))
    query = f"?{parts.query}" if parts.query else ""
    return f"{parts.scheme.lower()}://{parts.netloc.lower()}{parts.path.rstrip('/')}{query}"


def _number(raw: str) -> float:
    return float(raw.replace(",", "."))


def _is_recorded(raw: str, record: RunRecord) -> bool:
    value = _number(raw)
    return any(abs(value - known) <= PERCENT_TOLERANCE for known in record.percentages)


def check_figures(draft: str, record: RunRecord) -> list[Objection]:
    """Every percentage and every answer count must be one the run recorded."""
    found: list[Objection] = []
    for sentence in sentences(draft):
        wrong = [f"{raw}%" for raw in _PERCENT.findall(sentence) if not _is_recorded(raw, record)]
        wrong += [
            f"{raw} answers"
            for raw in _ANSWER_COUNT.findall(sentence)
            if int(re.sub(r"[.,]", "", raw)) not in record.answer_counts
        ]
        if wrong:
            reason = f"{', '.join(wrong)} not among the recorded figures"
            found.append(_objection(ObjectionKind.UNRECORDED_FIGURE, sentence, reason))
    return found


def check_sources(draft: str, record: RunRecord) -> list[Objection]:
    """Every URL and every [S#] marker must point to a recorded source."""
    known = {_normalize_url(url) for url in record.sources}
    found: list[Objection] = []
    for sentence in sentences(draft):
        problems = [
            url.rstrip(_URL_TRAILING)
            for url in _URL.findall(sentence)
            if _normalize_url(url) not in known
        ]
        problems += [
            f"[S{number}]"
            for number in _MARKER.findall(sentence)
            if not 1 <= int(number) <= len(record.sources)
        ]
        if problems:
            reason = f"{', '.join(problems)} not among the recorded sources"
            found.append(_objection(ObjectionKind.UNKNOWN_SOURCE, sentence, reason))
    return found


def check_causality(draft: str) -> list[Objection]:
    """A source or an image can show what an assistant may rely on, not cause a recommendation."""
    reason = "claims a cause that the evidence cannot show, only a correlation"
    found: list[Objection] = []

    for sentence in sentences(draft):
        for clause in _clauses(sentence):
            causal_match = _CAUSAL.search(clause)
            if (
                causal_match
                and _EVIDENCE.search(clause)
                and not _is_negated_before(clause, causal_match.start())
            ):
                found.append(_objection(ObjectionKind.CAUSAL_CLAIM, sentence, reason))
                break

    return found


def check_market_share(draft: str) -> list[Objection]:
    """The report measures visibility in AI answers; it never states a market share."""
    reason = "visibility in AI answers is not market share"
    found: list[Objection] = []

    for sentence in sentences(draft):
        if any(
            (match := _MARKET_SHARE.search(clause))
            and not _is_negated_before(clause, match.start())
            for clause in _clauses(sentence)
        ):
            found.append(_objection(ObjectionKind.MARKET_SHARE_CLAIM, sentence, reason))

    return found


def run_checks(draft: str, record: RunRecord) -> list[Objection]:
    """Run every check once, without repeating the same claim and kind."""
    if not draft.strip():
        return [
            Objection(
                kind=ObjectionKind.EMPTY_DRAFT,
                claim="(empty draft)",
                reason="there is nothing to review",
            )
        ]
    checks: list[Callable[[], list[Objection]]] = [
        lambda: check_figures(draft, record),
        lambda: check_sources(draft, record),
        lambda: check_causality(draft),
        lambda: check_market_share(draft),
    ]
    seen: set[tuple[ObjectionKind, str]] = set()
    found: list[Objection] = []
    for check in checks:
        for objection in check():
            key = (objection.kind, objection.claim)
            if key not in seen:
                seen.add(key)
                found.append(objection)
    return found
