from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from ai_share_of_voice.schemas.sources import (
    CheckUrlInput,
    ReadSourceInput,
    SearchSourcesInput,
    SourceDocument,
    SourceHit,
    UrlCheck,
)

NOW = datetime(2026, 10, 21, tzinfo=UTC)


def test_search_input_defaults_and_bounds() -> None:
    assert SearchSourcesInput(query="best pizza madrid").max_results == 5
    with pytest.raises(ValidationError):
        SearchSourcesInput(query="x", max_results=11)
    with pytest.raises(ValidationError):
        SearchSourcesInput(query="")


def test_source_hit_requires_a_valid_url() -> None:
    assert SourceHit(url="https://example.com/a", title="A").snippet == ""
    with pytest.raises(ValidationError):
        SourceHit(url="not a url", title="A")


def test_read_and_check_inputs_validate_urls() -> None:
    ReadSourceInput(url="https://example.com")
    CheckUrlInput(url="https://example.com")
    with pytest.raises(ValidationError):
        ReadSourceInput(url="ftp://example.com/file")


def test_source_document_is_always_untrusted() -> None:
    document = SourceDocument(url="https://example.com", text="content", fetched_at=NOW)
    assert document.untrusted is True
    with pytest.raises(ValidationError):
        SourceDocument(url="https://example.com", text="content", fetched_at=NOW, untrusted=False)


def test_url_check_existing_url_needs_a_status() -> None:
    with pytest.raises(ValidationError):
        UrlCheck(url="https://example.com", exists=True, allowed_by_robots=True, checked_at=NOW)


def test_url_check_missing_url_may_have_no_status() -> None:
    check = UrlCheck(
        url="https://example.com", exists=False, allowed_by_robots=True, checked_at=NOW
    )
    assert check.status_code is None
