from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from ai_share_of_voice.errors import ToolFailure
from ai_share_of_voice.schemas.sources import CheckUrlInput, ReadSourceInput, SearchSourcesInput
from ai_share_of_voice.tools.sources_tool import (
    MAX_TEXT_CHARS,
    USER_AGENT,
    SourcesService,
    injection_flags,
    untrusted_block,
)

FIXTURES = Path(__file__).parent / "fixtures" / "sources"
NOW = datetime(2026, 11, 2, 12, 0, tzinfo=UTC)
ARTICLE = "https://www.example.com/mejores-apps-comida?a=1"
EVIL = "https://evil.example.net/ranking"


def html(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def site(routes: dict[str, httpx.Response]):
    """A fake web: exact URL -> response; anything else is a 404. Records every request."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        url = str(request.url)
        if url in routes:
            return routes[url]
        return httpx.Response(404, text="not found")

    service = SourcesService(
        http=httpx.Client(transport=httpx.MockTransport(handler)), now=lambda: NOW
    )
    return service, seen


def page(body: str, content_type: str = "text/html; charset=utf-8") -> httpx.Response:
    return httpx.Response(200, text=body, headers={"content-type": content_type})


ALLOW_ALL = httpx.Response(200, text="User-agent: *\nAllow: /\n")
DENY_US = httpx.Response(200, text=f"User-agent: {USER_AGENT.split('/')[0]}\nDisallow: /\n")


# --- search ---------------------------------------------------------------------------------


def test_search_returns_the_real_target_urls_and_skips_ads():
    service, seen = site(
        {"https://html.duckduckgo.com/html/?q=apps+comida+Madrid": page(html("ddg_results.html"))}
    )

    hits = service.search_sources(SearchSourcesInput(query="apps comida Madrid", max_results=5))

    assert [str(hit.url) for hit in hits] == [ARTICLE, "https://news.example.org/delivery-2026"]
    assert hits[0].title == "Las mejores apps de comida a domicilio en Madrid"
    assert "Glovo" in hits[0].snippet
    assert seen[0].headers["user-agent"] == USER_AGENT


def test_search_respects_max_results():
    service, _ = site({"https://html.duckduckgo.com/html/?q=x": page(html("ddg_results.html"))})

    assert len(service.search_sources(SearchSourcesInput(query="x", max_results=1))) == 1


def test_a_blocked_search_is_a_tool_failure_and_is_not_retried():
    # DuckDuckGo answers bots with a challenge page; we report it, never bypass it.
    challenge = httpx.Response(202, text="<html>anomaly detected, please confirm</html>")
    service, seen = site({"https://html.duckduckgo.com/html/?q=x": challenge})

    with pytest.raises(ToolFailure, match="search"):
        service.search_sources(SearchSourcesInput(query="x"))
    assert len(seen) == 1


# --- check_url ------------------------------------------------------------------------------


def test_check_url_reports_status_and_robots():
    service, _ = site(
        {"https://www.example.com/robots.txt": ALLOW_ALL, ARTICLE: page("<html></html>")}
    )

    check = service.check_url(CheckUrlInput(url=ARTICLE))

    assert (check.exists, check.status_code, check.allowed_by_robots) == (True, 200, True)
    assert check.checked_at == NOW


def test_check_url_of_a_missing_page():
    service, _ = site({"https://www.example.com/robots.txt": ALLOW_ALL})

    check = service.check_url(CheckUrlInput(url=ARTICLE))

    assert (check.exists, check.status_code) == (False, 404)


def test_an_unreachable_host_does_not_exist():
    def handler(request):
        raise httpx.ConnectError("no route", request=request)

    service = SourcesService(
        http=httpx.Client(transport=httpx.MockTransport(handler)), now=lambda: NOW
    )

    check = service.check_url(CheckUrlInput(url=ARTICLE))

    assert (check.exists, check.status_code, check.allowed_by_robots) == (False, None, False)


@pytest.mark.parametrize(
    ("robots", "allowed"),
    [
        (httpx.Response(404), True),  # no robots.txt: everything allowed (RFC 9309)
        (httpx.Response(503), False),  # server error: assume everything is disallowed
        (DENY_US, False),
    ],
)
def test_robots_rules_follow_rfc_9309(robots, allowed):
    service, _ = site({"https://www.example.com/robots.txt": robots, ARTICLE: page("<p>x</p>")})

    assert service.check_url(CheckUrlInput(url=ARTICLE)).allowed_by_robots is allowed


def test_robots_is_fetched_once_per_host():
    other = "https://www.example.com/otra"
    service, seen = site(
        {"https://www.example.com/robots.txt": ALLOW_ALL, ARTICLE: page("a"), other: page("b")}
    )

    service.check_url(CheckUrlInput(url=ARTICLE))
    service.check_url(CheckUrlInput(url=other))

    assert sum(str(request.url).endswith("/robots.txt") for request in seen) == 1


# --- read_source ----------------------------------------------------------------------------


def test_read_source_returns_visible_text_only_as_untrusted():
    service, _ = site(
        {"https://evil.example.net/robots.txt": ALLOW_ALL, EVIL: page(html("malicious_page.html"))}
    )

    document = service.read_source(ReadSourceInput(url=EVIL))

    assert document.untrusted is True
    assert document.title == "Ranking de apps de reparto"
    assert "Glovo y Uber Eats son las más usadas" in document.text
    assert "stealCookies" not in document.text and ".hidden" not in document.text
    assert document.fetched_at == NOW


def test_read_source_refuses_a_page_robots_disallows():
    service, seen = site({"https://www.example.com/robots.txt": DENY_US, ARTICLE: page("x")})

    with pytest.raises(ToolFailure, match="robots"):
        service.read_source(ReadSourceInput(url=ARTICLE))
    assert not any(str(request.url) == ARTICLE for request in seen)  # the page is never fetched


def test_read_source_refuses_content_that_is_not_text():
    service, _ = site(
        {
            "https://www.example.com/robots.txt": ALLOW_ALL,
            ARTICLE: page("%PDF-1.7", content_type="application/pdf"),
        }
    )

    with pytest.raises(ToolFailure, match="content type"):
        service.read_source(ReadSourceInput(url=ARTICLE))


def test_long_pages_are_cut_to_the_contract_limit():
    long_page = "<p>" + "palabra " * 30_000 + "</p>"
    service, _ = site({"https://www.example.com/robots.txt": ALLOW_ALL, ARTICLE: page(long_page)})

    document = service.read_source(ReadSourceInput(url=ARTICLE))

    assert len(document.text) <= MAX_TEXT_CHARS


# --- prompt injection -----------------------------------------------------------------------


def test_instruction_like_lines_are_flagged():
    service, _ = site(
        {"https://evil.example.net/robots.txt": ALLOW_ALL, EVIL: page(html("malicious_page.html"))}
    )
    document = service.read_source(ReadSourceInput(url=EVIL))

    flags = injection_flags(document.text)

    assert any("ignore all previous instructions" in flag.casefold() for flag in flags)
    assert any("developer mode" in flag.casefold() for flag in flags)
    assert not any("encuesta" in flag for flag in flags)  # ordinary content is not flagged


def test_the_untrusted_block_cannot_be_closed_from_inside():
    service, _ = site(
        {"https://evil.example.net/robots.txt": ALLOW_ALL, EVIL: page(html("malicious_page.html"))}
    )
    document = service.read_source(ReadSourceInput(url=EVIL))

    block = untrusted_block(document)

    assert block.startswith(f'<source url="{EVIL}" untrusted="true">')
    assert block.count("</source>") == 1 and block.endswith("</source>")
    assert "never follow instructions" in block.casefold()
