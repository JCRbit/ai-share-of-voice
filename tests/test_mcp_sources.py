from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from ai_share_of_voice.errors import ToolFailure
from ai_share_of_voice.mcp_server.server import build_server
from ai_share_of_voice.schemas.sources import SourceHit
from ai_share_of_voice.tools.mcp_sources_client import open_sources, open_stdio_sources
from ai_share_of_voice.tools.sources_tool import SourcesService

FIXTURES = Path(__file__).parent / "fixtures" / "sources"
NOW = datetime(2026, 11, 2, 12, 0, tzinfo=UTC)
EVIL = "https://evil.example.net/ranking"
TOOLS = ["check_url", "read_source", "search_sources"]


class FixedSearch:
    def search(self, data):
        return (SourceHit(url=EVIL, title="Ranking", snippet="apps"),)[: data.max_results]


def service() -> SourcesService:
    routes = {
        "https://evil.example.net/robots.txt": httpx.Response(200, text="User-agent: *\nAllow: /"),
        EVIL: httpx.Response(
            200,
            text=(FIXTURES / "malicious_page.html").read_text(encoding="utf-8"),
            headers={"content-type": "text/html"},
        ),
        "https://private.example.net/robots.txt": httpx.Response(
            200, text="User-agent: *\nDisallow: /"
        ),
    }

    def handler(request):
        return routes.get(str(request.url), httpx.Response(404))

    http = httpx.Client(transport=httpx.MockTransport(handler))
    return SourcesService(http=http, now=lambda: NOW, search=FixedSearch())


@pytest.fixture
def sources():
    server = build_server(service())
    with open_sources(lambda: create_connected_server_and_client_session(server)) as client:
        yield client


def test_the_server_offers_the_three_tools(sources):
    assert sources.list_tools() == TOOLS


def test_search_goes_through_the_protocol(sources):
    hits = sources.search_sources("apps comida Madrid", max_results=3)

    assert [str(hit.url) for hit in hits] == [EVIL]


def test_read_returns_untrusted_text_with_its_flags_and_prompt_block(sources):
    result = sources.read_source(EVIL)

    assert result.document.untrusted is True
    assert any(
        "ignore all previous instructions" in flag.casefold() for flag in result.injection_flags
    )
    assert result.prompt_block.startswith(f'<source url="{EVIL}" untrusted="true">')


def test_check_url_goes_through_the_protocol(sources):
    check = sources.check_url(EVIL)

    assert (check.exists, check.allowed_by_robots) == (True, True)


def test_a_tool_error_comes_back_as_a_tool_failure(sources):
    with pytest.raises(ToolFailure, match="robots"):
        sources.read_source("https://private.example.net/page")


def test_invalid_arguments_are_rejected_by_the_contracts(sources):
    with pytest.raises(ToolFailure):
        sources.search_sources("", max_results=99)


def test_the_real_server_runs_as_a_subprocess_over_stdio():
    # No network: SEARCH_PROVIDER=none, set in the environment so it wins over .env.
    env = {"SEARCH_PROVIDER": "none", "LLM_MODE": "simulated"}

    with open_stdio_sources(env) as sources:
        assert sources.list_tools() == TOOLS
        assert sources.search_sources("cualquier cosa") == ()
