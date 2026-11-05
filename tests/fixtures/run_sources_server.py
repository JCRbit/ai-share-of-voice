"""Test-only entry point: the real MCP server wired to a canned web, no network needed.

``SourcesClient`` starts it as a subprocess, so the whole MCP path is exercised.
"""

from pathlib import Path

from ai_share_of_voice.mcp_server.server import build_server
from ai_share_of_voice.mcp_server.sources import (
    FetchResponse,
    SourcesService,
    StaticSearchProvider,
)

FIXTURES = Path(__file__).parent
HTML = "text/html; charset=utf-8"


class CannedFetcher:
    def fetch(
        self,
        url,
        *,
        method="GET",
        max_bytes,
        timeout,
        headers=None,
        validated_addresses=None,
    ):
        if url.endswith("/robots.txt"):
            body = b"User-agent: *\nDisallow: /private/\n"
            return FetchResponse(url, 200, "text/plain", None, body)
        if "/private/" in url:
            return FetchResponse(url, 200, HTML, None, b"<p>secret</p>")
        if url.endswith("/missing"):
            return FetchResponse(url, 404, HTML, None, b"")
        page = (FIXTURES / "malicious_page.html").read_bytes()
        return FetchResponse(url, 200, HTML, None, page if method == "GET" else b"")


if __name__ == "__main__":
    index = StaticSearchProvider(
        [
            {
                "url": "https://example.com/madrid",
                "title": "Delivery apps in Madrid",
                "snippet": "Glovo",
            }
        ]
    )
    service = SourcesService(
        fetcher=CannedFetcher(), search=index, resolver=lambda host: ["93.184.216.34"]
    )
    build_server(service).run()
