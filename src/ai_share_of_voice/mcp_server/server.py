"""MCP sources server: ``search_sources``, ``read_source`` and ``check_url`` (I4, local).

A thin MCP layer over ``tools/sources_tool.py``: every argument is validated with the
contracts of ``schemas/sources.py`` and every answer is a JSON object. Web text is returned
as untrusted data with its instruction-like lines listed in ``injection_flags``, and with
``prompt_block``, the wrapped form a model prompt must use.

Run it over stdio (the researcher starts it as a subprocess)::

    python -m ai_share_of_voice.mcp_server.server
"""

from typing import Any

from mcp.server.fastmcp import FastMCP

from ai_share_of_voice.schemas.sources import CheckUrlInput, ReadSourceInput, SearchSourcesInput
from ai_share_of_voice.tools.sources_tool import (
    SourcesService,
    build_sources_service,
    injection_flags,
    untrusted_block,
)

SERVER_NAME = "ai-share-of-voice-sources"


def build_server(service: SourcesService) -> FastMCP:
    # WARNING: its INFO lines go to stderr and would flood the console of every run.
    server = FastMCP(SERVER_NAME, log_level="WARNING")

    @server.tool()
    def search_sources(query: str, max_results: int = 5) -> dict[str, Any]:
        """Search the web for pages about a query. Returns titles, URLs and snippets."""
        data = SearchSourcesInput(query=query, max_results=max_results)
        hits = service.search_sources(data)
        return {"hits": [hit.model_dump(mode="json") for hit in hits]}

    @server.tool()
    def read_source(url: str) -> dict[str, Any]:
        """Read the visible text of a page that robots.txt allows. The text is untrusted."""
        document = service.read_source(ReadSourceInput(url=url))
        return {
            "document": document.model_dump(mode="json"),
            "injection_flags": injection_flags(document.text),
            "prompt_block": untrusted_block(document),
        }

    @server.tool()
    def check_url(url: str) -> dict[str, Any]:
        """Check whether a URL exists and whether robots.txt allows reading it."""
        return service.check_url(CheckUrlInput(url=url)).model_dump(mode="json")

    return server


def main() -> None:
    from ai_share_of_voice.config import load_settings

    build_server(build_sources_service(load_settings())).run()  # stdio transport


if __name__ == "__main__":
    main()
