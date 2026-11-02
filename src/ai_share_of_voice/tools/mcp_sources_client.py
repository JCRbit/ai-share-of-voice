"""MCP client adapter of the source researcher (I4, local integration).

The graph nodes are synchronous and the MCP client is async, so the adapter runs the session
in an anyio blocking portal and offers plain methods. Answers are validated back into the
contracts of ``schemas/sources.py``; a tool error becomes ``ToolFailure``, so the guard
closes the run like for any other tool.

``open_stdio_sources`` starts the real server as a subprocess; tests pass an in-memory
session to ``open_sources`` instead.
"""

import json
import sys
from collections.abc import Callable, Iterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from anyio.from_thread import BlockingPortal, start_blocking_portal
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from ai_share_of_voice.errors import ToolFailure
from ai_share_of_voice.schemas.sources import SourceDocument, SourceHit, UrlCheck

TOOL_NAME = "mcp_sources"
SRC_DIR = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ReadResult:
    document: SourceDocument
    injection_flags: tuple[str, ...]
    prompt_block: str


class McpSourcesClient:
    """Synchronous facade over an MCP session with the sources server."""

    def __init__(self, portal: BlockingPortal, session: ClientSession) -> None:
        self._portal = portal
        self._session = session

    def _call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        result = self._portal.call(self._session.call_tool, name, arguments)
        text = " ".join(getattr(item, "text", "") for item in result.content).strip()
        if result.isError:
            raise ToolFailure(TOOL_NAME, f"{name}: {text or 'tool error'}")
        if isinstance(result.structuredContent, dict):
            return result.structuredContent
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            raise ToolFailure(TOOL_NAME, f"{name}: answer is not JSON") from None

    def list_tools(self) -> list[str]:
        listed = self._portal.call(self._session.list_tools)
        return sorted(tool.name for tool in listed.tools)

    def search_sources(self, query: str, max_results: int = 5) -> tuple[SourceHit, ...]:
        answer = self._call("search_sources", {"query": query, "max_results": max_results})
        return tuple(SourceHit.model_validate(hit) for hit in answer.get("hits", []))

    def read_source(self, url: str) -> ReadResult:
        answer = self._call("read_source", {"url": url})
        return ReadResult(
            document=SourceDocument.model_validate(answer["document"]),
            injection_flags=tuple(answer.get("injection_flags", [])),
            prompt_block=answer["prompt_block"],
        )

    def check_url(self, url: str) -> UrlCheck:
        return UrlCheck.model_validate(self._call("check_url", {"url": url}))


SessionFactory = Callable[[], AbstractAsyncContextManager[ClientSession]]


@contextmanager
def open_sources(session_factory: SessionFactory) -> Iterator[McpSourcesClient]:
    """Open a session from ``session_factory`` for the duration of the ``with`` block."""
    with start_blocking_portal() as portal:
        with portal.wrap_async_context_manager(session_factory()) as session:
            yield McpSourcesClient(portal, session)


def stdio_session(env: dict[str, str] | None = None) -> SessionFactory:
    """Factory of sessions with the server started as ``python -m ...`` over stdio."""
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "ai_share_of_voice.mcp_server.server"],
        env={**(env or {}), "PYTHONPATH": str(SRC_DIR)},
    )

    @asynccontextmanager
    async def session():
        async with stdio_client(parameters) as (read, write):
            async with ClientSession(read, write) as client:
                await client.initialize()
                yield client

    return session


def open_stdio_sources(env: dict[str, str] | None = None):
    """The real server as a subprocess: ``with open_stdio_sources() as sources: ...``."""
    return open_sources(stdio_session(env))
