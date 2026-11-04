import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from ai_share_of_voice.agents.specs import AGENT_SPECS, KNOWN_TOOLS
from ai_share_of_voice.errors import ToolFailure
from ai_share_of_voice.mcp_server.sources import SourceError
from ai_share_of_voice.tools.sources_tool import (
    CheckUrlTool,
    ReadSourceTool,
    SearchSourcesTool,
    SourcesClient,
    source_tools,
)

ROOT = Path(__file__).resolve().parent.parent
SERVER = ROOT / "tests" / "fixtures" / "run_sources_server.py"
BROKEN_SERVER = ROOT / "tests" / "fixtures" / "run_broken_sources_server.py"


@pytest.fixture
def client() -> SourcesClient:
    """A client wired to the real MCP server over stdio, with a canned web behind it."""
    return SourcesClient(
        command=(sys.executable, str(SERVER)),
        env={"PYTHONPATH": str(ROOT / "src")},
        cwd=str(ROOT),
        timeout_s=30,
    )


def test_the_tools_match_the_names_the_researcher_is_given() -> None:
    tools = source_tools(object())  # type: ignore[arg-type]
    assert set(tools) == {"search_sources", "read_source", "check_url"}
    assert set(AGENT_SPECS["source_researcher"].tools) == set(tools)
    assert all(KNOWN_TOOLS[name] == "I4" for name in tools)


def test_tool_arguments_are_validated_before_reaching_the_backend() -> None:
    calls: list[str] = []

    class Backend:
        def read_source(self, url):
            calls.append(url)

    with pytest.raises(ValidationError):
        ReadSourceTool(Backend()).run(url="not a url")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        SearchSourcesTool(Backend()).run(query="", max_results=99)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        CheckUrlTool(Backend()).run(url="x", extra=1)  # type: ignore[arg-type]
    assert calls == []


def test_search_over_mcp(client: SourcesClient) -> None:
    hits = SearchSourcesTool(client).run(query="delivery apps Madrid", max_results=3)
    assert [str(h.url) for h in hits] == ["https://example.com/madrid"]


def test_read_source_over_mcp_keeps_the_malicious_page_inert(client: SourcesClient) -> None:
    tool = ReadSourceTool(client)
    result = tool.run(url="https://example.com/madrid")
    assert result.document.untrusted is True
    assert "HIDDEN-PAYLOAD" not in result.document.text
    assert {"instruction_like", "hidden_content"} <= {f.kind for f in result.flags}
    assert tool.render(result).count("</untrusted_web_content>") == 1


def test_check_url_over_mcp(client: SourcesClient) -> None:
    tool = CheckUrlTool(client)
    assert tool.run(url="https://example.com/madrid").exists is True
    assert tool.run(url="https://example.com/missing").status_code == 404
    assert tool.run(url="https://example.com/private/x").allowed_by_robots is False


def test_client_filters_explicit_environment() -> None:
    client = SourcesClient(
        env={
            "PYTHONPATH": "src",
            "BRAVE_API_KEY": "search-key",
            "OPENAI_API_KEY": "must-not-forward",
            "LLM_API_KEY": "must-not-forward",
        }
    )

    assert client._env["PYTHONPATH"] == "src"
    assert client._env["BRAVE_API_KEY"] == "search-key"
    assert "OPENAI_API_KEY" not in client._env
    assert "LLM_API_KEY" not in client._env


def test_server_refusals_become_source_errors(client: SourcesClient) -> None:
    with pytest.raises(SourceError, match="robots.txt"):
        client.read_source("https://example.com/private/x")
    with pytest.raises(SourceError, match="port"):
        client.read_source("https://example.com:8080/x")


def test_a_server_that_cannot_start_is_a_tool_failure() -> None:
    broken = SourcesClient(command=(sys.executable, "-c", "raise SystemExit(3)"), timeout_s=10)
    with pytest.raises(ToolFailure) as info:
        broken.check_url("https://example.com/")
    assert info.value.tool == "check_url"


def test_a_silent_server_times_out() -> None:
    hung = SourcesClient(
        command=(sys.executable, "-c", "import time; time.sleep(30)"), timeout_s=1.5
    )
    with pytest.raises(ToolFailure, match="no answer"):
        hung.read_source("https://example.com/")


def test_only_allowlisted_environment_reaches_the_server(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "model-key")
    monkeypatch.setenv("BRAVE_API_KEY", "search-key")
    env = SourcesClient()._env
    assert "LLM_API_KEY" not in env and env["BRAVE_API_KEY"] == "search-key"


def test_unexpected_server_errors_become_tool_failures() -> None:
    broken = SourcesClient(
        command=(sys.executable, str(BROKEN_SERVER)),
        env={"PYTHONPATH": str(ROOT / "src")},
        cwd=str(ROOT),
        timeout_s=30,
    )

    with pytest.raises(ToolFailure, match="unexpected tool error") as info:
        broken.read_source("https://example.com/")

    assert info.value.tool == "read_source"
