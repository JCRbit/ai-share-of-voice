"""Test-only MCP server that deliberately raises an unexpected error."""

from ai_share_of_voice.mcp_server.server import build_server


class BrokenService:
    def read_source(self, url: str):
        raise RuntimeError("simulated internal server error")


if __name__ == "__main__":
    build_server(BrokenService()).run()
