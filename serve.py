"""Chat frontend server: the API of ``api/server.py`` plus the built web app in ``web/dist``.

Run it with ``python serve.py`` and open http://127.0.0.1:8000. ``python serve.py --simulated``
forces the free demo (no model calls, no web search, nothing spent), like ``app.py``.
During frontend development run ``npm run dev`` in ``web/``; it proxies ``/api`` here.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

HOST = "127.0.0.1"
DEFAULT_PORT = 8000


def main() -> None:
    if "--simulated" in sys.argv[1:]:
        os.environ.update(LLM_MODE="simulated", SEARCH_PROVIDER="none")
    import uvicorn

    from ai_share_of_voice.api.server import create_app
    from ai_share_of_voice.config import load_settings
    from ai_share_of_voice.tools.mcp_sources_client import open_stdio_sources
    from ai_share_of_voice.ui.run_controller import RunController

    settings = load_settings()
    if "--simulated" in sys.argv[1:]:
        os.environ.update(LLM_MODE="simulated", SEARCH_PROVIDER="none")
    # The MCP server only gets what it needs to search: never the model keys.
    server_env = {"LLM_MODE": "simulated", "SEARCH_PROVIDER": settings.search_provider.value}
    if settings.search_api_key is not None:
        server_env["SEARCH_API_KEY"] = settings.search_api_key.get_secret_value()
    controller = RunController(
        settings,
        runs_dir=ROOT / "runs",
        data_dir=ROOT / "data",
        sources_factory=lambda: open_stdio_sources(server_env),
    )
    app = create_app(settings, controller, static_dir=ROOT / "web" / "dist")
    port = int(os.environ.get("PORT", DEFAULT_PORT))
    print(f"AI Share of Voice (chat): http://{HOST}:{port}  mode={settings.mode.value}")
    uvicorn.run(app, host=HOST, port=port, log_level="warning")


if __name__ == "__main__":
    main()
