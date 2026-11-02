"""Check the .env configuration with one tiny call per model role.

Usage: python scripts/ping_llm.py            one call per role
       python scripts/ping_llm.py --models   list the models each provider offers (free)
Prints mode, models, tokens and cost. Never prints the key. Costs a fraction of a cent.
"""

import sys
from pathlib import Path

from pydantic import BaseModel

# The package is not installed: make src/ importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ai_share_of_voice.config import ConfigError, LlmMode, load_settings  # noqa: E402
from ai_share_of_voice.errors import ToolFailure  # noqa: E402
from ai_share_of_voice.llm_client import build_clients  # noqa: E402

PING_CAP_USD = 0.02


class Pong(BaseModel):
    ok: bool


def list_models(clients, settings) -> int:
    """Print the model ids of each provider in use. Listing models costs nothing."""
    providers = {
        "agent": settings.provider,
        "subject": settings.subject_provider,
        "vision": settings.vision_provider,
    }
    seen = set()
    for role, provider in providers.items():
        sdk = getattr(clients, role)._sdk
        if id(sdk) in seen:
            continue
        seen.add(id(sdk))
        try:
            ids = sorted(model.id for model in sdk.models.list())
        except Exception as error:  # report and go on with the next provider
            print(f"FAIL {provider}: {type(error).__name__}: {error}")
            continue
        print(f"== {provider} ({len(ids)} models)")
        for model_id in ids:
            print(f"   {model_id}")
    return 0


def main() -> int:
    try:
        settings = load_settings()
    except ConfigError as error:
        print(f"config error: {error}")
        return 1
    print(
        f"mode={settings.mode} agents={settings.provider} "
        f"subject={settings.subject_provider} cap={settings.spend_cap_usd} USD"
    )
    if settings.mode is not LlmMode.LIVE:
        print("LLM_MODE is not 'live': nothing to ping.")
        return 0

    clients = build_clients(settings.model_copy(update={"spend_cap_usd": PING_CAP_USD}))
    if "--models" in sys.argv[1:]:
        return list_models(clients, settings)
    checks = [
        ("subject", lambda: clients.subject.answer("Reply with the single word: pong")),
        ("agent", lambda: clients.agent.complete('Return {"ok": true}.', Pong)),
        ("vision", lambda: clients.vision.answer("Reply with the single word: pong")),
    ]
    failed = 0
    for role, call in checks:
        model = getattr(clients, role).model
        try:
            result, usage = call()
        except (ToolFailure, ValueError) as error:
            failed += 1
            print(f"FAIL {role:<8} {model}: {error}")
            continue
        print(
            f"OK   {role:<8} {model}: {str(result)[:40]!r} "
            f"({usage.input_tokens}+{usage.output_tokens} tokens, {usage.cost_usd:.6f} USD)"
        )
    print(f"total spent: {clients.budget.spent_usd:.6f} USD")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
