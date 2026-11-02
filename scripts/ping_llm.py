"""Check the .env configuration with one tiny call per model role.

Usage: python scripts/ping_llm.py
Prints mode, models, tokens and cost. Never prints the key. Costs a fraction of a cent.
"""

from pydantic import BaseModel

from ai_share_of_voice.config import ConfigError, LlmMode, load_settings
from ai_share_of_voice.errors import ToolFailure
from ai_share_of_voice.llm_client import build_clients

PING_CAP_USD = 0.02


class Pong(BaseModel):
    ok: bool


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
