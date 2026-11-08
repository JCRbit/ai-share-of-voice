"""Command line for a real run, up to the human approval (owner: An, task C2).

    python scripts/real_run.py                          # stage client, run cap 1.00 USD
    python scripts/real_run.py --stages client,mcp      # also check the cited sources
    python scripts/real_run.py --stages client,mcp,vision --cap 2

The stages are added in order (client, mcp, vision), as in the plan: first make the real run
work with the model, then with the sources service, then with the vision model. It needs
``LLM_MODE=live`` and your key in ``.env``; the vision stage also needs ``VISION_MODEL`` and its
prices. Every model call counts against the run cap and the person cap and is logged in
``runs/llm_calls.jsonl``. At the end it shows what the run found and asks for the approval; use
``--yes`` or ``--no`` to answer without being asked.

Exit status: 0 when the run completed, 1 when it was closed in a partial state or rejected, 2 for
a problem with the setup. The writer is still simulated, so no report is written.
"""

import argparse
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ai_share_of_voice.exceptions import ConfigurationError
from ai_share_of_voice.llm_client import Transport
from ai_share_of_voice.real_nodes import STAGES
from ai_share_of_voice.real_run import brief_from_data, run_real


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the graph with real stages (task C2).")
    parser.add_argument("--stages", default="client", help=f"comma list, in order: {STAGES}")
    parser.add_argument("--cap", type=float, default=1.0, help="run spending cap in USD")
    parser.add_argument("--max-batches", type=int, default=3)
    parser.add_argument("--target-half-width", type=float, default=0.05)
    parser.add_argument("--language", default="es", choices=("es", "en"))
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    decision = parser.add_mutually_exclusive_group()
    decision.add_argument("--yes", action="store_true", help="approve without asking")
    decision.add_argument("--no", action="store_true", help="reject without asking")
    return parser.parse_args(argv)


def describe(state: Mapping[str, Any]) -> list[str]:
    """What the run found, as lines of text."""
    lines = [f"spend: {state.get('spend_usd', 0.0):.4f} USD, answers: {state.get('n_answers', 0)}"]
    history = state.get("history") or []
    if history:
        last = history[-1]
        lines.append(
            f"mention rate of the client: {last.brand_share:.0%} "
            f"[{last.ci_low:.0%}, {last.ci_high:.0%}] after {len(history)} batches"
        )
    decision = state.get("stop_decision")
    if decision is not None:
        lines.append(f"stopped because: {decision.explanation}")
    if "sources" in state:
        lines.append(f"verified sources: {len(state['sources'])}")
    lines += [f"vision: {finding}" for finding in state.get("findings", [])]
    if state.get("partial_reason"):
        lines.append(f"CLOSED IN A PARTIAL STATE: {state['partial_reason']}")
    return lines


def main(
    argv: Sequence[str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
    transport: Transport | None = None,
    ask: Callable[[str], str] = input,
) -> int:
    """Run the script. Returns 0, 1 or 2 as described in the module docstring."""
    args = _parse_args(argv)
    stages = tuple(part.strip() for part in args.stages.split(",") if part.strip())

    def approve(state: Mapping[str, Any]) -> bool:
        print("\n".join(describe(state)))
        if args.yes or args.no:
            return args.yes
        return ask("Approve the draft? [y/N] ").strip().lower() in {"y", "yes", "s", "si", "sí"}

    try:
        brief = brief_from_data(
            args.data_dir,
            language=args.language,
            spend_cap_usd=args.cap,
            max_batches=args.max_batches,
            target_half_width=args.target_half_width,
        )
        print(f"stages {list(stages)}, cap {args.cap:.2f} USD, up to {args.max_batches} batches")
        result = run_real(
            brief,
            stages,
            approve=approve,
            env=env,
            runs_dir=args.runs_dir,
            data_dir=args.data_dir,
            transport=transport,
        )
    except (ConfigurationError, ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    state = result.state
    if result.approved is None:
        print("\n".join(describe(state)), file=sys.stderr)
    print(f"run {result.run_id}: {state.get('status')}, spent {result.spent_usd:.4f} USD")
    return 0 if state.get("status") == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
