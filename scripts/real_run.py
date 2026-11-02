"""Run the graph with the real nodes that exist so far (local integration, E).

Usage: python scripts/real_run.py [--questions N] [--batches N] [--language es|en]
       python scripts/real_run.py --sector "clothing stores" --city Valencia
           --brand Zara --competitors "Mango,H&M" --generate
       python scripts/real_run.py --panel runs/<previous run>/questions.csv
Defaults are cheap on purpose: 5 questions, 2 batches, the demo brief and panel.
Uses .env (LLM_MODE=live for real calls). Nodes without a real version yet stay simulated.
The approval is automatic here, because the final action is still the simulated one.
"""

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# The package is not installed: make src/ importable when run as a plain script.
sys.path.insert(0, str(ROOT / "src"))

from langgraph.types import Command  # noqa: E402

from ai_share_of_voice.automation.finalize import ApprovalRequired, make_finalize_node  # noqa: E402
from ai_share_of_voice.checkpoint import memory_checkpointer  # noqa: E402
from ai_share_of_voice.config import ConfigError, load_settings  # noqa: E402
from ai_share_of_voice.event_log import EventLog  # noqa: E402
from ai_share_of_voice.graph import build_graph, initial_state, new_run_id, run_config  # noqa: E402
from ai_share_of_voice.guard import guard_nodes  # noqa: E402
from ai_share_of_voice.llm_client import build_clients  # noqa: E402
from ai_share_of_voice.nodes import NodeDeps  # noqa: E402
from ai_share_of_voice.nodes_real import RealNodeOptions, make_real_nodes  # noqa: E402
from ai_share_of_voice.schemas.brief import BrandSpec, Brief  # noqa: E402
from ai_share_of_voice.stats.stopping import should_stop  # noqa: E402
from ai_share_of_voice.tools.mcp_sources_client import open_stdio_sources  # noqa: E402
from ai_share_of_voice.writing import make_client_report_finalize  # noqa: E402

RUNS_DIR = ROOT / "runs"
DATA_DIR = ROOT / "data"


def demo_brief(language: str, cap_usd: float, max_batches: int) -> Brief:
    """The demo brief: client and competitors from data/competitors.csv."""
    with (DATA_DIR / "competitors.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    brands = {
        row["name"]: BrandSpec(name=row["name"], variants=tuple(row["variants"].split("|")))
        for row in rows
    }
    client = next(row["name"] for row in rows if row["is_client"].strip().lower() == "true")
    return Brief(
        brand=brands[client],
        competitors=tuple(spec for name, spec in brands.items() if name != client),
        sector="food delivery",
        city="Madrid",
        language=language,
        spend_cap_usd=cap_usd,
        max_batches=max_batches,
    )


def custom_brief(args: argparse.Namespace, cap_usd: float) -> Brief:
    """A brief for any sector from the command line; names are split on commas."""
    return Brief(
        brand=BrandSpec(name=args.brand),
        competitors=tuple(
            BrandSpec(name=name.strip()) for name in args.competitors.split(",") if name.strip()
        ),
        sector=args.sector,
        city=args.city,
        language=args.language,
        spend_cap_usd=cap_usd,
        max_batches=args.batches,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--questions", type=int, default=5)
    parser.add_argument("--batches", type=int, default=2)
    parser.add_argument("--language", default="es")
    parser.add_argument("--sector")
    parser.add_argument("--city")
    parser.add_argument("--brand")
    parser.add_argument("--competitors", help="at least two, separated by commas")
    parser.add_argument("--generate", action="store_true", help="draft a panel for the brief")
    parser.add_argument("--panel", type=Path, help="reuse an approved questions.csv")
    parser.add_argument("--no-discovery", action="store_true")
    parser.add_argument("--country", help="added to the city in web searches")
    args = parser.parse_args()
    custom = [args.sector, args.city, args.brand, args.competitors]
    if any(custom) and not all(custom):
        parser.error("--sector, --city, --brand and --competitors go together")

    try:
        settings = load_settings()
    except ConfigError as error:
        print(f"config error: {error}")
        return 1

    run_id = new_run_id()
    log = EventLog(RUNS_DIR)
    clients = build_clients(settings, usage_log=RUNS_DIR / run_id / "usage.jsonl")
    brief = (
        custom_brief(args, settings.spend_cap_usd)
        if args.sector
        else demo_brief(args.language, settings.spend_cap_usd, args.batches)
    )
    deps = NodeDeps(event_log=log, should_stop=should_stop)
    # The MCP server only gets what it needs to search: never the model keys.
    server_env = {"LLM_MODE": "simulated", "SEARCH_PROVIDER": settings.search_provider.value}
    if settings.search_api_key is not None:
        server_env["SEARCH_API_KEY"] = settings.search_api_key.get_secret_value()
    nodes = make_real_nodes(
        deps,
        subject=clients.subject,
        agent=clients.agent,
        budget=clients.budget,
        runs_dir=RUNS_DIR,
        data_dir=DATA_DIR,
        options=RealNodeOptions(
            max_questions=args.questions,
            panel_file=args.panel,
            generate_panel=args.generate,
            discover=not args.no_discovery,
            location=f"{brief.city}, {args.country}" if args.country else None,
        ),
        sources=lambda: open_stdio_sources(server_env),
        vision=clients.vision,
    )
    # Audit report, tracking CSV and, after them, the client report in the brief language.
    nodes["finalize"] = make_client_report_finalize(make_finalize_node(log), RUNS_DIR)
    graph = build_graph(
        guard_nodes(nodes, log, fatal=(ApprovalRequired,)), checkpointer=memory_checkpointer()
    )
    config = run_config(run_id, brief)

    print(
        f"run_id={run_id} mode={settings.mode} subject={clients.subject.model} "
        f"agent={clients.agent.model} questions={args.questions} batches<={args.batches} "
        f"cap={settings.spend_cap_usd} USD search={settings.search_provider}\n"
        f"brief: {brief.brand.name} vs "
        f"{', '.join(c.name for c in brief.competitors)} · {brief.sector} · {brief.city}"
    )
    graph.invoke(initial_state(brief, run_id), config)
    if graph.get_state(config).next == ("approval_gate",):
        graph.invoke(Command(resume={"approved": True, "comment": "Local smoke run"}), config)

    for event in log.read(run_id):
        detail = " | ".join(part for part in (event.result, event.reason) if part)
        print(f"{event.status.value:<17} {event.node:<16} {detail}")
    final = graph.get_state(config).values
    print(f"status={final.get('status')} spent={clients.budget.spent_usd:.6f} USD")
    print(f"files: {RUNS_DIR / run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
