"""Runs the graph with real stages, up to the human approval (owner: An, task C2).

``run_real`` is the real counterpart of ``ui/mock_run.py``: it builds one budgeted model client
for the run, the real nodes of the chosen stages (``real_nodes.py``) and the graph with the
failure guard, saves the brief in the run folder, runs to the approval gate and resumes with the
decision of ``approve``. ``scripts/real_run.py`` is the command line around it.

* It refuses to run with the simulated client: invented answers must never look like a real run.
* The client is built from the environment (``.env``) with the run cap of the brief, and every
  call is written to ``<runs_dir>/llm_calls.jsonl``, so the person cap sums over runs.
* The writer is still simulated, and so are the Skeptic and ``finalize`` that come after it. The
  approval is a real pause, but what it approves is a simulated draft: the person sees the real
  numbers, sources and findings, and no report is written.
"""

import csv
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langgraph.types import Command

from ai_share_of_voice.checkpoint import memory_checkpointer
from ai_share_of_voice.event_log import EventLog
from ai_share_of_voice.exceptions import ConfigurationError
from ai_share_of_voice.graph import build_graph, initial_state, new_run_id, run_config
from ai_share_of_voice.guard import guard_nodes
from ai_share_of_voice.llm_client import SimulatedClient, Transport, client_from_env
from ai_share_of_voice.mcp_server.sources import SourcesService
from ai_share_of_voice.nodes import NodeDeps, fake_should_stop
from ai_share_of_voice.real_nodes import RealDeps, check_stages, make_real_nodes
from ai_share_of_voice.run_store import RunStore
from ai_share_of_voice.schemas.brief import BrandSpec, Brief
from ai_share_of_voice.schemas.llm import ModelClient, VisionModel

LOG_NAME = "llm_calls.jsonl"
Approver = Callable[[dict[str, Any]], bool]


@dataclass(frozen=True)
class RealRunResult:
    """What a real run left: its id, the final state and what it spent."""

    run_id: str
    state: dict[str, Any]
    spent_usd: float
    approved: bool | None


def brief_from_data(
    data_dir: Path | str = "data",
    *,
    sector: str = "food delivery",
    city: str = "Madrid",
    language: str = "es",
    spend_cap_usd: float = 1.0,
    max_batches: int = 3,
    target_half_width: float = 0.05,
) -> Brief:
    """The brief of the project's brands, from ``<data_dir>/competitors.csv``.

    The file has the columns ``name,variants,is_client``; variants are separated by ``|`` and
    exactly one row is the client.
    """
    with (Path(data_dir) / "competitors.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    clients = [row for row in rows if row["is_client"].strip().lower() == "true"]
    if len(clients) != 1:
        raise ConfigurationError(
            f"competitors.csv must mark exactly one client, found {len(clients)}"
        )

    def spec(row: Mapping[str, str]) -> BrandSpec:
        variants = tuple(item for item in row["variants"].split("|") if item.strip())
        return BrandSpec(name=row["name"], variants=variants)

    return Brief(
        brand=spec(clients[0]),
        competitors=tuple(spec(row) for row in rows if row is not clients[0]),
        sector=sector,
        city=city,
        language=language,
        spend_cap_usd=spend_cap_usd,
        max_batches=max_batches,
        target_half_width=target_half_width,
    )


def run_real(
    brief: Brief,
    stages: tuple[str, ...],
    *,
    approve: Approver,
    env: Mapping[str, str] | None = None,
    runs_dir: Path | str = "runs",
    data_dir: Path | str = "data",
    client: ModelClient | None = None,
    sources: Any = None,
    vision: VisionModel | None = None,
    transport: Transport | None = None,
) -> RealRunResult:
    """Run ``brief`` with the real nodes of ``stages`` and ask ``approve`` at the gate.

    ``client``, ``sources``, ``vision`` and ``transport`` exist for tests; a real run leaves
    them out and everything is built from the environment.
    """
    chosen = check_stages(stages)
    if not chosen:
        raise ConfigurationError("choose at least the client stage; with none, use mock_run")
    runs, data, run_id = Path(runs_dir), Path(data_dir), new_run_id()
    if client is None:
        client = client_from_env(
            env,
            run_id=run_id,
            log_path=runs / LOG_NAME,
            cap_usd=brief.spend_cap_usd,
            transport=transport,
        )
        if isinstance(client, SimulatedClient):
            raise ConfigurationError("set LLM_MODE=live: simulated answers are not a real run")
    if "mcp" in chosen and sources is None:
        sources = SourcesService.from_env(env if env is not None else _environment())
    if "vision" in chosen and vision is None and getattr(client, "vision_model", None):
        vision = client

    log = EventLog(runs)
    real = RealDeps(
        client=client,
        store=RunStore(runs),
        runs_dir=runs,
        data_dir=data,
        sources=sources,
        vision=vision,
    )
    deps = NodeDeps(event_log=log, should_stop=fake_should_stop(2))
    nodes = make_real_nodes(deps, real, chosen)
    graph = build_graph(guard_nodes(nodes, log), checkpointer=memory_checkpointer())
    config = run_config(run_id, brief)
    graph.invoke(initial_state(brief, run_id), config)

    approved: bool | None = None
    if graph.get_state(config).next == ("approval_gate",):
        approved = approve(graph.get_state(config).values)
        resume = {"approved": approved, "comment": "decided in the real run"}
        graph.invoke(Command(resume=resume), config)
    state = graph.get_state(config).values
    return RealRunResult(
        run_id=run_id,
        state=state,
        spent_usd=float(state.get("spend_usd", 0.0)),
        approved=approved,
    )


def _environment() -> Mapping[str, str]:
    """The process environment after loading ``.env``, as the other builders read it."""
    import os

    from dotenv import load_dotenv

    load_dotenv()
    return os.environ
