"""Mock run generator (Task I2) backed by the canonical simulated LangGraph flow."""

from collections.abc import Iterator
from pathlib import Path
from tempfile import TemporaryDirectory

from langgraph.types import Command

from ai_share_of_voice.checkpoint import memory_checkpointer
from ai_share_of_voice.event_log import EventLog
from ai_share_of_voice.graph import build_graph, initial_state, new_run_id, run_config
from ai_share_of_voice.nodes import NodeDeps, fake_should_stop, make_fake_nodes
from ai_share_of_voice.schemas.brief import BrandSpec, Brief
from ai_share_of_voice.schemas.events import GraphEvent


def build_demo_brief() -> Brief:
    """Create the default demo brief used for UI mock runs."""
    return Brief(
        brand=BrandSpec(name="Acme"),
        competitors=(BrandSpec(name="Beta"), BrandSpec(name="Gamma")),
        sector="food delivery",
        city="Madrid",
        spend_cap_usd=2.0,
        max_batches=5,
    )


def simulate_run(
    run_id: str | None = None,
    *,
    brief: Brief | None = None,
    log_dir: Path | None = None,
    auto_approve: bool = False,
) -> Iterator[GraphEvent]:
    """Run the canonical graph with fake nodes and yield resulting GraphEvents."""
    active_brief = brief or build_demo_brief()
    active_run_id = run_id or new_run_id()

    with TemporaryDirectory() as tmp_dir:
        base_dir = log_dir if log_dir is not None else Path(tmp_dir)
        log = EventLog(base_dir)
        deps = NodeDeps(event_log=log, should_stop=fake_should_stop(2))
        graph = build_graph(make_fake_nodes(deps), checkpointer=memory_checkpointer())
        config = run_config(active_run_id, active_brief)

        graph.invoke(initial_state(active_brief, active_run_id), config)
        events = log.read(active_run_id)
        seen = len(events)
        yield from events

        state = graph.get_state(config)
        if auto_approve and state.next == ("approval_gate",):
            graph.invoke(
                Command(resume={"approved": True, "comment": "Demo approval"}),
                config,
            )
            resumed_events = log.read(active_run_id)
            yield from resumed_events[seen:]


if __name__ == "__main__":
    for event in simulate_run():
        print(f"[{event.status}] {event.node}: {event.result or event.reason}")
