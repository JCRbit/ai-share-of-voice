"""Run the A1 simulated graph end-to-end and print its audit events.

This is the C0/I2 smoke runner: all nodes are simulated, the graph pauses at the
human approval gate, then resumes with an automatic approval for demonstration.
"""

from pathlib import Path

from langgraph.types import Command

from ai_share_of_voice.checkpoint import memory_checkpointer
from ai_share_of_voice.event_log import EventLog
from ai_share_of_voice.graph import build_graph, initial_state, new_run_id, run_config
from ai_share_of_voice.nodes import NodeDeps, fake_should_stop, make_fake_nodes
from ai_share_of_voice.schemas.brief import BrandSpec, Brief


def build_demo_brief() -> Brief:
    return Brief(
        brand=BrandSpec(name="Acme"),
        competitors=(BrandSpec(name="Beta"), BrandSpec(name="Gamma")),
        sector="food delivery",
        city="Madrid",
        spend_cap_usd=2.0,
        max_batches=5,
    )


def print_new_events(log: EventLog, run_id: str, seen: int) -> int:
    events = log.read(run_id)
    for event in events[seen:]:
        print(event.model_dump_json())
    return len(events)


def main() -> None:
    brief = build_demo_brief()
    run_id = new_run_id()
    log = EventLog(Path("runs"))
    deps = NodeDeps(event_log=log, should_stop=fake_should_stop(2))
    graph = build_graph(make_fake_nodes(deps), checkpointer=memory_checkpointer())
    config = run_config(run_id, brief)

    graph.invoke(initial_state(brief, run_id), config)
    seen = print_new_events(log, run_id, 0)

    state = graph.get_state(config)
    if state.next == ("approval_gate",):
        print('{"type":"approval_requested","approved":true}')
        graph.invoke(Command(resume={"approved": True, "comment": "Demo approval"}), config)
        print_new_events(log, run_id, seen)

    final_state = graph.get_state(config).values
    print(f"run_id={run_id}")
    print(f"status={final_state.get('status', 'unknown')}")
    print(f"report_path={final_state.get('report_path', '')}")


if __name__ == "__main__":
    main()
