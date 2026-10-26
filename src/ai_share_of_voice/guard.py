"""Turns node failures into a partial close (owner: J).

A guarded node never crashes the run. When it raises, the guard classifies the error,
writes a ``FAILED`` event with the reason to the state and the JSONL log, and records
``partial_reason`` and ``failure_kind``. The graph routes every guarded node to
``close_partial`` when it sees them. Use it like this::

    nodes = guard_nodes(make_fake_nodes(deps), deps.event_log, fatal=(ApprovalRequired,))
    graph = build_graph(nodes, checkpointer=checkpointer)
"""

import logging
from collections.abc import Mapping
from typing import Any

from langgraph.errors import GraphBubbleUp

from ai_share_of_voice.errors import FailureKind, classify
from ai_share_of_voice.event_log import EventLog
from ai_share_of_voice.nodes import NodeFn
from ai_share_of_voice.schemas.events import EventStatus, GraphEvent
from ai_share_of_voice.state import GraphState

logger = logging.getLogger(__name__)

# The human pause and the closing nodes must never be guarded: a pause is not a failure,
# and a closing node that fails has nowhere left to go.
DEFAULT_UNGUARDED = frozenset({"approval_gate", "close_rejected", "close_partial"})


def guard_node(
    name: str,
    node: NodeFn,
    event_log: EventLog,
    *,
    fatal: tuple[type[BaseException], ...] = (),
) -> NodeFn:
    """Wrap ``node`` so a failure closes the run in a partial state.

    ``fatal`` lists exceptions that signal a bug or a broken safety rule, such as running
    the final action without approval. They are raised again instead of being absorbed.
    """

    def guarded(state: GraphState) -> dict[str, Any]:
        try:
            return node(state)
        except GraphBubbleUp:
            raise  # LangGraph uses it for the human pause and for control flow
        except fatal:
            raise
        except Exception as error:
            failure = classify(error)
            logger.warning(
                "node %s failed (%s): %s",
                name,
                failure.kind.value,
                failure.reason,
                exc_info=failure.kind is FailureKind.UNEXPECTED,
            )
            event = GraphEvent(
                run_id=state["run_id"],
                node=name,
                tool=failure.tool,
                status=EventStatus.FAILED,
                result=failure.kind.value,
                reason=failure.reason,
                cumulative_spend_usd=state.get("spend_usd"),
            )
            event_log.append(event)
            return {
                "partial_reason": failure.reason,
                "failure_kind": failure.kind.value,
                "events": [event],
            }

    return guarded


def guard_nodes(
    nodes: Mapping[str, NodeFn],
    event_log: EventLog,
    *,
    unguarded: frozenset[str] = DEFAULT_UNGUARDED,
    fatal: tuple[type[BaseException], ...] = (),
) -> dict[str, NodeFn]:
    """Guard every node except the ones in ``unguarded``."""
    return {
        name: node if name in unguarded else guard_node(name, node, event_log, fatal=fatal)
        for name, node in nodes.items()
    }
