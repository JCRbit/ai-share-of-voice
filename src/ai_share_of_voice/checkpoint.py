"""Checkpointers for the graph (owner: J).

LangGraph only deserializes types it knows are safe. The contracts that live in the
graph state are registered explicitly, so resuming a run never relies on warn-and-allow.
Add a type here whenever a new contract is stored in ``GraphState``.
"""

import sqlite3
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver

from ai_share_of_voice.schemas.brief import BrandSpec, Brief
from ai_share_of_voice.schemas.events import EventStatus, GraphEvent
from ai_share_of_voice.schemas.questions import Question, QuestionPanel
from ai_share_of_voice.schemas.stopping import BatchSummary, StopDecision, StopReason

_STATE_TYPES: tuple[type, ...] = (
    BatchSummary,
    BrandSpec,
    Brief,
    EventStatus,
    GraphEvent,
    Question,
    QuestionPanel,
    StopDecision,
    StopReason,
)


def contract_serde() -> JsonPlusSerializer:
    """Serializer that allows exactly the contracts stored in the graph state."""
    allowed = [(cls.__module__, cls.__qualname__) for cls in _STATE_TYPES]
    return JsonPlusSerializer(allowed_msgpack_modules=allowed)


def memory_checkpointer() -> InMemorySaver:
    """In-memory checkpointer for tests and simulated runs."""
    return InMemorySaver(serde=contract_serde())


def sqlite_checkpointer(path: Path | str) -> SqliteSaver:
    """SQLite checkpointer that survives restarts.

    The connection can be used from several threads (the UI runs the graph in the
    background). Close it with ``saver.conn.close()`` when done.
    """
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(db_path), check_same_thread=False)
    return SqliteSaver(connection, serde=contract_serde())
