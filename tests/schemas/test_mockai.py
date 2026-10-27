from pathlib import Path

from ai_share_of_voice.event_log import EventLog
from ai_share_of_voice.ui.mock_run import simulate_run


def test_simulate_run_yields_only_new_events_on_reuse(tmp_path: Path):
    """Ensure reusing a run_id does not re-yield past events."""
    run_id = "test-reuse-run"

    # First invocation populates the persistent log
    first_run_events = list(simulate_run(run_id=run_id, log_dir=tmp_path))
    assert len(first_run_events) > 0

    # Second invocation creates a new checkpointer, so it runs the graph again.
    # It should yield ONLY the newly generated events, not combined with the old ones.
    second_run_events = list(simulate_run(run_id=run_id, log_dir=tmp_path))

    # Verify the second run yielded the expected number of new events (19)
    assert len(second_run_events) == len(first_run_events)

    # Verify the underlying persistent log holds ALL events from both runs (38 total)
    log = EventLog(tmp_path)
    total_persisted_events = len(log.read(run_id))
    assert total_persisted_events == len(first_run_events) + len(second_run_events)
