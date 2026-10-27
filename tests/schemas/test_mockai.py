from pathlib import Path

from ai_share_of_voice.ui.mock_run import simulate_run


def test_simulate_run_yields_only_new_events_on_reuse(tmp_path: Path):
    """Ensure reusing a run_id does not re-yield past events."""
    run_id = "test-reuse-run"

    # First invocation populates the persistent log
    first_run_events = list(simulate_run(run_id=run_id, log_dir=tmp_path))
    assert len(first_run_events) > 0

    # Second invocation with the same run_id and persistent log_dir
    # Since the graph is paused at 'approval_gate', invoking it again without
    # auto_approve=True will do nothing, producing 0 *new* events.
    second_run_events = list(simulate_run(run_id=run_id, log_dir=tmp_path))

    assert len(second_run_events) == 0, "Should not yield events from the first run"
