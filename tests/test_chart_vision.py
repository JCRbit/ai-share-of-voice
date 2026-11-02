import pytest
from tests.test_writing import finished_state  # Glovo 70%, Just Eat 80%, Uber Eats 0%

from ai_share_of_voice.chart_vision import ChartReading, make_vision_node
from ai_share_of_voice.errors import ToolFailure
from ai_share_of_voice.event_log import EventLog
from ai_share_of_voice.exceptions import BudgetExceeded
from ai_share_of_voice.llm_client import Budget, QuotaExhausted, SimulatedClient
from ai_share_of_voice.schemas.vision import Confidence


def reading(highest, rank, observation="Just Eat has the highest bar; Glovo is second."):
    return ChartReading(
        highest_brand=highest,
        client_rank=rank,
        observation=observation,
        confidence=Confidence.HIGH,
    )


def node_with(tmp_path, responder=None, budget=None):
    budget = budget or Budget(1.0)
    responders = {ChartReading: responder} if responder else {}
    client = SimulatedClient(budget, responders=responders)
    return make_vision_node(EventLog(tmp_path / "runs"), client, tmp_path / "runs")


def test_a_reading_consistent_with_the_data_becomes_a_hypothesis(tmp_path):
    state, run_dir = finished_state(tmp_path)
    prompts = []

    def read(prompt):
        prompts.append(prompt)
        return reading("Just Eat", 2)

    update = node_with(tmp_path, read)(state)

    (finding,) = update["findings"]
    assert "Just Eat has the highest bar" in finding and "hypothesis" in finding.casefold()
    assert (run_dir / "charts" / "vision_rates.png").is_file()
    assert "Glovo" in prompts[0] and "never" in prompts[0].casefold()


def test_a_reading_that_contradicts_the_data_is_discarded(tmp_path):
    state, _ = finished_state(tmp_path)

    update = node_with(tmp_path, lambda prompt: reading("Uber Eats", 1))(state)

    assert update["findings"] == []
    assert "discarded" in update["events"][-1].result


def test_brand_names_are_compared_without_case_or_accents(tmp_path):
    state, _ = finished_state(tmp_path)

    update = node_with(tmp_path, lambda prompt: reading("just eat", 2))(state)

    assert len(update["findings"]) == 1


@pytest.mark.parametrize(
    "error",
    [
        QuotaExhausted("llm_client", "429 per day"),
        ToolFailure("llm_client", "503"),
        ValueError("bad"),
    ],
)
def test_a_vision_failure_skips_the_step_without_closing_the_run(tmp_path, error):
    state, _ = finished_state(tmp_path)

    def fail(prompt):
        raise error

    update = node_with(tmp_path, fail)(state)

    assert update["findings"] == []
    assert "skipped" in update["events"][-1].result


def test_an_exhausted_budget_still_stops_the_run(tmp_path):
    state, _ = finished_state(tmp_path)

    with pytest.raises(BudgetExceeded):
        node_with(tmp_path, lambda prompt: reading("Just Eat", 2), budget=Budget(0.0001))(state)
