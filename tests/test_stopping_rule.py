from collections.abc import Sequence
from pathlib import Path

import pytest

from ai_share_of_voice.checkpoint import memory_checkpointer
from ai_share_of_voice.event_log import EventLog
from ai_share_of_voice.graph import build_graph, initial_state, run_config
from ai_share_of_voice.nodes import NodeDeps, make_fake_nodes
from ai_share_of_voice.schemas.brief import BrandSpec, Brief
from ai_share_of_voice.schemas.stopping import BatchSummary, StopConfig, StopReason, StopRule
from ai_share_of_voice.stats.metrics import MIN_RELIABLE_QUESTIONS
from ai_share_of_voice.stats.stopping import should_stop

ANSWERS_PER_BATCH = 30


def make_history(
    half_widths: Sequence[float],
    spends: Sequence[float] | None = None,
    n_questions: int | None = 30,
) -> list[BatchSummary]:
    """Cumulative summaries with a share of 0.5 and the given half-widths and spends."""
    spends = spends if spends is not None else [0.1 * (i + 1) for i in range(len(half_widths))]
    return [
        BatchSummary(
            batch_index=index,
            n_answers=ANSWERS_PER_BATCH * (index + 1),
            brand_share=0.5,
            ci_low=0.5 - half_width,
            ci_high=0.5 + half_width,
            cumulative_spend_usd=spend,
            n_questions=n_questions,
        )
        for index, (half_width, spend) in enumerate(zip(half_widths, spends, strict=True))
    ]


def make_config(**overrides: object) -> StopConfig:
    data: dict[str, object] = {
        "target_half_width": 0.05,
        "min_batches": 2,
        "max_batches": 10,
        "max_spend_usd": 5.0,
        "marginal_gain_threshold": 0.005,
        "patience": 2,
    }
    data.update(overrides)
    return StopConfig(**data)


# ---- precision ----------------------------------------------------------------------------


def test_precision_reached_stops_and_explains_the_numbers() -> None:
    decision = should_stop(make_history([0.08, 0.04]), make_config())
    assert decision.stop is True
    assert decision.reason is StopReason.PRECISION_REACHED
    assert "0.040" in decision.explanation
    assert "0.050" in decision.explanation
    assert "2 batches" in decision.explanation


def test_precision_exactly_at_the_target_is_enough() -> None:
    decision = should_stop(make_history([0.08, 0.05]), make_config())
    assert decision.reason is StopReason.PRECISION_REACHED


def test_precision_waits_for_the_minimum_number_of_batches() -> None:
    decision = should_stop(make_history([0.03]), make_config(min_batches=2))
    assert decision.stop is False
    assert decision.reason is StopReason.CONTINUE


def test_precision_is_not_trusted_with_too_few_questions() -> None:
    few = MIN_RELIABLE_QUESTIONS - 1
    decision = should_stop(make_history([0.08, 0.04], n_questions=few), make_config())
    assert decision.reason is StopReason.CONTINUE
    assert f"{few} questions" in decision.explanation


def test_unknown_question_count_does_not_block_precision() -> None:
    decision = should_stop(make_history([0.08, 0.04], n_questions=None), make_config())
    assert decision.reason is StopReason.PRECISION_REACHED


# ---- marginal return ----------------------------------------------------------------------


def test_marginal_return_stops_after_patience_small_gains() -> None:
    decision = should_stop(make_history([0.100, 0.098, 0.097]), make_config())
    assert decision.reason is StopReason.MARGINAL_RETURN
    assert "0.005" in decision.explanation


def test_one_small_gain_is_not_enough_with_patience_two() -> None:
    decision = should_stop(make_history([0.120, 0.100, 0.098]), make_config())
    assert decision.reason is StopReason.CONTINUE


def test_a_large_gain_resets_the_patience() -> None:
    decision = should_stop(make_history([0.120, 0.119, 0.090]), make_config())
    assert decision.reason is StopReason.CONTINUE


def test_a_wider_interval_counts_as_no_gain() -> None:
    decision = should_stop(make_history([0.100, 0.101, 0.102]), make_config())
    assert decision.reason is StopReason.MARGINAL_RETURN


def test_marginal_return_needs_patience_plus_one_batches() -> None:
    decision = should_stop(make_history([0.100, 0.099]), make_config(min_batches=1))
    assert decision.reason is StopReason.CONTINUE


# ---- budget -------------------------------------------------------------------------------


def test_budget_stops_when_the_next_batch_would_exceed_the_cap() -> None:
    # 3 batches cost 1.50, so the next one is expected to cost 0.50: 2.00 > 1.90.
    decision = should_stop(
        make_history([0.2, 0.18, 0.16], [0.5, 1.0, 1.5]), make_config(max_spend_usd=1.9)
    )
    assert decision.reason is StopReason.BUDGET_EXHAUSTED
    assert "1.50" in decision.explanation
    assert "1.90" in decision.explanation


def test_budget_allows_a_next_batch_that_lands_exactly_on_the_cap() -> None:
    decision = should_stop(
        make_history([0.2, 0.18, 0.16], [0.5, 1.0, 1.5]), make_config(max_spend_usd=2.0)
    )
    assert decision.reason is StopReason.CONTINUE


def test_budget_overrides_the_minimum_number_of_batches() -> None:
    decision = should_stop(
        make_history([0.2], [1.2]), make_config(min_batches=3, max_spend_usd=2.0)
    )
    assert decision.reason is StopReason.BUDGET_EXHAUSTED


def test_a_free_run_never_stops_for_budget() -> None:
    decision = should_stop(make_history([0.2, 0.18], [0.0, 0.0]), make_config())
    assert decision.reason is StopReason.CONTINUE


# ---- max batches --------------------------------------------------------------------------


def test_max_batches_stops_a_run_that_is_still_too_wide() -> None:
    decision = should_stop(make_history([0.20, 0.15, 0.10]), make_config(max_batches=3))
    assert decision.reason is StopReason.MAX_BATCHES
    assert "3 batches" in decision.explanation


# ---- precedence and continue --------------------------------------------------------------


def test_precision_wins_over_the_hard_limits() -> None:
    history = make_history([0.08, 0.04], [1.0, 2.0])
    decision = should_stop(history, make_config(max_batches=2, max_spend_usd=2.5))
    assert decision.reason is StopReason.PRECISION_REACHED


def test_marginal_return_wins_over_the_budget() -> None:
    history = make_history([0.100, 0.098, 0.097], [0.5, 1.0, 1.5])
    decision = should_stop(history, make_config(max_spend_usd=1.9))
    assert decision.reason is StopReason.MARGINAL_RETURN


def test_continue_explains_how_far_from_the_target() -> None:
    decision = should_stop(make_history([0.20, 0.15]), make_config())
    assert decision.stop is False
    assert decision.reason is StopReason.CONTINUE
    assert "0.150" in decision.explanation
    assert "0.050" in decision.explanation


# ---- input checks and contract ------------------------------------------------------------


def test_an_empty_history_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one batch"):
        should_stop([], make_config())


def test_a_history_whose_spend_goes_down_is_rejected() -> None:
    with pytest.raises(ValueError, match="cumulative"):
        should_stop(make_history([0.2, 0.18], [1.0, 0.5]), make_config())


def test_a_history_whose_answers_go_down_is_rejected() -> None:
    history = make_history([0.2, 0.18])
    shrunk = history[1].model_copy(update={"n_answers": 10})
    with pytest.raises(ValueError, match="cumulative"):
        should_stop([history[0], shrunk], make_config())


def test_should_stop_satisfies_the_stop_rule_protocol() -> None:
    rule: StopRule = should_stop
    assert rule(make_history([0.08, 0.04]), make_config()).stop is True


def test_the_simulated_graph_stops_with_the_real_rule(tmp_path: Path) -> None:
    brief = Brief(
        brand=BrandSpec(name="Acme"),
        competitors=(BrandSpec(name="Beta"), BrandSpec(name="Gamma")),
        sector="food delivery",
        city="Madrid",
        spend_cap_usd=0.12,
        max_batches=10,
    )
    # Fake batches cost 0.05: after two batches a third would reach 0.15 > 0.12.
    deps = NodeDeps(event_log=EventLog(tmp_path), should_stop=should_stop)
    graph = build_graph(make_fake_nodes(deps), checkpointer=memory_checkpointer())
    config = run_config("run-d2", brief)
    graph.invoke(initial_state(brief, "run-d2"), config)
    state = graph.get_state(config).values
    decision = state["stop_decision"]
    assert decision.reason is StopReason.BUDGET_EXHAUSTED
    assert decision.explanation.startswith("Budget exhausted")
    assert len(state["history"]) == 2
