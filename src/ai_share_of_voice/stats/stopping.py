"""Stopping rule for the sampling loop (owner: E).

``should_stop(history, config)`` looks at the cumulative summary of the client brand after
each batch and decides whether another batch is worth its cost. It is deterministic code:
no model is asked whether to continue.

Rules, in order of precedence (the first one that holds is reported):

1. Precision reached: the interval half-width is at or below the target, after at least
   ``min_batches`` batches. Not accepted when the summary reports fewer than
   ``MIN_RELIABLE_QUESTIONS`` questions, because the question-level bootstrap is then too
   narrow and the precision would be an illusion.
2. Marginal return: the half-width shrank by less than ``marginal_gain_threshold`` in each
   of the last ``patience`` batches, after at least ``min_batches`` batches. More answers
   are no longer buying precision.
3. Budget exhausted: the next batch, at the average cost of the batches so far, would take
   the cumulative spend above the sampling budget (``config.sampling_budget_usd``: the part
   of the run cap left after the reserve for the agents that run after sampling). A safety
   rule, applied even before ``min_batches``.
4. Max batches reached. Also a safety rule.

Statistical reasons come first so the explanation says why the result is good enough,
not only that a limit was hit.
"""

from collections.abc import Sequence

from ai_share_of_voice.schemas.stopping import BatchSummary, StopConfig, StopDecision, StopReason
from ai_share_of_voice.stats.metrics import MIN_RELIABLE_QUESTIONS

# Half-widths and costs are floats computed elsewhere; equality at a limit counts as reached.
_TOLERANCE = 1e-9


def _check_history(history: Sequence[BatchSummary]) -> None:
    if not history:
        raise ValueError("the stopping rule needs at least one batch")
    for previous, current in zip(history, history[1:], strict=False):
        if current.n_answers < previous.n_answers:
            raise ValueError("n_answers must be cumulative and never decrease")
        if current.cumulative_spend_usd < previous.cumulative_spend_usd - _TOLERANCE:
            raise ValueError("cumulative_spend_usd must be cumulative and never decrease")


def _status(history: Sequence[BatchSummary], config: StopConfig) -> str:
    last = history[-1]
    return (
        f"half-width {last.half_width:.3f} (target {config.target_half_width:.3f}) "
        f"after {len(history)} batches and {last.n_answers} answers, "
        f"spent {last.cumulative_spend_usd:.2f} of the {config.sampling_budget_usd:.2f} USD "
        f"sampling budget (run cap {config.max_spend_usd:.2f} USD)"
    )


def _too_few_questions(summary: BatchSummary) -> bool:
    return summary.n_questions is not None and summary.n_questions < MIN_RELIABLE_QUESTIONS


def _small_gains(history: Sequence[BatchSummary], config: StopConfig) -> bool:
    """True if each of the last ``patience`` batches shrank the half-width too little."""
    if len(history) < config.patience + 1:
        return False
    recent = history[-(config.patience + 1) :]
    return all(
        previous.half_width - current.half_width < config.marginal_gain_threshold - _TOLERANCE
        for previous, current in zip(recent, recent[1:], strict=False)
    )


def _next_batch_exceeds_budget(history: Sequence[BatchSummary], config: StopConfig) -> bool:
    spent = history[-1].cumulative_spend_usd
    expected_next = spent / len(history)
    return spent + expected_next > config.sampling_budget_usd + _TOLERANCE


def should_stop(history: Sequence[BatchSummary], config: StopConfig) -> StopDecision:
    """Decide whether to run another batch; the decision always explains itself."""
    _check_history(history)
    last = history[-1]
    status = _status(history, config)
    enough_batches = len(history) >= config.min_batches
    precise = last.half_width <= config.target_half_width + _TOLERANCE

    if enough_batches and precise and not _too_few_questions(last):
        return StopDecision(
            stop=True,
            reason=StopReason.PRECISION_REACHED,
            explanation=f"Precision reached: {status}.",
        )
    if enough_batches and _small_gains(history, config):
        return StopDecision(
            stop=True,
            reason=StopReason.MARGINAL_RETURN,
            explanation=(
                f"Marginal return: the half-width shrank by less than "
                f"{config.marginal_gain_threshold:.3f} in each of the last {config.patience} "
                f"batches; {status}."
            ),
        )
    if _next_batch_exceeds_budget(history, config):
        return StopDecision(
            stop=True,
            reason=StopReason.BUDGET_EXHAUSTED,
            explanation=(
                "Budget exhausted: another batch would exceed the sampling budget, which keeps "
                f"a reserve for the agents that run after sampling; {status}."
            ),
        )
    if len(history) >= config.max_batches:
        return StopDecision(
            stop=True,
            reason=StopReason.MAX_BATCHES,
            explanation=f"Max batches reached: {status}.",
        )

    note = ""
    if precise and _too_few_questions(last):
        note = (
            f" The target is met, but with only {last.n_questions} questions the interval is "
            f"too optimistic (at least {MIN_RELIABLE_QUESTIONS} are needed)."
        )
    return StopDecision(
        stop=False,
        reason=StopReason.CONTINUE,
        explanation=f"Continue: {status}.{note}",
    )
