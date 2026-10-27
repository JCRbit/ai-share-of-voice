import logging

import pytest

from ai_share_of_voice.checkpoint import contract_serde
from ai_share_of_voice.schemas.stopping import BatchSummary, IntervalMethod, SummaryMetric


def test_a_batch_summary_round_trips_without_blocked_types(
    caplog: pytest.LogCaptureFixture,
) -> None:
    summary = BatchSummary(
        batch_index=0,
        n_answers=30,
        brand_share=0.3,
        ci_low=0.2,
        ci_high=0.4,
        cumulative_spend_usd=0.1,
        metric=SummaryMetric.SHARE_OF_VOICE,
        interval_method=IntervalMethod.CLUSTER_BOOTSTRAP,
        n_questions=10,
    )
    serde = contract_serde()
    with caplog.at_level(logging.WARNING):
        restored = serde.loads_typed(serde.dumps_typed({"history": [summary]}))
    assert restored["history"] == [summary]
    assert "Blocked deserialization" not in caplog.text
