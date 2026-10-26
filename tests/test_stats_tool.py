import shutil
from pathlib import Path
from typing import get_args

import pandas as pd
import pytest
from pydantic import ValidationError

from ai_share_of_voice.stats.metrics import METRICS, brand_metrics, load_answers
from ai_share_of_voice.tools.stats_tool import (
    ANSWERS_FILE_NAME,
    MetricName,
    StatsTool,
    StatsToolError,
    StatsToolInput,
    run_stats,
)

FIXTURE = Path(__file__).parent / "fixtures" / "sample_answers.csv"
RUN_ID = "run-20261026-120000-abc123"


@pytest.fixture
def runs_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "runs" / RUN_ID
    run_dir.mkdir(parents=True)
    shutil.copy(FIXTURE, run_dir / ANSWERS_FILE_NAME)
    return tmp_path / "runs"


# ---- results ------------------------------------------------------------------------------


def test_figures_come_from_the_vetted_metrics_function(runs_dir: Path) -> None:
    result = run_stats(StatsToolInput(run_id=RUN_ID, metric="mention_rate"), runs_dir)
    expected = brand_metrics(load_answers(FIXTURE))
    expected = expected[expected["metric"] == "mention_rate"].set_index("brand")
    assert [row.brand for row in result.rows] == list(expected.index)
    for row in result.rows:
        assert row.estimate == pytest.approx(expected.loc[row.brand, "estimate"])
        assert row.ci_low == pytest.approx(expected.loc[row.brand, "ci_low"])
        assert row.ci_high == pytest.approx(expected.loc[row.brand, "ci_high"])
        assert row.ci_method == "wilson_bootstrap_hull"


def test_the_result_flags_too_few_questions(runs_dir: Path) -> None:
    result = run_stats(StatsToolInput(run_id=RUN_ID), runs_dir)
    assert result.n_questions == 3
    assert result.few_questions is True


def test_the_chart_is_written_inside_the_run_folder(runs_dir: Path) -> None:
    result = run_stats(StatsToolInput(run_id=RUN_ID, metric="share_of_voice"), runs_dir)
    assert len(result.chart_paths) == 1
    chart = Path(result.chart_paths[0])
    assert chart.parent == runs_dir / RUN_ID / "charts"
    assert chart.name == "share_of_voice_by_brand.png"
    assert chart.exists()


def test_the_chart_can_be_skipped(runs_dir: Path) -> None:
    result = run_stats(StatsToolInput(run_id=RUN_ID, chart=False), runs_dir)
    assert result.chart_paths == ()
    assert not (runs_dir / RUN_ID / "charts").exists()


@pytest.mark.parametrize("metric", METRICS)
def test_every_metric_can_be_computed_and_charted(runs_dir: Path, metric: str) -> None:
    result = run_stats(StatsToolInput(run_id=RUN_ID, metric=metric), runs_dir)
    assert result.metric == metric
    assert len(result.rows) == 3


# ---- input validation ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "arguments",
    [
        {"run_id": "../evil"},
        {"run_id": "a/b"},
        {"run_id": ".."},
        {"run_id": ""},
        {"run_id": RUN_ID, "metric": "market_share"},
        {"run_id": RUN_ID, "code": "import os"},
    ],
    ids=["parent_dir", "nested", "dot_dot", "empty", "unknown_metric", "unknown_field"],
)
def test_invalid_input_is_rejected(arguments: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        StatsToolInput(**arguments)


def test_a_run_without_answers_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(StatsToolError, match="no analyzed answers"):
        run_stats(StatsToolInput(run_id=RUN_ID), tmp_path / "runs")


def test_a_malformed_answers_file_is_a_clear_error(runs_dir: Path) -> None:
    pd.DataFrame({"response_id": ["r1"]}).to_csv(runs_dir / RUN_ID / ANSWERS_FILE_NAME)
    with pytest.raises(StatsToolError, match="missing columns"):
        run_stats(StatsToolInput(run_id=RUN_ID), runs_dir)


# ---- the agent-facing wrapper -------------------------------------------------------------


def test_the_tool_offers_exactly_the_metrics_of_the_stats_module() -> None:
    assert set(get_args(MetricName)) == set(METRICS)


def test_the_tool_has_the_name_used_by_the_analyst_agent() -> None:
    assert StatsTool.name == "stats_tool"
    assert StatsTool.args_schema is StatsToolInput
    assert StatsTool.description


def test_the_tool_validates_arguments_before_running(runs_dir: Path) -> None:
    tool = StatsTool(runs_dir)
    result = tool.run(run_id=RUN_ID, metric="top1_rate", chart=False)
    assert result.metric == "top1_rate"
    with pytest.raises(ValidationError):
        tool.run(run_id="../evil")
