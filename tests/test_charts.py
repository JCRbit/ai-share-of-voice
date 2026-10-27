from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from ai_share_of_voice.schemas.stopping import BatchSummary
from ai_share_of_voice.stats.charts import _brand_figure, brand_comparison_chart, convergence_chart
from ai_share_of_voice.stats.metrics import brand_metrics, load_answers

FIXTURE = Path(__file__).parent / "fixtures" / "sample_answers.csv"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@pytest.fixture
def metrics() -> pd.DataFrame:
    return brand_metrics(load_answers(FIXTURE))


def make_history(half_widths: list[float]) -> list[BatchSummary]:
    return [
        BatchSummary(
            batch_index=index,
            n_answers=30 * (index + 1),
            brand_share=0.5,
            ci_low=0.5 - half_width,
            ci_high=0.5 + half_width,
            cumulative_spend_usd=0.1 * (index + 1),
        )
        for index, half_width in enumerate(half_widths)
    ]


# ---- brand comparison ---------------------------------------------------------------------


def test_brand_chart_is_written_as_png(metrics: pd.DataFrame, tmp_path: Path) -> None:
    path = brand_comparison_chart(metrics, "mention_rate", tmp_path / "charts" / "mention.png")
    assert path == tmp_path / "charts" / "mention.png"
    assert path.read_bytes().startswith(PNG_SIGNATURE)


def test_brand_chart_is_reproducible(metrics: pd.DataFrame, tmp_path: Path) -> None:
    first = brand_comparison_chart(metrics, "share_of_voice", tmp_path / "a.png")
    second = brand_comparison_chart(metrics, "share_of_voice", tmp_path / "b.png")
    assert first.read_bytes() == second.read_bytes()


def test_brand_chart_warns_when_there_are_too_few_questions(
    metrics: pd.DataFrame, tmp_path: Path
) -> None:
    assert metrics["few_questions"].all()  # the fixture has 3 questions
    warned = brand_comparison_chart(metrics, "mention_rate", tmp_path / "warned.png")
    reliable = metrics.assign(few_questions=False)
    plain = brand_comparison_chart(reliable, "mention_rate", tmp_path / "plain.png")
    assert warned.read_bytes() != plain.read_bytes()


def with_undefined(metrics: pd.DataFrame, brand: str, metric: str) -> pd.DataFrame:
    """Copy of the table where one brand's metric cannot be computed (e.g. never mentioned)."""
    table = metrics.copy()
    row = (table["brand"] == brand) & (table["metric"] == metric)
    table.loc[row, ["estimate", "ci_low", "ci_high"]] = float("nan")
    return table


def test_an_undefined_estimate_gets_no_bar_and_is_marked_unavailable(
    metrics: pd.DataFrame,
) -> None:
    table = with_undefined(metrics, "Uber Eats", "recommendation_conversion")
    figure = _brand_figure(table, "recommendation_conversion")
    axes = figure.axes[0]
    drawn = [patch.get_height() for patch in axes.patches]
    assert len(drawn) == 2  # Just Eat and Glovo only
    labels = [text.get_text() for text in axes.texts]
    assert labels == ["n/a"]
    notes = " ".join(text.get_text() for text in figure.texts)
    assert "cannot be computed" in notes
    plt.close(figure)


def test_a_real_zero_is_still_drawn_as_a_bar(metrics: pd.DataFrame) -> None:
    # In the fixture Uber Eats is mentioned but never recommended: a true 0%.
    figure = _brand_figure(metrics, "recommendation_conversion")
    axes = figure.axes[0]
    assert len(axes.patches) == 3
    assert not axes.texts
    plt.close(figure)


def test_undefined_and_zero_give_different_charts(metrics: pd.DataFrame, tmp_path: Path) -> None:
    undefined = with_undefined(metrics, "Uber Eats", "recommendation_conversion")
    first = brand_comparison_chart(undefined, "recommendation_conversion", tmp_path / "a.png")
    second = brand_comparison_chart(metrics, "recommendation_conversion", tmp_path / "b.png")
    assert first.read_bytes() != second.read_bytes()


def test_a_chart_where_no_brand_has_a_value_still_renders(
    metrics: pd.DataFrame, tmp_path: Path
) -> None:
    table = metrics.copy()
    table.loc[:, ["estimate", "ci_low", "ci_high"]] = float("nan")
    path = brand_comparison_chart(table, "recommendation_conversion", tmp_path / "empty.png")
    assert path.read_bytes().startswith(PNG_SIGNATURE)


def test_brand_chart_handles_negative_net_tone(metrics: pd.DataFrame, tmp_path: Path) -> None:
    path = brand_comparison_chart(metrics, "net_tone", tmp_path / "tone.png")
    assert path.read_bytes().startswith(PNG_SIGNATURE)


def test_brand_chart_rejects_an_unknown_metric(metrics: pd.DataFrame, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown metric"):
        brand_comparison_chart(metrics, "market_share", tmp_path / "x.png")


def test_brand_chart_rejects_a_table_without_intervals(tmp_path: Path) -> None:
    table = pd.DataFrame({"brand": ["A"], "metric": ["mention_rate"], "estimate": [0.5]})
    with pytest.raises(ValueError, match="missing columns"):
        brand_comparison_chart(table, "mention_rate", tmp_path / "x.png")


def test_brand_chart_only_writes_png_files(metrics: pd.DataFrame, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=".png"):
        brand_comparison_chart(metrics, "mention_rate", tmp_path / "chart.svg")


# ---- convergence --------------------------------------------------------------------------


def test_convergence_chart_is_written_as_png(tmp_path: Path) -> None:
    path = convergence_chart(make_history([0.12, 0.08, 0.05]), 0.05, tmp_path / "conv.png")
    assert path.read_bytes().startswith(PNG_SIGNATURE)


def test_convergence_chart_is_reproducible(tmp_path: Path) -> None:
    history = make_history([0.12, 0.08])
    first = convergence_chart(history, 0.05, tmp_path / "a.png")
    second = convergence_chart(history, 0.05, tmp_path / "b.png")
    assert first.read_bytes() == second.read_bytes()


def test_convergence_chart_needs_at_least_one_batch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one batch"):
        convergence_chart([], 0.05, tmp_path / "conv.png")


@pytest.mark.parametrize("target", [0.0, -0.1, 0.6])
def test_convergence_chart_rejects_an_invalid_target(target: float, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="target"):
        convergence_chart(make_history([0.1]), target, tmp_path / "conv.png")
