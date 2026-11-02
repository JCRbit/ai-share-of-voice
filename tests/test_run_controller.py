from pathlib import Path

import pytest

from ai_share_of_voice.config import load_settings
from ai_share_of_voice.ui.run_controller import BriefForm, RunController

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SIMULATED = {"LLM_MODE": "simulated", "SPEND_CAP_USD": "0.5"}


def form(**overrides) -> BriefForm:
    values = {
        "brand": "Glovo",
        "competitors": "Just Eat, Uber Eats",
        "sector": "food delivery",
        "city": "Madrid",
        "country": "España",
        "language": "es",
        "panel": "demo",
        "questions": 10,
        "batches": 2,
    }
    values.update(overrides)
    return BriefForm(**values)


@pytest.fixture
def controller(tmp_path):
    return RunController(load_settings(SIMULATED), runs_dir=tmp_path / "runs", data_dir=DATA_DIR)


def test_a_run_stops_at_the_human_approval(controller):
    run_id = controller.start(form())

    view = controller.wait(run_id, timeout=60)

    assert view.phase == "waiting_approval"
    nodes = [event.node for event in view.events]
    assert "sampling" in nodes and "metrics" in nodes and "approval_request" in nodes
    assert view.draft
    assert 0 < view.spend_usd <= view.cap_usd == 0.5


def test_approving_writes_the_report_and_the_downloads(controller, tmp_path):
    run_id = controller.start(form())
    controller.wait(run_id, timeout=60)

    controller.decide(run_id, approved=True, comment="Looks right")
    view = controller.wait(run_id, timeout=60)

    assert view.phase == "completed"
    assert (tmp_path / "runs" / run_id / "report.md").is_file()
    results = controller.results(run_id)
    assert results.view is not None and results.view.headline is not None
    assert len(results.view.downloads) == 3
    assert results.view.downloads[0].endswith("client_report.md")  # the client's first
    assert results.discovered  # demo answers name an untracked business
    assert (results.brand, results.city) == ("Glovo", "Madrid")
    assert results.recommendation is not None and results.spend_usd > 0


def test_rejecting_closes_without_a_report(controller, tmp_path):
    run_id = controller.start(form())
    controller.wait(run_id, timeout=60)

    controller.decide(run_id, approved=False, comment="Numbers look off")
    view = controller.wait(run_id, timeout=60)

    assert view.phase == "rejected"
    assert not (tmp_path / "runs" / run_id / "report.md").exists()


def test_a_decision_is_only_accepted_while_waiting(controller):
    run_id = controller.start(form())
    controller.wait(run_id, timeout=60)
    controller.decide(run_id, approved=True)
    controller.wait(run_id, timeout=60)

    with pytest.raises(ValueError, match="not waiting"):
        controller.decide(run_id, approved=True)


def test_a_generated_panel_for_another_sector(controller, tmp_path):
    run_id = controller.start(
        form(
            brand="Zara",
            competitors="Mango, H&M",
            sector="clothing stores",
            city="Valencia",
            panel="generate",
        )
    )

    view = controller.wait(run_id, timeout=60)

    assert view.phase == "waiting_approval"
    assert (tmp_path / "runs" / run_id / "questions.csv").is_file()
    assert any("generated panel" in (event.detail or "") for event in view.events)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("competitors", "Just Eat", "two competitors"),
        ("brand", "  ", "brand"),
        ("questions", 2, "questions"),
    ],
)
def test_an_invalid_form_is_rejected_before_starting(field, value, message):
    with pytest.raises(ValueError, match=message):
        form(**{field: value})


def test_the_cap_of_a_run_never_exceeds_the_settings_cap(controller):
    brief = form().to_brief(cap_usd=0.5)

    assert brief.spend_cap_usd == 0.5


def test_a_failure_inside_the_run_is_shown_as_an_error(tmp_path):
    def broken_clients(*args, **kwargs):
        raise RuntimeError("provider down")

    controller = RunController(
        load_settings(SIMULATED),
        runs_dir=tmp_path / "runs",
        data_dir=DATA_DIR,
        clients_factory=broken_clients,
    )

    run_id = controller.start(form())
    view = controller.wait(run_id, timeout=30)

    assert view.phase == "error"
    assert "provider down" in view.error


@pytest.mark.parametrize("lang", ["es", "en"])
def test_the_app_charts_are_drawn_in_the_interface_language(controller, lang):
    run_id = controller.start(form())
    controller.wait(run_id, timeout=60)

    rates, precision = controller.charts(run_id, lang)

    assert rates.endswith(f"app_rates_{lang}.png") and precision.endswith(
        f"app_precision_{lang}.png"
    )
    assert Path(rates).read_bytes().startswith(b"\x89PNG")
