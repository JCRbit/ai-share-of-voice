import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from ai_share_of_voice.exceptions import ConfigurationError
from ai_share_of_voice.llm_client import Pricing, SimulatedClient
from ai_share_of_voice.real_run import LOG_NAME, brief_from_data, run_real
from helpers_real_run import (
    LIVE_ENV,
    VISION_ENV,
    FakeSources,
    LiveTransport,
    respond,
    write_brands,
    write_data,
)

ROOT = Path(__file__).parents[1]


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("real_run_script", ROOT / "scripts/real_run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def project(tmp_path: Path, *, screenshots: bool = False) -> tuple[Path, Path]:
    """A runs folder and a data folder with the questions and the three brands."""
    data = tmp_path / "data"
    write_data(data, screenshots=screenshots)
    write_brands(data)
    return tmp_path / "runs", data


def small_brief(data: Path, **options):
    return brief_from_data(data, language="en", max_batches=2, target_half_width=0.01, **options)


# ---- the brief -------------------------------------------------------------------------------


def test_the_brief_comes_from_the_project_brands() -> None:
    brief = brief_from_data(ROOT / "data")
    assert brief.brand.name == "Glovo" and brief.brand.variants == ("GlovoApp", "Glovo App")
    assert [c.name for c in brief.competitors] == ["Just Eat", "Uber Eats"]
    assert brief.language == "es" and brief.spend_cap_usd == 1.0


@pytest.mark.parametrize("clients", [0, 2])
def test_exactly_one_client_is_required(tmp_path: Path, clients: int) -> None:
    rows = ["name,variants,is_client"]
    flags = ["true"] * clients + ["false"] * (3 - clients)
    rows += [
        f"{name},,{flag}" for name, flag in zip(("A One", "B Two", "C Three"), flags, strict=True)
    ]
    (tmp_path / "competitors.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="exactly one client"):
        brief_from_data(tmp_path)


# ---- run_real --------------------------------------------------------------------------------


def test_a_real_run_refuses_the_simulated_client(tmp_path: Path) -> None:
    runs, data = project(tmp_path)
    with pytest.raises(ConfigurationError, match="LLM_MODE=live"):
        run_real(
            small_brief(data),
            ("client",),
            approve=lambda state: True,
            env={"LLM_MODE": "simulated"},
            runs_dir=runs,
            data_dir=data,
        )
    assert not runs.exists()


def test_it_needs_at_least_the_client_stage(tmp_path: Path) -> None:
    runs, data = project(tmp_path)
    with pytest.raises(ConfigurationError, match="at least the client stage"):
        run_real(small_brief(data), (), approve=lambda s: True, runs_dir=runs, data_dir=data)


@pytest.mark.parametrize(("decision", "status"), [(True, "completed"), (False, "rejected")])
def test_the_run_pauses_for_the_approval_and_follows_the_decision(
    tmp_path: Path, decision: bool, status: str
) -> None:
    runs, data = project(tmp_path)
    client = SimulatedClient(responder=respond, pricing=Pricing(1.0, 1.0), cap_usd=5.0)
    seen: list[dict] = []

    def approve(state: dict) -> bool:
        seen.append(state)
        return decision

    result = run_real(
        small_brief(data), ("client",), approve=approve, client=client, runs_dir=runs, data_dir=data
    )
    assert len(seen) == 1 and "status" not in seen[0] and seen[0]["n_answers"] == 24
    assert result.state["status"] == status and result.approved is decision
    assert result.spent_usd == pytest.approx(client.spent_usd) and result.spent_usd > 0
    assert (runs / result.run_id / "brief.json").is_file()
    assert (runs / result.run_id / "events.jsonl").is_file()


def test_every_stage_goes_through_the_live_client_and_its_call_log(tmp_path: Path) -> None:
    runs, data = project(tmp_path, screenshots=True)
    transport = LiveTransport()
    result = run_real(
        small_brief(data),
        ("client", "mcp", "vision"),
        approve=lambda state: True,
        env=LIVE_ENV | VISION_ENV,
        runs_dir=runs,
        data_dir=data,
        sources=FakeSources(),
        transport=transport,
    )
    assert result.state["status"] == "completed" and len(result.state["findings"]) == 4
    log_lines = (runs / LOG_NAME).read_text(encoding="utf-8").splitlines()
    entries = [json.loads(line) for line in log_lines]
    assert len(entries) == len(transport.payloads)
    assert {e["run_id"] for e in entries} == {result.run_id}
    images = [p for p in transport.payloads if isinstance(p["messages"][0]["content"], list)]
    assert len(images) == 4 and {p["model"] for p in images} == {"vision-model"}
    assert {e["model"] for e in entries} == {"openai/gpt-oss-120b", "vision-model"}
    assert result.spent_usd == pytest.approx(sum(e["cost_usd"] for e in entries))


def test_the_vision_stage_without_a_vision_model_is_refused_before_any_call(
    tmp_path: Path,
) -> None:
    runs, data = project(tmp_path, screenshots=True)
    transport = LiveTransport()
    with pytest.raises(ConfigurationError, match="vision model"):
        run_real(
            small_brief(data),
            ("client", "mcp", "vision"),
            approve=lambda state: True,
            env=LIVE_ENV,
            runs_dir=runs,
            data_dir=data,
            sources=FakeSources(),
            transport=transport,
        )
    assert transport.payloads == []


def test_the_run_cap_closes_a_real_run_without_going_over(tmp_path: Path) -> None:
    runs, data = project(tmp_path)
    prices = {"LLM_PRICE_INPUT_PER_M": "200", "LLM_PRICE_OUTPUT_PER_M": "200"}
    result = run_real(
        small_brief(data),
        ("client",),
        approve=lambda state: pytest.fail("a closed run must not ask for approval"),
        env=LIVE_ENV | prices,
        runs_dir=runs,
        data_dir=data,
        transport=LiveTransport(),
    )
    assert result.state["status"] == "partial"
    assert result.state["failure_kind"] == "budget_exhausted"
    assert 0 < result.spent_usd <= 1.0 and result.approved is None


# ---- the script ------------------------------------------------------------------------------


def run_script(tmp_path: Path, *args: str, env=None, ask=lambda prompt: "y", screenshots=False):
    runs, data = project(tmp_path, screenshots=screenshots)
    module = load_script()
    argv = ["--language", "en", "--max-batches", "2", "--target-half-width", "0.01"]
    argv += ["--runs-dir", str(runs), "--data-dir", str(data), *args]
    return module.main(argv, env=env or LIVE_ENV, transport=LiveTransport(), ask=ask)


def test_the_script_runs_and_shows_what_the_run_found(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    assert run_script(tmp_path, "--yes") == 0
    out = capsys.readouterr().out
    assert "mention rate of the client: 50%" in out and "after 2 batches" in out
    assert "completed, spent" in out


def test_the_script_asks_and_a_no_closes_the_run_without_publishing(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    questions: list[str] = []

    def ask(prompt: str) -> str:
        questions.append(prompt)
        return "n"

    assert run_script(tmp_path, ask=ask) == 1
    assert len(questions) == 1 and "rejected" in capsys.readouterr().out


def test_the_script_reports_setup_problems_with_status_two(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    assert run_script(tmp_path, "--yes", env={"LLM_MODE": "simulated"}) == 2
    assert "LLM_MODE=live" in capsys.readouterr().err
    assert run_script(tmp_path / "b", "--yes", "--stages", "vision") == 2
    assert run_script(tmp_path / "c", "--yes", "--cap", "0") == 2


def test_the_script_accepts_yes_with_spanish_accent(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    assert run_script(tmp_path, ask=lambda prompt: "sí") == 0
    assert "completed, spent" in capsys.readouterr().out
