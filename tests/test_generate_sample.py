import csv
import importlib.util
import json
from pathlib import Path

import pytest

from ai_share_of_voice.errors import ToolFailure
from ai_share_of_voice.exceptions import BudgetExceeded
from ai_share_of_voice.llm_client import SimulatedClient
from ai_share_of_voice.schemas.analysis import RawAnswer
from ai_share_of_voice.schemas.llm import Usage

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "generate_sample.py"
QUESTIONS = ROOT / "data" / "questions_es.csv"
spec = importlib.util.spec_from_file_location("generate_sample", SCRIPT)
sample = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sample)

LIVE_ENV = {"LLM_MODE": "live", "LLM_PROVIDER": "groq", "LLM_API_KEY": "key"}


def panel_of(count: int):
    rows = [
        sample.Question(id=f"q{i:02d}", text=f"Pregunta número {i}", intent="recommendation")
        for i in range(1, count + 1)
    ]
    return sample.QuestionPanel(questions=tuple(rows))


def simulated(text=lambda prompt: f"Respuesta a: {prompt[-20:]}") -> SimulatedClient:
    def respond(prompt, schema):
        return schema(answer=text(prompt))

    return SimulatedClient(responder=respond)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class FlakyClient:
    """Fake client: the scripted calls raise, the others answer."""

    def __init__(self, errors: dict[int, Exception]) -> None:
        self.errors = errors
        self.calls = 0

    def complete(self, prompt, schema):
        self.calls += 1
        if self.calls in self.errors:
            raise self.errors[self.calls]
        usage = Usage(model="fake", input_tokens=1, output_tokens=1, cost_usd=0.0)
        return schema(answer=f"respuesta {self.calls}"), usage


class Transport:
    def __init__(self) -> None:
        self.payloads: list[dict] = []

    def __call__(self, payload: dict) -> dict:
        self.payloads.append(payload)
        body = json.dumps({"answer": f"Respuesta {len(self.payloads)} de prueba"})
        usage = {"prompt_tokens": 120, "completion_tokens": 40}
        return {"choices": [{"message": {"content": body}}], "usage": usage}


# ---- the questions and the plan ------------------------------------------------------------


@pytest.mark.parametrize("name", ["questions_es.csv", "questions.csv"])
def test_the_project_questions_load_as_a_panel(name: str) -> None:
    panel = sample.load_questions(ROOT / "data" / name)
    assert len(panel.questions) == 30


def test_sixty_answers_are_two_complete_batches_of_thirty_questions() -> None:
    pairs = sample.plan(panel_of(30), 60)
    assert len(pairs) == 60
    assert [q.id for q, r in pairs[:30]] == [f"q{i:02d}" for i in range(1, 31)]
    assert [q.id for q, r in pairs[30:]] == [f"q{i:02d}" for i in range(1, 31)]
    assert {r for q, r in pairs[:30]} == {0}
    assert {r for q, r in pairs[30:]} == {1}
    assert len({sample.response_id(q, r) for q, r in pairs}) == 60


def test_default_sample_size_is_two_complete_batches() -> None:
    assert sample.DEFAULT_ANSWERS == 60


def test_the_prompt_is_the_question_and_names_no_brand() -> None:
    question = panel_of(1).questions[0]
    prompt = sample.build_prompt(question)
    assert "Pregunta número 1" in prompt and "answer field" in prompt
    names = []
    for row in read_rows(ROOT / "data" / "competitors.csv"):
        names += [row["name"], *row["variants"].split("|")]
    for real_question in sample.load_questions(QUESTIONS).questions:
        prompt = sample.build_prompt(real_question).lower()
        assert not any(name.lower() in prompt for name in names)


# ---- writing, resuming and stopping --------------------------------------------------------


def test_the_file_has_the_columns_of_a_raw_answer_and_reads_back(tmp_path: Path) -> None:
    out = tmp_path / "sample.csv"
    report = sample.generate(simulated(), panel_of(3), out, 5)
    rows = read_rows(out)
    assert report.written == 5 and report.stopped is None and len(rows) == 5
    assert set(sample.COLUMNS) == set(RawAnswer.model_fields)
    answers = [RawAnswer(**(row | {"batch_index": int(row["batch_index"])})) for row in rows]
    assert [a.batch_index for a in answers] == [0, 0, 0, 1, 1]
    assert all(a.model and a.text.startswith("Respuesta") for a in answers)


def test_a_rerun_only_asks_for_what_is_missing(tmp_path: Path) -> None:
    out = tmp_path / "sample.csv"
    sample.generate(simulated(), panel_of(3), out, 3)
    client = FlakyClient({})
    report = sample.generate(client, panel_of(3), out, 5)
    assert (report.skipped, report.written, client.calls) == (3, 2, 2)
    assert len(read_rows(out)) == 5
    assert sample.count_pending(panel_of(3), out, 5) == 0


def test_a_spending_cap_stops_the_run_and_keeps_what_was_written(tmp_path: Path) -> None:
    out = tmp_path / "sample.csv"
    client = FlakyClient({3: BudgetExceeded(0.49, 0.5)})
    report = sample.generate(client, panel_of(5), out, 5)
    assert report.written == 2 and "spending cap" in report.stopped
    assert len(read_rows(out)) == 2
    resumed = sample.generate(FlakyClient({}), panel_of(5), out, 5)
    assert resumed.written == 3 and resumed.stopped is None


def test_a_failure_is_skipped_but_three_in_a_row_stop_the_run(tmp_path: Path) -> None:
    out = tmp_path / "sample.csv"
    one_bad = FlakyClient({2: ToolFailure("groq", "timeout")})
    report = sample.generate(one_bad, panel_of(3), out, 3)
    assert report.written == 2 and len(report.failed) == 1 and report.stopped is None
    always = FlakyClient({i: ToolFailure("groq", "down") for i in range(1, 10)})
    stopped = sample.generate(always, panel_of(5), tmp_path / "b.csv", 5)
    assert always.calls == 3 and "in a row" in stopped.stopped and stopped.written == 0


def test_a_file_that_is_not_a_sample_is_never_appended_to(tmp_path: Path) -> None:
    out = tmp_path / "sample.csv"
    out.write_text("id,answer\n1,hola\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not have the columns"):
        sample.generate(simulated(), panel_of(2), out, 2)
    assert out.read_text(encoding="utf-8") == "id,answer\n1,hola\n"


# ---- untrusted text in a spreadsheet -------------------------------------------------------


@pytest.mark.parametrize("text", ["=1+1", "+34 600", "- Glovo es una opción", "@usuario", "-1"])
def test_text_that_would_run_as_a_formula_is_shown_as_text(tmp_path: Path, text: str) -> None:
    out = tmp_path / "sample.csv"
    sample.generate(simulated(lambda prompt: text), panel_of(1), out, 1)
    assert read_rows(out)[0]["text"] == "'" + text


def test_normal_text_is_saved_untouched_even_with_commas_and_lines(tmp_path: Path) -> None:
    out = tmp_path / "sample.csv"
    text = "Hola, mundo\nSegunda línea con \"comillas\""
    sample.generate(simulated(lambda prompt: text), panel_of(1), out, 1)
    assert read_rows(out)[0]["text"] == text


# ---- the command line ----------------------------------------------------------------------


def test_the_script_refuses_to_run_with_the_simulated_client(tmp_path: Path) -> None:
    out = tmp_path / "sample.csv"
    args = ["--answers", "2", "--output", str(out), "--questions", str(QUESTIONS)]
    code = sample.main(args, env={"LLM_MODE": "simulated"})
    assert code == 2 and not out.exists()


@pytest.mark.parametrize("flag", [["--answers", "0"], ["--cap", "0"], ["--cap", "-1"]])
def test_the_script_rejects_nonsense_limits(tmp_path: Path, flag: list[str]) -> None:
    args = [*flag, "--output", str(tmp_path / "s.csv"), "--questions", str(QUESTIONS)]
    assert sample.main(args, env=LIVE_ENV) == 2


def test_the_script_needs_an_api_key_in_live_mode(tmp_path: Path) -> None:
    env = {"LLM_MODE": "live", "LLM_PROVIDER": "groq"}
    args = ["--output", str(tmp_path / "s.csv"), "--questions", str(QUESTIONS)]
    assert sample.main(args, env=env) == 2


def test_a_live_run_writes_the_sample_and_logs_every_call(tmp_path: Path) -> None:
    out, log, transport = tmp_path / "sample.csv", tmp_path / "calls.jsonl", Transport()
    args = ["--answers", "3", "--output", str(out), "--log", str(log), "--cap", "0.5"]
    args += ["--questions", str(QUESTIONS)]
    assert sample.main(args, env=LIVE_ENV, transport=transport) == 0
    rows = read_rows(out)
    assert [row["question_id"] for row in rows] == ["q01", "q02", "q03"]
    assert {row["model"] for row in rows} == {"openai/gpt-oss-120b"}
    assert len(log.read_text(encoding="utf-8").splitlines()) == 3
    assert len(transport.payloads) == 3
    again = sample.main(args, env=LIVE_ENV, transport=transport)
    assert again == 0 and len(transport.payloads) == 3 and len(read_rows(out)) == 3


def test_a_tiny_cap_stops_a_live_run_before_it_spends(tmp_path: Path) -> None:
    out, transport = tmp_path / "sample.csv", Transport()
    args = ["--answers", "3", "--output", str(out), "--log", str(tmp_path / "c.jsonl")]
    args += ["--questions", str(QUESTIONS)]
    code = sample.main([*args, "--cap", "0.0000001"], env=LIVE_ENV, transport=transport)
    assert code == 1 and not transport.payloads and not out.exists()
