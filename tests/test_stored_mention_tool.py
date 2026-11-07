from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from ai_share_of_voice.agents.specs import AGENT_SPECS
from ai_share_of_voice.errors import ToolFailure
from ai_share_of_voice.exceptions import BudgetExceeded
from ai_share_of_voice.run_store import RunStore
from ai_share_of_voice.schemas.analysis import RawAnswer, Tone
from ai_share_of_voice.schemas.brief import BrandSpec, Brief
from ai_share_of_voice.schemas.llm import Usage
from ai_share_of_voice.tools.mention_tool import (
    MAX_ANSWER_CHARS,
    BrandJudgement,
    MentionExtractionInput,
    MentionTool,
    StoredAnswerRequest,
    StoredMentionTool,
    extract_mentions,
    extract_stored_mentions,
)

RUN = "run-20261101-120000-abc123"
TEXT = "Para pedir cena, Acme Food es la mejor opción; Beta Eats es correcta. Ver https://x.es/a."


class ScriptedClient:
    """Fake model client: judges every mentioned brand the same way and records the prompts."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def complete[T: BaseModel](self, prompt: str, schema: type[T]) -> tuple[T, Usage]:
        self.prompts.append(prompt)
        listed = prompt.split("Brands to judge: ")[1].split("\n")[0]
        names = [name.strip(' "[]') for name in listed.split(",")]
        judgements = tuple(
            BrandJudgement(brand=name, recommended=name == "Acme Food", tone=Tone.POSITIVE)
            for name in names
        )
        usage = Usage(model="fake", input_tokens=10, output_tokens=5, cost_usd=0.001)
        return schema(judgements=judgements), usage


class BrokenClient:
    def complete[T: BaseModel](self, prompt: str, schema: type[T]) -> tuple[T, Usage]:
        raise BudgetExceeded(spent_usd=2.5, cap_usd=2.0)


def make_brief() -> Brief:
    return Brief(
        brand=BrandSpec(name="Acme Food", variants=("Acme",)),
        competitors=(BrandSpec(name="Beta Eats"), BrandSpec(name="Gamma Bites")),
        sector="food delivery",
        city="Madrid",
        spend_cap_usd=2.0,
        max_batches=5,
    )


def stored_run(tmp_path: Path, text: str = TEXT, response_id: str = "r1") -> RunStore:
    store = RunStore(tmp_path)
    store.save_brief(RUN, make_brief())
    answer = RawAnswer(
        response_id=response_id, question_id="q01", batch_index=0, model="fake", text=text
    )
    store.append_answer(RUN, answer)
    return store


# ---- what the agent has to pass ------------------------------------------------------------


def test_the_agent_passes_two_ids_and_never_the_text() -> None:
    assert set(StoredAnswerRequest.model_fields) == {"run_id", "response_id"}
    assert StoredMentionTool.args_schema is StoredAnswerRequest
    with pytest.raises(ValidationError):
        StoredAnswerRequest(run_id=RUN, response_id="r1", answer_text="made up text")


@pytest.mark.parametrize("bad", [{"run_id": "", "response_id": "r1"}, {"run_id": RUN}])
def test_the_tool_validates_its_arguments(tmp_path: Path, bad: dict[str, str]) -> None:
    tool = StoredMentionTool(ScriptedClient(), RunStore(tmp_path))
    with pytest.raises(ValidationError):
        tool.run(**bad)


def test_it_replaces_the_full_text_tool_under_the_same_name() -> None:
    assert StoredMentionTool.name == MentionTool.name == "extract_brand_mentions"
    assert StoredMentionTool.name in AGENT_SPECS["mention_extractor"].tools
    assert len(StoredMentionTool.description) < 500


# ---- reading the answer and the brief from the run -------------------------------------------


def test_the_stored_answer_gives_the_same_analysis_as_passing_the_text(tmp_path: Path) -> None:
    store = stored_run(tmp_path)
    by_id = StoredMentionTool(ScriptedClient(), store).run(run_id=RUN, response_id="r1")
    answer = store.load_answer(RUN, "r1")
    by_text = extract_mentions(
        MentionExtractionInput.from_answer(answer, make_brief()), ScriptedClient()
    )
    assert by_id.analysis == by_text.analysis
    rows = {m.brand: (m.mentioned, m.recommended, m.position) for m in by_id.analysis.mentions}
    assert rows == {
        "Acme Food": (True, True, 1),
        "Beta Eats": (True, False, 2),
        "Gamma Bites": (False, False, None),
    }
    assert tuple(str(source) for source in by_id.analysis.sources) == ("https://x.es/a",)
    assert by_id.analysis.response_id == "r1" and by_id.usage is not None


def test_the_graph_can_call_the_extraction_without_an_agent(tmp_path: Path) -> None:
    store = stored_run(tmp_path)
    result = extract_stored_mentions(RUN, "r1", store, ScriptedClient())
    assert [m.brand for m in result.analysis.mentions if m.mentioned] == ["Acme Food", "Beta Eats"]


def test_the_answer_reaches_the_model_only_inside_the_delimited_prompt(tmp_path: Path) -> None:
    client = ScriptedClient()
    StoredMentionTool(client, stored_run(tmp_path)).run(run_id=RUN, response_id="r1")
    assert len(client.prompts) == 1
    assert TEXT in client.prompts[0] and "<answer>" in client.prompts[0]


def test_no_model_call_is_made_when_no_brand_is_mentioned(tmp_path: Path) -> None:
    client = ScriptedClient()
    store = stored_run(tmp_path, text="No conozco ninguna aplicación concreta.")
    result = StoredMentionTool(client, store).run(run_id=RUN, response_id="r1")
    assert client.prompts == [] and result.usage is None
    assert not any(m.mentioned for m in result.analysis.mentions)


# ---- failures --------------------------------------------------------------------------------


def test_missing_run_answer_or_brief_are_tool_failures_and_cost_nothing(tmp_path: Path) -> None:
    client = ScriptedClient()
    tool = StoredMentionTool(client, stored_run(tmp_path))
    with pytest.raises(ToolFailure, match="no stored answer 'r9'"):
        tool.run(run_id=RUN, response_id="r9")
    with pytest.raises(ToolFailure, match="no stored answer"):
        tool.run(run_id="run-other", response_id="r1")
    (tmp_path / RUN / "brief.json").unlink()
    with pytest.raises(ToolFailure, match="no stored brief"):
        tool.run(run_id=RUN, response_id="r1")
    assert client.prompts == []


@pytest.mark.parametrize("run_id", ["..", "../x", "a/b"])
def test_an_agent_cannot_make_the_tool_read_outside_the_runs_folder(
    tmp_path: Path, run_id: str
) -> None:
    tool = StoredMentionTool(ScriptedClient(), RunStore(tmp_path / "runs"))
    with pytest.raises(ToolFailure, match="invalid run id"):
        tool.run(run_id=run_id, response_id="r1")


def test_an_answer_too_long_to_analyze_fails_without_echoing_it(tmp_path: Path) -> None:
    long_text = "Acme Food " + "palabra " * (MAX_ANSWER_CHARS // 4)
    client = ScriptedClient()
    store = stored_run(tmp_path, text=long_text)
    with pytest.raises(ToolFailure, match="cannot be analyzed") as caught:
        StoredMentionTool(client, store).run(run_id=RUN, response_id="r1")
    assert "palabra palabra" not in str(caught.value) and client.prompts == []


def test_a_spending_cap_error_reaches_the_graph(tmp_path: Path) -> None:
    tool = StoredMentionTool(BrokenClient(), stored_run(tmp_path))
    with pytest.raises(BudgetExceeded):
        tool.run(run_id=RUN, response_id="r1")
