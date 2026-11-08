import threading
from pathlib import Path

import pytest

from ai_share_of_voice.errors import ToolFailure
from ai_share_of_voice.run_store import (
    ANALYSES_FILE_NAME,
    ANSWERS_FILE_NAME,
    BRIEF_FILE_NAME,
    RunStore,
)
from ai_share_of_voice.schemas.analysis import AnalyzedResponse, BrandMention, RawAnswer, Tone
from ai_share_of_voice.schemas.brief import BrandSpec, Brief

RUN = "run-20261101-120000-abc123"


def make_brief(sector: str = "food delivery") -> Brief:
    return Brief(
        brand=BrandSpec(name="Acme Food", variants=("Acme",)),
        competitors=(BrandSpec(name="Beta Eats"), BrandSpec(name="Gamma Bites")),
        sector=sector,
        city="Madrid",
        spend_cap_usd=2.0,
        max_batches=5,
    )


def make_answer(response_id: str = "r1", text: str = "Acme es una buena opción.") -> RawAnswer:
    return RawAnswer(
        response_id=response_id, question_id="q01", batch_index=0, model="fake-model", text=text
    )


def make_analysis(response_id: str = "r1", url: str = "https://example.com/a") -> AnalyzedResponse:
    acme = BrandMention(
        brand="Acme Food", mentioned=True, recommended=True, position=1, tone=Tone.POSITIVE
    )
    mentions = (acme, BrandMention(brand="Beta Eats", mentioned=False))
    return AnalyzedResponse(
        response_id=response_id,
        question_id="q01",
        batch_index=0,
        mentions=mentions,
        sources=(url,),
    )


def test_the_brief_is_saved_in_the_run_folder_and_read_back(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    brief = make_brief()
    path = store.save_brief(RUN, brief)
    assert path == tmp_path / RUN / BRIEF_FILE_NAME and path.is_file()
    assert store.load_brief(RUN) == brief


def test_saving_the_brief_again_replaces_it(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    store.save_brief(RUN, make_brief("food delivery"))
    store.save_brief(RUN, make_brief("grocery delivery"))
    assert store.load_brief(RUN).sector == "grocery delivery"
    assert [p.name for p in (tmp_path / RUN).iterdir()] == [BRIEF_FILE_NAME]


def test_answers_are_stored_in_order_and_found_by_id(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    assert store.answer_ids(RUN) == []
    first, second = make_answer("r1"), make_answer("r2", "Beta Eats, sin duda.")
    store.append_answer(RUN, first)
    store.append_answer(RUN, second)
    assert store.answer_ids(RUN) == ["r1", "r2"]
    assert store.load_answer(RUN, "r2") == second
    assert (tmp_path / RUN / ANSWERS_FILE_NAME).read_text(encoding="utf-8").count("\n") == 2


def test_text_with_lines_quotes_and_accents_survives_the_round_trip(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    text = 'Línea uno\n\nLínea "dos", con coma y ñ\t— fin'
    store.append_answer(RUN, make_answer("r1", text))
    assert store.load_answer(RUN, "r1").text == text


def test_an_answer_id_that_is_already_stored_is_refused(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    store.append_answer(RUN, make_answer("r1"))
    with pytest.raises(ToolFailure, match="already stored"):
        store.append_answer(RUN, make_answer("r1", "otro texto"))
    assert store.load_answer(RUN, "r1").text == "Acme es una buena opción."


def test_a_missing_run_brief_or_answer_is_a_tool_failure(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    with pytest.raises(ToolFailure, match="no stored brief"):
        store.load_brief(RUN)
    with pytest.raises(ToolFailure, match="no stored answer 'r9'"):
        store.load_answer(RUN, "r9")
    store.append_answer(RUN, make_answer("r1"))
    with pytest.raises(ToolFailure, match="no stored answer 'r9'"):
        store.load_answer(RUN, "r9")


@pytest.mark.parametrize("run_id", ["", ".", "..", "../outside", "a/b", "a\\b", "/etc"])
def test_a_run_id_that_leaves_the_runs_folder_is_refused(tmp_path: Path, run_id: str) -> None:
    store = RunStore(tmp_path / "runs")
    with pytest.raises(ToolFailure, match="invalid run id"):
        store.load_brief(run_id)
    with pytest.raises(ToolFailure, match="invalid run id"):
        store.load_answer(run_id, "r1")
    with pytest.raises(ToolFailure, match="invalid run id"):
        store.append_answer(run_id, make_answer())
    with pytest.raises(ToolFailure, match="invalid run id"):
        store.save_brief(run_id, make_brief())
    assert not (tmp_path / "outside").exists() and not (tmp_path / "runs").exists()


def test_damaged_files_are_reported_with_the_line_not_with_their_content(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    store.append_answer(RUN, make_answer("r1"))
    answers = tmp_path / RUN / ANSWERS_FILE_NAME
    answers.write_text(
        answers.read_text(encoding="utf-8") + '{"secret": "do not echo me"}\n', encoding="utf-8"
    )
    with pytest.raises(ToolFailure, match="damaged at line 2") as caught:
        store.load_answer(RUN, "r1")
    assert "do not echo me" not in str(caught.value)
    (tmp_path / RUN / BRIEF_FILE_NAME).write_text("{not json", encoding="utf-8")
    with pytest.raises(ToolFailure, match="cannot be read"):
        store.load_brief(RUN)


def test_blank_lines_in_the_answers_file_are_ignored(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    store.append_answer(RUN, make_answer("r1"))
    answers = tmp_path / RUN / ANSWERS_FILE_NAME
    answers.write_text("\n" + answers.read_text(encoding="utf-8") + "\n\n", encoding="utf-8")
    assert store.answer_ids(RUN) == ["r1"]


def test_runs_do_not_see_each_others_answers(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    store.append_answer("run-a", make_answer("r1"))
    store.append_answer("run-b", make_answer("r1", "Gamma Bites."))
    assert store.load_answer("run-a", "r1").text != store.load_answer("run-b", "r1").text
    assert store.answer_ids("run-c") == []


def test_threads_can_append_at_the_same_time_without_losing_or_mixing_answers(
    tmp_path: Path,
) -> None:
    store = RunStore(tmp_path)
    ids = [f"r{i:02d}" for i in range(20)]
    threads = [
        threading.Thread(target=store.append_answer, args=(RUN, make_answer(i, f"texto {i}")))
        for i in ids
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(store.answer_ids(RUN)) == ids
    assert all(store.load_answer(RUN, i).text == f"texto {i}" for i in ids)


def test_analyses_are_stored_in_order_and_read_back(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    assert store.load_analyses(RUN) == []
    first, second = make_analysis("r1"), make_analysis("r2", "https://example.com/b")
    store.append_analysis(RUN, first)
    store.append_analysis(RUN, second)
    assert store.load_analyses(RUN) == [first, second]
    assert (tmp_path / RUN / ANALYSES_FILE_NAME).read_text(encoding="utf-8").count("\n") == 2


def test_an_analysis_that_is_already_stored_is_refused(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    store.append_analysis(RUN, make_analysis("r1"))
    with pytest.raises(ToolFailure, match="already analyzed"):
        store.append_analysis(RUN, make_analysis("r1", "https://example.com/other"))
    assert len(store.load_analyses(RUN)) == 1


def test_answers_and_analyses_are_kept_apart_and_a_damaged_one_names_its_file(
    tmp_path: Path,
) -> None:
    store = RunStore(tmp_path)
    store.append_answer(RUN, make_answer("r1"))
    store.append_analysis(RUN, make_analysis("r1"))
    assert store.answer_ids(RUN) == ["r1"] and len(store.load_analyses(RUN)) == 1
    (tmp_path / RUN / ANALYSES_FILE_NAME).write_text("not json\n", encoding="utf-8")
    with pytest.raises(ToolFailure, match=r"analyzed_answers\.jsonl.*damaged at line 1"):
        store.load_analyses(RUN)
    assert store.answer_ids(RUN) == ["r1"]


def test_analyses_are_refused_for_an_id_that_leaves_the_runs_folder(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs")
    with pytest.raises(ToolFailure, match="invalid run id"):
        store.append_analysis("../x", make_analysis())
    with pytest.raises(ToolFailure, match="invalid run id"):
        store.load_analyses("a/b")
