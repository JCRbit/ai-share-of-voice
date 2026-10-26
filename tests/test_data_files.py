import csv
from collections import Counter
from pathlib import Path

import pytest

from ai_share_of_voice.schemas.brief import BrandSpec, Brief
from ai_share_of_voice.schemas.questions import Question, QuestionPanel

DATA = Path(__file__).resolve().parents[1] / "data"
QUESTION_FILES = ("questions.csv", "questions_es.csv")
INTENTS = ("recommendation", "comparison", "price", "service_area", "trust_reviews")


def read_rows(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_brands() -> list[tuple[BrandSpec, bool]]:
    return [
        (
            BrandSpec(name=row["name"], variants=tuple(row["variants"].split("|"))),
            row["is_client"] == "true",
        )
        for row in read_rows("competitors.csv")
    ]


def load_panel(name: str) -> QuestionPanel:
    return QuestionPanel(questions=tuple(Question(**row) for row in read_rows(name)))


def test_competitors_build_a_valid_brief() -> None:
    brands = load_brands()
    clients = [brand for brand, is_client in brands if is_client]
    competitors = tuple(brand for brand, is_client in brands if not is_client)
    assert len(clients) == 1
    Brief(
        brand=clients[0],
        competitors=competitors,
        sector="food delivery",
        city="Madrid",
        spend_cap_usd=2.0,
    )


def test_is_client_is_an_explicit_boolean() -> None:
    assert {row["is_client"] for row in read_rows("competitors.csv")} <= {"true", "false"}


def test_ambiguous_uber_is_not_a_variant() -> None:
    names = {name.casefold() for brand, _ in load_brands() for name in brand.all_names()}
    assert "uber" not in names


@pytest.mark.parametrize("name", QUESTION_FILES)
def test_question_panel_has_thirty_balanced_questions(name: str) -> None:
    panel = load_panel(name)
    assert len(panel.questions) == 30
    assert Counter(question.intent for question in panel.questions) == dict.fromkeys(INTENTS, 6)


def test_languages_are_paired_by_id_and_intent() -> None:
    english, spanish = (load_panel(name) for name in QUESTION_FILES)
    pairs = [(question.id, question.intent) for question in english.questions]
    assert pairs == [(question.id, question.intent) for question in spanish.questions]


@pytest.mark.parametrize("file_name", QUESTION_FILES)
def test_questions_do_not_name_any_brand(file_name: str) -> None:
    names = [name.casefold() for brand, _ in load_brands() for name in brand.all_names()]
    for question in load_panel(file_name).questions:
        text = question.text.casefold()
        assert not [brand for brand in names if brand in text], question.id
