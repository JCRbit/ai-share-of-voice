import pytest

from ai_share_of_voice.errors import ToolFailure
from ai_share_of_voice.exceptions import BudgetExceeded
from ai_share_of_voice.llm_client import Budget, SimulatedClient
from ai_share_of_voice.panel import (
    INTENTS,
    DraftQuestion,
    DraftQuestions,
    build_prompt,
    generate_panel,
    read_panel_file,
    write_panel_file,
)
from ai_share_of_voice.schemas.brief import BrandSpec, Brief

BRIEF = Brief(
    brand=BrandSpec(name="Zara", variants=("Zara Home",)),
    competitors=(BrandSpec(name="Mango"), BrandSpec(name="H&M", variants=("HM",))),
    sector="clothing stores",
    city="Valencia",
    language="es",
    spend_cap_usd=1.0,
)


def drafts(*texts, intent="recommendation"):
    return DraftQuestions(questions=[DraftQuestion(text=text, intent=intent) for text in texts])


def scripted(*answers, budget=None):
    """A client that returns each DraftQuestions in turn and records the prompts."""
    queue = list(answers)
    prompts = []

    def respond(prompt):
        prompts.append(prompt)
        return queue.pop(0)

    client = SimulatedClient(budget or Budget(1.0), responders={DraftQuestions: respond})
    return client, prompts


def numbered(count, start=1, intent="recommendation"):
    return [
        f"¿Dónde compro ropa buena opción {index} en Valencia?"
        for index in range(start, start + count)
    ]


def test_the_prompt_asks_for_the_sector_city_language_and_intents():
    prompt = build_prompt(BRIEF, 30)

    for text in ("clothing stores", "Valencia", "es", "30", *INTENTS):
        assert text in prompt


def test_the_prompt_never_lists_the_tracked_brands():
    # Naming them would invite the model to put them in the questions.
    prompt = build_prompt(BRIEF, 30)

    assert "Zara" not in prompt and "Mango" not in prompt


def test_a_valid_panel_gets_sequential_ids():
    client, _ = scripted(drafts(*numbered(12)))

    panel = generate_panel(BRIEF, client, size=12)

    assert [question.id for question in panel.questions][:3] == ["q01", "q02", "q03"]
    assert len(panel.questions) == 12


def test_questions_that_name_a_tracked_brand_are_rejected():
    texts = [*numbered(10), "¿Es mejor Zara que otras tiendas?", "¿Qué tal la tienda de hm?"]
    client, _ = scripted(drafts(*texts), drafts())

    panel = generate_panel(BRIEF, client, size=12)

    assert all("zara" not in q.text.casefold() for q in panel.questions)
    assert len(panel.questions) == 10


def test_a_brand_inside_another_word_is_not_a_mention():
    # "hm" inside "hmm" or "Mango" inside "mangos" must not reject a question.
    client, _ = scripted(drafts(*numbered(9), "¿Hay tiendas de ropa con estampado de mangos?"))

    panel = generate_panel(BRIEF, client, size=10)

    assert len(panel.questions) == 10


def test_duplicates_and_unknown_intents_are_dropped():
    texts = numbered(10)
    answer = DraftQuestions(
        questions=[
            *[DraftQuestion(text=text, intent="price") for text in texts],
            DraftQuestion(text=texts[0].upper(), intent="price"),  # same question, other case
            DraftQuestion(text="¿Qué tienda me recomiendas hoy?", intent="weather"),
        ]
    )
    client, _ = scripted(answer, drafts())

    panel = generate_panel(BRIEF, client, size=12)

    assert len(panel.questions) == 10


def test_missing_questions_are_asked_for_once_more():
    client, prompts = scripted(drafts(*numbered(8)), drafts(*numbered(4, start=9)))

    panel = generate_panel(BRIEF, client, size=12)

    assert len(prompts) == 2
    assert "4" in prompts[1]  # the second call asks only for what is missing
    assert len(panel.questions) == 12


def test_too_few_valid_questions_is_a_tool_failure():
    # Under 10 questions the bootstrap interval is not reliable: do not start a run on it.
    client, _ = scripted(drafts(*numbered(3)), drafts(*numbered(2, start=4)))

    with pytest.raises(ToolFailure, match="5 valid questions"):
        generate_panel(BRIEF, client, size=30)


def test_an_exhausted_budget_propagates():
    client, _ = scripted(drafts(*numbered(30)), budget=Budget(0.0001))

    with pytest.raises(BudgetExceeded):
        generate_panel(BRIEF, client, size=30)


def test_a_panel_file_round_trips(tmp_path):
    client, _ = scripted(drafts(*numbered(10)))
    panel = generate_panel(BRIEF, client, size=10)

    path = write_panel_file(panel, tmp_path / "questions.csv")

    assert read_panel_file(path) == panel
    assert path.read_text("utf-8").splitlines()[0] == "id,text,intent"


def test_the_panel_alternates_intents_so_a_cut_keeps_the_mix():
    answer = DraftQuestions(
        questions=[
            DraftQuestion(text=text, intent=intent)
            for intent in INTENTS
            for text in [f"¿Pregunta {intent} número {index} en Valencia?" for index in range(2)]
        ]
    )
    client, _ = scripted(answer)

    panel = generate_panel(BRIEF, client, size=10)

    assert [question.intent for question in panel.questions[:5]] == list(INTENTS)


# --- the user's own panel (R9) --------------------------------------------------------------

from ai_share_of_voice.panel import MAX_USER_QUESTIONS, PanelProblem, check_user_panel  # noqa: E402


def rows(*texts, intent=""):
    return [(text, intent) for text in texts]


VALID = [f"¿Dónde compro ropa buena opción {index} en Valencia?" for index in range(1, 6)]


def test_a_valid_user_panel_is_numbered_and_keeps_its_intents():
    panel, problems = check_user_panel([(VALID[0], "price"), *rows(*VALID[1:])], BRIEF)

    assert problems == []
    assert [question.id for question in panel.questions] == ["q01", "q02", "q03", "q04", "q05"]
    assert panel.questions[0].intent == "price"
    assert panel.questions[1].intent == "custom"  # a question the user wrote


def test_empty_rows_are_ignored():
    panel, problems = check_user_panel([*rows(*VALID), ("  ", ""), ("", "price")], BRIEF)

    assert problems == [] and len(panel.questions) == 5


def test_each_problem_points_at_its_row():
    texts = [*VALID, "¿Es buena Zara?", "Hola", VALID[0].upper()]

    panel, problems = check_user_panel(rows(*texts), BRIEF)

    assert panel is None
    assert problems == [
        PanelProblem(row=6, kind="names_brand"),
        PanelProblem(row=7, kind="too_short"),
        PanelProblem(row=8, kind="duplicate"),
    ]


def test_too_few_or_too_many_questions():
    _, few = check_user_panel(rows(*VALID[:2]), BRIEF)
    many = [
        f"Pregunta de prueba número {index} sobre tiendas"
        for index in range(MAX_USER_QUESTIONS + 1)
    ]
    _, too_many = check_user_panel(rows(*many), BRIEF)

    assert few == [PanelProblem(row=0, kind="too_few")]
    assert too_many == [PanelProblem(row=0, kind="too_many")]
