import pytest

from ai_share_of_voice.agents.skeptic_checks import (
    MAX_CLAIM_CHARS,
    ObjectionKind,
    RunRecord,
    check_causality,
    check_figures,
    check_market_share,
    check_sources,
    run_checks,
)

RECORD = RunRecord(
    sources=("https://example.com/source-1", "https://example.org/b"),
    percentages=(50.0, 25.0, 75.0, 32.3, 67.7),
    answer_counts=(4, 8),
)


def kinds(objections: list) -> list[ObjectionKind]:
    return [objection.kind for objection in objections]


# ---- figures ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Acme is mentioned in 50% of the answers.",
        "The interval runs from 32% to 68%.",
        "The lower bound is 32,3%.",
        "We analyzed 8 answers in total.",
        "In 2026 there were 3 competitors.",
    ],
    ids=["recorded", "rounded", "decimal_comma", "answer_count", "plain_numbers"],
)
def test_recorded_figures_are_accepted(text: str) -> None:
    assert check_figures(text, RECORD) == []


def test_an_unrecorded_percentage_is_an_objection_tied_to_its_sentence() -> None:
    (objection,) = check_figures("Acme is mentioned in 99% of the answers.", RECORD)
    assert objection.kind is ObjectionKind.UNRECORDED_FIGURE
    assert "99%" in objection.reason
    assert "99%" in objection.claim


def test_an_unrecorded_answer_count_is_an_objection() -> None:
    (objection,) = check_figures("We analyzed 120 answers in total.", RECORD)
    assert "120 answers" in objection.reason


def test_two_bad_figures_in_one_sentence_make_one_objection() -> None:
    assert len(check_figures("It reached 90% and then 91%.", RECORD)) == 1


def test_without_recorded_metrics_any_percentage_is_unverifiable() -> None:
    assert kinds(check_figures("The share is 50%.", RunRecord())) == [
        ObjectionKind.UNRECORDED_FIGURE
    ]


# ---- sources ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "See https://example.com/source-1/.",
        "See HTTPS://EXAMPLE.ORG/b",
        "Acme leads [S1] and [S2].",
    ],
    ids=["trailing_slash_and_dot", "host_case", "markers"],
)
def test_recorded_sources_are_accepted(text: str) -> None:
    assert check_sources(text, RECORD) == []


def test_an_unknown_url_is_an_objection() -> None:
    (objection,) = check_sources("See https://evil.example/x for details.", RECORD)
    assert objection.kind is ObjectionKind.UNKNOWN_SOURCE
    assert "https://evil.example/x" in objection.reason


@pytest.mark.parametrize("marker", ["[S0]", "[S3]", "[S9]"])
def test_a_marker_outside_the_recorded_sources_is_an_objection(marker: str) -> None:
    (objection,) = check_sources(f"Acme leads {marker}.", RECORD)
    assert marker in objection.reason


# ---- causality and market share ---------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Beta ranks first because of the review at [S1].",
        "Beta wins thanks to its menu screenshot.",
        "Beta lidera debido a la fuente [S1].",
    ],
    ids=["source", "screenshot", "spanish"],
)
def test_a_cause_resting_on_evidence_is_an_objection(text: str) -> None:
    assert kinds(check_causality(text)) == [ObjectionKind.CAUSAL_CLAIM]


@pytest.mark.parametrize(
    "text",
    [
        "This does not show that the source caused the recommendation [S1].",
        "Answers vary because the questions are broad.",
    ],
    ids=["hedged", "no_evidence"],
)
def test_hedged_or_unrelated_causes_are_not_objections(text: str) -> None:
    assert check_causality(text) == []


def test_a_market_share_claim_is_an_objection() -> None:
    assert kinds(check_market_share("Acme holds 12% market share.")) == [
        ObjectionKind.MARKET_SHARE_CLAIM
    ]
    assert len(check_market_share("Acme tiene una cuota de mercado alta.")) == 1


def test_denying_market_share_is_fine() -> None:
    assert check_market_share("This is not market share.") == []


# ---- all checks together ----------------------------------------------------------------


def test_an_empty_draft_is_one_objection() -> None:
    assert kinds(run_checks("   ", RECORD)) == [ObjectionKind.EMPTY_DRAFT]


def test_a_clean_draft_has_no_objections() -> None:
    draft = "Acme is mentioned in 50% of the 8 answers [S1]. This is not market share."
    assert run_checks(draft, RECORD) == []


def test_one_sentence_can_break_several_rules_and_each_is_reported_once() -> None:
    found = run_checks("Acme holds 99% market share because of [S9].", RECORD)
    assert set(kinds(found)) == {
        ObjectionKind.UNRECORDED_FIGURE,
        ObjectionKind.UNKNOWN_SOURCE,
        ObjectionKind.CAUSAL_CLAIM,
        ObjectionKind.MARKET_SHARE_CLAIM,
    }
    assert len(found) == 4


def test_claims_are_bounded_and_rendered_on_one_line() -> None:
    long_sentence = "Acme is mentioned in 99% of answers " + "very " * 100 + "often."
    (objection,) = run_checks(long_sentence, RECORD)
    assert len(objection.claim) <= MAX_CLAIM_CHARS
    rendered = objection.render()
    assert "\n" not in rendered
    assert rendered.startswith("[unrecorded_figure] ")


def test_markdown_bullets_are_not_part_of_the_claim() -> None:
    (objection,) = run_checks("- Acme is at 99%.", RECORD)
    assert objection.claim == "Acme is at 99%."
