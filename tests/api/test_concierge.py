"""The chat's rule-based reader and its figure-only answers."""

from ai_share_of_voice.api.concierge import answer_question, read_brief


def test_reads_a_full_brief_without_a_model():
    d = read_brief("Glovo frente a Just Eat y Uber Eats, comida a domicilio en Madrid")
    assert d.brand == "Glovo"
    assert d.competitors == ["Just Eat", "Uber Eats"]
    assert d.sector == "Comida a domicilio"
    assert (d.city, d.country) == ("Madrid", "España")
    assert d.missing == []


def test_english_brief_and_comma_list():
    d = read_brief("Gold Gym versus Basic-Fit and McFit, gyms in Barcelona", "en")
    assert d.brand == "Gold Gym"
    assert d.competitors == ["Basic-Fit", "McFit"]
    assert d.sector == "Gyms"


def test_missing_fields_are_asked_for():
    d = read_brief("Glovo en Madrid")
    assert "competitors" in d.missing and "sector" in d.missing
    assert "al menos dos competidores" in d.reply


def facts(client_rec, other_rec):
    return {
        "brands": [
            {
                "name": "Glovo",
                "is_client": True,
                "mention": {"estimate": 0.7, "ci_low": 0.4, "ci_high": 1.0},
                "recommend": client_rec,
            },
            {"name": "Just Eat", "is_client": False, "recommend": other_rec},
        ],
        "stop": {"reason_text": "Paré porque el margen ya era suficiente."},
        "n_questions": 10,
        "headline": "Titular.",
        "research": {"sources": []},
    }


def test_who_wins_only_claims_a_lead_without_overlap():
    clear = answer_question(
        "¿quién gana?",
        facts(
            {"estimate": 0.8, "ci_low": 0.7, "ci_high": 0.9},
            {"estimate": 0.2, "ci_low": 0.1, "ci_high": 0.3},
        ),
        "es",
    )
    assert "es clara" in clear
    unclear = answer_question(
        "¿quién gana?",
        facts(
            {"estimate": 0.6, "ci_low": 0.4, "ci_high": 0.8},
            {"estimate": 0.5, "ci_low": 0.3, "ci_high": 0.7},
        ),
        "es",
    )
    assert "no puedo decir que gane" in unclear


def test_other_intents_use_the_figures():
    f = facts(
        {"estimate": 0.5, "ci_low": 0.3, "ci_high": 0.7},
        {"estimate": 0.5, "ci_low": 0.3, "ci_high": 0.7},
    )
    assert (
        answer_question("¿por qué paraste?", f, "es") == "Paré porque el margen ya era suficiente."
    )
    assert "10 preguntas" in answer_question("¿cuánto me puedo fiar?", f, "es")
    assert "búsqueda web estaba apagada" in answer_question("¿qué fuentes?", f, "es")
    assert answer_question("hola", f, "es").startswith("Titular.")
    assert "don't have results" in answer_question("hi", {"brands": []}, "en")


def test_more_sectors_and_unlisted_cities():
    d = read_brief("Mercadona frente a Carrefour, Lidl y Dia, supermercados en Valencia")
    assert d.competitors == ["Carrefour", "Lidl", "Dia"]
    assert d.sector == "Supermercados y alimentación"
    d = read_brief("Banco Santander frente a BBVA y CaixaBank, banca en Granada")
    assert d.brand == "Banco Santander" and d.city == "Granada"


def test_beating_a_rival_is_a_who_wins_question():
    f = facts(
        {"estimate": 0.6, "ci_low": 0.4, "ci_high": 0.8},
        {"estimate": 0.5, "ci_low": 0.3, "ci_high": 0.7},
    )
    assert "no puedo decir que gane" in answer_question("¿Le gano a Just Eat?", f, "es")
