"""The product voice: fixed sentences never claim more than the intervals allow."""

from ai_share_of_voice.voice import event_line, headline, join_names, objections_intro, overlaps


def iv(estimate, low, high):
    return {"estimate": estimate, "ci_low": low, "ci_high": high}


def test_overlap_is_unknown_without_intervals():
    assert overlaps(None, iv(0.5, 0.4, 0.6)) is None
    assert overlaps(iv(0.5, None, 0.6), iv(0.5, 0.4, 0.6)) is None


def test_overlap_true_and_false():
    assert overlaps(iv(0.5, 0.4, 0.6), iv(0.55, 0.5, 0.7)) is True
    assert overlaps(iv(0.8, 0.7, 0.9), iv(0.3, 0.2, 0.4)) is False


def test_headline_claims_a_gap_only_when_intervals_do_not_overlap():
    clear = headline("es", "Glovo", iv(0.8, 0.7, 0.9), iv(0.3, 0.2, 0.4), 30)
    assert "te conoce, pero no siempre te elige" in clear
    assert "24 de 30" in clear
    unclear = headline("es", "Glovo", iv(0.7, 0.4, 1.0), iv(0.5, 0.3, 0.8), 30)
    assert "no siempre te elige" not in unclear
    assert "proporciones parecidas" in unclear


def test_headline_low_mention():
    assert "casi no te nombra" in headline("es", "X", iv(0.1, 0.0, 0.2), iv(0.0, 0.0, 0.1), 10)
    assert "hardly mentions you" in headline("en", "X", None, None, 10)


def test_event_lines_never_leak_raw_logs():
    assert event_line("es", "finalize", "succeeded") == "Informe y descargas listos."
    assert event_line("en", "sampling", "started", 2) == "I'm asking the AI, round 2."
    assert "registro" in event_line("es", "metrics", "failed")
    assert event_line("xx", "questions", "succeeded") == "Preguntas listas."


def test_small_helpers():
    assert join_names(["a", "b", "c"], "es") == "a, b y c"
    assert join_names(["a"], "en") == "a"
    assert "una cosa" in objections_intro("es", 1)
    assert "3 things" in objections_intro("en", 3)
