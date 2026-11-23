"""The product's voice: how every sentence a person reads should sound (chat, loading, report).

One place for tone, so the chat, the live progress and the report writer speak as the same
person. ``VOICE_GUIDE`` is added to every model prompt that writes text for people; the helpers
below write the fixed sentences in that same voice, without a model call.

The rules, in short: first person, short sentences, plain words, numbers as people say them
("28 de 30"), honest about doubt, never hype, never "as an AI", no emoji.
"""

from collections.abc import Sequence

VOICE_GUIDE = {
    "es": (
        "Escribe como una analista de marketing con experiencia que habla con su cliente. "
        "Primera persona, frases cortas, palabras de todos los días. "
        "Di las cifras como las diría una persona («28 de 30 respuestas»), y redondea. "
        "Si un resultado no es concluyente, dilo sin rodeos. "
        "No exageres, no vendas, no uses emojis ni frases de asistente como «¡Claro!» o "
        "«Como IA…». Nunca llames cuota de mercado a una muestra."
    ),
    "en": (
        "Write like an experienced marketing analyst talking to a client. "
        "First person, short sentences, everyday words. "
        "Say numbers the way people say them ('28 out of 30 answers') and round them. "
        "If a result is not conclusive, say so plainly. "
        "No hype, no selling, no emoji, no assistant phrases like 'Sure!' or 'As an AI…'. "
        "Never call a sample market share."
    ),
}

# What each agent is doing, said the way a colleague would say it while working.
STAGE_LINES = {
    "es": {
        "questions": "Estoy preparando preguntas de cliente que no nombran ninguna marca.",
        "sampling": "Estoy preguntando a la IA, ronda {round}.",
        "metrics": "Estoy contando quién sale y quién se recomienda.",
        "research": "Estoy buscando fuentes para comprobar los nombres que aparecen.",
        "screenshots": "Estoy releyendo nuestros gráficos para que no digan otra cosa.",
        "write_draft": "Estoy escribiendo el borrador del informe.",
        "skeptic": "Le paso el borrador a mi revisor escéptico.",
        "approval_request": "El borrador está listo. Te toca a ti.",
        "finalize": "Publicando el informe aprobado.",
    },
    "en": {
        "questions": "I'm writing customer questions that don't name any brand.",
        "sampling": "I'm asking the AI, round {round}.",
        "metrics": "I'm counting who shows up and who gets recommended.",
        "research": "I'm looking for sources to check the names that came up.",
        "screenshots": "I'm rereading our charts so they don't say something else.",
        "write_draft": "I'm writing the draft report.",
        "skeptic": "I'm handing the draft to my sceptical reviewer.",
        "approval_request": "The draft is ready. Over to you.",
        "finalize": "Publishing the approved report.",
    },
}


# What each agent says once its step is done. The raw log (English, file paths, tool errors)
# stays in the run folder; people read these lines instead.
DONE_LINES = {
    "es": {
        "questions": "Preguntas listas.",
        "sampling": "Ronda hecha.",
        "metrics": "Recuento hecho, con su margen de error.",
        "research": "Fuentes revisadas.",
        "screenshots": "Gráficos revisados.",
        "write_draft": "Borrador escrito.",
        "skeptic": "Revisión hecha.",
        "approval_request": "Esperando tu decisión.",
        "finalize": "Informe y descargas listos.",
    },
    "en": {
        "questions": "Questions ready.",
        "sampling": "Round done.",
        "metrics": "Counted, with its margin of error.",
        "research": "Sources checked.",
        "screenshots": "Charts checked.",
        "write_draft": "Draft written.",
        "skeptic": "Review done.",
        "approval_request": "Waiting for your decision.",
        "finalize": "Report and downloads ready.",
    },
}
FAILED_LINE = {
    "es": "Esto no ha salido bien. Lo tienes en el registro de la medición.",
    "en": "This didn't go well. It's in the run log.",
}


def event_line(lang: str, node: str, status: str, round_number: int = 1) -> str:
    """The sentence a person reads for one agent event, in the product voice."""
    lang = lang if lang in DONE_LINES else "es"
    if status == "failed":
        return FAILED_LINE[lang]
    if status == "started":
        return stage_line(lang, node, round_number)
    return DONE_LINES[lang].get(node, "")


def stage_line(lang: str, node: str, round_number: int = 1) -> str:
    lines = STAGE_LINES.get(lang, STAGE_LINES["es"])
    return lines.get(node, "").format(round=round_number)


def out_of(count: int, total: int, lang: str) -> str:
    """«28 de 30» / "28 out of 30"."""
    return f"{count} de {total}" if lang == "es" else f"{count} out of {total}"


def percent(value: float | None) -> str:
    return "n/a" if value is None else f"{round(value * 100)}%"


def overlaps(a: dict | None, b: dict | None) -> bool | None:
    """True when two intervals overlap (the difference is not conclusive); None if unknown."""
    if (
        not a
        or not b
        or None in (a.get("ci_low"), a.get("ci_high"), b.get("ci_low"), b.get("ci_high"))
    ):
        return None
    return a["ci_low"] <= b["ci_high"] and b["ci_low"] <= a["ci_high"]


def headline(
    lang: str, brand: str, mention: dict | None, recommend: dict | None, n_answers: int
) -> str:
    """The one sentence that sums up a run. ``mention``/``recommend``: estimate and interval.

    The «known but not picked» line is only said when the two intervals do not overlap, so
    the sentence never claims a gap that the sample cannot support.
    """
    m = (mention or {}).get("estimate")
    r = (recommend or {}).get("estimate")
    named = round((m or 0) * n_answers)
    chosen = round((r or 0) * n_answers)
    clear_gap = overlaps(mention, recommend) is False and (m or 0) > (r or 0)
    if lang == "es":
        if m is None or m < 0.2:
            lead = "la IA casi no te nombra"
        elif clear_gap:
            lead = "la IA te conoce, pero no siempre te elige"
        else:
            lead = "la IA te nombra y te recomienda en proporciones parecidas"
        return (
            f"{lead[0].upper()}{lead[1:]}. {brand} sale en {out_of(named, n_answers, lang)} "
            f"respuestas y se recomienda en {chosen}."
        )
    if m is None or m < 0.2:
        lead = "the AI hardly mentions you"
    elif clear_gap:
        lead = "the AI knows you, but doesn't always pick you"
    else:
        lead = "the AI mentions and recommends you at similar rates"
    return (
        f"{lead[0].upper()}{lead[1:]}. {brand} shows up in {out_of(named, n_answers, lang)} "
        f"answers and is recommended in {chosen}."
    )


def objections_intro(lang: str, count: int) -> str:
    if lang == "es":
        if count == 0:
            return "Mi revisor escéptico no ha encontrado nada que objetar. Aun así, léelo antes."
        if count == 1:
            return "Antes de que lo apruebes, mi revisor escéptico ha encontrado una cosa:"
        return f"Antes de que lo apruebes, mi revisor escéptico ha encontrado {count} cosas:"
    if count == 0:
        return "My sceptical reviewer found nothing to object to. Read it anyway."
    if count == 1:
        return "Before you approve it, my sceptical reviewer found one thing:"
    return f"Before you approve it, my sceptical reviewer found {count} things:"


def join_names(names: Sequence[str], lang: str) -> str:
    names = list(names)
    if len(names) <= 1:
        return "".join(names)
    word = " y " if lang == "es" else " and "
    return ", ".join(names[:-1]) + word + names[-1]
