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


def stage_line(lang: str, node: str, round_number: int = 1) -> str:
    lines = STAGE_LINES.get(lang, STAGE_LINES["es"])
    return lines.get(node, "").format(round=round_number)


def out_of(count: int, total: int, lang: str) -> str:
    """«28 de 30» / "28 out of 30"."""
    return f"{count} de {total}" if lang == "es" else f"{count} out of {total}"


def percent(value: float | None) -> str:
    return "n/a" if value is None else f"{round(value * 100)}%"


def headline(
    lang: str, brand: str, mention: float | None, recommend: float | None, n_answers: int
) -> str:
    """The one sentence that sums up a run, in the product's voice."""
    named = round((mention or 0) * n_answers)
    chosen = round((recommend or 0) * n_answers)
    gap = (mention or 0) - (recommend or 0)
    if lang == "es":
        if mention is None or mention < 0.2:
            lead = "la IA casi no te nombra"
        elif gap >= 0.25:
            lead = "la IA te conoce, pero no te elige"
        elif (recommend or 0) >= 0.5:
            lead = "la IA te nombra y además te recomienda"
        else:
            lead = "la IA te nombra a menudo y te recomienda a veces"
        return (
            f"Te lo resumo en una frase: {lead}. {brand} sale en {out_of(named, n_answers, lang)} "
            f"respuestas y se recomienda en {chosen}."
        )
    if mention is None or mention < 0.2:
        lead = "the AI hardly mentions you"
    elif gap >= 0.25:
        lead = "the AI knows you, but doesn't pick you"
    elif (recommend or 0) >= 0.5:
        lead = "the AI mentions you and recommends you too"
    else:
        lead = "the AI mentions you often and recommends you sometimes"
    return (
        f"In one sentence: {lead}. {brand} shows up in {out_of(named, n_answers, lang)} answers "
        f"and is recommended in {chosen}."
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
