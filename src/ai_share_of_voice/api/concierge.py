"""The chat's own words: turn a sentence into a brief, and answer questions about a run.

Both work without a model (simulated mode and tests): a small rule-based reader for the brief and
answers built from the run's own figures. In live mode the agent model is asked first, with the
product voice (``voice.VOICE_GUIDE``) and only the run's figures as context, and the rule-based
answer is the fallback if the call fails. Figures never come from the model's memory.
"""

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from ai_share_of_voice.voice import VOICE_GUIDE, join_names, percent

SECTOR_WORDS = {
    "Comida a domicilio": ("comida", "domicilio", "delivery", "reparto", "cena", "food"),
    "Academias de idiomas": ("idioma", "inglés", "language school", "academia"),
    "Cafeterías": ("café", "cafeter", "coffee"),
    "Gimnasios": ("gimnas", "gym", "fitness"),
    "Coworking": ("coworking",),
}
SECTOR_EN = {
    "Comida a domicilio": "Food delivery",
    "Academias de idiomas": "Language schools",
    "Cafeterías": "Coffee shops",
    "Gimnasios": "Gyms",
    "Coworking": "Coworking",
}
CITIES = (
    "Madrid",
    "Barcelona",
    "Valencia",
    "Sevilla",
    "Bilbao",
    "Málaga",
    "Zaragoza",
    "Lisboa",
    "Oporto",
    "París",
    "Lyon",
)
_VERSUS = r"(?:frente a|contra|vs\.?|versus|compared (?:to|with)|against)"
_STOP = r"(?:\s+(?:en|in|para|for|de|del|como)\s+|[.?!]|$)"


class BriefDraft(BaseModel):
    """What the chat understood. Empty fields are asked for in the reply."""

    brand: str = ""
    competitors: list[str] = Field(default_factory=list)
    sector: str = ""
    city: str = ""
    country: str = ""
    language: Literal["es", "en"] = "es"
    reply: str = ""
    missing: list[str] = Field(default_factory=list)


class ChatAnswer(BaseModel):
    reply: str


def _clean(name: str) -> str:
    name = re.sub(r"^(?:a|de|la|el|the|si)\s+", "", name.strip(" ,.;:¿?¡!\"'«»"), flags=re.I)
    return name.strip(" ,.;:¿?¡!\"'«»")


def read_brief(text: str, lang: str = "es") -> BriefDraft:
    """A rule-based reading of «Glovo frente a Just Eat y Uber Eats en Madrid»."""
    draft = BriefDraft(language="en" if lang == "en" else "es")
    lowered = text.lower()
    for sector, words in SECTOR_WORDS.items():
        if any(word in lowered for word in words):
            draft.sector = sector if draft.language == "es" else SECTOR_EN[sector]
            break
    for city in CITIES:
        if city.lower() in lowered or city.lower().replace("á", "a") in lowered:
            draft.city = city
            draft.country = "Portugal" if city in ("Lisboa", "Oporto") else (
                "Francia" if city in ("París", "Lyon") else "España"
            )
            break
    match = re.search(rf"(.+?)\s+{_VERSUS}\s+(.+?){_STOP}", text, flags=re.I)
    if match:
        before, after = match.group(1), match.group(2)
        brand_words = re.findall(r"[A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑáéíóúñ&'.-]*(?:\s+[A-ZÁÉÍÓÚÑ][\w&'.-]*)*", before)
        if brand_words:
            draft.brand = _clean(brand_words[-1])
        parts = re.split(r",|\s+y\s+|\s+and\s+|\s+o\s+|\s+or\s+", after)
        draft.competitors = [_clean(p) for p in parts if _clean(p)][:5]
    if not draft.brand:
        capital = re.findall(r"\b[A-ZÁÉÍÓÚÑ][\w&'.-]+(?:\s+[A-ZÁÉÍÓÚÑ][\w&'.-]+)*", text)
        names = [c for c in capital if c not in CITIES and c.lower() not in ("quiero", "i", "la")]
        if names:
            draft.brand = names[0]
            draft.competitors = draft.competitors or names[1:6]
    draft.missing = [
        label
        for label, value in (
            ("brand", draft.brand),
            ("competitors", len(draft.competitors) >= 2),
            ("sector", draft.sector),
            ("city", draft.city),
        )
        if not value
    ]
    draft.reply = brief_reply(draft)
    return draft


def brief_reply(draft: BriefDraft) -> str:
    es = draft.language == "es"
    if not draft.missing:
        return (
            "Entendido. Esto es lo que voy a medir. Toca cualquier dato si quieres cambiarlo."
            if es
            else "Got it. Here's what I'll measure. Tap anything you want to change."
        )
    asks_es = {
        "brand": "qué marca es la tuya",
        "competitors": "al menos dos competidores",
        "sector": "el sector",
        "city": "la ciudad",
    }
    asks_en = {
        "brand": "which brand is yours",
        "competitors": "at least two competitors",
        "sector": "the sector",
        "city": "the city",
    }
    asks = [(asks_es if es else asks_en)[item] for item in draft.missing]
    lang = "es" if es else "en"
    return (
        f"Casi lo tengo. Me falta saber {join_names(asks, lang)}. Puedes escribirlo o rellenarlo en la tarjeta."
        if es
        else f"Almost there. I still need {join_names(asks, lang)}. Type it or fill it in on the card."
    )


def llm_brief(agent: Any, text: str, lang: str) -> BriefDraft:
    """Ask the agent model to read the sentence; the rule-based reader fills any gap."""
    prompt = (
        f"{VOICE_GUIDE[lang]}\n\nRead this request from a business owner and extract the brief. "
        "brand = their own brand; competitors = 2 to 5 brands to compare with; sector and city "
        "as they would say them; country in the same language. Leave a field empty if it is not "
        "said. reply = one or two sentences to them in their language, in the voice above.\n\n"
        f"Request: {text}"
    )
    draft, _ = agent.complete(prompt, BriefDraft)
    fallback = read_brief(text, lang)
    for name in ("brand", "sector", "city", "country"):
        if not getattr(draft, name):
            setattr(draft, name, getattr(fallback, name))
    if len(draft.competitors) < 2:
        draft.competitors = fallback.competitors
    draft.language = "en" if lang == "en" else "es"
    draft.missing = [
        label
        for label, ok in (
            ("brand", draft.brand),
            ("competitors", len(draft.competitors) >= 2),
            ("sector", draft.sector),
            ("city", draft.city),
        )
        if not ok
    ]
    if draft.missing or not draft.reply:
        draft.reply = brief_reply(draft)
    return draft


# --- questions about a finished run -------------------------------------------------------------


def answer_question(question: str, facts: dict[str, Any], lang: str) -> str:
    """Answer from the run's figures only (no model)."""
    es = lang == "es"
    q = question.lower()
    brands = facts.get("brands", [])
    client = next((b for b in brands if b.get("is_client")), None)
    others = [b for b in brands if not b.get("is_client")]
    rec = lambda b: (b.get("recommend") or {}).get("estimate") or 0  # noqa: E731
    men = lambda b: (b.get("mention") or {}).get("estimate") or 0  # noqa: E731
    if client is None:
        return "Todavía no tengo resultados de esta medición." if es else "I don't have results yet."
    if any(w in q for w in ("gana", "win", "compet", "mejor que", "better")):
        best = max(others, key=rec, default=None)
        if best is None:
            return "No hay competidores en esta medición." if es else "No competitors here."
        best_q = facts.get("best_questions", {}).get(best["name"])
        tail = (
            f" Donde más se nota es en «{best_q}»." if es and best_q else
            (f" It shows most in “{best_q}”." if best_q else "")
        )
        if es:
            return (
                f"Quien más se recomienda es {best['name']}: {percent(rec(best))} de las respuestas, "
                f"frente a tu {percent(rec(client))}.{tail} Ábrelo en la pestaña Respuestas para leerlas."
            )
        return (
            f"The most recommended is {best['name']}: {percent(rec(best))} of answers, "
            f"against your {percent(rec(client))}.{tail} Open the Answers tab to read them."
        )
    if any(w in q for w in ("fuente", "source", "web", "link")):
        n = len((facts.get("research") or {}).get("sources", []))
        if es:
            return (
                f"He consultado {n} fuentes. Ojo: sirven de contexto, no demuestran por qué la IA recomienda a nadie."
                if n
                else "En esta medición no se consultaron fuentes externas (modo sin búsqueda web)."
            )
        return (
            f"I checked {n} sources. They are context, not proof of why the AI recommends anyone."
            if n
            else "No external sources were checked in this run (web search was off)."
        )
    if any(w in q for w in ("fiable", "confi", "margen", "trust", "reliable", "margin", "error")):
        m = client.get("mention") or {}
        lo, hi = m.get("ci_low"), m.get("ci_high")
        n = facts.get("n_answers", 0)
        if es:
            return (
                f"Con {n} respuestas, el «te nombra» de {client['name']} está entre {percent(lo)} y {percent(hi)}. "
                "Si dos marcas se solapan en ese margen, no te diría que una gana a la otra."
            )
        return (
            f"With {n} answers, {client['name']}'s mention rate is between {percent(lo)} and {percent(hi)}. "
            "If two brands overlap in that range, I wouldn't say one beats the other."
        )
    gap = men(client) - rec(client)
    if es:
        return (
            f"{client['name']} sale en el {percent(men(client))} de las respuestas y se recomienda en el "
            f"{percent(rec(client))}. "
            + ("La distancia entre salir y ser elegido es grande: ahí está el trabajo." if gap >= 0.25 else
               "Salir y ser elegido van bastante parejos.")
            + " Pregúntame quién gana, de dónde salen los datos o cuánto fiarte."
        )
    return (
        f"{client['name']} shows up in {percent(men(client))} of answers and is recommended in "
        f"{percent(rec(client))}. "
        + ("The gap between showing up and being picked is wide: that's where the work is." if gap >= 0.25 else
           "Showing up and being picked are close.")
        + " Ask me who wins, where the data comes from, or how much to trust it."
    )


def llm_answer(agent: Any, question: str, facts: dict[str, Any], lang: str) -> str:
    prompt = (
        f"{VOICE_GUIDE[lang]}\n\nAnswer the client's question using ONLY these measured facts "
        "(JSON). If the facts do not answer it, say what would. Two to four sentences.\n\n"
        f"Facts: {json.dumps(facts, ensure_ascii=False)[:6000]}\n\nQuestion: {question}"
    )
    answer, _ = agent.complete(prompt, ChatAnswer)
    return answer.reply
