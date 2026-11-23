"""The chat's own words: turn a sentence into a brief, and answer questions about a run.

Both work without a model (simulated mode and tests): a small rule-based reader for the brief and
answers built from the run's own figures. In live mode the agent model is asked first, with the
product voice (``voice.VOICE_GUIDE``) and only the run's figures as context, and the rule-based
answer is the fallback if the call fails. Figures never come from the model's memory.
"""

import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from ai_share_of_voice.voice import VOICE_GUIDE, join_names, overlaps, percent

# Words that name a sector, checked in order. The first five are the demo's own sectors; the
# rest use the same labels as the sector picker (``ui.catalog.SECTORS``) so the card preselects it.
SECTOR_WORDS = {
    "Comida a domicilio": ("comida", "domicilio", "delivery", "reparto", "cena", "food"),
    "Academias de idiomas": ("idioma", "inglés", "language school", "academia"),
    "Cafeterías": ("café", "cafeter", "coffee"),
    "Gimnasios": ("gimnas", "gym", "fitness"),
    "Coworking": ("coworking",),
    "Supermercados y alimentación": ("supermercado", "supermarket", "grocer", "alimentación"),
    "Restaurantes y bares": ("restaurante", "restaurant", " bares", " bars", "tapas"),
    "Moda y tiendas de ropa": ("moda", "ropa", "fashion", "clothing", "clothes"),
    "Banca y seguros": ("banca", "bancos", "banco ", "seguros", "banking", "bank", "insurance"),
    "Telefonía e internet": ("telefon", "móvil", "fibra", "internet", "telecom", "mobile"),
    "Viajes y hoteles": ("viaje", "hotel", "travel", "vuelo", "flight"),
    "Salud y clínicas": ("clínica", "dentist", "salud", "health", "clinic"),
    "Educación y formación": ("educación", "formación", "universidad", "education", "training"),
    "Tecnología y electrónica": ("tecnolog", "electrónica", "electronics", "tech"),
    "Hogar y reformas": ("hogar", "reforma", "muebles", "home", "furniture", "renovation"),
    "Automoción y movilidad": ("coche", "automoción", "movilidad", "car ", "cars", "mobility"),
}
SECTOR_EN = {
    "Comida a domicilio": "Food delivery",
    "Academias de idiomas": "Language schools",
    "Cafeterías": "Coffee shops",
    "Gimnasios": "Gyms",
    "Coworking": "Coworking",
    "Supermercados y alimentación": "Supermarkets and groceries",
    "Restaurantes y bares": "Restaurants and bars",
    "Moda y tiendas de ropa": "Fashion and clothing stores",
    "Banca y seguros": "Banking and insurance",
    "Telefonía e internet": "Mobile and internet providers",
    "Viajes y hoteles": "Travel and hotels",
    "Salud y clínicas": "Health and clinics",
    "Educación y formación": "Education and training",
    "Tecnología y electrónica": "Technology and electronics",
    "Hogar y reformas": "Home and renovation",
    "Automoción y movilidad": "Cars and mobility",
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
_PLACE = r"[A-ZÁÉÍÓÚÑ][\wáéíóúñ]+"
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


def _clean(name: str) -> str:
    name = re.sub(r"^(?:a|de|la|el|the|si)\s+", "", name.strip(" ,.;:¿?¡!\"'«»"), flags=re.I)
    return name.strip(" ,.;:¿?¡!\"'«»")


_SECTOR_HINTS = tuple(w for words in SECTOR_WORDS.values() for w in words)


def _is_name(part: str) -> bool:
    """A competitor is a short name, not a trailing phrase like «comida a domicilio»."""
    if not part or len(part.split()) > 4 or part in CITIES:
        return False
    # «supermercados» or «comida a domicilio» after the list is the sector, not a rival; a
    # capitalised name such as «Banco Santander» stays a brand even if it contains a sector word.
    lowered = f" {part.lower()} "
    return part[0].isupper() or not any(hint in lowered for hint in _SECTOR_HINTS)


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
            draft.country = (
                "Portugal"
                if city in ("Lisboa", "Oporto")
                else ("Francia" if city in ("París", "Lyon") else "España")
            )
            break
    if not draft.city:
        # A city we don't list: the capitalised words after the last «en» / «in».
        found = re.findall(rf"\b(?:en|in)\s+({_PLACE}(?:\s+{_PLACE})?)", text)
        if found:
            draft.city = found[-1]
    match = re.search(rf"(.+?)\s+{_VERSUS}\s+(.+?){_STOP}", text, flags=re.I)
    if match:
        before, after = match.group(1), match.group(2)
        brand_words = re.findall(
            r"[A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑáéíóúñ&'.-]*(?:\s+[A-ZÁÉÍÓÚÑ][\w&'.-]*)*", before
        )
        if brand_words:
            draft.brand = _clean(brand_words[-1])
        parts = re.split(r",|\s+y\s+|\s+and\s+|\s+o\s+|\s+or\s+", after)
        draft.competitors = [_clean(p) for p in parts if _is_name(_clean(p))][:5]
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
        f"Casi lo tengo. Me falta saber {join_names(asks, lang)}. "
        "Puedes escribirlo o rellenarlo en la tarjeta."
        if es
        else f"Almost there. I still need {join_names(asks, lang)}. "
        "Type it or fill it in on the card."
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
#
# Every answer below is built from the computed figures of the run (``facts``): no model writes
# them, so the chat can never state a number the statistics did not produce. Conclusions about
# small subgroups (one question, a handful of answers) are never drawn.


def _pct(b: dict[str, Any] | None, key: str) -> float:
    return ((b or {}).get(key) or {}).get("estimate") or 0.0


def answer_question(question: str, facts: dict[str, Any], lang: str) -> str:
    es = lang == "es"
    q = question.lower()
    brands = facts.get("brands", [])
    client = next((b for b in brands if b.get("is_client")), None)
    others = [b for b in brands if not b.get("is_client")]
    if client is None:
        return (
            "Todavía no tengo resultados de esta medición." if es else "I don't have results yet."
        )
    stop = facts.get("stop") or {}
    if any(
        w in q for w in ("sigue", "sigues", "para", "paras", "parado", "stop", "continu", "keep")
    ):
        reason = stop.get("reason_text")
        if reason:
            return reason
        return (
            "Sigo mientras el margen de error pueda mejorar lo bastante y quede presupuesto."
            if es
            else "I keep going while the margin can still improve enough and there is budget left."
        )
    wins = (
        "gana",
        "gano",
        "ganamos",
        "ganar",
        "win",
        "beat",
        "ahead",
        "compet",
        "mejor que",
        "más que",
        "better",
        "quién",
        "who",
    )
    if any(w in q for w in wins):
        if not others:
            return "No hay competidores en esta medición." if es else "No competitors in this run."
        ranked = sorted(brands, key=lambda b: _pct(b, "recommend"), reverse=True)
        best, runner = ranked[0], ranked[1]
        clear = overlaps(best.get("recommend"), runner.get("recommend")) is False
        mine = bool(best.get("is_client"))
        rival, rival_pct = runner["name"], percent(_pct(runner, "recommend"))
        if es:
            who = (
                "Tú eres la marca más recomendada"
                if best.get("is_client")
                else f"La más recomendada es {best['name']}"
            )
            verdict = (
                f"y la ventaja sobre {rival} ({rival_pct}) es clara con esta muestra."
                if clear
                else f"pero {'tu' if mine else 'su'} margen se solapa con el de {rival} "
                f"({rival_pct}): "
                "con esta muestra no puedo decir que gane."
            )
            return (
                f"{who}, en el {percent(_pct(best, 'recommend'))} de las respuestas, {verdict} "
                "No saco conclusiones pregunta a pregunta: hay muy pocas respuestas por pregunta."
            )
        who = (
            "You are the most recommended brand"
            if best.get("is_client")
            else f"The most recommended is {best['name']}"
        )
        verdict = (
            f"and the lead over {rival} ({rival_pct}) is clear in this sample."
            if clear
            else f"but {'your' if mine else 'its'} margin overlaps with {rival} ({rival_pct}): "
            "I can't say it wins with this sample."
        )
        return (
            f"{who}, in {percent(_pct(best, 'recommend'))} of answers, {verdict} "
            "I don't draw conclusions question by question: there are too few answers per question."
        )
    if any(w in q for w in ("fuente", "source", "web", "link")):
        n = len((facts.get("research") or {}).get("sources", []))
        if es:
            return (
                f"He leído {n} fuentes. Sirven de contexto: "
                "no demuestran por qué la IA recomienda a nadie."
                if n
                else "En esta medición no se leyeron fuentes externas "
                "(la búsqueda web estaba apagada)."
            )
        return (
            f"I read {n} sources. They are context, not proof of why the AI recommends anyone."
            if n
            else "No external sources were read in this run (web search was off)."
        )
    if any(
        w in q
        for w in (
            "fiable",
            "fiar",
            "fío",
            "confi",
            "margen",
            "trust",
            "reliable",
            "margin",
            "error",
        )
    ):
        m = client.get("mention") or {}
        nq = facts.get("n_questions", 0)
        if es:
            return (
                f"El «te nombra» de {client['name']} está entre {percent(m.get('ci_low'))} y "
                f"{percent(m.get('ci_high'))}. Lo calculo por pregunta: "
                f"con {nq} preguntas tengo {nq} datos independientes, aunque haya más respuestas. "
                "Si dos marcas se solapan, no diría que una gana."
            )
        return (
            f"{client['name']}'s mention rate is between {percent(m.get('ci_low'))} and "
            f"{percent(m.get('ci_high'))}. I compute it per question: "
            f"{nq} questions are {nq} independent "
            "data points, even with more answers. If two brands overlap, I wouldn't say one wins."
        )
    lead = facts.get("headline", "")
    return lead + (
        " Pregúntame por qué paré, quién se recomienda más o cuánto fiarte."
        if es
        else " Ask me why I stopped, who gets recommended most, or how much to trust it."
    )
