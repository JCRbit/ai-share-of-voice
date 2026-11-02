"""Interface texts in Spanish (default) and English, written for non-technical users.

Gradio's own i18n follows the browser locale and cannot be switched at runtime, so the app
keeps its texts here and redraws them when the user changes the language. Every key exists
in both languages (a test checks it). Code, logs and the audit trail stay in English; only
what the user reads is translated, in plain words: "margin of error" rather than
"confidence interval", "how often the AI names you" rather than "mention rate".
"""

import re
from typing import Literal

Lang = Literal["es", "en"]
DEFAULT_LANG: Lang = "es"

TEXTS: dict[str, dict[str, str]] = {
    "es": {
        "app_title": "AI Share of Voice",
        "app_subtitle": "Tu visibilidad en las respuestas de la IA",
        "hero_title": "¿Te recomienda la inteligencia artificial?",
        "hero_text": "Cada vez más clientes preguntan a ChatGPT, Gemini o similares dónde "
        "comprar o qué servicio elegir. Descubre si la IA te nombra, si te recomienda y a "
        "quién recomienda en tu lugar.",
        "how_title": "Cómo funciona",
        "how_1_title": "Preguntamos como un cliente",
        "how_1_text": "Hacemos a la IA las preguntas que haría alguien que busca lo que vendes, "
        "sin nombrar ninguna marca.",
        "how_2_title": "Contamos y comprobamos",
        "how_2_text": "Medimos cuántas veces te nombra y te recomienda frente a tu competencia, "
        "con su margen de error.",
        "how_3_title": "Tú decides",
        "how_3_text": "Un revisor automático señala lo dudoso y nada se publica sin tu aprobación.",
        "mode_simulated": "Modo demostración · sin coste",
        "mode_live": "Modo real",
        "tech_details": "Detalles técnicos",
        "measured_model": "IA que se mide",
        "agent_model": "IA que hace el análisis",
        "search": "Buscador de fuentes",
        "cap": "Gasto máximo por medición",
        "language": "Idioma",
        "tab_brief": "1 · Tu marca",
        "tab_activity": "2 · Progreso",
        "tab_results": "3 · Resultados",
        "section_market": "¿Qué quieres medir?",
        "brand": "Tu marca",
        "brand_placeholder": "Por ejemplo: Glovo",
        "competitors": "Tus competidores",
        "competitors_info": "Entre 2 y 5, separados por comas",
        "sector": "Tu sector",
        "sector_info": "Cómo lo diría un cliente: «comida a domicilio», «tiendas de ropa»…",
        "city": "Ciudad",
        "country": "País",
        "advanced": "Opciones avanzadas",
        "question_language": "Idioma de las preguntas",
        "panel": "Preguntas",
        "panel_demo": "Usar las preguntas de ejemplo (comida a domicilio en Madrid)",
        "panel_generate": "Crear preguntas para mi sector",
        "questions": "Número de preguntas",
        "batches": "Rondas máximas",
        "advanced_help": "Más preguntas y rondas dan un resultado más preciso, pero tardan más. "
        "La medición se detiene sola cuando el resultado ya es suficientemente preciso.",
        "estimate": "Haremos hasta **{answers} preguntas** a la IA · nunca gastará más de "
        "**{cap:.2f} USD**",
        "start": "Medir mi marca",
        "started": "¡En marcha! Sigue el progreso en la pestaña **2 · Progreso**.",
        "form_error": "**Revisa los datos:** {error}",
        "no_run": "Todavía no has lanzado ninguna medición. Empieza en **1 · Tu marca**.",
        "spend": "Gasto",
        "spend_of": "de un máximo de {cap:.2f} USD",
        "log": "Registro técnico completo",
        "log_time": "Hora",
        "log_status": "Estado",
        "log_step": "Paso",
        "log_detail": "Detalle",
        "approval_title": "Revisa y aprueba el informe",
        "approval_help": "Nada se publica hasta que tú lo decidas.",
        "draft": "Borrador del informe",
        "objections": "El revisor automático tiene dudas sobre:",
        "no_objections": "El revisor automático no ha encontrado problemas.",
        "comment": "Comentario (opcional)",
        "approve": "Aprobar y publicar",
        "reject": "Rechazar",
        "decision_ok": "Decisión guardada.",
        "decision_error": "No se pudo guardar la decisión: {error}",
        "load_results": "Actualizar",
        "results_empty": "Aquí verás los resultados cuando termine tu medición.",
        "results_pending": "Los resultados aún no están listos.",
        "kpi_mention": "Te nombra",
        "kpi_mention_help": "de las respuestas",
        "kpi_recommend": "Te recomienda",
        "kpi_answers": "Respuestas analizadas",
        "kpi_spend": "Coste de la medición",
        "margin": "Margen de error: entre {low} y {high}",
        "plain_sentence": "De cada 10 respuestas de la IA, **{brand} aparece en {of_ten}**.",
        "plain_leader": "La marca más nombrada es **{leader}** ({rate}).",
        "plain_you_lead": "**Eres la marca más nombrada** de tu comparativa.",
        "sample_label": "Resultado de una muestra del {date}: {answers} respuestas sobre "
        "{sector} en {city}. Es una foto de lo que dice la IA, no una cuota de mercado.",
        "by_brand": "Tú frente a tu competencia",
        "brand_col": "Marca",
        "mention_col": "Te nombra",
        "range_col": "Margen de error",
        "charts": "Gráficos",
        "discovered_title": "Otros negocios que la IA recomienda",
        "discovered_help": "La IA también nombra a estos negocios, que no estás vigilando. Los "
        "marcados con ★ aparecen a menudo y en preguntas distintas: quizá deberías "
        "incluirlos como competidores.",
        "business_col": "Negocio",
        "kind_col": "Tipo",
        "rate_col": "Aparece en",
        "questions_col": "En cuántas preguntas",
        "in_questions": "en {n} preguntas",
        "suggested_col": "",
        "sources_title": "De dónde sale la información",
        "no_sources": "No se consultaron fuentes externas en esta medición.",
        "checks_title": "¿Existen de verdad los negocios más nombrados?",
        "evidence": "encontrado en internet",
        "no_evidence": "no lo hemos encontrado (puede existir igualmente)",
        "downloads": "Descarga tu informe",
        "running": "Midiendo…",
        "waiting_approval": "Esperando tu aprobación",
        "completed": "¡Listo! Informe publicado",
        "partial": "La medición se detuvo antes de tiempo",
        "rejected": "Rechazado: no se ha publicado nada",
        "error": "Algo ha fallado",
        "notes_title": "Ten en cuenta",
        "chart_rates_title": "¿Qué marcas nombra y recomienda la IA?",
        "chart_named": "La nombra",
        "chart_recommended": "La recomienda",
        "chart_axis": "% de respuestas de la IA",
        "chart_rates_note": "Las líneas finas sobre cada barra son el margen de error.",
        "chart_rates_caption": "**Cómo leerlo:** la barra oscura es cuántas veces la IA nombra "
        "cada marca; la clara, cuántas veces la recomienda. Que te nombre no basta: lo valioso "
        "es que te recomiende. Tu marca va en color.",
        "chart_precision_title": "¿Es fiable el resultado?",
        "chart_rounds": "Ronda de preguntas",
        "chart_margin": "Margen de error",
        "chart_target": "Objetivo de precisión",
        "chart_precision_note": "Cada ronda repite las preguntas; el margen se estrecha al "
        "acumular respuestas.",
        "chart_precision_caption": "**Cómo leerlo:** cada punto es el margen de error tras una "
        "ronda de preguntas. Cuanto más baja la línea, más fiable el resultado. La medición se "
        "para sola al llegar al objetivo, o antes si el margen deja de mejorar o se acaba el "
        "presupuesto.",
        "sector_other": "Escribe tu sector",
        "need_city": "elige país, región y ciudad",
        "need_sector": "escribe tu sector o elige uno de la lista",
        "region": "Región",
        "region_info": "Comunidad autónoma en España; región en Portugal y Francia",
    },
    "en": {
        "app_title": "AI Share of Voice",
        "app_subtitle": "Your visibility in AI answers",
        "hero_title": "Does artificial intelligence recommend you?",
        "hero_text": "More and more customers ask ChatGPT, Gemini and similar tools where to "
        "buy or which service to choose. Find out whether the AI names you, recommends you "
        "and who it recommends instead.",
        "how_title": "How it works",
        "how_1_title": "We ask like a customer",
        "how_1_text": "We ask the AI the questions someone looking for what you sell would ask, "
        "without naming any brand.",
        "how_2_title": "We count and check",
        "how_2_text": "We measure how often it names and recommends you against your "
        "competitors, with its margin of error.",
        "how_3_title": "You decide",
        "how_3_text": "An automatic reviewer flags anything doubtful and nothing is published "
        "without your approval.",
        "mode_simulated": "Demo mode · no cost",
        "mode_live": "Live mode",
        "tech_details": "Technical details",
        "measured_model": "AI being measured",
        "agent_model": "AI doing the analysis",
        "search": "Source search",
        "cap": "Maximum spend per measurement",
        "language": "Language",
        "tab_brief": "1 · Your brand",
        "tab_activity": "2 · Progress",
        "tab_results": "3 · Results",
        "section_market": "What do you want to measure?",
        "brand": "Your brand",
        "brand_placeholder": "For example: Glovo",
        "competitors": "Your competitors",
        "competitors_info": "Two to five, separated by commas",
        "sector": "Your sector",
        "sector_info": 'As a customer would say it: "food delivery", "clothing stores"…',
        "city": "City",
        "country": "Country",
        "advanced": "Advanced options",
        "question_language": "Language of the questions",
        "panel": "Questions",
        "panel_demo": "Use the sample questions (food delivery in Madrid)",
        "panel_generate": "Create questions for my sector",
        "questions": "Number of questions",
        "batches": "Maximum rounds",
        "advanced_help": "More questions and rounds give a more precise result but take longer. "
        "The measurement stops on its own once the result is precise enough.",
        "estimate": "We will ask the AI up to **{answers} questions** · it will never spend "
        "more than **{cap:.2f} USD**",
        "start": "Measure my brand",
        "started": "Started! Follow the progress in the **2 · Progress** tab.",
        "form_error": "**Check the details:** {error}",
        "no_run": "You have not started a measurement yet. Begin in **1 · Your brand**.",
        "spend": "Spend",
        "spend_of": "of at most {cap:.2f} USD",
        "log": "Full technical log",
        "log_time": "Time",
        "log_status": "Status",
        "log_step": "Step",
        "log_detail": "Detail",
        "approval_title": "Review and approve the report",
        "approval_help": "Nothing is published until you decide.",
        "draft": "Report draft",
        "objections": "The automatic reviewer has doubts about:",
        "no_objections": "The automatic reviewer found no problems.",
        "comment": "Comment (optional)",
        "approve": "Approve and publish",
        "reject": "Reject",
        "decision_ok": "Decision saved.",
        "decision_error": "Cannot save the decision: {error}",
        "load_results": "Refresh",
        "results_empty": "Your results will appear here when the measurement finishes.",
        "results_pending": "The results are not ready yet.",
        "kpi_mention": "Names you",
        "kpi_mention_help": "of the answers",
        "kpi_recommend": "Recommends you",
        "kpi_answers": "Answers analyzed",
        "kpi_spend": "Cost of the measurement",
        "margin": "Margin of error: between {low} and {high}",
        "plain_sentence": "Out of every 10 AI answers, **{brand} appears in {of_ten}**.",
        "plain_leader": "The most named brand is **{leader}** ({rate}).",
        "plain_you_lead": "**You are the most named brand** in your comparison.",
        "sample_label": "Result of a sample taken on {date}: {answers} answers about {sector} "
        "in {city}. It is a snapshot of what the AI says, not a market share.",
        "by_brand": "You against your competitors",
        "brand_col": "Brand",
        "mention_col": "Names it",
        "range_col": "Margin of error",
        "charts": "Charts",
        "discovered_title": "Other businesses the AI recommends",
        "discovered_help": "The AI also names these businesses, which you are not tracking. The "
        "ones marked ★ appear often and in different questions: you may want to add them "
        "as competitors.",
        "business_col": "Business",
        "kind_col": "Kind",
        "rate_col": "Appears in",
        "questions_col": "In how many questions",
        "in_questions": "in {n} questions",
        "suggested_col": "",
        "sources_title": "Where the information comes from",
        "no_sources": "No external sources were consulted in this measurement.",
        "checks_title": "Do the most named businesses really exist?",
        "evidence": "found online",
        "no_evidence": "not found (it may still exist)",
        "downloads": "Download your report",
        "running": "Measuring…",
        "waiting_approval": "Waiting for your approval",
        "completed": "Done! Report published",
        "partial": "The measurement stopped early",
        "rejected": "Rejected: nothing was published",
        "error": "Something went wrong",
        "notes_title": "Keep in mind",
        "chart_rates_title": "Which brands does the AI name and recommend?",
        "chart_named": "Named",
        "chart_recommended": "Recommended",
        "chart_axis": "% of AI answers",
        "chart_rates_note": "The thin lines on each bar are the margin of error.",
        "chart_rates_caption": "**How to read it:** the dark bar is how often the AI names each "
        "brand; the light one, how often it recommends it. Being named is not enough: what "
        "matters is being recommended. Your brand is in colour.",
        "chart_precision_title": "How reliable is the result?",
        "chart_rounds": "Round of questions",
        "chart_margin": "Margin of error",
        "chart_target": "Precision target",
        "chart_precision_note": "Each round repeats the questions; the margin narrows as answers "
        "add up.",
        "chart_precision_caption": "**How to read it:** each point is the margin of error after a "
        "round of questions. The lower the line, the more reliable the result. The measurement "
        "stops on its own at the target, or earlier if the margin stops improving or the budget "
        "runs out.",
        "sector_other": "Type your sector",
        "need_city": "choose a country, region and city",
        "need_sector": "type your sector or choose one from the list",
        "region": "Region",
        "region_info": "Autonomous community in Spain; region in Portugal and France",
    },
}

STAGES: dict[str, dict[str, str]] = {
    "es": {
        "questions": "Preparar preguntas",
        "sampling": "Preguntar a la IA",
        "metrics": "Calcular resultados",
        "research": "Buscar fuentes",
        "screenshots": "Revisar gráficos",
        "write_draft": "Redactar informe",
        "skeptic": "Revisión crítica",
        "approval_gate": "Tu aprobación",
        "finalize": "Publicar informe",
    },
    "en": {
        "questions": "Prepare questions",
        "sampling": "Ask the AI",
        "metrics": "Compute results",
        "research": "Find sources",
        "screenshots": "Review charts",
        "write_draft": "Write report",
        "skeptic": "Critical review",
        "approval_gate": "Your approval",
        "finalize": "Publish report",
    },
}

EVENT_STATUS: dict[str, dict[str, str]] = {
    "es": {
        "started": "iniciado",
        "succeeded": "completado",
        "failed": "fallido",
        "waiting_approval": "esperando aprobación",
        "approved": "aprobado",
        "rejected": "rechazado",
    },
    "en": {
        "started": "started",
        "succeeded": "succeeded",
        "failed": "failed",
        "waiting_approval": "waiting for approval",
        "approved": "approved",
        "rejected": "rejected",
    },
}

KINDS: dict[str, dict[str, str]] = {
    "es": {
        "app_or_platform": "app o plataforma",
        "store_or_restaurant": "tienda o local",
        "product_or_plan": "producto o plan",
        "website": "web",
        "other": "otro",
    },
    "en": {
        "app_or_platform": "app or platform",
        "store_or_restaurant": "store or venue",
        "product_or_plan": "product or plan",
        "website": "website",
        "other": "other",
    },
}

# Notes written by ui/results_tab.py, recognised by pattern and translated to plain words.
_NOTES: list[tuple[re.Pattern[str], dict[str, str]]] = [
    (
        re.compile(r"Few questions \((\d+)\)"),
        {
            "es": "Solo se usaron {0} preguntas: toma el resultado como orientativo. Con 10 o "
            "más preguntas el margen de error es fiable.",
            "en": "Only {0} questions were used: take the result as indicative. With 10 or more "
            "questions the margin of error is reliable.",
        },
    ),
    (
        re.compile(r"Downloads appear after the report is approved"),
        {
            "es": "Podrás descargar el informe cuando lo apruebes.",
            "en": "You can download the report once you approve it.",
        },
    ),
    (
        re.compile(r"No analyzed answers for this run yet"),
        {
            "es": "Aún no hay respuestas analizadas.",
            "en": "There are no analyzed answers yet.",
        },
    ),
    (
        re.compile(r"(.+) is not in the per-brand table"),
        {
            "es": "No encontramos «{0}» en la comparativa: revisa cómo escribiste la marca.",
            "en": 'We could not find "{0}" in the comparison: check how the brand is written.',
        },
    ),
    (
        re.compile(r"The headline and the table differ for the client brand \(([^)]*)\)"),
        {
            "es": "La cifra principal y la tabla no coinciden ({0}). Es normal en el modo "
            "demostración.",
            "en": "The headline figure and the table differ ({0}). This is normal in demo mode.",
        },
    ),
]


def t(lang: str, key: str, **values: object) -> str:
    """The text of ``key`` in ``lang`` (Spanish when the language is unknown)."""
    texts = TEXTS.get(lang, TEXTS[DEFAULT_LANG])
    return texts[key].format(**values) if values else texts[key]


def stage(lang: str, node: str) -> str:
    return STAGES.get(lang, STAGES[DEFAULT_LANG]).get(node, node)


def event_status(lang: str, status: str) -> str:
    return EVENT_STATUS.get(lang, EVENT_STATUS[DEFAULT_LANG]).get(status, status)


def kind(lang: str, value: str) -> str:
    return KINDS.get(lang, KINDS[DEFAULT_LANG]).get(value, value)


def note(lang: str, text: str) -> str:
    """Translate a results note into plain words; unknown notes are shown as they are."""
    for pattern, versions in _NOTES:
        match = pattern.search(text)
        if match:
            return versions.get(lang, versions[DEFAULT_LANG]).format(*match.groups())
    return text
