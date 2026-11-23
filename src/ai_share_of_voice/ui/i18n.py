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
        "access_denied": "**Contraseña incorrecta.** Pídela a quien te dio el enlace.",
        "access_locked": "**Demasiados intentos.** Espera un minuto y vuelve a probar.",
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
        "sample_label": "Resultado de una muestra del {date} con la IA {model}: {answers} "
        "respuestas sobre {sector} en {city}. Es una foto de lo que dice esa IA, no una cuota "
        "de mercado.",
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
        "evidence": "encontrado en internet en esta zona",
        "no_evidence": "no lo hemos encontrado (puede existir igualmente)",
        "downloads": "Descarga tu informe",
        "running": "Midiendo…",
        "waiting_approval": "Esperando tu aprobación",
        "completed": "¡Listo! Informe publicado",
        "partial": "La medición se detuvo antes de tiempo",
        "rejected": "Rechazado: no se ha publicado nada",
        "error": "Algo ha fallado",
        "notes_title": "Ten en cuenta",
        "round_progress": "Ronda {done} de {total}",
        "time_live": "Tardará hasta unos {minutes} min (se para antes si ya es preciso).",
        "time_demo": "En modo demostración tarda unos segundos.",
        "fail_quota": "El proveedor de IA ha agotado su cuota gratuita por hoy. Vuelve a "
        "intentarlo "
        "más tarde o usa el modo demostración.",
        "fail_budget": "Se alcanzó el gasto máximo antes de terminar. Usa menos preguntas o "
        "sube el "
        "gasto máximo.",
        "fail_sources": "No se pudo buscar fuentes en internet. Inténtalo de nuevo más tarde.",
        "fail_no_evidence": "No hubo respuestas suficientes para calcular resultados.",
        "fail_unexpected": "Algo falló durante la medición. Los detalles están en el registro "
        "técnico.",
        "actions_title": "Qué puedes hacer ahora",
        "act_recommend_gap": "La IA te nombra en el {m} de las respuestas, pero solo te "
        "recomienda en "
        "el {r}. Lee las respuestas reales para ver qué dice de ti.",
        "act_leader": "{leader} aparece más que tú ({rate}). Mira en «Una respuesta real de la IA» "
        "en qué preguntas gana.",
        "act_you_lead": "Eres la marca más nombrada. Repite la medición cada mes para comprobar "
        "que "
        "se mantiene.",
        "act_suggested": "La IA también recomienda {names}, que no estás siguiendo: añádelos como "
        "competidores en la próxima medición.",
        "act_few": "Usa 10 o más preguntas para que el resultado sea fiable.",
        "act_wide": "El margen de error es amplio (±{hw}): más preguntas o rondas lo estrecharán.",
        "glossary_title": "¿Qué significa cada cosa?",
        "g_named": "Te nombra: en qué parte de las respuestas aparece tu marca, aunque sea "
        "de pasada.",
        "g_recommended": "Te recomienda: en qué parte la IA sugiere tu marca como buena opción. "
        "Es lo "
        "que más importa.",
        "g_margin": "Margen de error: el rango en el que probablemente está el valor real. Con más "
        "preguntas y rondas se estrecha.",
        "g_sample": "Muestra fechada: una foto de lo que respondió una IA concreta en una "
        "fecha. Otro "
        "día u otra IA pueden responder distinto.",
        "history_title": "Mis mediciones",
        "history_pick": "Abrir una medición anterior",
        "history_open": "Abrir",
        "history_empty": "Aún no hay mediciones aprobadas.",
        "chart_trend_title": "¿Cómo evoluciona {brand} con el tiempo?",
        "chart_trend_axis": "% de respuestas que la nombran",
        "chart_trend_note": "La banda es el margen de error; la letra, el conjunto de "
        "preguntas usado.",
        "trend_caption": "**Cómo leerlo:** cada punto es una medición de {brand} en {city} con la "
        "misma IA ({model}); la banda es su margen de error. Solo se comparan mediciones con la "
        "misma marca, sector, ciudad e IA. La letra indica qué conjunto de preguntas se usó.",
        "trend_set_tag": "Preguntas {letter}",
        "trend_n_questions": "{n} preguntas",
        "trend_set_changed_same": "Se mantiene en el {after}, pero las preguntas cambiaron "
        "(Preguntas {before_set} → {after_set}), así que no es una comparación exacta. Para "
        "comparar, repite la medición con las mismas preguntas.",
        "trend_set_changed": "Entre estas dos mediciones las preguntas cambiaron (Preguntas "
        "{before_set} → {after_set}): el paso de {before} a {after} puede venir de las preguntas y "
        "no de tu marca. Para comparar, repite la medición con las mismas preguntas.",
        "trend_table_date": "Medición",
        "trend_table_set": "Preguntas usadas",
        "trend_table_named": "Te nombra",
        "trend_table_margin": "Margen de error",
        "trend_set_summary": "Preguntas {letter} · {n} preguntas",
        "trend_set_unknown": "Sin registro de preguntas",
        "trend_up": "Desde la medición anterior te nombra más ({before} → {after}) y los márgenes "
        "de error no se solapan: es un cambio claro.",
        "trend_down": "Desde la medición anterior te nombra menos ({before} → {after}) y los "
        "márgenes de error no se solapan: es un cambio claro.",
        "trend_same": "Se mantiene en el {after}, igual que en la medición anterior.",
        "trend_unclear": "Desde la medición anterior ha pasado de {before} a {after}, pero los "
        "márgenes de error se solapan: el cambio puede deberse al azar. Repite la medición para "
        "confirmarlo.",
        "history_unavailable": "Esta medición no se puede abrir aquí: se hizo fuera de la app. "
        "Su informe está en la carpeta runs.",
        "review_title": "Ver y editar las preguntas que se harán a la IA (opcional)",
        "review_help": "Estas son las preguntas que haremos a la IA en cada ronda, como las haría "
        "un cliente. Puedes cambiar el texto, borrar filas o añadir las tuyas (fila nueva al "
        "final). Ninguna debe nombrar tu marca ni a tus competidores. Si no las tocas, se usan "
        "las propuestas.",
        "review_load": "Cargar las preguntas propuestas",
        "review_loaded": "{n} preguntas cargadas. Edítalas en la tabla; se usarán al medir.",
        "col_question": "Pregunta",
        "col_intent": "Tipo (opcional)",
        "problem_too_short": "es demasiado corta",
        "problem_too_long": "es demasiado larga (máximo 500 caracteres)",
        "problem_duplicate": "está repetida",
        "problem_names_brand": "nombra una de las marcas: así no medimos si la IA la menciona "
        "por sí sola",
        "problem_too_few": "Hacen falta al menos 3 preguntas.",
        "problem_too_many": "Como máximo 30 preguntas.",
        "problem_row": "Pregunta {row}: {problem}.",
        "example_title": "Una respuesta real de la IA",
        "example_help": "Elige una pregunta para ver lo que respondió la IA, o mira una al azar. "
        "Las marcas que seguimos aparecen resaltadas.",
        "example_pick": "Pregunta",
        "example_random": "Otra al azar",
        "example_none": "Aún no hay respuestas para mostrar.",
        "example_question": "Pregunta",
        "example_answer": "Respuesta de la IA",
        "example_analysis": "Lo que hemos contado en esta respuesta",
        "named_and_recommended": "La nombra y la recomienda",
        "named_only": "La nombra, sin recomendarla",
        "not_named": "No la nombra",
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
        "access_denied": "**Wrong password.** Ask whoever gave you the link.",
        "access_locked": "**Too many attempts.** Wait a minute and try again.",
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
        "sample_label": "Result of a sample taken on {date} with the AI {model}: {answers} "
        "answers about {sector} in {city}. It is a snapshot of what that AI says, not a market "
        "share.",
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
        "evidence": "found online in this area",
        "no_evidence": "not found (it may still exist)",
        "downloads": "Download your report",
        "running": "Measuring…",
        "waiting_approval": "Waiting for your approval",
        "completed": "Done! Report published",
        "partial": "The measurement stopped early",
        "rejected": "Rejected: nothing was published",
        "error": "Something went wrong",
        "notes_title": "Keep in mind",
        "round_progress": "Round {done} of {total}",
        "time_live": "It will take up to about {minutes} min (it stops earlier when precise).",
        "time_demo": "In demo mode it takes a few seconds.",
        "fail_quota": "The AI provider has used up its free quota for today. Try again later or "
        "use "
        "demo mode.",
        "fail_budget": "The maximum spend was reached before the end. Use fewer questions or raise "
        "the maximum spend.",
        "fail_sources": "Web sources could not be searched. Try again later.",
        "fail_no_evidence": "There were not enough answers to compute results.",
        "fail_unexpected": "Something went wrong during the measurement. The details are in the "
        "technical log.",
        "actions_title": "What you can do now",
        "act_recommend_gap": "The AI names you in {m} of its answers but recommends you in only "
        "{r}. "
        "Read the real answers to see what it says about you.",
        "act_leader": '{leader} appears more than you ({rate}). See in "A real answer from the AI" '
        "which questions it wins.",
        "act_you_lead": "You are the most named brand. Repeat the measurement every month to check "
        "that it holds.",
        "act_suggested": "The AI also recommends {names}, which you do not track: add them as "
        "competitors in the next measurement.",
        "act_few": "Use 10 or more questions for a reliable result.",
        "act_wide": "The margin of error is wide (±{hw}): more questions or rounds will narrow it.",
        "glossary_title": "What does each thing mean?",
        "g_named": "Names you: in what share of the answers your brand appears, even in passing.",
        "g_recommended": "Recommends you: in what share the AI suggests your brand as a good "
        "choice. "
        "This is what matters most.",
        "g_margin": "Margin of error: the range where the real value probably is. More questions "
        "and rounds narrow it.",
        "g_sample": "Dated sample: a snapshot of what one AI answered on one date. Another day or "
        "another AI may answer differently.",
        "history_title": "My measurements",
        "history_pick": "Open a past measurement",
        "history_open": "Open",
        "history_empty": "There are no approved measurements yet.",
        "chart_trend_title": "How is {brand} doing over time?",
        "chart_trend_axis": "% of answers that name it",
        "chart_trend_note": "The band is the margin of error; the letter, the question set used.",
        "trend_caption": "**How to read it:** each point is one measurement of {brand} in {city} "
        "with the same AI ({model}); the band is its margin of error. Only measurements with the "
        "same brand, sector, city and AI are compared. The letter shows which question set was "
        "used.",
        "trend_set_tag": "Questions {letter}",
        "trend_n_questions": "{n} questions",
        "trend_set_changed_same": "It holds at {after}, but the questions changed "
        "(Questions {before_set} → {after_set}), so it is not an exact comparison. To compare, "
        "repeat the measurement with the same questions.",
        "trend_set_changed": "The questions changed between these two measurements (Questions "
        "{before_set} → {after_set}): the move from {before} to {after} may come from the "
        "questions, not from your brand. To compare, repeat the measurement with the same "
        "questions.",
        "trend_table_date": "Measurement",
        "trend_table_set": "Questions used",
        "trend_table_named": "Names you",
        "trend_table_margin": "Margin of error",
        "trend_set_summary": "Questions {letter} · {n} questions",
        "trend_set_unknown": "No record of the questions",
        "trend_up": "Since the previous measurement the AI names you more ({before} → {after}) and "
        "the margins of error do not overlap: it is a clear change.",
        "trend_down": "Since the previous measurement the AI names you less ({before} → {after}) "
        "and the margins of error do not overlap: it is a clear change.",
        "trend_same": "It holds at {after}, the same as in the previous measurement.",
        "trend_unclear": "Since the previous measurement it went from {before} to {after}, but the "
        "margins of error overlap: the change may be chance. Repeat the measurement to confirm it.",
        "history_unavailable": "This measurement cannot be opened here: it was made outside "
        "the app. Its report is in the runs folder.",
        "review_title": "See and edit the questions the AI will be asked (optional)",
        "review_help": "These are the questions we will ask the AI in every round, as a customer "
        "would. You can change the text, delete rows or add your own (new row at the end). None "
        "may name your brand or your competitors. If you do not touch them, the proposed ones "
        "are used.",
        "review_load": "Load the proposed questions",
        "review_loaded": "{n} questions loaded. Edit them in the table; they will be used.",
        "col_question": "Question",
        "col_intent": "Type (optional)",
        "problem_too_short": "is too short",
        "problem_too_long": "is too long (500 characters at most)",
        "problem_duplicate": "is repeated",
        "problem_names_brand": "names one of the brands: then we cannot measure whether the AI "
        "brings it up on its own",
        "problem_too_few": "At least 3 questions are needed.",
        "problem_too_many": "30 questions at most.",
        "problem_row": "Question {row}: {problem}.",
        "example_title": "A real answer from the AI",
        "example_help": "Pick a question to see what the AI answered, or look at a random one. "
        "The brands we track are highlighted.",
        "example_pick": "Question",
        "example_random": "Another random one",
        "example_none": "There are no answers to show yet.",
        "example_question": "Question",
        "example_answer": "The AI's answer",
        "example_analysis": "What we counted in this answer",
        "named_and_recommended": "Names it and recommends it",
        "named_only": "Names it, without recommending it",
        "not_named": "Does not name it",
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
