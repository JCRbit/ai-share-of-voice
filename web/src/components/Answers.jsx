import { useMemo, useState } from "react";
import { t } from "../i18n.js";

const escapeRe = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

// The answer text with every tracked brand marked: client in rust, recommended ones underlined.
const TONES = {
  es: { positive: "en positivo", neutral: "neutral", negative: "en negativo", mixed: "mixto" },
  en: { positive: "positive", neutral: "neutral", negative: "negative", mixed: "mixed" },
};

export function Highlighted({ text, mentions, client }) {
  const named = (mentions || []).filter((m) => m.mentioned);
  if (!named.length) return <>{text}</>;
  const byName = Object.fromEntries(named.map((m) => [m.brand.toLowerCase(), m]));
  const re = new RegExp(`(${named.map((m) => escapeRe(m.brand)).join("|")})`, "gi");
  return (
    <>
      {text.split(re).map((part, i) => {
        const m = byName[part.toLowerCase()];
        if (!m) return <span key={i}>{part}</span>;
        const cls = `${m.brand === client ? "b-client" : "b-other"}${m.recommended ? " b-rec" : ""}`;
        return <mark key={i} className={cls}>{part}</mark>;
      })}
    </>
  );
}

export function AnswersBrowser({ answers, client, lang }) {
  const [filter, setFilter] = useState("all");
  const [question, setQuestion] = useState("");
  const status = (a) => {
    const m = a.mentions.find((x) => x.brand === client);
    if (!m || !m.mentioned) return "notNamed";
    return m.recommended ? "recOnly" : "namedOnly";
  };
  const counts = useMemo(() => {
    const c = { all: answers.length, recOnly: 0, namedOnly: 0, notNamed: 0 };
    answers.forEach((a) => (c[status(a)] += 1));
    return c;
  }, [answers]);
  const questions = [...new Set(answers.map((a) => a.question))];
  const shown = answers.filter((a) => (filter === "all" || status(a) === filter) && (!question || a.question === question));
  const es = lang === "es";
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      <div className="filters" role="group" aria-label={es ? "Filtrar respuestas" : "Filter answers"}>
        {["all", "recOnly", "namedOnly", "notNamed"].map((k) => (
          <button key={k} className={filter === k ? "on" : ""} onClick={() => setFilter(k)}>
            {t(lang, k)} ({counts[k]})
          </button>
        ))}
      </div>
      <label className="field">
        <span className="sr-only">{es ? "Pregunta" : "Question"}</span>
        <select value={question} onChange={(e) => setQuestion(e.target.value)}>
          <option value="">{es ? "Todas las preguntas" : "All questions"}</option>
          {questions.map((q) => <option key={q} value={q}>{q}</option>)}
        </select>
      </label>
      <p className="notice" style={{ margin: 0 }}>
        {es
          ? "Cada pregunta tiene pocas respuestas: léelas como ejemplos, no como resultados por pregunta."
          : "Each question has few answers: read them as examples, not per-question results."}
      </p>
      {shown.slice(0, 60).map((a) => (
        <article key={a.id} className="ans">
          <div className="mono muted" style={{ fontSize: 11 }}>{a.question_id} · {t(lang, "round")} {a.round} · {a.model === "simulated" ? (es ? "IA simulada" : "simulated AI") : a.model}</div>
          <div className="q">{a.question}</div>
          <div className="t"><Highlighted text={a.text} mentions={a.mentions} client={client} /></div>
          <div className="tags">
            {a.mentions.filter((m) => m.mentioned).map((m) => (
              <span key={m.brand} className={m.recommended ? "rec" : m.brand === client ? "cl" : ""}>
                {m.brand} · {m.recommended ? (es ? "recomendada" : "recommended") : (es ? "nombrada" : "mentioned")}
                {m.position ? ` · ${m.position}º` : ""}{m.tone ? ` · ${(TONES[lang] || TONES.es)[m.tone] || m.tone}` : ""}
              </span>
            ))}
          </div>
        </article>
      ))}
    </div>
  );
}
