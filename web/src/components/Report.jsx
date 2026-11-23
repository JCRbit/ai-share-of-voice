import { marked } from "marked";
import { useMemo, useRef, useState } from "react";
import { api } from "../api.js";
import { t } from "../i18n.js";
import { BrandBars } from "./Charts.jsx";

marked.setOptions({ gfm: true, breaks: false });

// Split the report's Markdown into A4 pages: the title and summary first (with the chart), then
// roughly two sections per page. Pure layout; the text is the writer's, untouched.
function paginate(md) {
  const parts = md.split(/\n(?=## )/);
  const pages = [];
  let current = parts.shift() || "";
  let sections = 0;
  for (const part of parts) {
    const firstPage = pages.length === 0;
    if ((firstPage && sections >= 2) || (!firstPage && sections >= 2) || current.length + part.length > 2600) {
      pages.push(current);
      current = part;
      sections = 1;
    } else {
      current += "\n" + part;
      sections += 1;
    }
  }
  if (current.trim()) pages.push(current);
  return pages;
}

const safeHtml = (md) =>
  // The report text can quote model answers and web pages: drop any raw HTML before rendering.
  marked.parse(md.replace(/<[^>]*>/g, ""));

export function ReportViewer({ markdown, results, draft, lang }) {
  const hasChart = results?.brands?.length > 0;
  // The chart gets a sheet of its own, right after the summary, so it is never cut off.
  const pages = useMemo(() => {
    const text = paginate(markdown || "");
    return hasChart ? [text[0], null, ...text.slice(1)] : text;
  }, [markdown, hasChart]);
  const [on, setOn] = useState(0);
  const refs = useRef([]);
  const es = lang === "es";
  if (!markdown) return <p className="muted">{es ? "El informe aparecerá aquí." : "The report will appear here."}</p>;
  const go = (i) => {
    setOn(i);
    refs.current[i]?.scrollIntoView({ behavior: "smooth", block: "start" });
  };
  return (
    <div className="pdf">
      <div className="thumbs no-print" aria-label={t(lang, "page")}>
        {pages.map((_, i) => (
          <button key={i} className={on === i ? "on" : ""} onClick={() => go(i)} aria-label={`${t(lang, "page")} ${i + 1}`}>{i + 1}</button>
        ))}
      </div>
      <div className="sheets">
        {pages.map((p, i) => (
          <section key={i} className="sheet" ref={(el) => (refs.current[i] = el)}>
            <div className="runhead">
              <span>AI Share of Voice · {draft ? (es ? "borrador" : "draft") : (es ? "informe aprobado" : "approved report")}</span>
              <span>{results?.date}</span>
            </div>
            <div className="content">
              {p !== null && <div dangerouslySetInnerHTML={{ __html: safeHtml(p) }} />}
              {p === null && (
                <figure style={{ margin: 0 }}>
                  <h2>{es ? "Te nombra frente a te recomienda" : "Mentions vs recommends"}</h2>
                  <figcaption style={{ fontSize: 13, color: "var(--muted)", marginBottom: 14 }}>
                    {es ? "Cada barra lleva su margen de error. Si dos márgenes se tocan, no digo que una marca gane." : "Each bar carries its margin of error. If two margins touch, I don't say one brand wins."}
                  </figcaption>
                  <BrandBars brands={results.brands} lang={lang} />
                </figure>
              )}
            </div>
            <div className="pfoot">
              <span>{t(lang, "sample")}</span>
              <span>{t(lang, "page")} {i + 1} / {pages.length}</span>
            </div>
          </section>
        ))}
      </div>
    </div>
  );
}

export function Downloads({ runId, names, lang }) {
  const label = {
    "client_report.md": lang === "es" ? "Informe para el cliente" : "Client report",
    "report.md": lang === "es" ? "Informe técnico completo" : "Full technical report",
    "answers.csv": lang === "es" ? "Respuestas analizadas (CSV)" : "Analysed answers (CSV)",
    "tracking_row.csv": lang === "es" ? "Fila de seguimiento (CSV)" : "Tracking row (CSV)",
    "discovered_businesses.csv": lang === "es" ? "Otros nombres (CSV)" : "Other names (CSV)",
    "questions.csv": lang === "es" ? "Preguntas usadas (CSV)" : "Questions used (CSV)",
  };
  return (
    <div className="row">
      {names.map((n) => (
        <a key={n} className="btn" href={api.file(runId, n)} download>{label[n] || n}</a>
      ))}
    </div>
  );
}
