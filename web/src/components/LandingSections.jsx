import { useState } from "react";
import { t } from "../i18n.js";
import { Funnel } from "./Charts.jsx";

// ---------------------------------------------------------------------------------------------
// El método: every step says what we do, why, and where you can check it in the app.
// ---------------------------------------------------------------------------------------------

const EXAMPLE_FUNNEL = [
  { round: 1, estimate: 0.62, low: 0.38, high: 0.84, half_width: 0.23 },
  { round: 2, estimate: 0.66, low: 0.49, high: 0.81, half_width: 0.16 },
  { round: 3, estimate: 0.67, low: 0.55, high: 0.78, half_width: 0.115 },
];

function steps(es, rules) {
  const minRounds = rules?.min_batches ?? 2;
  const gain = rules?.marginal_gain_threshold != null ? Math.round(rules.marginal_gain_threshold * 1000) / 10 : 0.5;
  const share = rules?.sampling_budget_share != null ? Math.round(rules.sampling_budget_share * 100) : 70;
  return [
    {
      key: "questions",
      title: es ? "Preguntas neutrales" : "Neutral questions",
      what: es
        ? "Escribimos preguntas como las haría un cliente de tu sector y tu ciudad. Ninguna nombra una marca: así vemos a quién trae la IA por sí sola."
        : "We write questions the way a customer in your sector and city would ask. None names a brand, so we see who the AI brings up on its own.",
      why: es
        ? "Si la pregunta dijera «¿es buena Glovo?», la IA hablaría de Glovo por obligación. Eso no mide nada."
        : "If the question said 'is Glovo any good?', the AI would have to talk about Glovo. That measures nothing.",
      check: es
        ? "Las ves y las editas antes de empezar. Si una nombra una marca medida, el botón se bloquea."
        : "You see and edit them before starting. If one names a measured brand, the button locks.",
      visual: "questions",
    },
    {
      key: "sampling",
      title: es ? "Preguntamos en rondas" : "We ask in rounds",
      what: es
        ? "Cada ronda hace todas las preguntas otra vez al mismo modelo. La IA no contesta siempre igual, así que una sola respuesta no basta."
        : "Each round asks every question again to the same model. The AI doesn't always answer the same way, so one answer isn't enough.",
      why: es
        ? "Repetir nos dice si tu marca sale por costumbre o por casualidad."
        : "Repeating tells us whether your brand shows up by habit or by chance.",
      check: es
        ? "En «En vivo» ves llegar cada respuesta, con su pregunta y su ronda."
        : "In 'Live' you see every answer arrive, with its question and round.",
      visual: "rounds",
    },
    {
      key: "count",
      title: es ? "Te nombra frente a te recomienda" : "Mentioned vs recommended",
      what: es
        ? "En cada respuesta miramos dos cosas distintas: si aparece tu marca y si la IA la aconseja de verdad."
        : "In each answer we look at two different things: whether your brand appears, and whether the AI actually advises it.",
      why: es
        ? "Salir no es lo mismo que ser elegido. Una marca puede salir en casi todas las respuestas y casi nunca ser la recomendada."
        : "Showing up is not being picked. A brand can appear in almost every answer and hardly ever be the one recommended.",
      check: es
        ? "En «Respuestas» cada marca va subrayada: amarillo si sale, verde si se recomienda."
        : "In 'Answers' each brand is marked: yellow when it appears, green when it's recommended.",
      visual: "marks",
    },
    {
      key: "margin",
      title: es ? "El margen, pregunta a pregunta" : "The margin, question by question",
      what: es
        ? "El margen de error se calcula por pregunta, no por respuesta. Con 10 preguntas tenemos 10 datos independientes, aunque haya 30 respuestas."
        : "The margin of error is computed per question, not per answer. With 10 questions we have 10 independent data points, even with 30 answers.",
      why: es
        ? "Las respuestas a una misma pregunta se parecen entre sí. Contarlas como si fueran independientes daría un margen falsamente estrecho."
        : "Answers to the same question look alike. Counting them as independent would give a falsely narrow margin.",
      check: es
        ? "Cada cifra lleva su margen. Si dos márgenes se tocan, no decimos que una marca gane."
        : "Every figure carries its margin. If two margins touch, we don't say one brand wins.",
      visual: "clusters",
    },
    {
      key: "stop",
      title: es ? "Paramos solos, y te decimos por qué" : "We stop on our own, and say why",
      what: es
        ? `Hacemos al menos ${minRounds} rondas. Luego paramos por la primera de estas razones: el margen ya es suficiente, otra ronda lo mejoraría menos de ${gain} puntos, se acaba el presupuesto (${share}% para preguntar) o llegamos al máximo de rondas.`
        : `We run at least ${minRounds} rounds. Then we stop for the first of these reasons: the margin is good enough, another round would improve it by less than ${gain} points, the budget runs out (${share}% for asking) or we reach the round limit.`,
      why: es
        ? "Así no gastas en rondas que ya no cambian el resultado, ni paramos antes de tiempo."
        : "So you don't pay for rounds that no longer change the result, and we don't stop too early.",
      check: es
        ? "El embudo de la certeza muestra el margen tras cada ronda y la regla que hizo parar."
        : "The certainty funnel shows the margin after each round and the rule that stopped it.",
      visual: "funnel",
    },
    {
      key: "review",
      title: es ? "Un escéptico, y luego tú" : "A sceptic, then you",
      what: es
        ? "Un revisor automático lee el borrador buscando frases que digan más de lo que muestran las cifras. Después decides tú: nada se guarda sin tu aprobación."
        : "An automatic reviewer reads the draft looking for sentences that claim more than the figures show. Then you decide: nothing is saved without your approval.",
      why: es
        ? "Un informe bonito que exagera es peor que ninguno."
        : "A nice report that overstates is worse than none.",
      check: es
        ? "Sus objeciones aparecen junto al botón «Aprobar y descargar». Todo queda en un registro: qué agente hizo qué, con qué herramienta."
        : "Its objections appear next to 'Approve and download'. Everything is logged: which agent did what, with which tool.",
      visual: "review",
    },
  ];
}

function Visual({ kind, es, lang }) {
  if (kind === "questions") {
    const qs = es
      ? ["¿Qué app me recomiendas para pedir la cena?", "¿Cuál es la más rápida en mi zona?", "¿Cuál tiene más restaurantes?"]
      : ["Which app would you recommend for dinner?", "Which is fastest in my area?", "Which has the most restaurants?"];
    return (
      <ol className="mv-questions">
        {qs.map((q, i) => <li key={q} style={{ "--d": `${i * 0.12}s` }}><span className="mono">0{i + 1}</span>{q}<span className="ok">{es ? "sin marcas" : "no brands"}</span></li>)}
      </ol>
    );
  }
  if (kind === "rounds") {
    return (
      <div className="mv-rounds" aria-hidden="true">
        {[1, 2, 3].map((r) => (
          <div key={r} className="mv-round" style={{ "--d": `${(r - 1) * 0.25}s` }}>
            <span className="mono">{t(lang, "round")} {r}</span>
            <div>{Array.from({ length: 10 }).map((_, i) => <i key={i} style={{ "--i": i }} />)}</div>
          </div>
        ))}
      </div>
    );
  }
  if (kind === "marks") {
    return (
      <p className="mv-marks">
        «{es ? "Las más usadas son " : "The most used are "}<span className="mark-named">Glovo</span>, Just Eat {es ? "y" : "and"} Uber Eats.{" "}
        {es ? "Para ofertas, " : "For deals, "}<span className="mark-rec">{es ? "te recomiendo Just Eat" : "I'd recommend Just Eat"}</span>.»
        <span className="mv-legend"><span><span className="mark-named">Glovo</span> {es ? "te nombra" : "mentioned"}</span><span><span className="mark-rec">Just Eat</span> {es ? "recomendada" : "recommended"}</span></span>
      </p>
    );
  }
  if (kind === "clusters") {
    return (
      <div className="mv-clusters" aria-hidden="true">
        {Array.from({ length: 10 }).map((_, q) => (
          <div key={q} style={{ "--d": `${q * 0.06}s` }}>
            {[0, 1, 2].map((r) => <i key={r} className={(q * 3 + r) % 4 === 0 ? "" : "on"} />)}
            <span className="mono">p{q + 1}</span>
          </div>
        ))}
        <p>{es ? "30 respuestas, agrupadas en 10 preguntas = 10 datos" : "30 answers grouped in 10 questions = 10 data points"}</p>
      </div>
    );
  }
  if (kind === "funnel") {
    return (
      <div>
        <span className="tag-example">{t(lang, "exampleTag")}</span>
        <Funnel points={EXAMPLE_FUNNEL} target={0.12} maxRounds={3} stop={{ stop: true }} lang={lang} height={190} />
      </div>
    );
  }
  return (
    <div className="mv-review">
      <p><span className="mono">{es ? "BORRADOR" : "DRAFT"}</span> {es ? "«Glovo domina el mercado de Madrid.»" : "'Glovo dominates the Madrid market.'"}</p>
      <p className="obj">{es ? "Objeción: una muestra de respuestas no es cuota de mercado. Reformular." : "Objection: a sample of answers is not market share. Rephrase."}</p>
      <div className="row"><span className="btn btn-ink" aria-hidden="true">{t(lang, "approve")}</span><span className="btn" aria-hidden="true">{t(lang, "reject")}</span></div>
    </div>
  );
}

export function MethodSection({ lang, rules }) {
  const es = lang === "es";
  const items = steps(es, rules);
  const [on, setOn] = useState(0);
  const item = items[on];
  return (
    <section className="section method" id="metodo">
      <div className="wrap">
        <h2 data-reveal>{es ? "El método, sin cajas negras." : "The method, no black boxes."}</h2>
        <p className="lead-sm" data-reveal>
          {es
            ? "Cada número del informe se puede seguir hasta las respuestas que lo produjeron. Así lo hacemos, paso a paso."
            : "Every number in the report can be traced back to the answers that produced it. Here's how, step by step."}
        </p>
        <div className="method-grid" data-reveal>
          <ol className="method-steps" role="tablist" aria-label={es ? "Pasos del método" : "Method steps"}>
            {items.map((s, i) => (
              <li key={s.key}>
                <button role="tab" aria-selected={on === i} className={on === i ? "on" : ""} onClick={() => setOn(i)}>
                  <span className="mono">0{i + 1}</span>{s.title}
                </button>
              </li>
            ))}
          </ol>
          <div className="method-detail" role="tabpanel" key={item.key}>
            <h3>{item.title}</h3>
            <p>{item.what}</p>
            <div className="method-visual"><Visual kind={item.visual} es={es} lang={lang} /></div>
            <dl>
              <div><dt>{es ? "Por qué" : "Why"}</dt><dd>{item.why}</dd></div>
              <div><dt>{es ? "Dónde lo compruebas" : "Where you check it"}</dt><dd>{item.check}</dd></div>
            </dl>
            <div className="method-nav">
              <button className="btn-link" disabled={on === 0} onClick={() => setOn(on - 1)}>← {es ? "Anterior" : "Previous"}</button>
              <span className="mono muted">{on + 1} / {items.length}</span>
              <button className="btn-link" disabled={on === items.length - 1} onClick={() => setOn(on + 1)}>{es ? "Siguiente" : "Next"} →</button>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

// ---------------------------------------------------------------------------------------------
// Quiénes somos. Only real links: a person without a known link shows none.
// ---------------------------------------------------------------------------------------------

export const REPO_URL = "https://github.com/adryeli/ai-share-of-voice";

// Roles as the team gave them. GitHub handles come from the README contributors table;
// empty fields are not shown.
const TEAM = [
  { name: "Anas", summary: { es: "Especialista en UX", en: "UX Expert" }, github: "https://github.com/Anasfady", linkedin: "" },
  { name: "Javier", summary: { es: "Especialista en backend", en: "Backend Expert" }, github: "https://github.com/JCRbit", linkedin: "" },
  { name: "Eli", summary: { es: "Especialista en producto", en: "Product Expert" }, github: "https://github.com/adryeli", linkedin: "" },
  { name: "Abdur", summary: { es: "Especialista en testing", en: "Testing Expert" }, github: "", linkedin: "" },
];

const GitHubIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M12 .5a11.5 11.5 0 0 0-3.64 22.41c.58.1.79-.25.79-.56v-2c-3.2.7-3.88-1.37-3.88-1.37-.52-1.33-1.28-1.69-1.28-1.69-1.04-.71.08-.7.08-.7 1.15.08 1.76 1.19 1.76 1.19 1.03 1.76 2.69 1.25 3.35.96.1-.75.4-1.25.73-1.54-2.55-.29-5.24-1.28-5.24-5.69 0-1.26.45-2.29 1.19-3.1-.12-.29-.52-1.46.11-3.05 0 0 .97-.31 3.17 1.18a11 11 0 0 1 5.77 0c2.2-1.49 3.17-1.18 3.17-1.18.63 1.59.23 2.76.11 3.05.74.81 1.19 1.84 1.19 3.1 0 4.42-2.69 5.39-5.26 5.68.41.36.78 1.06.78 2.14v3.17c0 .31.21.67.8.56A11.5 11.5 0 0 0 12 .5Z" /></svg>
);
const LinkedInIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M20.45 20.45h-3.55v-5.57c0-1.33-.02-3.04-1.85-3.04-1.86 0-2.14 1.45-2.14 2.94v5.67H9.36V9h3.41v1.56h.05c.48-.9 1.64-1.85 3.37-1.85 3.6 0 4.27 2.37 4.27 5.46v6.28ZM5.34 7.43a2.06 2.06 0 1 1 0-4.12 2.06 2.06 0 0 1 0 4.12ZM7.12 20.45H3.56V9h3.56v11.45ZM22.22 0H1.77C.79 0 0 .77 0 1.73v20.54C0 23.23.79 24 1.77 24h20.45c.98 0 1.78-.77 1.78-1.73V1.73C24 .77 23.2 0 22.22 0Z" /></svg>
);

export function TeamSection({ lang, go }) {
  const es = lang === "es";
  return (
    <section className="section team" id="equipo">
      <div className="wrap">
        <h2 data-reveal>{es ? "Quiénes somos." : "Who we are."}</h2>
        <p className="lead-sm" data-reveal>
          {es
            ? "Un equipo pequeño en Madrid que quería saber qué dice la IA de los negocios, y contarlo sin exagerar."
            : "A small team in Madrid that wanted to know what AI says about businesses, and to say it without overstating."}
        </p>
        <ul className="team-grid">
          {TEAM.map((m, i) => {
            const summary = m.summary[lang] || m.summary.es;
            return (
              <li key={m.name} data-reveal style={{ "--d": `${i * 0.08}s` }}>
                <div className="initial" aria-hidden="true">{m.name[0]}</div>
                <h3>{m.name}</h3>
                {summary ? <p className="role">{summary}</p> : <p className="muted"><span className="tag-soon">{t(lang, "soon")}</span></p>}
                <div className="links">
                  {m.github && <a href={m.github} target="_blank" rel="noreferrer noopener"><GitHubIcon /> GitHub</a>}
                  {m.linkedin && <a href={m.linkedin} target="_blank" rel="noreferrer noopener"><LinkedInIcon /> LinkedIn</a>}
                </div>
              </li>
            );
          })}
        </ul>
        <div className="team-foot" data-reveal>
          <a className="btn" href={REPO_URL} target="_blank" rel="noreferrer noopener"><GitHubIcon /> {es ? "El código, en GitHub" : "The code, on GitHub"}</a>
          <button className="btn btn-cta" onClick={() => go("#/app")}>{es ? "Haz tu primera medición" : "Run your first measurement"} →</button>
        </div>
      </div>
    </section>
  );
}
