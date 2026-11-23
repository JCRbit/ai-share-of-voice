import { useEffect, useState } from "react";
import { api, pct, duration } from "../api.js";
import { t } from "../i18n.js";
import { Highlighted } from "./Answers.jsx";
import { CountUp, useReveal, useTypewriter } from "../motion.jsx";

// Example answers for the card while there is no approved measurement. Tagged "Ejemplo".
// Each part is [text, kind]: kind "named" or "rec" gets a marker that sweeps in after the text.
const EXAMPLE_ANSWERS = {
  es: [
    { q: "¿Qué app me recomiendas para pedir la cena en Madrid?",
      parts: [["«Las más usadas son ", null], ["Glovo", "named"], [", Just Eat y Uber Eats. Si buscas ofertas, ", null], ["te recomiendo Just Eat", "rec"], ["…»", null]] },
    { q: "Me acabo de mudar a Madrid. ¿Cómo pido comida a casa?",
      parts: [["«Casi todo el mundo usa ", null], ["Glovo", "named"], [" o Uber Eats. Para restaurantes de barrio, ", null], ["yo probaría Glovo", "rec"], [".»", null]] },
    { q: "¿Cuál es la app de reparto más rápida en Madrid?",
      parts: [["«Depende de la zona. En el centro, ", null], ["Uber Eats suele tardar menos", "rec"], [". ", null], ["Glovo", "named"], [" tiene más variedad.»", null]] },
  ],
  en: [
    { q: "Which app would you recommend to order dinner in Madrid?",
      parts: [["“The most used are ", null], ["Glovo", "named"], [", Just Eat and Uber Eats. For deals, ", null], ["I'd recommend Just Eat", "rec"], ["…”", null]] },
    { q: "I just moved to Madrid. How do I get food delivered?",
      parts: [["“Almost everyone uses ", null], ["Glovo", "named"], [" or Uber Eats. For local places, ", null], ["I'd try Glovo", "rec"], [".”", null]] },
    { q: "What's the fastest delivery app in Madrid?",
      parts: [["“It depends on the area. In the centre, ", null], ["Uber Eats is usually quicker", "rec"], [". ", null], ["Glovo", "named"], [" has more choice.”", null]] },
  ],
};

// The card asks its question again and again, the way the product does.
function AnswerCard({ lang, answer, simulatedRun }) {
  const es = lang === "es";
  const examples = EXAMPLE_ANSWERS[lang] || EXAMPLE_ANSWERS.es;
  const [i, setI] = useState(0);
  const item = answer ? null : examples[i % examples.length];
  const question = answer ? answer.question : item.q;
  const [typed, done] = useTypewriter(`«${question}»`, 26, 700);
  useEffect(() => {
    if (answer || !done) return undefined;
    const id = setTimeout(() => setI((n) => n + 1), 5200);
    return () => clearTimeout(id);
  }, [done, answer]);
  return (
    <figure className="answer-card enter" style={{ "--d": "0.35s" }}>
      <div className="mono muted" style={{ fontSize: 12, display: "flex", justifyContent: "space-between", gap: 8 }}>
        <span>{answer && !simulatedRun ? (es ? "UNA RESPUESTA REAL DE LA IA" : "A REAL AI ANSWER") : (es ? "UNA RESPUESTA DE LA IA" : "AN AI ANSWER")}</span>
        {(!answer || simulatedRun) && <span className="tag-example">{t(lang, "exampleTag")}</span>}
      </div>
      <p className="q">{typed}<span className={`caret ${done ? "off" : ""}`} aria-hidden="true" /></p>
      <p key={i} className={`a ${done ? "show" : ""}`} style={{ margin: 0, fontSize: 15, color: "var(--ink-2)" }}>
        {answer ? (
          <Highlighted text={answer.text.length > 260 ? answer.text.slice(0, 260) + "…" : answer.text} mentions={answer.mentions} client={answer.client} />
        ) : (
          item.parts.map(([text, kind], k) => (kind
            ? <span key={k} className={`sweep ${kind === "rec" ? "mark-rec" : "mark-named"}`} style={{ "--d": `${0.5 + k * 0.25}s` }}>{text}</span>
            : <span key={k}>{text}</span>))
        )}
      </p>
      <figcaption style={{ marginTop: 14, paddingTop: 10, borderTop: "1px dashed var(--rule)", fontSize: 13, display: "flex", gap: 14, flexWrap: "wrap", alignItems: "center" }}>
        <span><mark className="b-client">{es ? "tu marca" : "your brand"}</mark></span>
        <span><mark className="b-other b-rec">{es ? "recomendada" : "recommended"}</mark></span>
        {!answer && (
          <span className="dots" aria-hidden="true">
            {examples.map((_, k) => <i key={k} className={k === i % examples.length ? "on" : ""} />)}
          </span>
        )}
      </figcaption>
    </figure>
  );
}

// Example figures, shown only when no approved measurement exists yet — always tagged "Ejemplo".
const EXAMPLE = {
  brand: "Glovo", city: "Madrid", n_answers: 30, n_questions: 10,
  mention: { estimate: 0.93, ci_low: 0.8, ci_high: 1 }, recommend: { estimate: 0.37, ci_low: 0.2, ci_high: 0.55 },
};

export default function Landing({ lang, setLang, go }) {
  const [meta, setMeta] = useState(null);
  const [real, setReal] = useState(null); // the newest approved measurement, if any
  const [answer, setAnswer] = useState(null);

  useEffect(() => {
    api.meta().then(setMeta).catch(() => {});
    api.history().then(async (runs) => {
      if (!runs.length) return;
      const r = await api.results(runs[0].run_id, lang);
      const client = r.brands.find((b) => b.is_client);
      if (client) setReal({ ...r, mention: client.mention, recommend: client.recommend });
      const a = await api.answers(runs[0].run_id);
      const withRec = a.answers.find((x) => x.mentions.some((m) => m.recommended));
      if (withRec) setAnswer({ ...withRec, client: r.brand });
    }).catch(() => {});
  }, [lang]);

  // A run made with the simulated AI is still an example: only live runs count as real answers.
  const simulatedRun = real && real.model === "simulated";
  const data = real || EXAMPLE;
  const isExample = !real || simulatedRun;
  const plans = meta?.plans;
  const es = lang === "es";
  const revealRef = useReveal([lang, real]);

  return (
    <div className="landing" ref={revealRef}>
      <header className="l-head">
        <div className="wrap">
          <a className="logo" href="#/">AI Share of Voice<span>.</span></a>
          <nav className="l-nav" aria-label="Principal">
            <a href="#como">{es ? "Cómo funciona" : "How it works"}</a>
            <a href="#metodo">{es ? "El método" : "The method"}</a>
            <a href="#equipo">{es ? "Quiénes somos" : "Who we are"}</a>
            <span className="seg" style={{ borderColor: "var(--rule)" }}>
              <button className={lang === "es" ? "on" : ""} style={{ color: lang === "es" ? undefined : "var(--muted)" }} onClick={() => setLang("es")}>ES</button>
              <button className={lang === "en" ? "on" : ""} style={{ color: lang === "en" ? undefined : "var(--muted)" }} onClick={() => setLang("en")}>EN</button>
            </span>
          </nav>
          <a className="btn" href="#/app">{es ? "Abrir la app" : "Open the app"}</a>
        </div>
      </header>

      <div className="wrap">
        <div className="dateline enter" style={{ "--d": "0s" }}>
          <span>{es ? "Informe de campo" : "Field report"} {isExample && <span className="tag-example">{t(lang, "exampleTag")}</span>}</span>
          <span>{data.city} · {real ? real.date : "—"}</span>
          <span>{data.n_answers} {t(lang, "answers")} · {data.n_questions} {es ? "preguntas" : "questions"} · {es ? "1 modelo" : "1 model"}</span>
        </div>

        <section className="hero">
          <div className="copy">
            <h1 className="enter" style={{ "--d": "0.08s" }}>{es ? <>Le preguntamos a la IA por tu negocio. <em>Una y otra vez.</em></> : <>We ask the AI about your business. <em>Again and again.</em></>}</h1>
            <p className="lead enter" style={{ "--d": "0.2s" }}>
              {es
                ? "Tus clientes ya no buscan: preguntan. Hacemos esas mismas preguntas 10, 30 o 100 veces, contamos quién sale y quién se recomienda, y te decimos con honestidad cuánto puedes fiarte del resultado."
                : "Your customers don't search any more: they ask. We ask those same questions 10, 30 or 100 times, count who shows up and who gets recommended, and tell you honestly how far to trust it."}
            </p>
            <div className="row enter" style={{ gap: 16, "--d": "0.3s" }}>
              <button className="btn btn-ink" onClick={() => go("#/app")}>{es ? "Prueba con tu marca" : "Try it with your brand"}</button>
              <a href="#demo">{es ? "o mira cómo funciona →" : "or see how it works →"}</a>
            </div>
          </div>
          <AnswerCard lang={lang} answer={simulatedRun ? null : answer} simulatedRun={simulatedRun} />
        </section>
      </div>

      <section className="band">
        <div className="wrap">
          <div data-reveal>
            <div className="bignum"><CountUp value={Math.round(data.mention.estimate * 100)} /><small>%</small></div>
            <p>{es ? "de las respuestas " : "of answers "}<strong style={{ color: "#fff" }}>{es ? "nombran" : "mention"}</strong> {es ? `a ${data.brand}.` : `${data.brand}.`}</p>
            <div className="band-ci">{es ? "margen" : "margin"} {pct(data.mention.ci_low)}–{pct(data.mention.ci_high)}</div>
          </div>
          <div data-reveal style={{ "--d": "0.12s" }}>
            <div className="bignum" style={{ color: "#f08a5d" }}><CountUp value={Math.round(data.recommend.estimate * 100)} duration={1400} /><small>%</small></div>
            <p>{es ? "la " : ""}<strong style={{ color: "#fff" }}>{es ? "recomiendan" : "recommend it"}</strong>{es ? "." : "."}</p>
            <div className="band-ci">{es ? "margen" : "margin"} {pct(data.recommend.ci_low)}–{pct(data.recommend.ci_high)}</div>
          </div>
          <div data-reveal style={{ "--d": "0.24s" }}>
            <p className="serif" style={{ fontSize: 30, lineHeight: 1.2, fontStyle: "italic", color: "#f4efe6", margin: 0 }}>
              {es ? "«Salir no es lo mismo que ser elegido.»" : "“Showing up is not the same as being picked.”"}
            </p>
            <p style={{ fontSize: 13 }}>
              {isExample ? <span className="tag-example" style={{ color: "#cfc7ba", borderColor: "#cfc7ba" }}>{t(lang, "exampleTag")}</span> : null}{" "}
              {t(lang, "sample")}
            </p>
          </div>
        </div>
      </section>

      <section className="section" id="como">
        <div className="wrap">
          <h2 data-reveal>{es ? "Funciona como una buena entrevista: preguntas, escuchas, dudas, y luego decides." : "It works like a good interview: ask, listen, doubt, then decide."}</h2>
          <ol className="steps">
            <li data-reveal style={{ "--d": "0.0s" }}><div className="mono rust" style={{ fontSize: 13 }}>01</div><h3>{es ? "Nos cuentas tu caso" : "You tell us your case"}</h3><p>{es ? "En el chat, con tus palabras. Te enseñamos las preguntas antes: ninguna nombra una marca." : "In the chat, in your words. You see the questions first: none names a brand."}</p></li>
            <li data-reveal style={{ "--d": "0.1s" }}><div className="mono rust" style={{ fontSize: 13 }}>02</div><h3>{es ? "Preguntamos en rondas" : "We ask in rounds"}</h3><p>{es ? "Y paramos solos: margen suficiente, mejora marginal o presupuesto." : "And stop on our own: margin good enough, marginal gain or budget."}</p></li>
            <li data-reveal style={{ "--d": "0.2s" }}><div className="mono rust" style={{ fontSize: 13 }}>03</div><h3>{es ? "Alguien lo pone en duda" : "Someone doubts it"}</h3><p>{es ? "Un revisor escéptico ataca cada frase del borrador antes de que lo leas." : "A sceptical reviewer attacks every sentence of the draft before you read it."}</p></li>
            <li data-reveal style={{ "--d": "0.3s" }}><div className="mono rust" style={{ fontSize: 13 }}>04</div><h3>{es ? "Tú decides" : "You decide"}</h3><p>{es ? "Nada se guarda sin tu aprobación. Te llevas el informe, los gráficos y los datos." : "Nothing is saved without your approval. You take the report, the charts and the data."}</p></li>
          </ol>
        </div>
      </section>

      <section id="demo" style={{ paddingBottom: 88 }}>
        <div className="wrap">
          <div className="plans-row">
            {["quick", "standard", "deep"].map((k) => (
              <div key={k} data-reveal style={{ "--d": `${["quick", "standard", "deep"].indexOf(k) * 0.1}s` }}>
                <div className="mono muted" style={{ fontSize: 12, textTransform: "uppercase" }}>{t(lang, k)}</div>
                <div className="n"><CountUp value={plans ? plans[k].answers : { quick: 10, standard: 30, deep: 100 }[k]} duration={900} /></div>
                <div>{t(lang, "answers")}{plans ? ` · ${plans[k].questions} ${es ? "preguntas" : "questions"} × ${plans[k].rounds} ${es ? "rondas" : "rounds"}` : ""}</div>
                <div className="muted" style={{ fontSize: 14 }}>
                  {plans ? (meta.mode === "live" ? `${duration(plans[k].seconds, lang)} ${es ? "(medido: ~25 s por respuesta)" : "(measured: ~25 s per answer)"}` : (es ? "en modo demostración, unos segundos" : "in demo mode, a few seconds")) : ""}
                </div>
              </div>
            ))}
          </div>
          <div className="example-box" style={{ marginTop: 24, display: "flex", justifyContent: "space-between", gap: 12, flexWrap: "wrap", alignItems: "center" }}>
            <span>{t(lang, "exampleView")}</span>
            <span className="tag-example">{es ? "Vídeo de la demo · Próximamente" : "Demo video · Coming soon"}</span>
          </div>
        </div>
      </section>

      <section id="metodo" style={{ borderTop: "1px solid var(--ink)" }}>
        <div className="wrap" style={{ display: "flex", flexWrap: "wrap", gap: 48, padding: "72px 32px" }}>
          <h2 data-reveal className="serif" style={{ flex: "1 1 320px", margin: 0, fontSize: 44, lineHeight: 1.05 }}>{es ? <>Lo que <em>no</em> te vamos a decir.</> : <>What we <em>won't</em> tell you.</>}</h2>
          <ul className="nots" style={{ flex: "999 1 520px" }}>
            <li data-reveal style={{ "--d": "0.00s" }}>{es ? "No es tu cuota de mercado. Es lo que dijo un modelo, un día, ante unas preguntas." : "It isn't your market share. It's what one model said, one day, to some questions."}</li>
            <li data-reveal style={{ "--d": "0.08s" }}>{es ? "No te diremos que una web «causó» una recomendación. Te enseñamos fuentes, no culpables." : "We won't tell you a website 'caused' a recommendation. We show sources, not culprits."}</li>
            <li data-reveal style={{ "--d": "0.16s" }}>{es ? "No sacamos conclusiones de grupos pequeños ni de diferencias que caben en el margen de error." : "We don't draw conclusions from small groups or from differences within the margin of error."}</li>
            <li data-reveal style={{ "--d": "0.24s" }}>{es ? "No publicamos nada por ti. Apruebas y descargas tú." : "We don't publish anything for you. You approve and download."}</li>
          </ul>
        </div>
      </section>

      <footer className="l-foot" id="equipo">
        <div className="wrap">
          <p className="serif" style={{ margin: 0, fontSize: 26 }}>{es ? "Hecho en Madrid por Anas, Javier, Eli y Abdhur." : "Made in Madrid by Anas, Javier, Eli and Abdhur."}</p>
          <button className="btn btn-rust" onClick={() => go("#/app")}>{es ? "Haz tu primera medición" : "Run your first measurement"}</button>
        </div>
      </footer>
    </div>
  );
}
