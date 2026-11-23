import { useCallback, useEffect, useRef, useState } from "react";
import { api, ApiError, duration, pct, setPassword, usd } from "../api.js";
import { t } from "../i18n.js";
import { AnswersBrowser, Highlighted } from "./Answers.jsx";
import { BrandBars, Funnel, Trend } from "./Charts.jsx";
import { Downloads, ReportViewer } from "./Report.jsx";

const EMPTY_BRIEF = { brand: "", competitors: [], sector: "food_delivery", city: "", country: "España", plan: "standard" };
const REPLAY_MS = 380; // demo mode: events are revealed one by one so the run can be followed

function newConversation(lang) {
  return {
    key: Math.random().toString(36).slice(2),
    title: "",
    messages: [{ role: "ai", text: t(lang, "hello") }],
    stage: "brief", // brief → questions → running → review → done
    brief: null,
    questions: null,
    generated: false,
    runId: null,
  };
}

const PLACEHOLDER_EXAMPLES = {
  es: [
    "Glovo frente a Just Eat y Uber Eats, comida a domicilio en Madrid",
    "Mercadona frente a Carrefour, Lidl y Dia, supermercados en Valencia",
    "Basic-Fit frente a McFit y Anytime Fitness, gimnasios en Sevilla",
    "Banco Santander frente a BBVA y CaixaBank, banca en Bilbao",
  ],
  en: [
    "Glovo vs Just Eat and Uber Eats, food delivery in Madrid",
    "Mercadona vs Carrefour, Lidl and Dia, supermarkets in Valencia",
    "Basic-Fit vs McFit and Anytime Fitness, gyms in Seville",
    "Banco Santander vs BBVA and CaixaBank, banking in Bilbao",
  ],
};

// What I'm doing while you wait, said plainly, with a sketch of the card that is coming.
function Thinking({ lang, stage, hasBrief }) {
  const es = lang === "es";
  const preparing = stage === "brief" && hasBrief;
  const label = stage === "brief"
    ? (preparing ? (es ? "Preparando preguntas que no nombran ninguna marca" : "Writing questions that name no brand") : (es ? "Leyendo lo que me has escrito" : "Reading what you wrote"))
    : (es ? "Buscando la respuesta en las cifras" : "Looking for the answer in the figures");
  return (
    <div className="msg-ai thinking" role="status" aria-live="polite">
      <div className="avatar" aria-hidden="true">A</div>
      <div className="body">
        <span className="think-line">{label}<span className="wave" aria-hidden="true"><i /><i /><i /></span></span>
        {(stage === "brief") && (
          <div className="skeleton-card" aria-hidden="true">
            {(preparing ? [92, 78, 85, 70, 88] : [40, 64, 52]).map((w, k) => <i key={k} style={{ width: `${w}%`, animationDelay: `${k * 0.08}s` }} />)}
          </div>
        )}
      </div>
    </div>
  );
}

export default function ChatApp({ lang, setLang }) {
  const [meta, setMeta] = useState(null);
  const [history, setHistory] = useState([]);
  const [conv, setConv] = useState(() => newConversation(lang));
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [sent, setSent] = useState(false); // the send arrow flies off for a moment
  const [phIdx, setPhIdx] = useState(0); // which example sentence the empty chat box shows
  const inputRef = useRef(null);
  const [askPassword, setAskPassword] = useState(null); // a function to retry after the password
  const [run, setRun] = useState(null); // latest run view from the API
  const [revealed, setRevealed] = useState(0);
  const [results, setResults] = useState(null);
  const [answers, setAnswers] = useState([]);
  const [tab, setTab] = useState("live");
  const [panelOpen, setPanelOpen] = useState(false);
  const [eventMode, setEventMode] = useState(false);
  const threadRef = useRef(null);
  const es = lang === "es";

  useEffect(() => { api.meta().then(setMeta).catch(() => {}); }, []);
  const loadHistory = useCallback(() => api.history().then(setHistory).catch(() => {}), []);
  useEffect(() => { loadHistory(); }, [loadHistory]);
  useEffect(() => { threadRef.current?.scrollTo({ top: threadRef.current.scrollHeight, behavior: "smooth" }); }, [conv.messages.length, conv.stage, revealed, run?.phase]);

  const say = (role, text, extra = {}) => setConv((c) => ({ ...c, messages: [...c.messages, { role, text, ...extra }] }));

  // Run an API call; on "password required" ask for it and retry.
  const guarded = async (fn) => {
    try {
      return await fn();
    } catch (e) {
      if (e instanceof ApiError && (e.status === 401 || e.status === 423)) {
        return new Promise((resolve) => setAskPassword(() => async (pw) => {
          setPassword(pw);
          setAskPassword(null);
          resolve(await guarded(fn));
        }));
      }
      throw e;
    }
  };

  // ---- brief -------------------------------------------------------------------------------
  const sendBrief = async (text) => {
    say("user", text);
    setBusy(true);
    try {
      const d = await guarded(() => api.parse(text, lang));
      const brief = {
        ...EMPTY_BRIEF,
        brand: d.brand, competitors: d.competitors, city: d.city,
        country: d.country || "España", sector: d.sector_key || d.sector || "food_delivery",
      };
      setConv((c) => ({ ...c, brief, title: d.brand ? `${d.brand}${d.city ? " · " + d.city : ""}` : c.title,
        messages: [...c.messages, { role: "ai", text: d.reply, card: "brief" }] }));
    } catch (e) {
      say("ai", es ? `No he podido leerlo: ${e.message}` : `I couldn't read that: ${e.message}`);
    } finally {
      setBusy(false);
    }
  };

  const prepareQuestions = async (brief) => {
    setBusy(true);
    try {
      const q = await guarded(() => api.questions({ ...brief, language: lang }));
      setConv((c) => ({ ...c, brief, questions: q.questions, generated: q.generated, stage: "questions",
        title: `${brief.brand} · ${brief.city}`,
        messages: [...c.messages, { role: "ai", text: `${t(lang, "questionsIntro")} ${q.generated ? t(lang, "generated") : t(lang, "demoPanel")}`, card: "questions" }] }));
    } catch (e) {
      say("ai", es ? `No he podido preparar las preguntas: ${e.message}` : `I couldn't prepare the questions: ${e.message}`);
    } finally {
      setBusy(false);
    }
  };

  const startRun = async (questions) => {
    setBusy(true);
    try {
      const { run_id } = await guarded(() => api.start({ ...conv.brief, language: lang }, questions));
      setRevealed(0);
      setRun(null);
      setResults(null);
      setAnswers([]);
      setTab("live");
      // On a narrow window the panel would cover the chat box: open it only when there is room.
      setPanelOpen(window.matchMedia("(min-width: 1101px)").matches);
      setConv((c) => ({ ...c, runId: run_id, questions, stage: "running",
        messages: [...c.messages, { role: "user", text: t(lang, "start") }, { role: "ai", text: t(lang, "starting"), card: "live" }] }));
    } catch (e) {
      const detail = Array.isArray(e.detail) ? e.detail.map((p) => (p.row ? `${p.row}: ${p.kind}` : p)).join(", ") : e.message;
      say("ai", es ? `No puedo empezar todavía: ${detail}` : `I can't start yet: ${detail}`);
    } finally {
      setBusy(false);
    }
  };

  // ---- watch the run -------------------------------------------------------------------------
  useEffect(() => {
    if (!conv.runId || conv.stage === "done") return undefined;
    let stop = false;
    const tick = async () => {
      try {
        const v = await api.run(conv.runId, lang);
        if (!stop) setRun(v);
        const a = await api.answers(conv.runId);
        if (!stop) setAnswers(a.answers);
      } catch { /* keep polling */ }
    };
    tick();
    const id = setInterval(tick, 1200);
    return () => { stop = true; clearInterval(id); };
  }, [conv.runId, conv.stage, lang]);

  // A finished run is not polled, so a language switch fetches its texts once more.
  useEffect(() => {
    if (!conv.runId || conv.stage !== "done") return;
    api.run(conv.runId, lang).then(setRun).catch(() => {});
    api.results(conv.runId, lang).then(setResults).catch(() => {});
  }, [lang]); // eslint-disable-line react-hooks/exhaustive-deps

  const simulated = meta?.mode !== "live";
  const total = run?.events.length || 0;
  useEffect(() => {
    if (!run) return undefined;
    if (!simulated) { setRevealed(total); return undefined; }
    if (revealed >= total) return undefined;
    const id = setTimeout(() => setRevealed((r) => Math.min(total, r + 1)), REPLAY_MS);
    return () => clearTimeout(id);
  }, [run, revealed, total, simulated]);

  const caughtUp = run && revealed >= total;
  const phase = caughtUp ? run.phase : "running";
  const shownEvents = run ? run.events.slice(0, revealed) : [];
  const metricsShown = shownEvents.filter((e) => e.node === "metrics" && e.status === "succeeded").length;
  const funnel = run ? run.funnel.slice(0, metricsShown) : [];
  const roundsShown = shownEvents.filter((e) => e.node === "sampling" && e.status === "succeeded").length;
  const answersShown = answers.filter((a) => a.round <= Math.max(roundsShown, phase !== "running" ? 99 : 0));

  // When the run reaches approval (or ends), load the figures once and say it in the chat.
  useEffect(() => {
    if (!conv.runId || !caughtUp) return;
    if (conv.stage === "running" && ["waiting_approval", "completed", "partial", "rejected", "error"].includes(run.phase)) {
      api.results(conv.runId, lang).then((r) => {
        setResults(r);
        if (run.phase === "waiting_approval") {
          setTab("report");
          setConv((c) => ({ ...c, stage: "review", messages: [...c.messages, { role: "ai", text: r.headline || t(lang, "approveTitle"), card: "review" }] }));
        } else {
          setConv((c) => ({ ...c, stage: "done", messages: [...c.messages, { role: "ai", text: run.phase === "completed" ? r.headline : t(lang, "failed") }] }));
        }
      }).catch(() => {
        setConv((c) => ({ ...c, stage: run.phase === "waiting_approval" ? "review" : "done",
          messages: [...c.messages, { role: "ai", text: run.phase === "waiting_approval" ? t(lang, "approveTitle") : t(lang, "failed"), card: run.phase === "waiting_approval" ? "review" : undefined }] }));
      });
    }
  }, [caughtUp, run?.phase, conv.stage, conv.runId, lang]);

  const decide = async (approved, comment) => {
    setBusy(true);
    try {
      await api.decide(conv.runId, approved, comment);
      say("user", approved ? t(lang, "approve") : `${t(lang, "reject")}${comment ? ": " + comment : ""}`);
      for (let i = 0; i < 40; i += 1) {
        const v = await api.run(conv.runId, lang);
        if (!["running", "waiting_approval"].includes(v.phase)) { setRun(v); setRevealed(v.events.length); break; }
        await new Promise((r) => setTimeout(r, 400));
      }
      const r = await api.results(conv.runId, lang);
      setResults(r);
      setConv((c) => ({ ...c, stage: "done", messages: [...c.messages, { role: "ai", text: approved ? t(lang, "approved") : t(lang, "rejected"), card: approved ? "downloads" : undefined }] }));
      loadHistory();
    } catch (e) {
      say("ai", e.message);
    } finally {
      setBusy(false);
    }
  };

  const askAbout = async (question) => {
    say("user", question);
    setBusy(true);
    try {
      const { reply } = await api.ask(conv.runId, question, lang);
      say("ai", reply);
    } catch (e) {
      say("ai", e.message);
    } finally {
      setBusy(false);
    }
  };

  const openPast = async (item) => {
    const c = newConversation(lang);
    c.title = `${item.brand} · ${item.city}`;
    c.runId = item.run_id;
    c.stage = "done";
    setConv(c);
    setPanelOpen(true);
    setTab("report");
    setRevealed(9999);
    try {
      const [v, r, a] = await Promise.all([api.run(item.run_id, lang), api.results(item.run_id, lang), api.answers(item.run_id)]);
      setRun(v); setRevealed(v.events.length); setResults(r); setAnswers(a.answers);
      setConv((cc) => ({ ...cc, messages: [...cc.messages.slice(0, 0), { role: "ai", text: r.headline, card: "downloads" }] }));
    } catch (e) {
      setConv((cc) => ({ ...cc, messages: [{ role: "ai", text: e.message }] }));
    }
  };

  const reset = () => { setConv(newConversation(lang)); setRun(null); setResults(null); setAnswers([]); setPanelOpen(false); setRevealed(0); };

  // While the chat waits for a brief, the empty box cycles through sentences people can copy.
  const examples = PLACEHOLDER_EXAMPLES[lang] || PLACEHOLDER_EXAMPLES.es;
  const showExamples = conv.stage === "brief" && !conv.brief;
  useEffect(() => {
    if (!showExamples) return undefined;
    const id = setInterval(() => setPhIdx((n) => n + 1), 3600);
    return () => clearInterval(id);
  }, [showExamples]);
  useEffect(() => { if (!input && inputRef.current) inputRef.current.style.height = "auto"; }, [input]);

  const submit = (e) => {
    e?.preventDefault();
    const text = input.trim();
    if (!text || busy) return;
    setInput("");
    setSent(true);
    setTimeout(() => setSent(false), 450);
    if (conv.stage === "brief") sendBrief(text);
    else if (conv.runId && (conv.stage === "review" || conv.stage === "done")) askAbout(text);
    else say("ai", es ? "Te contesto en cuanto termine esta parte." : "I'll answer as soon as this part is done.");
  };

  const suggestions = conv.runId && (conv.stage === "review" || conv.stage === "done")
    ? (es ? ["¿Por qué paraste?", "¿Quién se recomienda más?", "¿Cuánto me puedo fiar?"] : ["Why did you stop?", "Who is recommended most?", "How much can I trust it?"])
    : [];

  return (
    <div className="app">
      <aside className="side" aria-label={es ? "Mediciones" : "Measurements"}>
        <a className="logo" href="#/">AI Share of Voice<span style={{ color: "#e0703f" }}>.</span></a>
        <button className="new" onClick={reset}>{t(lang, "newRun")}</button>
        <div>
          <h4>{t(lang, "past")}</h4>
          {history.length === 0 && <p className="muted" style={{ fontSize: 13, padding: "0 8px", color: "#a39b8f" }}>{t(lang, "none")}</p>}
          {history.map((h) => (
            <button key={h.run_id} className={`item ${conv.runId === h.run_id ? "on" : ""}`} onClick={() => openPast(h)}>
              <span>{h.brand} · {h.city}<br /><span style={{ fontSize: 11, color: "#a39b8f" }}>{h.date} {h.time}</span></span>
              <span className="pct">{pct(h.mention)}</span>
            </button>
          ))}
        </div>
        <div className="foot">
          <div className="row" style={{ justifyContent: "space-between" }}>
            <span>{t(lang, "eventMode")}</span>
            <span className="seg">
              <button className={eventMode ? "on" : ""} onClick={() => { setEventMode(true); if (conv.stage !== "brief" || conv.brief) reset(); }}>On</button>
              <button className={!eventMode ? "on" : ""} onClick={() => setEventMode(false)}>Off</button>
            </span>
          </div>
          <div className="row" style={{ justifyContent: "space-between" }}>
            <span>{es ? "Gasto del mes" : "Monthly spend"}</span>
            <span className="tag-example" style={{ color: "#a39b8f", borderColor: "#6b6358" }}>{t(lang, "soon")}</span>
          </div>
          <div className="row" style={{ justifyContent: "space-between" }}>
            <span style={{ fontSize: 12 }}>{meta ? `${meta.mode === "live" ? meta.subject_model : (es ? "IA simulada" : "simulated AI")}` : ""}</span>
            <span className="seg">
              <button className={lang === "es" ? "on" : ""} onClick={() => setLang("es")}>ES</button>
              <button className={lang === "en" ? "on" : ""} onClick={() => setLang("en")}>EN</button>
            </span>
          </div>
        </div>
      </aside>

      <main className="main">
        <div className="topbar">
          {(busy || run?.phase === "running") && <div className="progress-line" aria-hidden="true" />}
          <span className="title">{conv.title || t(lang, "newRun").replace("＋ ", "")}</span>
          <span className="row">
            <button className="btn-link only-mobile" onClick={reset}>{t(lang, "newRun").replace("＋ ", "")}</button>
            {meta && <span className={`mode ${meta.mode === "live" ? "live" : ""}`}>{meta.mode === "live" ? `${t(lang, "modeLive")} · ${es ? "tope" : "cap"} ${usd(meta.cap_usd)}` : t(lang, "modeSim")}</span>}
            {conv.runId && <button className="btn-link" onClick={() => setPanelOpen((o) => !o)}>{panelOpen ? t(lang, "hidePanel") : t(lang, "showPanel")}</button>}
          </span>
        </div>

        <div className="thread" ref={threadRef}>
          <div className="thread-inner">
            {eventMode && conv.stage === "brief" && !conv.brief && (
              <EventPicker meta={meta} lang={lang} onReady={(b) => { setConv((c) => ({ ...c, brief: b, messages: [...c.messages, { role: "ai", text: t(lang, "eventNote"), card: "brief" }] })); }} />
            )}
            {conv.messages.map((m, i) =>
              m.role === "user" ? (
                <div key={i} className="msg-user">{m.text}</div>
              ) : (
                <div key={i} className="msg-ai">
                  <div className="avatar" aria-hidden="true">A</div>
                  <div className="body">
                    {m.text && <p>{m.text}</p>}
                    {m.card === "brief" && conv.stage === "brief" && conv.brief && (
                      <BriefCard brief={conv.brief} meta={meta} lang={lang} busy={busy} onSubmit={prepareQuestions} />
                    )}
                    {m.card === "questions" && conv.stage === "questions" && (
                      <QuestionsCard initial={conv.questions} brief={conv.brief} meta={meta} lang={lang} busy={busy}
                        onBack={() => setConv((c) => ({ ...c, stage: "brief", messages: [...c.messages, { role: "ai", text: es ? "Cambia lo que quieras:" : "Change whatever you like:", card: "brief" }] }))}
                        onStart={startRun} />
                    )}
                    {m.card === "live" && run && (
                      <AgentsList meta={meta} events={shownEvents} lang={lang} />
                    )}
                    {m.card === "live" && caughtUp && run?.stop && (
                      <div className="warn"><strong>{t(lang, "stopWhy")}:</strong> {run.stop.reason_text}</div>
                    )}
                    {m.card === "review" && conv.stage === "review" && run && (
                      <ReviewCard run={run} lang={lang} busy={busy} onDecide={decide} onOpen={() => { setPanelOpen(true); setTab("report"); }} />
                    )}
                    {m.card === "downloads" && results?.downloads?.length > 0 && (
                      <Downloads runId={conv.runId} names={results.downloads.filter((n) => ["client_report.md", "answers.csv", "tracking_row.csv"].includes(n))} lang={lang} />
                    )}
                  </div>
                </div>
              ),
            )}
            {busy && <Thinking lang={lang} stage={conv.stage} hasBrief={!!conv.brief} />}
          </div>
        </div>

        <form className="composer" onSubmit={submit}>
          {suggestions.length > 0 && (
            <div className="suggest">
              {suggestions.map((s) => <button key={s} type="button" onClick={() => askAbout(s)} disabled={busy}>{s}</button>)}
            </div>
          )}
          <div className="inputbar">
            <label htmlFor="msg" className="sr-only">{t(lang, "send")}</label>
            {showExamples && <span id="msg-hint" className="sr-only">{es ? "Pulsa Tab para usar el ejemplo que aparece." : "Press Tab to use the example shown."}</span>}
            <div className="input-wrap">
              <textarea id="msg" ref={inputRef} rows={1} value={input}
                onChange={(e) => { setInput(e.target.value); const el = e.target; el.style.height = "auto"; el.style.height = `${Math.min(el.scrollHeight, 160)}px`; }}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) submit(e);
                  // Tab on an empty box takes the example it is showing.
                  if (e.key === "Tab" && !e.shiftKey && showExamples && !input) { e.preventDefault(); setInput(examples[phIdx % examples.length]); }
                }}
                aria-describedby={showExamples ? "msg-hint" : undefined}
                placeholder={showExamples ? "" : conv.runId ? t(lang, "placeholderRun") : t(lang, "placeholder")} />
              {showExamples && !input && (
                <span key={`${lang}-${phIdx}`} className="ph" aria-hidden="true">
                  <span className="ph-text">{examples[phIdx % examples.length]}</span>
                  <kbd>Tab</kbd>
                </span>
              )}
            </div>
            <button type="button" className="icon-btn" disabled aria-label={`${t(lang, "voice")} · ${t(lang, "soon")}`}
              title={es ? "Próximamente. La voz enviaría tu audio a un servicio externo; te avisaremos antes." : "Coming soon. Voice would send your audio to an external service; we'll tell you first."}>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true"><rect x="9" y="3" width="6" height="11" rx="3" /><path d="M5 11a7 7 0 0 0 14 0M12 18v3" /></svg>
            </button>
            <button type="submit" className={`icon-btn send ${input.trim() ? "ready" : ""} ${sent ? "sent" : ""}`} aria-label={t(lang, "send")} disabled={busy || !input.trim()}>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true"><path d="M5 12h14M13 6l6 6-6 6" /></svg>
            </button>
          </div>
        </form>
      </main>

      <aside className={`panel ${panelOpen && conv.runId ? "" : "closed"}`} aria-label={t(lang, "report")}>
        <div className="tabs" role="tablist">
          {["live", "report", "charts", "answersTab", "sources"].map((k) => (
            <button key={k} role="tab" aria-selected={tab === k} className={tab === k ? "on" : ""} onClick={() => setTab(k)}>{t(lang, k)}</button>
          ))}
          <span className="spacer" />
          {tab === "report" && <button className="btn-link" onClick={() => window.print()}>{t(lang, "printPdf")}</button>}
          <button className="btn-link" onClick={() => setPanelOpen(false)} aria-label={t(lang, "hidePanel")}>✕</button>
        </div>
        <div className="panel-body">
          {tab === "live" && (
            <LivePanel run={run} funnel={funnel} answers={answersShown} meta={meta} lang={lang} client={conv.brief?.brand || results?.brand} replay={simulated && !caughtUp} caughtUp={caughtUp} />
          )}
          {tab === "report" && (
            <ReportViewer markdown={results?.approved ? results.report_md : run?.draft} results={results} draft={!results?.approved} lang={lang} />
          )}
          {tab === "charts" && results && (
            <>
              {results.few_questions && <div className="warn">{t(lang, "few")}</div>}
              <figure className="fig"><figcaption>{es ? "Tú frente a tu competencia" : "You vs your competitors"}</figcaption><BrandBars brands={results.brands} lang={lang} />
                <p className="note">{t(lang, "sample")} {results.n_questions} {es ? "preguntas" : "questions"} · {results.n_answers} {t(lang, "answers")} · {results.model === "simulated" ? (es ? "IA simulada" : "simulated AI") : results.model}</p></figure>
              <figure className="fig"><figcaption>{t(lang, "funnel")}</figcaption><Funnel points={run?.funnel || []} target={run?.target_half_width} maxRounds={run?.max_rounds} stop={run?.stop} lang={lang} />
                <p className="note">{run?.stop?.reason_text}</p></figure>
              {results.trend?.length > 1 && <figure className="fig"><figcaption>{t(lang, "trend")}</figcaption><Trend points={results.trend} lang={lang} /></figure>}
            </>
          )}
          {tab === "answersTab" && <AnswersBrowser answers={answers} client={results?.brand || conv.brief?.brand} lang={lang} />}
          {tab === "sources" && <SourcesPanel results={results} lang={lang} />}
        </div>
      </aside>

      {askPassword && <PasswordModal lang={lang} onSubmit={askPassword} onCancel={() => setAskPassword(null)} />}
    </div>
  );
}

// ---- pieces ------------------------------------------------------------------------------------

function BriefCard({ brief, meta, lang, busy, onSubmit }) {
  const [b, setB] = useState(brief);
  const [extra, setExtra] = useState("");
  const es = lang === "es";
  const sectors = meta?.sectors || [];
  const known = sectors.some((s) => s.key === b.sector);
  const addCompetitor = () => {
    const name = extra.trim();
    if (name && b.competitors.length < 5 && !b.competitors.includes(name)) setB({ ...b, competitors: [...b.competitors, name] });
    setExtra("");
  };
  const ok = b.brand.trim() && b.competitors.length >= 2 && b.city.trim() && b.sector.trim();
  const plan = meta?.plans?.[b.plan];
  return (
    <div className="card">
      <div className="grid2">
        <label className="field"><h5>{t(lang, "brand")}</h5><input value={b.brand} onChange={(e) => setB({ ...b, brand: e.target.value })} /></label>
        <div className="field">
          <h5>{t(lang, "competitors")} <span style={{ textTransform: "none" }}>(2–5)</span></h5>
          <div className="chips">
            {b.competitors.map((c) => (
              <span className="chip" key={c}>{c}<button type="button" aria-label={`${es ? "Quitar" : "Remove"} ${c}`} onClick={() => setB({ ...b, competitors: b.competitors.filter((x) => x !== c) })}>✕</button></span>
            ))}
            {b.competitors.length < 5 && (
              <input className="chip-input" aria-label={t(lang, "add")} placeholder={t(lang, "add")} value={extra}
                onChange={(e) => setExtra(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter" || e.key === ",") { e.preventDefault(); addCompetitor(); } }} onBlur={addCompetitor} />
            )}
          </div>
        </div>
        <label className="field"><h5>{t(lang, "sector")}</h5>
          <select value={known ? b.sector : "__other"} onChange={(e) => setB({ ...b, sector: e.target.value === "__other" ? "" : e.target.value })}>
            {sectors.map((s) => <option key={s.key} value={s.key}>{s[lang]}</option>)}
            <option value="__other">{es ? "Otro (escríbelo)" : "Other (type it)"}</option>
          </select>
          {!known && <input placeholder={es ? "Tu sector" : "Your sector"} value={b.sector} onChange={(e) => setB({ ...b, sector: e.target.value })} />}
        </label>
        <div className="grid2" style={{ gap: 8 }}>
          <label className="field"><h5>{t(lang, "city")}</h5><input value={b.city} onChange={(e) => setB({ ...b, city: e.target.value })} /></label>
          <label className="field"><h5>{t(lang, "country")}</h5><input value={b.country} onChange={(e) => setB({ ...b, country: e.target.value })} /></label>
        </div>
      </div>
      <div className="field">
        <h5>{t(lang, "plan")}</h5>
        <div className="plans" role="radiogroup" aria-label={t(lang, "plan")}>
          {["quick", "standard", "deep"].map((k) => {
            const p = meta?.plans?.[k];
            return (
              <button type="button" role="radio" aria-checked={b.plan === k} key={k} className={`plan ${b.plan === k ? "on" : ""}`} onClick={() => setB({ ...b, plan: k })}>
                <b>{t(lang, k)}</b>
                <span>{p ? `${p.answers} ${t(lang, "answers")} · ${duration(p.seconds, lang)}` : ""}</span>
              </button>
            );
          })}
        </div>
        {plan && meta?.mode === "live" && <span className="notice">{es ? `Tiempo medido en real: ~${meta.seconds_per_answer} s por respuesta. Tope de gasto ${usd(meta.cap_usd)}.` : `Measured live: ~${meta.seconds_per_answer} s per answer. Spend cap ${usd(meta.cap_usd)}.`}</span>}
      </div>
      <div className="row">
        <button className="btn btn-ink" disabled={!ok || busy} onClick={() => onSubmit(b)}>{t(lang, "prepare")} →</button>
        {!ok && <span className="notice">{es ? "Falta marca, dos competidores, sector o ciudad." : "Brand, two competitors, sector or city missing."}</span>}
      </div>
    </div>
  );
}

const PROBLEM_TEXT = {
  es: { too_short: "muy corta", too_long: "muy larga", duplicate: "repetida", names_brand: "nombra una marca" },
  en: { too_short: "too short", too_long: "too long", duplicate: "repeated", names_brand: "names a brand" },
};

function QuestionsCard({ initial, brief, meta, lang, busy, onBack, onStart }) {
  const [qs, setQs] = useState(initial || []);
  const es = lang === "es";
  const names = [brief.brand, ...brief.competitors].filter(Boolean).map((n) => n.toLowerCase());
  const escape = (n) => n.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const patterns = names.map((n) => new RegExp(`(^|[^\\p{L}\\p{N}])${escape(n)}($|[^\\p{L}\\p{N}])`, "iu"));
  const flagged = (q) => patterns.some((re) => re.test(q));
  const plan = meta?.plans?.[brief.plan];
  return (
    <div className="card">
      <ol className="qlist">
        {qs.map((q, i) => (
          <li key={i}>
            <input className={flagged(q) ? "bad" : ""} value={q} aria-label={`${es ? "Pregunta" : "Question"} ${i + 1}`}
              onChange={(e) => setQs(qs.map((x, j) => (j === i ? e.target.value : x)))} />
          </li>
        ))}
      </ol>
      {qs.some(flagged) && <span className="problem">{es ? "Una pregunta nombra una marca que medimos: cámbiala para que la prueba sea neutral." : "A question names a brand we measure: change it to keep the test neutral."}</span>}
      <p className="notice" style={{ margin: 0 }}>
        {plan ? (es ? `Haré cada pregunta hasta ${plan.rounds} veces (${plan.answers} respuestas como máximo). Puedo parar antes si el margen ya no mejora.` : `I'll ask each question up to ${plan.rounds} times (${plan.answers} answers at most). I may stop earlier if the margin stops improving.`) : ""}
      </p>
      <div className="row">
        <button className="btn btn-rust" disabled={busy || qs.some(flagged) || qs.filter((q) => q.trim()).length < 3} onClick={() => onStart(qs.filter((q) => q.trim()))}>{t(lang, "start")} →</button>
        <button className="btn" onClick={onBack} disabled={busy}>{t(lang, "edit")}</button>
      </div>
    </div>
  );
}

function AgentsList({ meta, events, lang }) {
  const stages = meta?.stages || [];
  const state = (node) => {
    const evs = events.filter((e) => e.node === node || (node === "approval_request" && e.node === "approval_gate"));
    if (!evs.length) return { s: "todo" };
    const last = evs[evs.length - 1];
    const started = last.status === "started";
    const ok = evs.some((e) => ["succeeded", "approved", "waiting_approval"].includes(e.status));
    return { s: started ? "now" : ok ? "done" : last.status === "failed" ? "failed" : "done", last, count: evs.filter((e) => e.status === "started").length };
  };
  return (
    <ol className="agents" aria-label={lang === "es" ? "Agentes" : "Agents"}>
      {stages.map((st, i) => {
        const { s, last, count } = state(st.node);
        return (
          <li key={st.node} className={s}>
            <span className="dot">{s === "done" ? "✓" : s === "failed" ? "!" : i + 1}</span>
            <div className="what">
              <strong>{st[lang]}</strong>
              {st.node === "sampling" && count ? <span className="muted"> · {t(lang, "round")} {count}</span> : null}
              {last?.detail && s !== "todo" && <div style={{ fontSize: 13 }}>{last.detail.length > 120 ? last.detail.slice(0, 120) + "…" : last.detail}</div>}
              {last?.tool && s !== "todo" && <div className="tool">{last.tool}</div>}
            </div>
          </li>
        );
      })}
    </ol>
  );
}

function LivePanel({ run, funnel, answers, meta, lang, client, replay, caughtUp }) {
  const es = lang === "es";
  if (!run) return <p className="muted">{es ? "Preparando…" : "Getting ready…"}</p>;
  const last = funnel[funnel.length - 1];
  const spend = caughtUp ? run.spend_usd : (funnel.length ? last.spend : 0);
  const nQuestions = funnel[0]?.questions;
  const maxAnswers = nQuestions && run.max_rounds ? nQuestions * run.max_rounds : null;
  return (
    <>
      {replay && <span className="tag-example">{t(lang, "replay")}</span>}
      <div className="stats3">
        <div className="stat"><div className="k">{t(lang, "answers")}</div><div className="v">{answers.length}{maxAnswers ? <small> / {maxAnswers}</small> : null}</div></div>
        <div className="stat"><div className="k">{t(lang, "margin")}</div><div className="v">{last ? `±${Math.round(last.half_width * 100)}` : "—"}<small>{last ? " pts" : ""}</small></div></div>
        <div className="stat"><div className="k">{t(lang, "spend")}</div><div className="v">{usd(spend)}<small> / {usd(run.cap_usd)}</small></div></div>
      </div>
      <figure className="fig">
        <figcaption>{t(lang, "funnel")} · {client}</figcaption>
        <Funnel points={funnel} target={run.target_half_width} maxRounds={run.max_rounds} stop={caughtUp ? run.stop : null} lang={lang} />
        <p className="note">{t(lang, "funnelNote")}</p>
        {caughtUp && run.stop && <p className="note" style={{ color: "var(--ink)" }}><strong>{t(lang, "stopWhy")}:</strong> {run.stop.reason_text}</p>}
      </figure>
      <div className="fig">
        <div className="row" style={{ justifyContent: "space-between", marginBottom: 8 }}>
          <strong style={{ fontSize: 14 }}>{t(lang, "incoming")}</strong>
          {!caughtUp && <span className="mono" style={{ fontSize: 12, color: "var(--green)" }}>● {es ? "en directo" : "live"}</span>}
        </div>
        <div className="stream">
          {answers.slice(-4).reverse().map((a) => (
            <div key={a.id} className="a">
              <div className="meta">{a.question_id} · {t(lang, "round")} {a.round}</div>
              <Highlighted text={a.text.length > 220 ? a.text.slice(0, 220) + "…" : a.text} mentions={a.mentions} client={client} />
            </div>
          ))}
          {!answers.length && (
            <>
              <span className="muted" style={{ fontSize: 14 }}>{es ? "Esperando la primera ronda…" : "Waiting for the first round…"}</span>
              {[0, 1, 2].map((k) => <div key={k} className="skeleton-answer" aria-hidden="true"><i /><i /><i /></div>)}
            </>
          )}
        </div>
      </div>
    </>
  );
}

function ReviewCard({ run, lang, busy, onDecide, onOpen }) {
  const [comment, setComment] = useState("");
  const es = lang === "es";
  return (
    <div className="card">
      <p style={{ margin: 0 }}>{run.objections_intro}</p>
      {run.objections.length > 0 && (
        <ol style={{ margin: 0, paddingLeft: 20, display: "flex", flexDirection: "column", gap: 6 }}>
          {run.objections.map((o, i) => <li key={i}>{o}</li>)}
        </ol>
      )}
      <button className="btn-link" style={{ alignSelf: "flex-start" }} onClick={onOpen}>{es ? "Leer el borrador →" : "Read the draft →"}</button>
      <label className="field">
        <span className="notice">{t(lang, "commentPh")}</span>
        <input value={comment} onChange={(e) => setComment(e.target.value)} />
      </label>
      <div className="row">
        <button className="btn btn-ink" disabled={busy} onClick={() => onDecide(true, comment)}>{t(lang, "approve")}</button>
        <button className="btn" disabled={busy} onClick={() => onDecide(false, comment)}>{comment ? t(lang, "requestChanges") : t(lang, "reject")}</button>
      </div>
      <span className="notice">{es ? "Aprobar guarda el informe y el CSV de seguimiento en tu equipo. No publicamos nada." : "Approving saves the report and the tracking CSV on your computer. We publish nothing."}</span>
    </div>
  );
}

function SourcesPanel({ results, lang }) {
  const es = lang === "es";
  if (!results) return null;
  const sources = results.research?.sources || [];
  return (
    <>
      <figure className="fig">
        <figcaption>{t(lang, "sources")}</figcaption>
        <p className="note" style={{ marginTop: 0 }}>{t(lang, "sourcesNote")}</p>
        {sources.length === 0 ? <p className="muted">{t(lang, "noSources")}</p> : (
          <ul style={{ margin: 0, paddingLeft: 18 }}>
            {sources.map((s, i) => <li key={i}><a href={s.url || s} target="_blank" rel="noreferrer noopener">{s.title || s.url || s}</a></li>)}
          </ul>
        )}
      </figure>
      <figure className="fig">
        <figcaption>{t(lang, "discovered")}</figcaption>
        <p className="note" style={{ marginTop: 0 }}>{t(lang, "discoveredNote")}</p>
        {(results.discovered || []).length === 0 ? <p className="muted">—</p> : (
          <ul style={{ margin: 0, padding: 0, listStyle: "none", display: "flex", flexDirection: "column", gap: 8 }}>
            {results.discovered.map((d) => (
              <li key={d.name} className="ans" style={{ flexDirection: "row", justifyContent: "space-between", flexWrap: "wrap" }}>
                <span>{d.suggested ? "★ " : ""}{d.name}</span>
                <span className="muted" style={{ fontSize: 13 }}>
                  {pct(d.rate)} ({pct(d.ci_low)}–{pct(d.ci_high)}) · {d.n_questions_with} {es ? "preguntas" : "questions"} ·{" "}
                  {d.found_online ? <a href={d.evidence_url} target="_blank" rel="noreferrer noopener">{t(lang, "foundOnline")}</a> : t(lang, "notChecked")}
                </span>
              </li>
            ))}
          </ul>
        )}
      </figure>
    </>
  );
}

function EventPicker({ meta, lang, onReady }) {
  const [sector, setSector] = useState("");
  const es = lang === "es";
  return (
    <div className="card" style={{ borderColor: "var(--ink)" }}>
      <h2 className="serif" style={{ margin: 0, fontSize: 34, lineHeight: 1.1 }}>{t(lang, "eventTitle")}</h2>
      <p className="notice" style={{ margin: 0 }}>{t(lang, "eventNote")}</p>
      <div className="event-grid" role="radiogroup" aria-label={t(lang, "sector")}>
        {(meta?.sectors || []).map((s) => (
          <button key={s.key} role="radio" aria-checked={sector === s.key} className={sector === s.key ? "on" : ""} onClick={() => setSector(s.key)}>{s[lang]}</button>
        ))}
      </div>
      <div className="row">
        <button className="btn btn-ink" disabled={!sector} onClick={() => onReady({ ...EMPTY_BRIEF, sector, city: "Madrid" })}>{es ? "Siguiente: la empresa" : "Next: the company"} →</button>
        <span className="notice">{es ? "Plan B: las mediciones de ayer están a la izquierda, en «Mediciones guardadas»." : "Plan B: yesterday's runs are on the left, under “Saved measurements”."}</span>
      </div>
    </div>
  );
}

function PasswordModal({ lang, onSubmit, onCancel }) {
  const [pw, setPw] = useState("");
  return (
    <div className="modal-bg" role="dialog" aria-modal="true" aria-labelledby="pw-title">
      <form className="modal" onSubmit={(e) => { e.preventDefault(); onSubmit(pw); }}>
        <h2 id="pw-title" className="serif" style={{ margin: 0, fontSize: 28 }}>{t(lang, "password")}</h2>
        <p style={{ margin: 0 }}>{t(lang, "passwordWhy")}</p>
        <label className="field"><span className="sr-only">{t(lang, "password")}</span>
          <input type="password" autoFocus value={pw} onChange={(e) => setPw(e.target.value)} autoComplete="current-password" /></label>
        <div className="row"><button className="btn btn-ink" type="submit">{t(lang, "ok")}</button><button className="btn" type="button" onClick={onCancel}>{t(lang, "cancel")}</button></div>
      </form>
    </div>
  );
}
