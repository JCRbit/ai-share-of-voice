"""HTML pieces of the app, written for non-technical users.

Pure functions from plain data to HTML, so they are tested without a browser. Two rules:

* Everything that comes from outside the code (model drafts, objections, business names,
  URLs) is escaped: model output and web content are untrusted, and a draft must not be
  able to inject HTML or scripts into the page. Links are only made for http(s) URLs.
* Colours come from Gradio's CSS variables plus a few semantic ones, so the page works in
  light and dark mode.
"""

import re
from html import escape
from urllib.parse import urlsplit

from ai_share_of_voice.panel import PanelProblem
from ai_share_of_voice.stats.trend import Rate, compare
from ai_share_of_voice.ui.i18n import kind, stage, t
from ai_share_of_voice.ui.run_controller import AnswerExample, ResultsBundle, RunView

STAGE_ORDER = (
    "questions",
    "sampling",
    "metrics",
    "research",
    "screenshots",
    "write_draft",
    "skeptic",
    "approval_gate",
    "finalize",
)
PHASE_TONE = {
    "running": "info",
    "waiting_approval": "warn",
    "completed": "ok",
    "partial": "warn",
    "rejected": "neutral",
    "error": "bad",
}

APP_CSS = """
:root { --sov-ok:#15803d; --sov-warn:#b45309; --sov-bad:#b91c1c; --sov-info:#4338ca;
        --sov-accent:#4f46e5; --sov-soft:rgba(79,70,229,.08); }
.gradio-container { max-width: 1120px !important; margin: 0 auto !important; }
.sov-tabs > .tab-wrapper { display: none !important; }
.sov-nav .wrap { gap: 6px !important; border-bottom: 1px solid var(--border-color-primary);
  padding-bottom: 8px; }
.sov-nav label { border-radius: 999px !important; padding: 8px 16px !important;
  font-weight: 600; }
.sov-nav input[type=radio] { display: none; }
.sov-topbar { display:flex; align-items:center; gap: 12px; padding: 6px 2px; }
.sov-logo { width: 38px; height: 38px; border-radius: 10px; display:flex; align-items:center;
  justify-content:center; font-size: 20px; color:#fff;
  background: linear-gradient(135deg, #4f46e5, #0ea5e9); }
.sov-topbar b { font-size: 18px; display:block; }
.sov-topbar span { font-size: 13px; color: var(--body-text-color-subdued); }
.sov-hero { padding: 28px 28px 22px; border-radius: 18px; margin-bottom: 8px;
  background: linear-gradient(135deg, rgba(79,70,229,.12), rgba(14,165,233,.10));
  border: 1px solid var(--border-color-primary); }
.sov-hero h1 { font-size: 30px; line-height: 1.2; margin: 0 0 8px; }
.sov-hero p { font-size: 16px; margin: 0; max-width: 760px; color: var(--body-text-color-subdued); }
.sov-badge { display:inline-block; padding: 4px 10px; border-radius: 999px; font-size: 12px;
  font-weight: 600; margin-top: 14px; background: var(--sov-soft); color: var(--sov-accent); }
.sov-steps { display:grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-top: 18px; }
.sov-step { background: var(--block-background-fill); border: 1px solid var(--border-color-primary);
  border-radius: 14px; padding: 14px 16px; }
.sov-step b { display:block; margin-bottom: 4px; }
.sov-step span { font-size: 14px; color: var(--body-text-color-subdued); }
.sov-num { display:inline-flex; width: 26px; height: 26px; border-radius: 50%; align-items:center;
  justify-content:center; background: var(--sov-accent); color: #fff; font-weight: 700;
  font-size: 13px; margin-bottom: 8px; }
.sov-banner { display:flex; align-items:center; gap: 10px; padding: 14px 18px; border-radius: 14px;
  font-weight: 600; font-size: 16px; border: 1px solid var(--border-color-primary); }
.sov-banner.ok { background: rgba(21,128,61,.10); color: var(--sov-ok); }
.sov-banner.warn { background: rgba(180,83,9,.10); color: var(--sov-warn); }
.sov-banner.bad { background: rgba(185,28,28,.10); color: var(--sov-bad); }
.sov-banner.info { background: var(--sov-soft); color: var(--sov-info); }
.sov-banner.neutral { background: var(--block-background-fill); }
.sov-track { display:grid; grid-template-columns: repeat(auto-fit, minmax(92px, 1fr));
  gap: 8px; margin: 14px 0; }
.sov-stage { padding: 10px 10px; border-radius: 12px; font-size: 12.5px; line-height: 1.3;
  border: 1px solid var(--border-color-primary); background: var(--block-background-fill); }
.sov-stage .dot { display:inline-block; width: 9px; height: 9px; border-radius: 50%;
  margin-right: 6px; background: var(--border-color-primary); }
.sov-stage.done .dot { background: var(--sov-ok); }
.sov-stage.active { border-color: var(--sov-accent); box-shadow: 0 0 0 2px var(--sov-soft); }
.sov-stage.active .dot { background: var(--sov-accent); animation: sov-pulse 1.2s infinite; }
.sov-stage.waiting { border-color: var(--sov-warn); }
.sov-stage.waiting .dot { background: var(--sov-warn); }
.sov-stage.failed .dot { background: var(--sov-bad); }
.sov-stage.pending { opacity: .55; }
@keyframes sov-pulse { 50% { opacity: .3; } }
.sov-spend { margin-top: 6px; font-size: 14px; }
.sov-bar { background: var(--border-color-primary); border-radius: 999px; height: 10px;
  overflow: hidden; margin-top: 6px; }
.sov-bar > div { height: 10px; border-radius: 999px; }
.sov-kpis { display:grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 6px 0 14px; }
.sov-kpi { background: var(--block-background-fill); border: 1px solid var(--border-color-primary);
  border-radius: 16px; padding: 16px 18px; }
.sov-kpi .label { font-size: 13px; color: var(--body-text-color-subdued); }
.sov-kpi .value { font-size: 34px; font-weight: 800; line-height: 1.15; margin: 4px 0; }
.sov-kpi .hint { font-size: 12px; color: var(--body-text-color-subdued); }
.sov-kpi.main { border-color: var(--sov-accent); background: var(--sov-soft); }
.sov-kpi.main .value { color: var(--sov-accent); }
.sov-summary { font-size: 17px; line-height: 1.55; margin: 4px 0 10px; }
.sov-muted { font-size: 13px; color: var(--body-text-color-subdued); }
.sov-table { width: 100%; border-collapse: collapse; font-size: 15px; }
.sov-table td { padding: 10px 8px; border-bottom: 1px solid var(--border-color-primary);
  vertical-align: middle; }
.sov-table tr.me td { font-weight: 700; }
.sov-table .barcell { width: 46%; }
.sov-hbar { background: var(--border-color-primary); border-radius: 999px; height: 12px;
  position: relative; overflow: hidden; }
.sov-hbar > div { position:absolute; left:0; top:0; bottom:0; border-radius: 999px;
  background: #94a3b8; }
.sov-table tr.me .sov-hbar > div { background: var(--sov-accent); }
.sov-star { color: #d97706; }
.sov-draft { white-space: pre-wrap; font-size: 15px; line-height: 1.55; padding: 14px 16px;
  border-radius: 12px; border: 1px solid var(--border-color-primary);
  background: var(--block-background-fill); max-height: 360px; overflow: auto; }
.sov-list { margin: 6px 0 0 18px; }
.sov-answer { white-space: pre-wrap; font-size: 14.5px; line-height: 1.6; padding: 14px 16px;
  border-radius: 12px; border: 1px solid var(--border-color-primary);
  background: var(--block-background-fill); max-height: 380px; overflow: auto; }
.sov-answer mark { background: rgba(79,70,229,.18); color: inherit; border-radius: 4px;
  padding: 0 2px; font-weight: 600; }
/* The questions read as text, not code: Gradio draws table cells in a monospace font. */
.sov-review td, .sov-review td * { font-family: var(--font) !important; font-size: 14px; }
.sov-actions { padding: 14px 18px; border-radius: 14px; margin: 6px 0 12px;
  background: rgba(21,128,61,.08); border: 1px solid var(--border-color-primary); }
.sov-actions h3 { margin: 0 0 6px; }
.sov-chips { display:flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
.sov-chip { padding: 6px 12px; border-radius: 999px; font-size: 13px;
  border: 1px solid var(--border-color-primary); }
.sov-chip.rec { background: rgba(21,128,61,.12); color: var(--sov-ok); }
.sov-chip.named { background: var(--sov-soft); color: var(--sov-accent); }
@media (max-width: 760px) {
  .sov-steps, .sov-kpis { grid-template-columns: 1fr 1fr; }
  .sov-hero h1 { font-size: 24px; }
}
"""


def _money(lang: str, usd: float) -> str:
    text = f"{usd:.3f} USD"
    return text.replace(".", ",") if lang == "es" else text


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.0%}"


def _safe_link(url: str, label: str) -> str:
    """An escaped link, only for http(s) URLs; anything else is plain text."""
    if urlsplit(url).scheme in {"http", "https"}:
        return (
            f'<a href="{escape(url, quote=True)}" target="_blank" rel="noopener noreferrer">'
            f"{escape(label)}</a>"
        )
    return escape(label)


def topbar_html(lang: str) -> str:
    return (
        '<div class="sov-topbar"><div class="sov-logo">◎</div>'
        f"<div><b>{escape(t(lang, 'app_title'))}</b>"
        f"<span>{escape(t(lang, 'app_subtitle'))}</span></div></div>"
    )


def hero_html(lang: str, simulated: bool) -> str:
    steps = "".join(
        f'<div class="sov-step"><div class="sov-num">{index}</div>'
        f"<b>{escape(t(lang, f'how_{index}_title'))}</b>"
        f"<span>{escape(t(lang, f'how_{index}_text'))}</span></div>"
        for index in (1, 2, 3)
    )
    badge = t(lang, "mode_simulated" if simulated else "mode_live")
    return (
        f'<div class="sov-hero"><h1>{escape(t(lang, "hero_title"))}</h1>'
        f"<p>{escape(t(lang, 'hero_text'))}</p>"
        f'<div class="sov-steps">{steps}</div>'
        f'<span class="sov-badge">{escape(badge)}</span></div>'
    )


def stage_states(view: RunView) -> dict[str, str]:
    """State of every stage: done, active, waiting, failed or pending."""
    states = dict.fromkeys(STAGE_ORDER, "pending")
    for event in view.events:
        node = "approval_gate" if event.node == "approval_request" else event.node
        if node not in states:
            continue
        if event.status in {"succeeded", "approved"}:
            states[node] = "done"
        elif event.status == "started" and states[node] != "done":
            states[node] = "active"
        elif event.status == "waiting_approval":
            states[node] = "waiting"
        elif event.status in {"failed", "rejected"}:
            states[node] = "failed"
    if view.phase in {"partial", "error"}:
        for node, value in states.items():
            if value == "active":
                states[node] = "failed"
    return states


def progress_html(view: RunView, lang: str) -> str:
    tone = PHASE_TONE[view.phase]
    message = t(lang, view.phase)
    if view.phase == "partial":
        message += " — " + friendly_failure(lang, view.failure_kind, view.failure_reason)
    elif view.phase == "error":
        message += " — " + friendly_failure(lang, "unexpected", view.error)
    banner = f'<div class="sov-banner {tone}">{escape(message)}</div>'
    if view.max_rounds:
        rounds = t(lang, "round_progress", done=view.rounds_done, total=view.max_rounds)
        banner += f'<p class="sov-muted" style="margin:8px 2px 0">{escape(rounds)}</p>'
    track = "".join(
        f'<div class="sov-stage {state}"><span class="dot"></span>{escape(stage(lang, node))}</div>'
        for node, state in stage_states(view).items()
    )
    share = min(1.0, view.spend_usd / view.cap_usd) if view.cap_usd else 0.0
    color = (
        "var(--sov-ok)" if share < 0.7 else "var(--sov-warn)" if share < 0.9 else "var(--sov-bad)"
    )
    spend = (
        f'<div class="sov-spend">{escape(t(lang, "spend"))}: <b>{view.spend_usd:.3f} USD</b> '
        f"{escape(t(lang, 'spend_of', cap=view.cap_usd))}"
        f'<div class="sov-bar"><div style="width:{share:.1%};background:{color}"></div></div></div>'
    )
    return f'{banner}<div class="sov-track">{track}</div>{spend}'


def draft_html(draft: str, objections: tuple[str, ...], lang: str) -> str:
    body = f'<div class="sov-draft">{escape(draft)}</div>'
    if objections:
        items = "".join(f"<li>{escape(item)}</li>" for item in objections)
        review = f"<p><b>{escape(t(lang, 'objections'))}</b></p><ul class='sov-list'>{items}</ul>"
    else:
        review = f"<p>✔ {escape(t(lang, 'no_objections'))}</p>"
    return f"<p><b>{escape(t(lang, 'draft'))}</b></p>{body}{review}"


def kpis_html(
    lang: str,
    *,
    mention: tuple[float, float, float],
    recommendation: float | None,
    answers: int,
    spend_usd: float,
) -> str:
    estimate, low, high = mention
    cards = [
        (
            "main",
            t(lang, "kpi_mention"),
            _pct(estimate),
            t(lang, "margin", low=_pct(low), high=_pct(high)),
        ),
        ("", t(lang, "kpi_recommend"), _pct(recommendation), t(lang, "kpi_mention_help")),
        ("", t(lang, "kpi_answers"), str(answers), ""),
        ("", t(lang, "kpi_spend"), _money(lang, spend_usd), ""),
    ]
    return (
        '<div class="sov-kpis">'
        + "".join(
            f'<div class="sov-kpi {cls}"><div class="label">{escape(label)}</div>'
            f'<div class="value">{escape(value)}</div><div class="hint">{escape(hint)}</div></div>'
            for cls, label, value, hint in cards
        )
        + "</div>"
    )


def summary_html(lang: str, brand: str, rows: list[tuple[str, float | None]]) -> str:
    """One or two sentences a non-technical reader understands."""
    mine = next((value for name, value in rows if name == brand), None)
    parts = []
    if mine is not None:
        parts.append(t(lang, "plain_sentence", brand=brand, of_ten=round(mine * 10)))
    known = [(name, value) for name, value in rows if value is not None]
    if known:
        leader, rate = max(known, key=lambda item: item[1])
        parts.append(
            t(lang, "plain_you_lead")
            if leader == brand
            else t(lang, "plain_leader", leader=leader, rate=_pct(rate))
        )
    return _markdown_bold(f'<div class="sov-summary">{escape(" ".join(parts))}</div>')


def _markdown_bold(text: str) -> str:
    """Turn the **bold** of our own texts into <b> after escaping."""
    pieces = text.split("**")
    return "".join(f"<b>{piece}</b>" if index % 2 else piece for index, piece in enumerate(pieces))


def comparison_html(
    lang: str, brand: str, rows: list[tuple[str, float | None, float | None, float | None]]
) -> str:
    """Tracked brands with a bar each; the client row is highlighted."""
    ordered = sorted(rows, key=lambda row: -(row[1] or 0))
    body = "".join(
        f'<tr class="{"me" if name == brand else ""}"><td>{escape(name)}</td>'
        f"{_bar_cell(value)}<td><b>{_pct(value)}</b></td>"
        f'<td class="sov-muted">{escape(t(lang, "margin", low=_pct(low), high=_pct(high)))}'
        "</td></tr>"
        for name, value, low, high in ordered
    )
    return f'<table class="sov-table">{body}</table>'


def _bar_cell(share: float | None) -> str:
    width = f"{min(1.0, max(0.0, share or 0.0)):.0%}"
    return f'<td class="barcell"><div class="sov-hbar"><div style="width:{width}"></div></div></td>'


def discovered_html(lang: str, rows: list[dict[str, str]], limit: int = 8) -> str:
    if not rows:
        return ""
    star = '<span class="sov-star">★</span> '
    body = "".join(
        f"<tr><td>{star if row.get('suggested') == 'True' else ''}{escape(row['name'])}"
        f"<div class='sov-muted'>{escape(kind(lang, row['kind']))}</div></td>"
        f"{_bar_cell(float(row['rate']))}<td><b>{_pct(float(row['rate']))}</b></td>"
        f"<td class='sov-muted'>"
        f"{escape(t(lang, 'in_questions', n=row.get('n_questions_with', '?')))}</td></tr>"
        for row in rows[:limit]
    )
    return (
        f"<h3>{escape(t(lang, 'discovered_title'))}</h3>"
        f"<p class='sov-muted'>{escape(t(lang, 'discovered_help'))}</p>"
        f'<table class="sov-table">{body}</table>'
    )


def research_html(lang: str, research: dict | None) -> str:
    if research is None:
        return ""
    sources = research.get("sources", [])
    items = "".join(f"<li>{_safe_link(url, url)}</li>" for url in sources)
    html = f"<h3>{escape(t(lang, 'sources_title'))}</h3>" + (
        f"<ul class='sov-list'>{items}</ul>" if items else f"<p>{escape(t(lang, 'no_sources'))}</p>"
    )
    checks = research.get("checks", [])
    if checks:
        lines = "".join(
            f"<li><b>{escape(check['name'])}</b>: "
            + (
                f"✔ {_safe_link(check['evidence_url'], t(lang, 'evidence'))}"
                if check["found_online"] and check.get("evidence_url")
                else escape(t(lang, "no_evidence"))
            )
            + "</li>"
            for check in checks
        )
        html += f"<h3>{escape(t(lang, 'checks_title'))}</h3><ul class='sov-list'>{lines}</ul>"
    return html


def results_html(lang: str, bundle: ResultsBundle) -> str:
    """Sample label, key figures, plain summary and brand comparison of one run."""
    view = bundle.view
    if view is None or view.headline is None:
        return f"<p>{escape(t(lang, 'results_pending'))}</p>"
    head = view.headline
    rows = [(row.brand, row.estimate, row.ci_low, row.ci_high) for row in view.table]
    label = t(
        lang,
        "sample_label",
        date=bundle.date,
        model=bundle.subject_model or "—",
        answers=head.n_answers,
        sector=bundle.sector,
        city=bundle.city,
    )
    return (
        f"<p class='sov-muted'>{escape(label)}</p>"
        + kpis_html(
            lang,
            mention=(head.estimate, head.ci_low, head.ci_high),
            recommendation=bundle.recommendation,
            answers=head.n_answers,
            spend_usd=bundle.spend_usd,
        )
        + summary_html(lang, bundle.brand, [(name, value) for name, value, _, _ in rows])
        + f"<h3>{escape(t(lang, 'by_brand'))}</h3>"
        + comparison_html(lang, bundle.brand, rows)
    )


def answer_html(lang: str, example: AnswerExample | None) -> str:
    """A real answer (untrusted model text): escaped first, then tracked brands highlighted."""
    if example is None:
        return f"<p class='sov-muted'>{escape(t(lang, 'example_none'))}</p>"
    text = escape(example.answer)
    named = [mention.brand for mention in example.mentions if mention.mentioned]
    if named:
        pattern = "|".join(re.escape(escape(name)) for name in sorted(named, key=len, reverse=True))
        text = re.sub(rf"(?<!\w)({pattern})(?!\w)", r"<mark>\1</mark>", text, flags=re.IGNORECASE)
    chips = "".join(
        f'<span class="sov-chip {"rec" if m.recommended else "named" if m.mentioned else ""}">'
        f"<b>{escape(m.brand)}</b> · "
        + escape(
            t(
                lang,
                "named_and_recommended"
                if m.recommended
                else "named_only"
                if m.mentioned
                else "not_named",
            )
        )
        + "</span>"
        for m in example.mentions
    )
    return (
        f"<p><b>{escape(t(lang, 'example_question'))}:</b> {escape(example.question)}</p>"
        f"<p><b>{escape(t(lang, 'example_answer'))}:</b></p>"
        f'<div class="sov-answer">{text}</div>'
        f"<p style='margin-top:12px'><b>{escape(t(lang, 'example_analysis'))}</b></p>"
        f'<div class="sov-chips">{chips}</div>'
    )


def panel_problems_text(lang: str, problems: list[PanelProblem]) -> str:
    """The problems of the user's questions, one line each, in plain words."""
    lines = []
    for problem in problems:
        message = t(lang, f"problem_{problem.kind}")
        if problem.row:
            message = t(lang, "problem_row", row=problem.row, problem=message)
        lines.append(f"- {message}")
    return "\n".join(lines)


_QUOTA_HINTS = ("per day", "quota", "rate limit", "ratelimit", "429")


def friendly_failure(lang: str, kind: str | None, reason: str | None) -> str:
    """What happened, in plain words. The technical reason stays in the log."""
    text = (reason or "").casefold()
    if any(hint in text for hint in _QUOTA_HINTS):
        return t(lang, "fail_quota")
    if kind == "budget_exhausted":
        return t(lang, "fail_budget")
    if kind == "no_evidence":
        return t(lang, "fail_no_evidence")
    if "sources" in text or "search" in text:
        return t(lang, "fail_sources")
    return t(lang, "fail_unexpected")


def insights_html(lang: str, bundle: ResultsBundle) -> str:
    """Actions that follow from the figures, by fixed rules: no model writes advice."""
    view = bundle.view
    if view is None or view.headline is None:
        return ""
    rows = {row.brand: row.estimate for row in view.table if row.estimate is not None}
    mine = rows.get(bundle.brand)
    actions: list[str] = []
    rec = bundle.recommendation
    if mine and rec is not None and rec < 0.6 * mine:
        actions.append(t(lang, "act_recommend_gap", m=_pct(mine), r=_pct(rec)))
    if rows:
        top = max(rows.values())
        leaders = [name for name, value in rows.items() if value == top]
        if bundle.brand in leaders and top > 0:
            actions.append(t(lang, "act_you_lead"))
        elif top > 0:
            actions.append(t(lang, "act_leader", leader=leaders[0], rate=_pct(top)))
    suggested = [row["name"] for row in bundle.discovered if row.get("suggested") == "True"][:3]
    if suggested:
        actions.append(t(lang, "act_suggested", names=", ".join(suggested)))
    if any("Few questions" in item for item in view.notes):
        actions.append(t(lang, "act_few"))
    half_width = (view.headline.ci_high - view.headline.ci_low) / 2
    if half_width > 0.15:
        actions.append(t(lang, "act_wide", hw=_pct(half_width)))
    if not actions:
        return ""
    items = "".join(f"<li>{escape(action)}</li>" for action in actions)
    return (
        f'<div class="sov-actions"><h3>{escape(t(lang, "actions_title"))}</h3>'
        f'<ul class="sov-list">{items}</ul></div>'
    )


def glossary_html(lang: str) -> str:
    items = []
    for key in ("g_named", "g_recommended", "g_margin", "g_sample"):
        term, _, meaning = t(lang, key).partition(": ")
        items.append(f"<li><b>{escape(term)}</b>: {escape(meaning)}</li>")
    return f'<ul class="sov-list">{"".join(items)}</ul>'


_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")


def safe_markdown(text: str) -> str:
    """A model draft as markdown that cannot carry HTML or load images (tracking pixels)."""
    without_images = _IMAGE.sub("", text)
    return escape(without_images, quote=False)


def objections_html(objections: tuple[str, ...], lang: str) -> str:
    """The Skeptic's open objections under the draft (escaped: they quote the draft)."""
    if not objections:
        return f"<p>✔ {escape(t(lang, 'no_objections'))}</p>"
    items = "".join(f"<li>{escape(item)}</li>" for item in objections)
    return f"<p><b>{escape(t(lang, 'objections'))}</b></p><ul class='sov-list'>{items}</ul>"


def trend_caption(
    lang: str,
    brand: str,
    city: str,
    model: str,
    previous: Rate,
    latest: Rate,
    *,
    previous_set: str = "A",
    latest_set: str = "A",
) -> str:
    """How to read the trend, and whether the last change is clear or may be chance.

    A change is only judged between runs that asked the same questions: otherwise the
    questions, not the brand, may explain it (an unknown set counts as a change).
    """
    how = t(lang, "trend_caption", brand=brand, city=city, model=model or "—")
    if previous_set != latest_set or "?" in (previous_set, latest_set):
        same_figure = _pct(previous[0]) == _pct(latest[0])
        verdict = t(
            lang,
            "trend_set_changed_same" if same_figure else "trend_set_changed",
            before_set=previous_set,
            after_set=latest_set,
            before=_pct(previous[0]),
            after=_pct(latest[0]),
        )
        return f"{how}\n\n{verdict}"
    change = compare(previous, latest)
    if change == "unclear" and _pct(previous[0]) == _pct(latest[0]):
        change = "same"  # "from 70% to 70%" reads as a mistake
    verdict = t(lang, f"trend_{change}", before=_pct(previous[0]), after=_pct(latest[0]))
    return f"{how}\n\n{verdict}"


def trend_table_html(
    lang: str,
    rows: list[tuple[str, str, int, Rate]],
    sets: dict[str, list[str]],
) -> str:
    """Every trend point with its question set, and the questions of each set to check them.

    ``rows`` are (date and time, set letter, number of questions, mention rate).
    """
    if not rows:
        return ""
    head = "".join(
        f"<th>{escape(t(lang, key))}</th>"
        for key in (
            "trend_table_date",
            "trend_table_set",
            "trend_table_named",
            "trend_table_margin",
        )
    )
    body = "".join(
        f"<tr><td>{escape(when)}</td>"
        f"<td>{escape(t(lang, 'trend_set_tag', letter=letter))}"
        f"<div class='sov-muted'>{escape(t(lang, 'trend_n_questions', n=n))}</div></td>"
        f"<td><b>{_pct(rate[0])}</b></td>"
        f"<td class='sov-muted'>{escape(t(lang, 'margin', low=_pct(rate[1]), high=_pct(rate[2])))}"
        "</td></tr>"
        for when, letter, n, rate in rows
    )
    details = "".join(
        f"<details><summary>{escape(t(lang, 'trend_set_summary', letter=letter, n=len(texts)))}"
        f"</summary><ol class='sov-list'>"
        + "".join(f"<li>{escape(text)}</li>" for text in texts)
        + "</ol></details>"
        for letter, texts in sets.items()
    )
    if any(letter == "?" for _, letter, _, _ in rows):
        details += f"<p class='sov-muted'>? = {escape(t(lang, 'trend_set_unknown'))}</p>"
    return (
        f'<table class="sov-table"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'
        f"{details}"
    )
