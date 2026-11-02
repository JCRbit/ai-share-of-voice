"""AI Share of Voice app (I6, local integration): Brief, Activity and approval, Results.

Run it with ``python app.py`` and open http://127.0.0.1:7860. It reads ``.env``; with
``python app.py --simulated`` (or ``LLM_MODE=simulated``) every answer is a deterministic
demo answer, there is no web search and nothing is spent.

The UI only draws what ``RunController`` returns; all the logic is tested there.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# The package is not installed: make src/ importable when run as a plain script.
sys.path.insert(0, str(ROOT / "src"))

import gradio as gr  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from ai_share_of_voice.config import ConfigError, LlmMode, load_settings  # noqa: E402
from ai_share_of_voice.tools.mcp_sources_client import open_stdio_sources  # noqa: E402
from ai_share_of_voice.ui.results_tab import table_rows  # noqa: E402
from ai_share_of_voice.ui.run_controller import BriefForm, RunController, RunView  # noqa: E402

RUNS_DIR = ROOT / "runs"
DATA_DIR = ROOT / "data"
EVENT_HEADERS = ["Time", "Status", "Step", "Detail"]
PHASE_LABELS = {
    "running": "⏳ Running",
    "waiting_approval": "✋ Waiting for your approval",
    "completed": "✅ Completed: report written",
    "partial": "⚠️ Closed in a partial state",
    "rejected": "⛔ Rejected: nothing was published",
    "error": "❌ Error",
}


def build_controller() -> RunController:
    settings = load_settings()
    # The MCP server only gets what it needs to search: never the model keys.
    server_env = {"LLM_MODE": "simulated", "SEARCH_PROVIDER": settings.search_provider.value}
    if settings.search_api_key is not None:
        server_env["SEARCH_API_KEY"] = settings.search_api_key.get_secret_value()
    return RunController(
        settings,
        runs_dir=RUNS_DIR,
        data_dir=DATA_DIR,
        sources_factory=lambda: open_stdio_sources(server_env),
    )


def spend_bar(view: RunView) -> str:
    share = min(1.0, view.spend_usd / view.cap_usd) if view.cap_usd else 0.0
    color = "#2e7d32" if share < 0.7 else "#ef6c00" if share < 0.9 else "#c62828"
    return (
        f"<div style='font-size:14px;margin-bottom:4px'>Spend: <b>{view.spend_usd:.4f} USD</b>"
        f" of {view.cap_usd:.2f} USD cap</div>"
        "<div style='background:#e0e0e0;border-radius:6px;height:12px;width:100%'>"
        f"<div style='background:{color};width:{share:.1%};height:12px;border-radius:6px'></div>"
        "</div>"
    )


def build_app(controller: RunController) -> gr.Blocks:
    settings = controller.settings
    mode = "SIMULATED (no spending)" if settings.mode is LlmMode.SIMULATED else "LIVE"
    header = (
        "# AI Share of Voice\n"
        "How often AI assistants mention and recommend a brand, with intervals, sources and "
        "a human approval before anything is published.\n\n"
        f"**Mode:** {mode} · **Measured model:** `{settings.subject_model}` · "
        f"**Agents:** `{settings.agent_model}` · **Search:** {settings.search_provider} · "
        f"**Cap per run:** {settings.spend_cap_usd:.2f} USD"
    )

    with gr.Blocks(title="AI Share of Voice") as app:
        run_id = gr.State(None)
        gr.Markdown(header)

        with gr.Tab("1 · Brief"):
            with gr.Row():
                brand = gr.Textbox(label="Client brand", value="Glovo")
                competitors = gr.Textbox(
                    label="Competitors (2–5, separated by commas)", value="Just Eat, Uber Eats"
                )
            with gr.Row():
                sector = gr.Textbox(label="Sector", value="food delivery")
                city = gr.Textbox(label="City", value="Madrid")
                country = gr.Textbox(label="Country (helps web searches)", value="España")
            with gr.Row():
                language = gr.Radio(["es", "en"], value="es", label="Language of the questions")
                panel = gr.Radio(
                    [
                        ("Demo panel (food delivery, Madrid)", "demo"),
                        ("Generate for this brief", "generate"),
                    ],
                    value="demo",
                    label="Question panel",
                )
            with gr.Row():
                questions = gr.Slider(3, 30, value=10, step=1, label="Questions per batch")
                batches = gr.Slider(2, 10, value=3, step=1, label="Maximum batches")
            start = gr.Button("Start measurement", variant="primary")
            start_message = gr.Markdown()

        with gr.Tab("2 · Activity and approval"):
            phase = gr.Markdown("No run yet. Fill in the Brief and start a measurement.")
            spend = gr.HTML()
            events = gr.Dataframe(headers=EVENT_HEADERS, interactive=False, wrap=True)
            with gr.Group(visible=False) as approval:
                gr.Markdown("### Human approval\nNothing is published until you decide.")
                draft = gr.Markdown()
                objections = gr.Markdown()
                comment = gr.Textbox(label="Comment (optional)")
                with gr.Row():
                    approve = gr.Button("Approve and publish", variant="primary")
                    reject = gr.Button("Reject", variant="stop")
            timer = gr.Timer(1.5)

        with gr.Tab("3 · Results"):
            refresh = gr.Button("Load results")
            headline = gr.Markdown()
            table = gr.Dataframe(
                headers=["Brand", "Estimate (95% interval)", "Interval method"], interactive=False
            )
            charts = gr.Gallery(label="Charts", columns=2, height=420, object_fit="contain")
            gr.Markdown("### Businesses the AI names that you do not track")
            discovered = gr.Dataframe(interactive=False, wrap=True)
            research = gr.Markdown()
            downloads = gr.File(label="Downloads", file_count="multiple")

        def on_start(*values):
            names = ["brand", "competitors", "sector", "city", "country", "language", "panel"]
            data = dict(zip(names, values[:7], strict=True))
            data.update(questions=int(values[7]), batches=int(values[8]))
            try:
                new_id = controller.start(BriefForm(**data))
            except (ValidationError, ValueError) as error:
                return None, f"**Check the form:** {error}"
            return new_id, f"Started `{new_id}`. Follow it in **Activity and approval**."

        start.click(
            on_start,
            [brand, competitors, sector, city, country, language, panel, questions, batches],
            [run_id, start_message],
        )

        def on_tick(current):
            if not current:
                return gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()
            view = controller.view(current)
            label = PHASE_LABELS[view.phase]
            if view.error:
                label += f": {view.error}"
            rows = [[e.time, e.status, e.node, e.detail or ""] for e in view.events]
            waiting = view.phase == "waiting_approval"
            open_objections = (
                "**Open objections from the Skeptic:**\n"
                + "\n".join(f"- {o}" for o in view.objections)
                if view.objections
                else "The Skeptic has no open objections."
            )
            return (
                f"**Run** `{view.run_id}` · {label}",
                spend_bar(view),
                rows,
                gr.update(visible=waiting),
                f"**Draft:**\n\n{view.draft}" if waiting else "",
                open_objections if waiting else "",
            )

        timer.tick(on_tick, run_id, [phase, spend, events, approval, draft, objections])

        def decide(current, note, approved):
            try:
                controller.decide(current, approved=approved, comment=note)
            except (ValueError, KeyError, TypeError) as error:
                return f"Cannot record the decision: {error}"
            return "Decision recorded. The run continues."

        approve.click(lambda c, n: decide(c, n, True), [run_id, comment], start_message)
        reject.click(lambda c, n: decide(c, n, False), [run_id, comment], start_message)

        def on_refresh(current):
            if not current:
                return "Start a measurement first.", [], [], [], "", None
            bundle = controller.results(current)
            if bundle.view is None:
                text = "\n".join(bundle.notes) or "No results yet."
                return text, [], [], [], "", None
            view = bundle.view
            head = view.headline
            lines = [f"### {view.sample_label}"]
            if head is not None:
                lines.append(
                    f"**Mention rate of the client:** {head.estimate:.0%} "
                    f"(95% interval {head.ci_low:.0%}–{head.ci_high:.0%}) · "
                    f"{head.n_answers} answers, {head.n_questions} questions"
                )
            lines += [f"- {note}" for note in view.notes]
            ranking = [
                [
                    row["name"],
                    row["kind"],
                    f"{float(row['rate']):.0%}",
                    f"{float(row['ci_low']):.0%}–{float(row['ci_high']):.0%}",
                    row["n_questions_with"],
                    "yes" if row["suggested"] == "True" else "no",
                ]
                for row in bundle.discovered
            ]
            research_text = ""
            if bundle.research:
                sources = bundle.research.get("sources", [])
                checks = bundle.research.get("checks", [])
                research_text = "### Sources read (untrusted data)\n" + (
                    "\n".join(f"- {url}" for url in sources) or "No sources."
                )
                if checks:
                    research_text += (
                        "\n\n### Do the most named businesses exist online?\n"
                        + "\n".join(
                            f"- **{c['name']}**: "
                            + (
                                f"evidence at {c['evidence_url']}"
                                if c["found_online"]
                                else "no evidence found (not proof it does not exist)"
                            )
                            for c in checks
                        )
                    )
            return (
                "\n\n".join(lines),
                table_rows(view),
                list(view.chart_paths),
                gr.update(
                    value=ranking,
                    headers=["Business", "Kind", "Rate", "95% interval", "Questions", "Suggested"],
                ),
                research_text,
                list(view.downloads) or None,
            )

        refresh.click(
            on_refresh, run_id, [headline, table, charts, discovered, research, downloads]
        )

    return app


def main() -> None:
    if "--simulated" in sys.argv[1:]:
        # Process variables win over .env: a free demo without editing .env.
        os.environ.update(LLM_MODE="simulated", SEARCH_PROVIDER="none")
    try:
        controller = build_controller()
    except ConfigError as error:
        raise SystemExit(f"config error: {error}") from None
    build_app(controller).launch(server_port=7860, allowed_paths=[str(RUNS_DIR)])


if __name__ == "__main__":
    main()
