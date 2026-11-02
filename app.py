"""AI Share of Voice app (I6, local integration), for non-technical users, in Spanish/English.

Run it with ``python app.py`` and open http://127.0.0.1:7860. It reads ``.env``; with
``python app.py --simulated`` (or ``LLM_MODE=simulated``) every answer is a deterministic
demo answer, there is no web search and nothing is spent.

The UI only draws: the logic is in ``ui/run_controller.py``, the texts in ``ui/i18n.py`` and
the HTML pieces in ``ui/components.py``, all tested without a browser.
"""

import atexit
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
from ai_share_of_voice.ui import components as ui  # noqa: E402
from ai_share_of_voice.ui.i18n import event_status, note, stage, t  # noqa: E402
from ai_share_of_voice.ui.port_guard import PortGuard  # noqa: E402
from ai_share_of_voice.ui.run_controller import BriefForm, RunController  # noqa: E402

RUNS_DIR = ROOT / "runs"
DATA_DIR = ROOT / "data"
DEFAULT_PORT = 7860
LANGUAGES = [("Español", "es"), ("English", "en")]
THEME = gr.themes.Soft(
    primary_hue="indigo",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
)


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


def friendly_error(error: Exception) -> str:
    """Field errors of the form without pydantic's technical wording."""
    if isinstance(error, ValidationError):
        return "; ".join(str(item["msg"]).removeprefix("Value error, ") for item in error.errors())
    return str(error)


def build_app(controller: RunController) -> gr.Blocks:
    settings = controller.settings
    simulated = settings.mode is LlmMode.SIMULATED

    def tech_details(lang: str) -> str:
        return (
            f"- **{t(lang, 'measured_model')}:** `{settings.subject_model}`\n"
            f"- **{t(lang, 'agent_model')}:** `{settings.agent_model}`\n"
            f"- **{t(lang, 'search')}:** {settings.search_provider}\n"
            f"- **{t(lang, 'cap')}:** {settings.spend_cap_usd:.2f} USD"
        )

    def estimate(lang: str, questions: float, batches: float) -> str:
        return t(
            lang,
            "estimate",
            answers=int(questions) * int(batches),
            cap=settings.spend_cap_usd,
        )

    with gr.Blocks(title="AI Share of Voice") as app:
        lang = gr.State("es")
        run_id = gr.State(None)

        with gr.Row(equal_height=True):
            with gr.Column(scale=5):
                topbar = gr.HTML(ui.topbar_html("es"))
            with gr.Column(scale=1, min_width=200):
                language = gr.Radio(
                    LANGUAGES, value="es", label=t("es", "language"), container=False
                )

        with gr.Tabs() as tabs:
            # --- 1 · Your brand --------------------------------------------------------------
            with gr.Tab(t("es", "tab_brief"), id="brief") as tab_brief:
                hero = gr.HTML(ui.hero_html("es", simulated))
                market_title = gr.Markdown(f"### {t('es', 'section_market')}")
                with gr.Row():
                    brand = gr.Textbox(
                        label=t("es", "brand"),
                        value="Glovo",
                        placeholder=t("es", "brand_placeholder"),
                    )
                    competitors = gr.Textbox(
                        label=t("es", "competitors"),
                        info=t("es", "competitors_info"),
                        value="Just Eat, Uber Eats",
                    )
                with gr.Row():
                    sector = gr.Textbox(
                        label=t("es", "sector"),
                        info=t("es", "sector_info"),
                        value="comida a domicilio",
                    )
                    city = gr.Textbox(label=t("es", "city"), value="Madrid")
                    country = gr.Textbox(label=t("es", "country"), value="España")
                with gr.Accordion(t("es", "advanced"), open=False) as advanced:
                    advanced_help = gr.Markdown(t("es", "advanced_help"))
                    with gr.Row():
                        panel = gr.Radio(
                            [
                                (t("es", "panel_demo"), "demo"),
                                (t("es", "panel_generate"), "generate"),
                            ],
                            value="demo",
                            label=t("es", "panel"),
                        )
                        question_language = gr.Radio(
                            ["es", "en"], value="es", label=t("es", "question_language")
                        )
                    with gr.Row():
                        questions = gr.Slider(3, 30, value=10, step=1, label=t("es", "questions"))
                        batches = gr.Slider(2, 10, value=3, step=1, label=t("es", "batches"))
                cost = gr.Markdown(estimate("es", 10, 3))
                start = gr.Button(t("es", "start"), variant="primary", size="lg")
                start_message = gr.Markdown()

            # --- 2 · Progress ----------------------------------------------------------------
            with gr.Tab(t("es", "tab_activity"), id="activity") as tab_activity:
                progress = gr.HTML(f"<p>{t('es', 'no_run')}</p>")
                with gr.Group(visible=False) as approval:
                    approval_title = gr.Markdown(
                        f"### {t('es', 'approval_title')}\n{t('es', 'approval_help')}"
                    )
                    draft = gr.HTML()
                    comment = gr.Textbox(label=t("es", "comment"), lines=2)
                    with gr.Row():
                        approve = gr.Button(t("es", "approve"), variant="primary")
                        reject = gr.Button(t("es", "reject"), variant="stop")
                decision_message = gr.Markdown()
                with gr.Accordion(t("es", "log"), open=False) as log_box:
                    log = gr.Dataframe(interactive=False, wrap=True)
                timer = gr.Timer(1.5)

            # --- 3 · Results -----------------------------------------------------------------
            with gr.Tab(t("es", "tab_results"), id="results") as tab_results:
                refresh = gr.Button(t("es", "load_results"), size="sm")
                results = gr.HTML(f"<p>{t('es', 'results_empty')}</p>")
                discovered = gr.HTML()
                research = gr.HTML()
                notes = gr.Markdown()
                with gr.Accordion(t("es", "charts"), open=False) as charts_box:
                    charts = gr.Gallery(
                        show_label=False, columns=2, height=380, object_fit="contain"
                    )
                downloads = gr.File(label=t("es", "downloads"), file_count="multiple")

        with gr.Accordion(t("es", "tech_details"), open=False) as tech_box:
            tech = gr.Markdown(tech_details("es"))

        # --- start ------------------------------------------------------------------------
        def on_start(current_lang, *values):
            names = ["brand", "competitors", "sector", "city", "country", "language", "panel"]
            data = dict(zip(names, values[:7], strict=True))
            data.update(questions=int(values[7]), batches=int(values[8]))
            try:
                new_id = controller.start(BriefForm(**data))
            except (ValidationError, ValueError) as error:
                return (
                    gr.skip(),
                    t(current_lang, "form_error", error=friendly_error(error)),
                    gr.skip(),
                )
            return new_id, t(current_lang, "started"), gr.Tabs(selected="activity")

        start.click(
            on_start,
            [
                lang,
                brand,
                competitors,
                sector,
                city,
                country,
                question_language,
                panel,
                questions,
                batches,
            ],
            [run_id, start_message, tabs],
        )
        for slider in (questions, batches):
            slider.change(estimate, [lang, questions, batches], cost)

        # --- progress ---------------------------------------------------------------------
        def on_tick(current, current_lang):
            if not current:
                return gr.skip(), gr.skip(), gr.skip(), gr.skip()
            view = controller.view(current)
            waiting = view.phase == "waiting_approval"
            rows = [
                [
                    e.time,
                    event_status(current_lang, e.status),
                    stage(current_lang, e.node),
                    e.detail or "",
                ]
                for e in view.events
            ]
            headers = [
                t(current_lang, key) for key in ("log_time", "log_status", "log_step", "log_detail")
            ]
            return (
                ui.progress_html(view, current_lang),
                gr.update(visible=waiting),
                ui.draft_html(view.draft, view.objections, current_lang) if waiting else "",
                gr.update(value=rows, headers=headers),
            )

        timer.tick(on_tick, [run_id, lang], [progress, approval, draft, log])

        def decide(current, current_lang, text, approved):
            try:
                controller.decide(current, approved=approved, comment=text)
            except (ValueError, KeyError, TypeError) as error:
                return t(current_lang, "decision_error", error=error), gr.skip()
            return t(current_lang, "decision_ok"), gr.update(visible=False)

        approve.click(
            lambda c, lg, x: decide(c, lg, x, True),
            [run_id, lang, comment],
            [decision_message, approval],
        )
        reject.click(
            lambda c, lg, x: decide(c, lg, x, False),
            [run_id, lang, comment],
            [decision_message, approval],
        )

        # --- results ----------------------------------------------------------------------
        def render_results(current, current_lang):
            """Results of the run, computed once: html, businesses, sources, notes, files,
            charts."""
            if not current:
                empty = f"<p>{t(current_lang, 'results_empty')}</p>"
                return empty, "", "", "", None, []
            bundle = controller.results(current)
            view = bundle.view
            notes_text = ""
            if view is not None and view.notes:
                notes_text = f"**{t(current_lang, 'notes_title')}**\n" + "\n".join(
                    f"- {note(current_lang, item)}" for item in view.notes
                )
            return (
                ui.results_html(current_lang, bundle),
                ui.discovered_html(current_lang, bundle.discovered),
                ui.research_html(current_lang, bundle.research),
                notes_text,
                list(view.downloads) if view and view.downloads else None,
                list(view.chart_paths) if view else [],
            )

        result_outputs = [results, discovered, research, notes, downloads, charts]
        refresh.click(render_results, [run_id, lang], result_outputs)
        tab_results.select(render_results, [run_id, lang], result_outputs)

        # --- language ---------------------------------------------------------------------
        def switch_language(new, panel_value, current):
            labels = (
                new,
                ui.topbar_html(new),
                ui.hero_html(new, simulated),
                gr.update(label=t(new, "language")),
                gr.update(label=t(new, "tab_brief")),
                gr.update(label=t(new, "tab_activity")),
                gr.update(label=t(new, "tab_results")),
                f"### {t(new, 'section_market')}",
                gr.update(label=t(new, "brand"), placeholder=t(new, "brand_placeholder")),
                gr.update(label=t(new, "competitors"), info=t(new, "competitors_info")),
                gr.update(label=t(new, "sector"), info=t(new, "sector_info")),
                gr.update(label=t(new, "city")),
                gr.update(label=t(new, "country")),
                gr.update(label=t(new, "advanced")),
                t(new, "advanced_help"),
                gr.update(
                    label=t(new, "panel"),
                    choices=[
                        (t(new, "panel_demo"), "demo"),
                        (t(new, "panel_generate"), "generate"),
                    ],
                    value=panel_value,
                ),
                gr.update(label=t(new, "question_language")),
                gr.update(label=t(new, "questions")),
                gr.update(label=t(new, "batches")),
                gr.update(value=t(new, "start")),
                f"### {t(new, 'approval_title')}\n{t(new, 'approval_help')}",
                gr.update(label=t(new, "comment")),
                gr.update(value=t(new, "approve")),
                gr.update(value=t(new, "reject")),
                gr.update(label=t(new, "log")),
                gr.update(value=t(new, "load_results")),
                gr.update(label=t(new, "charts")),
                gr.update(label=t(new, "tech_details")),
                tech_details(new),
            )
            html, found, sources, notes_text, files, _charts = render_results(current, new)
            waiting_text = f"<p>{t(new, 'no_run')}</p>" if not current else gr.skip()
            return (
                *labels,
                waiting_text,
                html,
                found,
                sources,
                notes_text,
                gr.update(label=t(new, "downloads"), value=files),
            )

        language.change(
            switch_language,
            [language, panel, run_id],
            [
                lang,
                topbar,
                hero,
                language,
                tab_brief,
                tab_activity,
                tab_results,
                market_title,
                brand,
                competitors,
                sector,
                city,
                country,
                advanced,
                advanced_help,
                panel,
                question_language,
                questions,
                batches,
                start,
                approval_title,
                comment,
                approve,
                reject,
                log_box,
                refresh,
                charts_box,
                tech_box,
                tech,
                progress,
                results,
                discovered,
                research,
                notes,
                downloads,
            ],
        ).then(estimate, [lang, questions, batches], cost)

    return app


def main() -> None:
    if "--simulated" in sys.argv[1:]:
        # Process variables win over .env: a free demo without editing .env.
        os.environ.update(LLM_MODE="simulated", SEARCH_PROVIDER="none")
    try:
        controller = build_controller()
    except ConfigError as error:
        raise SystemExit(f"config error: {error}") from None
    # Close an old instance of this app that still holds the port (never another program).
    guard = PortGuard(RUNS_DIR / "app.pid")
    port = guard.claim(DEFAULT_PORT)
    guard.record(port)
    atexit.register(guard.release)
    print(f"AI Share of Voice: http://127.0.0.1:{port}")
    build_app(controller).launch(
        server_port=port, allowed_paths=[str(RUNS_DIR)], theme=THEME, css=ui.APP_CSS
    )


if __name__ == "__main__":
    main()
