"""Gradio entry point (owner: An, task I6). Run ``python app.py`` and open the printed address.

The skeleton has three tabs, driven by the simulated graph of ``ui/mock_run.py``:

* Brief: the form. Start validates it as a ``Brief`` and starts one background run.
* Activity: the events of the run (task A4 adds live approval buttons).
* Results: a summary of the run (task D4 adds the report and downloads).

A spend bar and the run status stay above the tabs. A timer refreshes the page while a run is
active; the run itself is a background thread (``ui/run_manager.py``), so the page never blocks.
The server only listens on this computer.
"""

from typing import Any

import gradio as gr

from ai_share_of_voice.ui import brief_tab, render
from ai_share_of_voice.ui.run_manager import RunInProgress, RunManager

HOST = "127.0.0.1"
POLL_SECONDS = 1.0


def build_app(manager: RunManager | None = None) -> gr.Blocks:
    """Build the interface. Pass a ``manager`` to run something other than the simulated graph."""
    manager = manager or RunManager()

    def refresh() -> tuple[str, str, list[list[str]], str, gr.Timer]:
        snapshot = manager.snapshot()
        return (
            render.spend_bar(snapshot.spent_usd, snapshot.cap_usd),
            render.status_line(snapshot),
            render.events_rows(snapshot.events),
            render.results_text(snapshot),
            gr.Timer(active=snapshot.running),
        )

    def start_run(*form_values: Any) -> tuple[str, str, str, list[list[str]], str, gr.Timer]:
        result = brief_tab.build_brief(*form_values)
        if result.brief is None:
            problems = "\n".join(f"- {render.escape_markdown(text)}" for text in result.errors)
            message = f"**Fix these before starting:**\n\n{problems}"
        else:
            try:
                manager.start(result.brief)
                message = "Run started. Follow it in the Activity tab."
            except RunInProgress:
                message = "A run is already in progress."
        return (message, *refresh())

    with gr.Blocks() as demo:
        gr.Markdown("# AI Share of Voice")
        gr.Markdown(render.SIMULATED_NOTICE)
        spend = gr.Markdown(render.spend_bar(0.0, None))
        status = gr.Markdown(render.status_line(manager.snapshot()))
        with gr.Tabs():
            with gr.Tab("Brief"):
                form = brief_tab.build_brief_tab()
            with gr.Tab("Activity"):
                table = gr.Dataframe(
                    headers=render.EVENT_HEADERS, datatype="str", interactive=False, wrap=True
                )
            with gr.Tab("Results"):
                results = gr.Markdown(render.results_text(manager.snapshot()))
        timer = gr.Timer(POLL_SECONDS, active=False)
        live = [spend, status, table, results, timer]
        form.start.click(start_run, inputs=form.inputs, outputs=[form.message, *live])
        timer.tick(refresh, outputs=live)
        demo.load(refresh, outputs=live)  # a reloaded page picks up a run that is still going
    return demo


def main() -> None:
    """Start the app on this computer only."""
    build_app().launch(server_name=HOST)


if __name__ == "__main__":
    main()
