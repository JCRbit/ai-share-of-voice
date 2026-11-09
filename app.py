"""Gradio entry point (owner: An, task I6). Run ``python app.py`` and open the printed address.

The skeleton has three tabs, driven by the simulated graph of ``ui/mock_run.py``:

* Brief: the form. Start validates it as a ``Brief`` and starts one background run.
* Activity: the events of the run (task A4 adds live approval buttons).
* Results: the figures with their intervals, the charts and the downloads of a run, read from
  its folder in ``runs/`` (task D4). A run that saved no analyzed answers, such as the simulated
  one, has none, and the tab says so.

A spend bar and the run status stay above the tabs. A timer refreshes the page while a run is
active; the run itself is a background thread (``ui/run_manager.py``), so the page never blocks.
The server only listens on this computer.
"""

from pathlib import Path
from typing import Any

import gradio as gr

from ai_share_of_voice.ui import brief_tab, render, results_tab
from ai_share_of_voice.ui.run_manager import RunInProgress, RunManager, RunSnapshot

HOST = "127.0.0.1"
POLL_SECONDS = 1.0
RUNS_DIR = Path("runs")


def _results_outputs(view: results_tab.ResultsView) -> tuple[Any, ...]:
    """The tab's outputs for ``view``; empty charts and files reset their components."""
    return view.summary, view.rows, view.charts or None, view.files or None


def build_app(manager: RunManager | None = None, runs_dir: Path | str = RUNS_DIR) -> gr.Blocks:
    """Build the interface. Pass a ``manager`` to run something other than the simulated graph."""
    manager = manager or RunManager()
    runs = Path(runs_dir)

    def head(snapshot: RunSnapshot) -> tuple[str, str, list[list[str]], gr.Timer]:
        return (
            render.spend_bar(snapshot.spent_usd, snapshot.cap_usd),
            render.status_line(snapshot),
            render.events_rows(snapshot.events),
            gr.Timer(active=snapshot.running),
        )

    def ended_run(snapshot: RunSnapshot) -> tuple[Any, ...] | None:
        """The run menu and the results to show once a run has ended, else ``None``."""
        ended = results_tab.view_for_ended_run(runs, snapshot)
        if ended is None:
            return None
        run_id, view = ended
        if run_id:
            menu = gr.Dropdown(choices=results_tab.list_runs(runs), value=run_id)
        else:
            menu = gr.skip()
        return (menu, *_results_outputs(view))

    def refresh() -> tuple[Any, ...]:
        """Timer and start: results change only when a run has just ended, never while it runs."""
        snapshot = manager.snapshot()
        return (*head(snapshot), *(ended_run(snapshot) or (gr.skip(),) * 5))

    def load() -> tuple[Any, ...]:
        """Page load: the run that ended, else the newest run that has results."""
        snapshot = manager.snapshot()
        shown = ended_run(snapshot)
        if shown is None:
            choices = results_tab.list_runs(runs)
            newest = choices[0] if choices else None
            view = results_tab.show_results(runs, newest)
            shown = (gr.Dropdown(choices=choices, value=newest), *_results_outputs(view))
        return (*head(snapshot), *shown)

    def show(run_id: str | None) -> tuple[Any, ...]:
        return _results_outputs(results_tab.show_results(runs, run_id))

    def reload_runs(run_id: str | None) -> tuple[Any, ...]:
        choices = results_tab.list_runs(runs)
        chosen = run_id if run_id in choices else (choices[0] if choices else None)
        return (gr.Dropdown(choices=choices, value=chosen), *show(chosen))

    def start_run(*form_values: Any) -> tuple[Any, ...]:
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
                panel = results_tab.build_results_tab(runs, render.results_text(manager.snapshot()))
        timer = gr.Timer(POLL_SECONDS, active=False)
        live = [spend, status, table, timer, panel.run, *panel.outputs]
        form.start.click(start_run, inputs=form.inputs, outputs=[form.message, *live])
        timer.tick(refresh, outputs=live)
        demo.load(load, outputs=live)  # a reloaded page picks up a run that is still going
        panel.run.change(show, inputs=panel.run, outputs=panel.outputs)
        panel.refresh.click(reload_runs, inputs=panel.run, outputs=[panel.run, *panel.outputs])
    return demo


def main() -> None:
    """Start the app on this computer only, allowing it to serve the files of ``runs/``."""
    build_app().launch(server_name=HOST, allowed_paths=[str(RUNS_DIR.resolve())])


if __name__ == "__main__":
    main()
