"""What the pages show, as plain strings and rows (owner: An, task I6).

No Gradio here, so the skeleton's content is tested directly. A4 and D4 replace the activity
and results content; the spend bar stays. Text that can come from a model or a web page is
shown in a table of plain text, and anything put in Markdown is escaped first.
"""

import re
from collections.abc import Sequence
from datetime import UTC

from ai_share_of_voice.schemas.events import GraphEvent
from ai_share_of_voice.ui.run_manager import (
    COMPLETED,
    FAILED,
    IDLE,
    PARTIAL,
    REJECTED,
    RUNNING,
    WAITING_APPROVAL,
    RunSnapshot,
)

SIMULATED_NOTICE = (
    "**Simulated mode.** Every step is a stand-in: no model, web source or image was used, "
    "so nothing shown here is a real measurement."
)
EVENT_HEADERS = ["Time (UTC)", "Node", "Agent", "Tool", "Status", "Detail", "Spend (USD)"]
STATUS_LABELS = {
    IDLE: "no run yet",
    RUNNING: "running",
    WAITING_APPROVAL: "waiting for approval",
    COMPLETED: "completed",
    PARTIAL: "closed in a partial state",
    REJECTED: "closed without publishing",
    FAILED: "failed",
}
BAR_WIDTH = 20
NEAR_CAP_RATIO = 0.8
MAX_DETAIL_CHARS = 160
_MARKDOWN_SPECIALS = re.compile(r"([\\`*_{}\[\]()#+\-.!|<>~&])")


def escape_markdown(text: str) -> str:
    """Escape Markdown and HTML characters so ``text`` is shown literally."""
    return _MARKDOWN_SPECIALS.sub(r"\\\1", text)


def shorten(text: str, limit: int = MAX_DETAIL_CHARS) -> str:
    """One line of at most ``limit`` characters."""
    clean = " ".join(text.split())
    return clean if len(clean) <= limit else clean[: limit - 3] + "..."


def spend_bar(spent_usd: float, cap_usd: float | None) -> str:
    """Spend against the cap, as a text bar."""
    if cap_usd is None or cap_usd <= 0:
        return "**Spend:** no run yet"
    ratio = max(0.0, spent_usd / cap_usd)
    filled = round(min(ratio, 1.0) * BAR_WIDTH)
    bar = "█" * filled + "░" * (BAR_WIDTH - filled)
    line = f"**Spend:** `{bar}` ${spent_usd:.2f} of ${cap_usd:.2f} ({ratio:.0%})"
    if ratio >= 1.0:
        return line + " — cap reached"
    return line + (" — near the cap" if ratio >= NEAR_CAP_RATIO else "")


def events_rows(events: Sequence[GraphEvent]) -> list[list[str]]:
    """One row per event, in order, for the activity table."""
    rows = []
    for event in events:
        spend = event.cumulative_spend_usd
        rows.append(
            [
                event.timestamp.astimezone(UTC).strftime("%H:%M:%S"),
                event.node,
                event.agent or "",
                event.tool or "",
                event.status.value,
                shorten(event.result or event.reason or ""),
                "" if spend is None else f"{spend:.2f}",
            ]
        )
    return rows


def status_line(snapshot: RunSnapshot) -> str:
    """Status of the run, in one line."""
    label = STATUS_LABELS.get(snapshot.status) or escape_markdown(snapshot.status)
    text = f"**Status:** {label}"
    if snapshot.run_id:
        text += f" — run `{snapshot.run_id}`"
    return text


def results_text(snapshot: RunSnapshot) -> str:
    """Text of the results tab while a run has no analyzed answers."""
    if snapshot.status == IDLE:
        return "No run yet. Fill in the Brief tab and press **Start run**."
    if snapshot.status == RUNNING:
        return "The run is in progress. Its results will appear here when it ends."

    lines = [
        f"**Run `{snapshot.run_id}` ended:** "
        f"{STATUS_LABELS.get(snapshot.status, snapshot.status)}.",
        f"{len(snapshot.events)} events, spend **${snapshot.spent_usd:.2f}**.",
        (
            "**This was a simulation.** It demonstrates the workflow, "
            "but does not query real AI models or produce analyzed answers."
        ),
    ]
    if snapshot.error:
        lines.append(f"Error: {escape_markdown(shorten(snapshot.error))}")
    lines.extend(
        [
            "**No results to display yet:** this run saved no analyzed answers "
            "(`answers.csv`), so there are no metrics, charts or downloads.",
            (
                "**Next step:** run the real analysis workflow to collect "
                "and analyze answers. Results will appear here once "
                "`answers.csv` has been generated."
            ),
        ]
    )
    return "\n\n".join(lines)
