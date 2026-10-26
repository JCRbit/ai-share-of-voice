"""Mock run generator (Task I2) to unblock frontend development."""

from uuid import uuid4

from ai_share_of_voice.schemas.analysis import AnalyzedResponse, BrandMention, Tone
from ai_share_of_voice.schemas.events import EventStatus, GraphEvent


def simulate_run(run_id: str = None) -> list[GraphEvent]:
    if not run_id:
        run_id = str(uuid4())

    events = []
    current_spend = 0.0

    # 1. Question Generation Step
    events.append(
        GraphEvent(
            run_id=run_id,
            node="questions",
            agent="Question generator",
            status=EventStatus.STARTED,
            cumulative_spend_usd=current_spend,
        )
    )

    current_spend += 0.05
    events.append(
        GraphEvent(
            run_id=run_id,
            node="questions",
            agent="Question generator",
            status=EventStatus.SUCCEEDED,
            result="Generated 30 sector questions.",
            cumulative_spend_usd=current_spend,
        )
    )

    # 2. Sampling & Extractor Step (Research)
    mock_brand_a = BrandMention(
        brand="Brand A", mentioned=True, recommended=True, position=1, tone=Tone.POSITIVE
    )

    mock_brand_b = BrandMention(
        brand="Brand B", mentioned=False, recommended=False, position=None, tone=None
    )

    mock_analysis = AnalyzedResponse(
        response_id="resp-001",
        question_id="q-001",
        batch_index=0,
        mentions=[mock_brand_a, mock_brand_b],
        sources=["https://example-source.com/article1"],
    )

    events.append(
        GraphEvent(
            run_id=run_id,
            node="research",
            agent="Mention extractor",
            tool="Pydantic tool",
            status=EventStatus.SUCCEEDED,
            result=mock_analysis.model_dump_json(),
            cumulative_spend_usd=current_spend + 0.15,
        )
    )
    current_spend += 0.15

    # 3. Skeptic Review Loop
    events.append(
        GraphEvent(
            run_id=run_id,
            node="skeptic",
            agent="Skeptic",
            status=EventStatus.SUCCEEDED,
            result="Draft passes all causality checks. No objections.",
            cumulative_spend_usd=current_spend + 0.15,
        )
    )
    current_spend += 0.15

    # 4. Final Approval Pause
    events.append(
        GraphEvent(
            run_id=run_id,
            node="approval_gate",
            status=EventStatus.WAITING_APPROVAL,
            result="Awaiting human confirmation before publishing report.",
            cumulative_spend_usd=current_spend,
        )
    )

    return events


if __name__ == "__main__":
    # Test the generator
    for event in simulate_run():
        print(f"[{event.status}] {event.node}: {event.result or event.reason}")
