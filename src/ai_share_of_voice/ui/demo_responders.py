"""Deterministic answers for ``LLM_MODE=simulated`` (UI work and tests, no spending).

They follow the brief, so a simulated run looks like a real one: the measured model names
some tracked brands and one local business, the judge recommends what is named, discovery
lists the names, and the panel generator writes neutral questions for every intent.
"""

import hashlib
import json
import re
from collections.abc import Callable

from pydantic import BaseModel

from ai_share_of_voice.discovery import BusinessKind, BusinessList, NamedBusiness
from ai_share_of_voice.panel import INTENTS, DraftQuestion, DraftQuestions
from ai_share_of_voice.schemas.brief import Brief
from ai_share_of_voice.tools.mention_tool import BrandJudgement, JudgementBatch

LOCAL_BUSINESS = "Tienda Local Ejemplo"


def _pick(prompt: str, names: list[str]) -> list[str]:
    """A stable subset of ``names`` for this prompt (never empty)."""
    digest = hashlib.sha256(prompt.encode("utf-8")).digest()
    chosen = [name for index, name in enumerate(names) if digest[index] % 3 != 0]
    return chosen or names[:1]


def demo_responders(
    brief: Brief,
) -> tuple[dict[type[BaseModel], Callable[[str], BaseModel]], Callable[[str], str]]:
    names = [brand.name for brand in (brief.brand, *brief.competitors)]

    def answer(prompt: str) -> str:
        named = ", ".join(_pick(prompt, names))
        if brief.language == "es":
            return (
                f"(Respuesta de demostración) En {brief.city} mucha gente usa {named}. "
                f"A otros les gusta {LOCAL_BUSINESS}."
            )
        return (
            f"(Demo answer) In {brief.city} people often use {named}. "
            f"Some also like {LOCAL_BUSINESS}."
        )

    def judge(prompt: str) -> JudgementBatch:
        listed = json.loads(re.search(r"Brands to judge: (\[.*\])", prompt).group(1))
        return JudgementBatch(
            judgements=tuple(
                BrandJudgement(brand=name, recommended=index == 0, tone="positive")
                for index, name in enumerate(listed)
            )
        )

    def discover(prompt: str) -> BusinessList:
        found = [name for name in [*names, LOCAL_BUSINESS] if name in prompt]
        return BusinessList(
            businesses=[NamedBusiness(name=name, kind=BusinessKind.OTHER) for name in found]
        )

    def draft(prompt: str) -> DraftQuestions:
        count = int(re.search(r"Number of questions: (\d+)", prompt).group(1))
        return DraftQuestions(
            questions=[
                DraftQuestion(
                    text=f"Simulated {INTENTS[index % len(INTENTS)]} question {index + 1} "
                    f"about {brief.sector} in {brief.city}?",
                    intent=INTENTS[index % len(INTENTS)],
                )
                for index in range(count)
            ]
        )

    responders = {JudgementBatch: judge, BusinessList: discover, DraftQuestions: draft}
    return responders, answer
