"""Shared fixtures of the tests that run the real nodes (not collected as tests)."""

import csv
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from ai_share_of_voice.sampling import SubjectAnswer
from ai_share_of_voice.schemas.analysis import Tone
from ai_share_of_voice.schemas.brief import BrandSpec, Brief
from ai_share_of_voice.schemas.sources import UrlCheck
from ai_share_of_voice.schemas.vision import Confidence
from ai_share_of_voice.tools.mention_tool import BrandJudgement, JudgementBatch
from ai_share_of_voice.vision import VisionJudgement

QUESTIONS = 12
LIVE_LINK = "https://example.com/guia"
DEAD_LINK = "https://dead.example.com/x"
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16


def respond(prompt: str, schema: type[BaseModel]) -> BaseModel:
    """Acme leads the odd questions (50%), Beta every fourth one (25%); Gamma is always there."""
    if schema is SubjectAnswer:
        number = int(prompt.rsplit(" ", 1)[1])
        beta_or_gamma = "Beta Eats" if number % 4 == 0 else "Gamma Bites"
        first = "Acme Food" if number % 2 else beta_or_gamma
        live = LIVE_LINK + " " if number % 3 == 0 else ""
        links = live + (DEAD_LINK if number % 4 == 0 else "")
        answer = f"Para pedir cena, {first} es la mejor opción; Gamma Bites también. {links}"
        return schema(answer=answer)
    listed = prompt.split("Brands to judge: ")[1].split("\n")[0]
    names = [name.strip(' "[]') for name in listed.split(",")]
    judged = tuple(
        BrandJudgement(brand=n, recommended=n != "Gamma Bites", tone=Tone.NEUTRAL) for n in names
    )
    return schema(judgements=judged)


class FakeSources:
    """Stand-in for the sources service: a table of what each URL answers."""

    def __init__(self) -> None:
        self.checked: list[str] = []

    def check_url(self, url: str) -> UrlCheck:
        self.checked.append(url)
        exists = url != DEAD_LINK
        return UrlCheck(
            url=url,
            exists=exists,
            status_code=200 if exists else 404,
            allowed_by_robots=True,
            checked_at=datetime.now(UTC),
        )


def make_brief(*, cap: float = 2.0, max_batches: int = 2) -> Brief:
    return Brief(
        brand=BrandSpec(name="Acme Food", variants=("Acme",)),
        competitors=(BrandSpec(name="Beta Eats"), BrandSpec(name="Gamma Bites")),
        sector="food delivery",
        city="Madrid",
        spend_cap_usd=cap,
        max_batches=max_batches,
        target_half_width=0.01,
    )


def write_data(data_dir: Path, *, screenshots: bool = False) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    with (data_dir / "questions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "text", "intent"])
        for number in range(1, QUESTIONS + 1):
            writer.writerow([f"q{number:02d}", f"Pregunta {number}", "recommendation"])
    if not screenshots:
        return
    folder = data_dir / "screenshots"
    folder.mkdir(exist_ok=True)
    rows = [
        ("beta-home.png", "Beta Eats"),
        ("acme-home.png", "Acme Food"),
        ("gamma-home.png", "Gamma Bites"),
        ("other-home.png", "Some Other App"),
        ("gone.png", "Acme Food"),
    ]
    for file, _ in rows:
        if file != "gone.png":
            (folder / file).write_bytes(PNG)
    with (folder / "index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file", "url", "captured_at", "competitor", "note"])
        for file, brand in rows:
            writer.writerow([file, "https://example.com", "2026-10-27", brand, ""])


def write_brands(data_dir: Path) -> None:
    """The three brands of the tests, in the format of ``data/competitors.csv``."""
    data_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        ["name", "variants", "is_client"],
        ["Acme Food", "Acme", "true"],
        ["Beta Eats", "", "false"],
        ["Gamma Bites", "", "false"],
    ]
    with (data_dir / "competitors.csv").open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)


LIVE_ENV = {"LLM_MODE": "live", "LLM_PROVIDER": "groq", "LLM_API_KEY": "key"}
VISION_ENV = {
    "VISION_MODEL": "vision-model",
    "VISION_PRICE_INPUT_PER_M": "2",
    "VISION_PRICE_OUTPUT_PER_M": "8",
}


class LiveTransport:
    """Fake HTTP layer that answers like the provider would, for the three kinds of request."""

    def __init__(self) -> None:
        self.payloads: list[dict] = []

    def __call__(self, payload: dict) -> dict:
        self.payloads.append(payload)
        content = payload["messages"][0]["content"]
        if isinstance(content, list):  # a request with an image
            judged = VisionJudgement(
                observation="Three bars, the first one is highest.",
                confidence=Confidence.MEDIUM,
            )
            body = judged.model_dump_json()
        elif "Brands to judge" in content:
            body = respond(content, JudgementBatch).model_dump_json()
        else:
            body = respond(content, SubjectAnswer).model_dump_json()
        usage = {"prompt_tokens": 150, "completion_tokens": 60}
        return {"choices": [{"message": {"content": body}}], "usage": usage}
