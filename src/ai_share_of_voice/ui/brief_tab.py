"""Brief tab: the form that becomes a ``Brief`` (owner: An, task I6).

``build_brief`` is plain Python: it turns the form values into the ``Brief`` contract and
returns readable messages when the contract refuses them, so the rules live in one place
(``schemas/brief.py``) and the tab only translates. ``build_brief_tab`` creates the Gradio
components; Gradio is imported there so everything else works and is tested without it.

Brands are typed as ``Name`` or ``Name | variant, variant``, one competitor per line. P4 reviews
the texts of this form.
"""

from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from ai_share_of_voice.schemas.brief import BrandSpec, Brief
from ai_share_of_voice.ui.mock_run import build_demo_brief

FIELD_LABELS = {
    "brand": "Brand",
    "competitors": "Competitors",
    "sector": "Sector",
    "city": "City",
    "language": "Language",
    "spend_cap_usd": "Spend cap (USD)",
    "max_batches": "Maximum batches",
    "target_half_width": "Target half-width",
}
LANGUAGES = ("en", "es")
_VALUE_ERROR_PREFIX = "Value error, "


@dataclass(frozen=True)
class BriefResult:
    """A valid ``Brief``, or the messages that say what to fix."""

    brief: Brief | None
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class BriefForm:
    """The components of the tab: ``inputs`` follow the parameters of ``build_brief``."""

    inputs: list[Any]
    start: Any
    message: Any


def format_brand(brand: BrandSpec) -> str:
    """The text a person types for ``brand``: ``Name`` or ``Name | variant, variant``."""
    return f"{brand.name} | {', '.join(brand.variants)}" if brand.variants else brand.name


def parse_brand(text: str) -> BrandSpec:
    """Read ``Name | variant, variant``. Raises ``ValidationError`` for an empty name."""
    name, _, variants = (text or "").partition("|")
    return BrandSpec(name=name, variants=tuple(variants.split(",")))


def parse_competitors(text: str) -> tuple[BrandSpec, ...]:
    """One competitor per non-empty line."""
    return tuple(parse_brand(line) for line in (text or "").splitlines() if line.strip())


def _messages(error: ValidationError, field: str | None = None) -> tuple[str, ...]:
    """Readable messages; ``field`` names the form field when the error's own path does not."""
    messages: dict[str, None] = {}
    for item in error.errors():
        location = item.get("loc") or ()
        key = field or (str(location[0]) if location else "")
        label = FIELD_LABELS.get(key, key or "Brief")
        messages[f"{label}: {str(item['msg']).removeprefix(_VALUE_ERROR_PREFIX)}"] = None
    return tuple(messages)


def build_brief(
    brand: str,
    competitors: str,
    sector: str,
    city: str,
    language: str,
    spend_cap_usd: float | None,
    max_batches: float | None,
    target_half_width: float | None,
) -> BriefResult:
    """Validate the form values and build the ``Brief``."""
    try:
        brand_spec = parse_brand(brand)
    except ValidationError as error:
        return BriefResult(None, _messages(error, "brand"))
    try:
        competitor_specs = parse_competitors(competitors)
    except ValidationError as error:
        return BriefResult(None, _messages(error, "competitors"))
    try:
        brief = Brief(
            brand=brand_spec,
            competitors=competitor_specs,
            sector=sector or "",
            city=city or "",
            language=language or "",
            spend_cap_usd=spend_cap_usd if spend_cap_usd is not None else 0,
            max_batches=max_batches if max_batches is not None else 0,
            target_half_width=target_half_width if target_half_width is not None else 0,
        )
    except ValidationError as error:
        return BriefResult(None, _messages(error))
    return BriefResult(brief)


def build_brief_tab(defaults: Brief | None = None) -> BriefForm:
    """Create the form components inside the current Gradio ``Blocks``."""
    import gradio as gr

    start_values = defaults or build_demo_brief()
    brand = gr.Textbox(
        label="Brand",
        value=format_brand(start_values.brand),
        info="Name, or Name | other spellings separated by commas",
    )
    competitors = gr.Textbox(
        label="Competitors",
        value="\n".join(format_brand(item) for item in start_values.competitors),
        lines=4,
        info="Two to five, one per line, in the same format as the brand",
    )
    sector = gr.Textbox(label="Sector", value=start_values.sector)
    city = gr.Textbox(label="City", value=start_values.city)
    language = gr.Dropdown(label="Language", choices=list(LANGUAGES), value=start_values.language)
    cap = gr.Number(label="Spend cap (USD)", value=start_values.spend_cap_usd)
    batches = gr.Number(label="Maximum batches", value=start_values.max_batches)
    half_width = gr.Number(label="Target half-width", value=start_values.target_half_width)
    start = gr.Button("Start run", variant="primary")
    message = gr.Markdown()
    inputs = [brand, competitors, sector, city, language, cap, batches, half_width]
    return BriefForm(inputs=inputs, start=start, message=message)
