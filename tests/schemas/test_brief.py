import pytest
from pydantic import ValidationError

from ai_share_of_voice.schemas.brief import BrandSpec, Brief


def make_brief(**overrides: object) -> Brief:
    data: dict[str, object] = {
        "brand": BrandSpec(name="Acme", variants=("Acme Inc",)),
        "competitors": (BrandSpec(name="Beta"), BrandSpec(name="Gamma")),
        "sector": "food delivery",
        "city": "Madrid",
        "spend_cap_usd": 2.0,
    }
    data.update(overrides)
    return Brief(**data)


def test_valid_brief_uses_defaults() -> None:
    brief = make_brief()
    assert brief.language == "en"
    assert brief.max_batches == 10
    assert brief.target_half_width == pytest.approx(0.05)


def test_max_batches_must_be_at_least_two() -> None:
    with pytest.raises(ValidationError):
        make_brief(max_batches=1)


def test_brief_is_immutable() -> None:
    brief = make_brief()
    with pytest.raises(ValidationError):
        brief.city = "Barcelona"  # type: ignore[misc]


def test_unknown_field_is_rejected() -> None:
    with pytest.raises(ValidationError):
        make_brief(unexpected="value")


@pytest.mark.parametrize("count", [0, 1, 6])
def test_competitor_count_must_be_between_two_and_five(count: int) -> None:
    competitors = tuple(BrandSpec(name=f"Rival {i}") for i in range(count))
    with pytest.raises(ValidationError):
        make_brief(competitors=competitors)


@pytest.mark.parametrize("cap", [0, -1, 101])
def test_spend_cap_must_be_positive_and_bounded(cap: float) -> None:
    with pytest.raises(ValidationError):
        make_brief(spend_cap_usd=cap)


def test_variants_are_cleaned() -> None:
    spec = BrandSpec(name="Acme", variants=(" Acme Inc ", "", "Acme Inc"))
    assert spec.variants == ("Acme Inc",)
    assert spec.all_names() == ("Acme", "Acme Inc")


def test_same_name_in_two_brands_is_rejected() -> None:
    with pytest.raises(ValidationError, match="both"):
        make_brief(competitors=(BrandSpec(name="acme"), BrandSpec(name="Gamma")))


def test_shared_variant_between_brands_is_rejected() -> None:
    with pytest.raises(ValidationError, match="both"):
        make_brief(
            competitors=(
                BrandSpec(name="Beta", variants=("Acme Inc",)),
                BrandSpec(name="Gamma"),
            )
        )
