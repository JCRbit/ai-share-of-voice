"""Brief contract: the request the user fills in the form (owners: J with Ab)."""

from typing import Self

from pydantic import Field, field_validator, model_validator

from ai_share_of_voice.schemas.base import Contract


class BrandSpec(Contract):
    """A brand and the name variants used to detect it in model answers."""

    name: str = Field(min_length=1, max_length=100)
    variants: tuple[str, ...] = ()

    @field_validator("variants")
    @classmethod
    def clean_variants(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Strip whitespace, drop empty entries and remove exact duplicates."""
        return tuple(dict.fromkeys(item.strip() for item in value if item.strip()))

    def all_names(self) -> tuple[str, ...]:
        """Return the brand name followed by its variants."""
        return (self.name, *self.variants)


class Brief(Contract):
    """What the user asks the system to measure.

    ``max_batches`` and ``target_half_width`` are the user-facing limits that the
    graph copies into the ``StopConfig`` used by the stopping rule.
    """

    brand: BrandSpec
    competitors: tuple[BrandSpec, ...] = Field(min_length=2, max_length=5)
    sector: str = Field(min_length=2, max_length=100)
    city: str = Field(min_length=2, max_length=100)
    language: str = Field(default="en", pattern=r"^[a-z]{2}$")
    spend_cap_usd: float = Field(gt=0, le=100)
    max_batches: int = Field(default=10, ge=2, le=50)
    target_half_width: float = Field(default=0.05, gt=0, le=0.5)

    @model_validator(mode="after")
    def names_must_not_collide(self) -> Self:
        """Reject brands that share a name or variant, which would make mentions ambiguous."""
        seen: dict[str, tuple[int, str]] = {}
        for index, brand in enumerate((self.brand, *self.competitors)):
            for name in brand.all_names():
                owner_index, owner_name = seen.setdefault(name.casefold(), (index, brand.name))
                if owner_index != index:
                    raise ValueError(f"'{name}' is used by both '{owner_name}' and '{brand.name}'")
        return self
