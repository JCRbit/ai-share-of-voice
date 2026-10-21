"""Base class shared by every contract."""

from pydantic import BaseModel, ConfigDict


class Contract(BaseModel):
    """Immutable model that rejects unknown fields.

    Every shared contract inherits from it, so a typo in a field name fails loudly
    instead of being silently dropped.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)
