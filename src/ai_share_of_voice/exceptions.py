"""Exceptions shared across the package."""


class ConfigurationError(Exception):
    """The environment is missing a value or has an invalid one (for example, no API key)."""


class BudgetExceeded(Exception):
    """Raised by the model client when a call would exceed the spending cap."""

    def __init__(self, spent_usd: float, cap_usd: float) -> None:
        super().__init__(
            f"Spending cap exceeded: {spent_usd:.4f} USD spent, cap is {cap_usd:.4f} USD"
        )
        self.spent_usd = spent_usd
        self.cap_usd = cap_usd
