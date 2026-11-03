"""Change between two measurements of one brand (owner: E).

The rule is deliberately conservative: a change is called clear only when the two
intervals do not overlap. Non-overlap implies a real difference at a stricter level than
each interval alone, so "up" or "down" is never claimed on noise; an overlap is reported
as "unclear", not as "no change", because overlapping intervals can still hide one.
"""

from typing import Literal

Rate = tuple[float, float, float]  # estimate, interval low, interval high
Change = Literal["up", "down", "unclear"]


def compare(previous: Rate, latest: Rate) -> Change:
    """Whether ``latest`` is clearly above or below ``previous`` (touching counts as overlap)."""
    _, previous_low, previous_high = previous
    _, latest_low, latest_high = latest
    if latest_low > previous_high:
        return "up"
    if latest_high < previous_low:
        return "down"
    return "unclear"
