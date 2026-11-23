"""A shared password that guards every call that spends model budget on a public demo.

The password comes from the ``DEMO_PASSWORD`` variable (a secret of the host, never the
repository). It is only required in live mode: the simulated demo costs nothing. Without the
variable the gate is open, which is what a developer wants on their own computer.

Wrong guesses are limited: after ``MAX_FAILURES`` in ``WINDOW_SECONDS`` the gate refuses
everything, even the right password, until the window passes. The count is shared by all
visitors because the demo has one password; the cost is that an attacker can delay others,
the gain is that the password cannot be brute-forced.
"""

import hmac
import threading
import time
from collections.abc import Callable, Mapping
from enum import StrEnum

from ai_share_of_voice.config import LlmMode

MAX_FAILURES = 5
WINDOW_SECONDS = 60.0


class Access(StrEnum):
    OK = "ok"
    WRONG = "wrong"
    LOCKED = "locked"


class AccessGate:
    def __init__(
        self, password: str | None, *, clock: Callable[[], float] = time.monotonic
    ) -> None:
        password = (password or "").strip()
        self._password = password.encode("utf-8") if password else None
        self._clock = clock
        self._failures: list[float] = []
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self._password is not None

    def check(self, attempt: str | None) -> Access:
        if self._password is None:
            return Access.OK
        with self._lock:
            now = self._clock()
            self._failures = [at for at in self._failures if now - at < WINDOW_SECONDS]
            if len(self._failures) >= MAX_FAILURES:
                return Access.LOCKED
            candidate = (attempt or "").encode("utf-8")
            if hmac.compare_digest(candidate, self._password):
                self._failures.clear()
                return Access.OK
            self._failures.append(now)
            return Access.WRONG


def gate_from_env(env: Mapping[str, str], mode: LlmMode) -> AccessGate:
    """The gate for this run: closed only in live mode and only when a password is set."""
    if mode is not LlmMode.LIVE:
        return AccessGate(None)
    return AccessGate(env.get("DEMO_PASSWORD"))
