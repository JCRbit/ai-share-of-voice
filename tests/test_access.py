"""The demo password gate: no model call without it, no guessing by brute force."""

import pytest

from ai_share_of_voice.config import LlmMode
from ai_share_of_voice.ui.access import MAX_FAILURES, Access, AccessGate, gate_from_env


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def make_gate(clock: Clock | None = None) -> AccessGate:
    return AccessGate("s3cret", clock=clock or Clock())


def test_the_right_password_is_allowed():
    assert make_gate().check("s3cret") is Access.OK


@pytest.mark.parametrize("attempt", ["", "S3CRET", "s3cret ", "other", None])
def test_any_other_text_is_refused(attempt):
    assert make_gate().check(attempt) is Access.WRONG


def test_a_gate_without_password_is_open_and_not_enabled():
    gate = AccessGate(None)
    assert not gate.enabled
    assert gate.check(None) is Access.OK
    assert gate.check("anything") is Access.OK


def test_an_empty_configured_password_means_no_gate():
    assert not AccessGate("   ").enabled


def test_non_ascii_passwords_work():
    assert AccessGate("contraseña-ñ").check("contraseña-ñ") is Access.OK


def test_too_many_failures_lock_the_gate_even_for_the_right_password():
    gate = make_gate()
    for _ in range(MAX_FAILURES):
        assert gate.check("bad") is Access.WRONG
    assert gate.check("s3cret") is Access.LOCKED
    assert gate.check("bad") is Access.LOCKED


def test_the_lock_expires_and_the_count_restarts():
    clock = Clock()
    gate = make_gate(clock)
    for _ in range(MAX_FAILURES):
        gate.check("bad")
    clock.now += 61
    assert gate.check("s3cret") is Access.OK
    # one more failure after the lock is a first failure again, not a new lock
    assert gate.check("bad") is Access.WRONG
    assert gate.check("s3cret") is Access.OK


def test_a_success_clears_the_failure_count():
    gate = make_gate()
    for _ in range(MAX_FAILURES - 1):
        gate.check("bad")
    assert gate.check("s3cret") is Access.OK
    for _ in range(MAX_FAILURES - 1):
        assert gate.check("bad") is Access.WRONG
    assert gate.check("s3cret") is Access.OK


def test_failures_forgotten_after_the_window():
    clock = Clock()
    gate = make_gate(clock)
    for _ in range(MAX_FAILURES - 1):
        gate.check("bad")
    clock.now += 61
    assert gate.check("bad") is Access.WRONG
    assert gate.check("s3cret") is Access.OK


def test_live_mode_with_a_password_enables_the_gate():
    gate = gate_from_env({"DEMO_PASSWORD": "pw"}, LlmMode.LIVE)
    assert gate.enabled
    assert gate.check("pw") is Access.OK


def test_simulated_mode_needs_no_password_because_it_costs_nothing():
    assert not gate_from_env({"DEMO_PASSWORD": "pw"}, LlmMode.SIMULATED).enabled


def test_live_mode_without_a_password_is_open_for_local_use():
    assert not gate_from_env({}, LlmMode.LIVE).enabled
