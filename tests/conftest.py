"""Shared test setup (task C0): the suite never reaches a real model or the internet.

* Every test runs in simulated mode, as CI does. A developer whose ``.env`` says
  ``LLM_MODE=live`` gets the same behavior as the CI run, so a test can never spend money.
* ``no_network`` is an opt-in fixture for tests that must prove they stay offline.
"""

import socket
from collections.abc import Iterator

import pytest

LOCAL_HOSTS = frozenset({"localhost", "::1"})


def is_local(address: object) -> bool:
    """True for a Unix socket path or a loopback address; False for anything else."""
    if isinstance(address, str | bytes):
        return True
    host = str(address[0]) if isinstance(address, tuple) and address else ""
    return host in LOCAL_HOSTS or host.startswith("127.")


@pytest.fixture(autouse=True)
def simulated_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force ``LLM_MODE=simulated``; ``load_dotenv`` never overrides a variable already set."""
    monkeypatch.setenv("LLM_MODE", "simulated")


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str]]:
    """Refuse connections and name lookups outside this computer.

    Local servers still work. The attempts are recorded and the test fails at the end if there
    was any, even when the code under test swallowed the error raised at the attempt.
    """
    attempts: list[str] = []
    real_connect = socket.socket.connect
    real_lookup = socket.getaddrinfo

    def connect(self: socket.socket, address: object) -> None:
        if not is_local(address):
            attempts.append(f"connect {address!r}")
            raise AssertionError(f"network access attempted: {address!r}")
        return real_connect(self, address)

    def lookup(host: object, *args: object, **kwargs: object) -> object:
        if host not in (None, "") and not is_local((host,)):
            attempts.append(f"lookup {host!r}")
            raise AssertionError(f"name lookup attempted: {host!r}")
        return real_lookup(host, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket, "getaddrinfo", lookup)
    yield attempts
    assert not attempts, f"the test reached the network: {attempts}"
