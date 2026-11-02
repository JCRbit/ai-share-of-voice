"""Free the app port from an old instance of this app, and only from it.

When ``app.py`` starts and its port is busy, a previous run of the app is usually still
alive (a closed terminal, a crashed session). ``PortGuard`` closes it, but only when both
are true:

* the server on the port answers as this app (its Gradio title), and
* ``runs/app.pid`` names the process that started it.

Anything else on the port is never touched: the app moves to the next free port instead.
Killing whatever holds a port could close an unrelated program without warning.
"""

import os
import signal
import socket
import time
from collections.abc import Callable
from pathlib import Path

import httpx

APP_TITLE = "AI Share of Voice"
PORT_RANGE = 20
WAIT_SECONDS = 5.0


def port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.5)
        return probe.connect_ex((host, port)) == 0


def is_our_app(port: int, host: str = "127.0.0.1") -> bool:
    """Whether the server on ``port`` is this app (Gradio exposes its title in /config)."""
    try:
        response = httpx.get(f"http://{host}:{port}/config", timeout=2.0)
        return response.status_code == 200 and response.json().get("title") == APP_TITLE
    except (httpx.HTTPError, ValueError):
        return False


def _terminate(pid: int) -> None:
    os.kill(pid, signal.SIGTERM)  # on Windows this is TerminateProcess


class PortGuard:
    def __init__(
        self,
        pid_file: Path,
        *,
        in_use: Callable[[int], bool] = port_in_use,
        is_our_app: Callable[[int], bool] = is_our_app,
        kill: Callable[[int], None] = _terminate,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.pid_file = Path(pid_file)
        self._in_use = in_use
        self._is_our_app = is_our_app
        self._kill = kill
        self._sleep = sleep

    def _recorded(self) -> tuple[int, int] | None:
        """(pid, port) of the instance that wrote the PID file, if it is readable."""
        try:
            pid, port = (int(value) for value in self.pid_file.read_text("utf-8").split()[:2])
        except (OSError, ValueError):
            return None
        return pid, port

    def claim(self, port: int, *, attempts: int = PORT_RANGE) -> int:
        """The port to use: ``port`` itself when free or freed, else the next free one."""
        if self._in_use(port):
            recorded = self._recorded()
            if recorded and recorded[1] == port and self._is_our_app(port):
                print(f"Closing the previous instance of the app (process {recorded[0]})...")
                try:
                    self._kill(recorded[0])
                except OSError:
                    pass  # already gone
                waited = 0.0
                while self._in_use(port) and waited < WAIT_SECONDS:
                    self._sleep(0.2)
                    waited += 0.2
        for candidate in range(port, port + attempts):
            if not self._in_use(candidate):
                if candidate != port:
                    print(f"Port {port} is used by another program: using {candidate} instead.")
                return candidate
        raise RuntimeError(f"no free port between {port} and {port + attempts - 1}")

    def record(self, port: int, *, pid: int | None = None) -> None:
        self.pid_file.parent.mkdir(parents=True, exist_ok=True)
        self.pid_file.write_text(f"{pid or os.getpid()}\n{port}\n", encoding="utf-8")

    def release(self, *, pid: int | None = None) -> None:
        """Remove the PID file, unless another process has written it since."""
        recorded = self._recorded()
        if recorded and recorded[0] == (pid or os.getpid()):
            self.pid_file.unlink(missing_ok=True)
