import socket

import pytest

from ai_share_of_voice.ui.port_guard import PortGuard, port_in_use


class FakeSystem:
    """A port table and a process table, so no real process is ever touched."""

    def __init__(self, busy=(), ours=(), alive=()):
        self.busy = set(busy)
        self.ours = set(ours)
        self.alive = set(alive)
        self.killed = []

    def in_use(self, port):
        return port in self.busy

    def is_our_app(self, port):
        return port in self.ours

    def kill(self, pid):
        self.killed.append(pid)
        self.alive.discard(pid)
        self.busy.discard(7860)  # our old instance held the default port


def guard(tmp_path, system, pid=None):
    pid_file = tmp_path / "app.pid"
    if pid is not None:
        pid_file.write_text(f"{pid}\n7860\n", encoding="utf-8")
    return PortGuard(
        pid_file,
        in_use=system.in_use,
        is_our_app=system.is_our_app,
        kill=system.kill,
        sleep=lambda _: None,
    ), pid_file


def test_a_free_port_is_used_as_it_is(tmp_path):
    system = FakeSystem()
    port_guard, _ = guard(tmp_path, system)

    assert port_guard.claim(7860) == 7860
    assert system.killed == []


def test_an_old_instance_of_our_app_is_closed(tmp_path):
    system = FakeSystem(busy={7860}, ours={7860}, alive={4242})
    port_guard, _ = guard(tmp_path, system, pid=4242)

    assert port_guard.claim(7860) == 7860
    assert system.killed == [4242]


def test_another_program_on_the_port_is_never_closed(tmp_path):
    # The port is busy, but whatever answers there is not our app.
    system = FakeSystem(busy={7860}, alive={4242})
    port_guard, _ = guard(tmp_path, system, pid=4242)

    assert port_guard.claim(7860) == 7861
    assert system.killed == []


def test_our_app_without_a_pid_file_is_left_alone(tmp_path):
    # Started by someone else (another terminal, another user): we cannot tell which process.
    system = FakeSystem(busy={7860}, ours={7860})
    port_guard, _ = guard(tmp_path, system)

    assert port_guard.claim(7860) == 7861
    assert system.killed == []


def test_the_next_free_port_is_searched_in_a_small_range(tmp_path):
    system = FakeSystem(busy={7860, 7861, 7862})
    port_guard, _ = guard(tmp_path, system)

    assert port_guard.claim(7860) == 7863


def test_no_free_port_in_the_range_is_a_clear_error(tmp_path):
    system = FakeSystem(busy=set(range(7860, 7900)))
    port_guard, _ = guard(tmp_path, system)

    with pytest.raises(RuntimeError, match="7860"):
        port_guard.claim(7860, attempts=5)


def test_a_corrupt_pid_file_is_ignored(tmp_path):
    system = FakeSystem(busy={7860}, ours={7860})
    port_guard, pid_file = guard(tmp_path, system)
    pid_file.write_text("not a pid", encoding="utf-8")

    assert port_guard.claim(7860) == 7861
    assert system.killed == []


def test_the_pid_file_records_this_process_and_is_removed_on_release(tmp_path):
    port_guard, pid_file = guard(tmp_path, FakeSystem())

    port_guard.record(7860, pid=999)
    assert pid_file.read_text(encoding="utf-8").split() == ["999", "7860"]

    port_guard.release(pid=999)
    assert not pid_file.exists()


def test_release_keeps_a_pid_file_written_by_another_process(tmp_path):
    port_guard, pid_file = guard(tmp_path, FakeSystem(), pid=111)

    port_guard.release(pid=999)

    assert pid_file.exists()


def test_port_in_use_sees_a_real_listening_socket():
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen()
        port = server.getsockname()[1]

        assert port_in_use(port) is True
    assert port_in_use(port) is False
