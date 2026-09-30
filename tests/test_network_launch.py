"""Both player seats must launch the host first without polling the human."""
from contextlib import ExitStack
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from network_runtime.launch import start_games


@pytest.mark.parametrize("seat", [1, 2])
def test_human_seat_starts_host_first_and_never_controls_human(tmp_path, seat):
    events = []

    class Worker:
        def __init__(self, log_path, mute_audio):
            self.name = log_path.name
            self.identity = {"kind": "network"}

        def request(self, operation, payload):
            events.append((self.name, operation, payload))
            return {"pid": 11 if self.name == "game-worker.log" else 22}

        def close(self):
            pass

    config = {"human": {"enabled": True, "seat": seat, "mute_audio": False, "automate_menu": False},
              "runtime": {"mute_audio": True},
              "network": {"role": "host", "port": 12345, "address": "127.0.0.1", "automate_menu": True}}
    with ExitStack() as stack:
        connection, games = start_games(stack, config, {}, tmp_path, Worker)
        assert connection.name == "game-worker.log"
    starts = [event for event in events if event[1] == "start"]
    assert [event[2]["network"]["role"] for event in starts] == ["host", "join"]
    assert starts[0][0] == ("human_game-worker.log" if seat == 1 else "game-worker.log")
    human = next(event for event in starts if event[0] == "human_game-worker.log")
    assert human[2]["network"]["automate_menu"] is False
    assert events.index(next(event for event in events if event[1] == "wait_host")) < events.index(starts[1])
    assert not any(event[1] in {"poll", "submit"} for event in events)
    assert games["human_game"]["pid"] == 22
