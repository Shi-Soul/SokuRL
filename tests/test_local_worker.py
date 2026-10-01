"""The local worker delivers original match replays before reset, close and errors."""
from dataclasses import asdict, replace
from io import BytesIO
import sys
from types import SimpleNamespace

import pytest

if sys.platform != "win32":
    pytest.skip("Windows worker adapter", allow_module_level=True)

import local_worker
from soku_rl.env.worker_pipe import PROTOCOL, receive, send
from soku_rl.play.match import MatchFrame, MatchState
from test_official_replay import example
from test_replay_rollout import config


@pytest.mark.parametrize("fail", (False, True))
def test_worker_preserves_replays_and_closes_only_its_game(monkeypatch, fail):
    games = []

    class Game:
        def __init__(self, episode, seed, timeout):
            self.seed, self.frame, self.closed = seed, 0, 0
            self.client = SimpleNamespace(snapshot=lambda: SimpleNamespace(game_frame=self.frame))
            games.append(self)

        def read(self):
            return MatchFrame(MatchState(1, 0, self.frame, (0, 0), (10000, 10000), "battle"), ((), ()))

        def step(self, inputs):
            self.frame += 1
            if fail:
                raise ValueError("injected step failure")

        def replay(self):
            replay = example()
            return replace(replay, matches=(replace(replay.matches[0], seed=self.seed),))

        def reset(self, seed):
            self.seed, self.frame = seed, 0

        def close(self):
            self.closed += 1

    source, destination = BytesIO(), BytesIO()
    requests = [("initialize", {"protocol": PROTOCOL, "game_directory": "fixture", "launch_timeout": 2.}),
                ("start", {"episode": asdict(config()), "seed": 1732}), ("step", {0: (0,) * 8}),
                ("finish", {}), ("reset", {"seed": 1733}), ("close", {})]
    for request in requests:
        send(source, request)
    source.seek(0)
    monkeypatch.setattr(local_worker, "LocalMatch", Game)
    monkeypatch.setattr(local_worker, "configure_game", lambda directory: None)
    monkeypatch.setattr(local_worker, "fingerprints", lambda: {})
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(buffer=source))
    monkeypatch.setattr(sys, "stdout", SimpleNamespace(buffer=destination))
    if fail:
        with pytest.raises(ValueError, match="injected step failure"):
            local_worker.main()
    else:
        local_worker.main()
    destination.seek(0)
    messages = []
    while destination.tell() < len(destination.getvalue()):
        messages.append(receive(destination))
    assert games[0].closed == 1
    assert messages[0]["value"]["kind"] == "local_match"
    if fail:
        assert messages[-1]["ok"] is False
        assert messages[-1]["replays"][0]["reason"] == "error"
        assert messages[-1]["replays"][0]["frames"] == 1
    else:
        replays = [replay for message in messages[1:] for replay in message["value"]["replays"]]
        assert [(value["seed"], value["reason"], value["frames"]) for value in replays] == [
            (1732, "match_finished", 1), (1733, "close", 0)]
        assert all(value["scope"] == "match" for value in replays)
