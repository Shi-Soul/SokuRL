"""Recorded episode boundaries must survive absent intro input and reject drift."""
from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace

import pytest

from soku_rl.pomg import Outcome, TimeStep
from soku_rl.replay.episode import RecordedEpisode
from test_official_replay import example


def state(frame, outcome):
    return TimeStep(frame, ((), ()), (0., 0.), outcome, {})


@pytest.mark.parametrize("frames", (0, 1, 30, 600))
def test_recorded_end_preserves_every_frame_including_empty_intro(frames):
    episode = RecordedEpisode(frames, "close")
    assert [episode.finished(state(i, Outcome.ONGOING)) for i in range(frames + 1)] == [False] * frames + [True]


@pytest.mark.parametrize("reason,outcome", (("p1_win", Outcome.P1_WIN),
    ("p2_win", Outcome.P2_WIN), ("double_ko", Outcome.DRAW)))
def test_recorded_knockout_requires_exact_frame_and_outcome(reason, outcome):
    episode = RecordedEpisode(20, reason)
    assert episode.finished(state(20, outcome))
    for actual in (state(19, outcome), state(21, outcome), state(20, Outcome.ONGOING)):
        with pytest.raises(RuntimeError, match="recorded episode"):
            episode.finished(actual)


@pytest.mark.parametrize("change", ({}, {"sha256": "wrong"}, {"seed": 7},
    {"frames": -1}, {"frames": True}, {"reason": "unknown"}, {"scope": "match"}))
def test_episode_metadata_is_bound_to_replay_bytes_and_seed(tmp_path, change):
    replay = replace(example(), matches=(replace(example().matches[0], input_words=()),))
    source = tmp_path / "episode.rep"
    source.write_bytes(replay.encode())
    metadata = {"sha256": hashlib.sha256(replay.encode()).hexdigest(),
                "seed": replay.matches[0].seed, "frames": 30, "reason": "reset"} | change
    source.with_suffix(".json").write_text(json.dumps(metadata), encoding="utf-8")
    if change:
        with pytest.raises(ValueError):
            RecordedEpisode.read(source, replay, 0)
    else:
        assert RecordedEpisode.read(source, replay, 0) == RecordedEpisode(30, "reset")


def test_recorded_replay_conversion_does_not_stop_at_empty_input_queue(tmp_path, monkeypatch):
    import sys
    if sys.platform != "win32":
        pytest.skip("Windows replay adapter")
    from bridge_shared import RawFrameState
    from game_runtime import trajectory
    from test_replay_rollout import config

    raw = RawFrameState()
    raw.sceneId, raw.battleMode = 5, 3
    raw.p1.hp = raw.p2.hp = 10000
    raw.p1.maxSpirit = raw.p2.maxSpirit = 1000
    closed = []

    def step(timeout):
        assert timeout > 0 and raw.frameId < 30
        raw.frameId += 1
        return raw

    instance = SimpleNamespace(pid=1, state=raw, step_native=step,
        client=SimpleNamespace(snapshot=lambda: SimpleNamespace(dropped_frames=0)),
        close=lambda: closed.append("game"))
    monkeypatch.setattr(trajectory, "launch_replay", lambda *args: instance)
    memory = SimpleNamespace(begin_frame=lambda: None,
        read=lambda address, size: (100 if address == 0x898718 + 0x104 else 0).to_bytes(size, "little"),
        close=lambda: closed.append("memory"))
    monkeypatch.setattr(trajectory, "ProcessMemory", lambda pid: memory)
    replay = replace(example(), matches=(replace(example().matches[0], input_words=()),))
    frames = list(trajectory.replay_frames(replay, 0, config(), 1., tmp_path, RecordedEpisode(30, "reset")))
    assert [frame.frame for frame in frames] == list(range(31))
    assert [frame.outcome for frame in frames] == ["ongoing"] * 30 + ["time_limit"]
    assert set(closed) == {"game", "memory"}
    assert not list(tmp_path.iterdir())
