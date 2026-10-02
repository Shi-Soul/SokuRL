"""Interactive matches survive late observations without inventing missing frames."""
import io
import queue
import threading
from types import SimpleNamespace

import pytest

from soku_rl.env import EpisodeConfig
from soku_rl.env.match import LEGACY_MATCH
from soku_rl.env.wrappers.learning import LearningConfig, LearningInterface
from soku_rl.play.match import MatchFrame, MatchState
from soku_rl.play.connection import PlayConnection
from soku_rl.play.realtime_session import RealtimePolicy, run_session
from test_env_timing import VISIBILITY
from test_live_policy import RecordingPolicy, observations


@pytest.mark.parametrize("timeout_frame", (0, 2))
def test_interactive_session_recovers_gaps_and_completes_the_original_match(timeout_frame, monkeypatch):
    interface = LearningInterface(EpisodeConfig(7200, 4, 1, 0, "state", VISIBILITY, LEGACY_MATCH),
                                  LearningConfig("combat", False, 0, 0.))
    policy = RecordingPolicy()
    controller = RealtimePolicy(policy, interface, 1, 1732)
    advance = controller.advance

    def timed_advance(frame):
        if frame.match.frame == timeout_frame:
            raise TimeoutError("temporary policy delay")
        return advance(frame)

    monkeypatch.setattr(controller, "advance", timed_advance)
    frames = iter((1, 2, 30, 31, 300, 500))
    commands, records = [], []

    class Connection:
        def request(self, operation, payload):
            if operation == "submit":
                commands.append(payload)
                return {"submitted": True}
            frame = next(frames)
            state = MatchState(1, int(frame >= 300), frame,
                (2 if frame == 500 else int(frame >= 300), 0),
                (10000, 0 if frame == 500 else 10000), "battle")
            item = {"frame": MatchFrame(state, observations(frame, 0)),
                    "engine_inputs": (), "characters": (0, 1)}
            return {"closed": False, "events": (), "records": (item,)}

    result = run_session(Connection(), controller, 1, 0., records.append)
    assert result["termination"] == "matches_completed"
    assert result["scores"] == (2, 0)
    assert result["recoveries"] == 3 and result["skipped_frames"] == 494 + bool(timeout_frame)
    assert result["policy_timeouts"] == bool(timeout_frame)
    assert [command["state"].frame for command in commands] == [f for f in (1, 2, 30, 31, 300) if f != timeout_frame]
    assert len(policy.episodes) == 3
    assert [record["skipped"] for record in records if record["kind"] == "policy_resync"] == [27 + bool(timeout_frame), 268, 199]


@pytest.mark.parametrize("operation", ("poll", "submit"))
def test_play_waits_for_a_late_reply_without_breaking_the_worker(operation):
    connection = object.__new__(PlayConnection)
    connection.closed = connection.broken = False
    connection.timeout = .01
    connection.lock = threading.Lock()
    connection.replies = queue.Queue()
    connection.process = SimpleNamespace(stdin=io.BytesIO())

    def reply_later():
        threading.Event().wait(.05)
        connection.replies.put({"ok": True, "value": {"ready": True}})

    reply = threading.Thread(target=reply_later)
    reply.start()
    try:
        assert connection.request(operation, {}) == {"ready": True}
        assert not connection.broken
    finally:
        reply.join()
    connection.replies.put(EOFError("worker exited"))
    with pytest.raises(EOFError, match="worker exited"):
        connection.request(operation, {})
