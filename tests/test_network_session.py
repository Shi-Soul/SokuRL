"""The policy runner must follow complete matches and stop on rejected input."""
import pytest

from test_env_timing import VISIBILITY
from test_live_policy import RecordingPolicy, observations
from soku_rl.env import EpisodeConfig
from soku_rl.learning_wrappers import LearningConfig, LearningInterface
from soku_rl.network_session import run_session


class Connection:
    def __init__(self, frames, reply):
        self.frames, self.reply, self.commands = iter(frames), reply, []

    def request(self, operation, payload):
        if operation == "poll":
            return {"menu_reply": "not_requested", "input_events": (), "records": [next(self.frames)]}
        assert operation == "submit"
        self.commands.append(payload)
        return {"reply": self.reply}


def frame(match, round_id, step, phase, events, scores):
    return {"match": match, "round": round_id, "frame": step, "seat": 0,
            "phase": phase, "scores": scores, "observations": observations(step, 0),
            "events": tuple({"kind": kind} for kind in events)}


def interface():
    return LearningInterface(EpisodeConfig(7200, 4, 3, 5, "state", VISIBILITY),
                             LearningConfig("combat", True, 8, 1.))


def test_three_rounds_and_a_rematch_reset_model_memory_without_offline_commands():
    frames = [
        frame(1, 0, 1, "battle", ("match_started", "round_started"), (0, 0)),
        frame(1, 0, 2, "between_rounds", (), (0, 0)),
        frame(1, 0, 3, "between_rounds", ("score_changed",), (1, 0)),
        frame(1, 1, 200, "battle", ("round_started",), (1, 0)),
        frame(1, 1, 201, "between_rounds", ("score_changed",), (1, 1)),
        frame(1, 2, 400, "battle", ("round_started",), (1, 1)),
        frame(1, 2, 401, "match_finished", ("score_changed", "match_finished"), (2, 1)),
        frame(1, 2, 402, "menu", (), (2, 1)),
        frame(2, 0, 1, "battle", ("match_started", "round_started"), (0, 0)),
        frame(2, 0, 2, "between_rounds", ("score_changed",), (0, 1)),
        frame(2, 1, 200, "battle", ("round_started",), (0, 1)),
        frame(2, 1, 201, "match_finished", ("score_changed", "match_finished"), (0, 2)),
    ]
    connection, policy, log = Connection(frames, "accepted"), RecordingPolicy(), []
    result = run_session(connection, policy, interface(), 0, 51, 2, 10., log.append)
    assert result["matches"] == 2 and result["rounds"] == result["decisions"] == 5
    assert result["last_scores"] == (0, 2)
    assert [seed for seed, _ in policy.episodes] == list(range(51, 56))
    assert [(c["match"], c["frame"], c["duration"]) for c in connection.commands] == [
        (1, 1, 3), (1, 200, 3), (1, 400, 3), (2, 1, 3), (2, 200, 3)]


@pytest.mark.parametrize("reply", ("late", "queue_full", "wrong_round"))
def test_rejected_command_is_logged_and_stops_before_another_decision(reply):
    connection = Connection([frame(1, 0, 1, "battle", ("round_started",), (0, 0))], reply)
    log = []
    with pytest.raises(RuntimeError, match=reply):
        run_session(connection, RecordingPolicy(), interface(), 0, 1, 1, 10., log.append)
    assert log[-1]["kind"] == "command" and log[-1]["reply"] == reply
    assert len(connection.commands) == 1


def test_disconnect_is_an_interruption_and_not_a_match_result():
    connection = Connection([
        frame(1, 0, 1, "battle", ("round_started",), (0, 0)),
        frame(1, 0, 2, "disconnected", ("match_interrupted",), (0, 0)),
    ], "accepted")
    with pytest.raises(ConnectionError, match="interrupted"):
        run_session(connection, RecordingPolicy(), interface(), 0, 1, 1, 10., [].append)


def test_accepted_input_that_later_expires_stops_inference():
    class ExpiredConnection:
        def request(self, operation, payload):
            assert operation == "poll"
            return {"input_events": ({"request": 7, "result": "expired", "command_type": 1},)}

    log = []
    with pytest.raises(RuntimeError, match="7 expired"):
        run_session(ExpiredConnection(), RecordingPolicy(), interface(), 0, 1, 1, 10., log.append)
    assert log == [{"kind": "input_event", "request": 7, "result": "expired", "command_type": 1}]
