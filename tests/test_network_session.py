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
            return {"closed": False, "menu_reply": "not_requested", "input_events": (), "records": [next(self.frames)]}
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


@pytest.mark.parametrize("reply", ("invalid", "queue_full", "too_frequent"))
def test_rejected_command_is_logged_and_stops_before_another_decision(reply):
    connection = Connection([frame(1, 0, 1, "battle", ("round_started",), (0, 0))], reply)
    log = []
    with pytest.raises(RuntimeError, match=reply):
        run_session(connection, RecordingPolicy(), interface(), 0, 1, 1, 10., log.append)
    assert log[-1]["kind"] == "command" and log[-1]["reply"] == reply
    assert len(connection.commands) == 1


@pytest.mark.parametrize("reply", ("late", "wrong_round"))
def test_game_advancing_during_inference_is_logged_without_ending_the_match(reply):
    connection = Connection([
        frame(1, 0, 1, "battle", ("round_started",), (0, 0)),
        frame(1, 0, 2, "battle", (), (0, 0)),
        frame(1, 0, 3, "battle", (), (0, 0)),
        frame(1, 0, 4, "battle", (), (0, 0)),
        frame(1, 0, 5, "match_finished", ("match_finished",), (2, 0)),
    ], reply)
    log = []
    result = run_session(connection, RecordingPolicy(), interface(), 0, 1, 1, 10., log.append)
    assert result["matches"] == 1 and result["dropped_commands"][reply] == 2
    assert result["accepted_commands"] == 0 and result["decisions"] == 2
    assert len([x for x in log if x["kind"] == "command" and x["reply"] == reply]) == 2


def test_backlog_keeps_observation_history_but_skips_obsolete_decisions():
    class BatchedConnection(Connection):
        def request(self, operation, payload):
            if operation == "poll":
                return {"closed": False, "menu_reply": "not_requested", "input_events": (), "records": next(self.frames)}
            return super().request(operation, payload)

    batch = [frame(1, 0, index, "battle", ("round_started",) if index == 1 else (), (0, 0))
             for index in range(1, 11)]
    connection = BatchedConnection([batch, [frame(1, 0, 11, "match_finished", ("match_finished",), (2, 0))]], "accepted")
    policy, log = RecordingPolicy(), []
    result = run_session(connection, policy, interface(), 0, 1, 1, 10., log.append)
    assert result["skipped_decisions"] == 3 and result["accepted_commands"] == 1
    assert len(policy.episodes[0][1]) == 1
    assert [x["frame"] for x in connection.commands] == [10]
    assert [x["frame"] for x in log if x["kind"] == "frame"] == list(range(1, 12))


def test_disconnect_is_an_interruption_and_not_a_match_result():
    connection = Connection([
        frame(1, 0, 1, "battle", ("round_started",), (0, 0)),
        frame(1, 0, 2, "disconnected", ("match_interrupted",), (0, 0)),
    ], "accepted")
    with pytest.raises(ConnectionError, match="interrupted"):
        run_session(connection, RecordingPolicy(), interface(), 0, 1, 1, 10., [].append)


def test_returning_to_selection_abandons_the_match_and_allows_a_new_one():
    connection = Connection([
        frame(1, 0, 1, "battle", ("round_started",), (0, 0)),
        frame(1, 0, 2, "menu", ("match_interrupted",), (0, 0)),
        frame(2, 0, 1, "battle", ("round_started",), (0, 0)),
        frame(2, 0, 2, "match_finished", ("match_finished",), (2, 0)),
    ], "accepted")
    policy = RecordingPolicy()
    result = run_session(connection, policy, interface(), 0, 11, 1, 10., [].append)
    assert result["matches"] == 1 and result["interrupted_matches"] == 1
    assert result["last_scores"] == (2, 0)
    assert [seed for seed, _ in policy.episodes] == [11, 12]


def test_accepted_input_that_later_expires_stops_inference():
    class ExpiredConnection:
        def request(self, operation, payload):
            assert operation == "poll"
            return {"closed": False, "input_events": ({"request": 7, "result": "expired", "command_type": 1},)}

    log = []
    with pytest.raises(RuntimeError, match="7 expired"):
        run_session(ExpiredConnection(), RecordingPolicy(), interface(), 0, 1, 1, 10., log.append)
    assert log == [{"kind": "input_event", "request": 7, "result": "expired", "command_type": 1}]


def test_closing_the_game_preserves_partial_results_without_inventing_a_win():
    class ClosingConnection(Connection):
        def request(self, operation, payload):
            if operation == "poll":
                try:
                    return super().request(operation, payload)
                except StopIteration:
                    return {"closed": True}
            return super().request(operation, payload)

    connection = ClosingConnection([
        frame(1, 0, 1, "battle", ("round_started",), (0, 0)),
        frame(1, 0, 2, "between_rounds", ("score_changed",), (1, 0)),
    ], "accepted")
    result = run_session(connection, RecordingPolicy(), interface(), 0, 1, 100, 10., [].append)
    assert result["termination"] == "game_closed"
    assert result["matches"] == 0 and result["rounds"] == 1
    assert result["last_scores"] == (1, 0) and result["accepted_commands"] == 1


def test_final_score_is_not_replaced_by_a_later_match_in_the_same_batch():
    class ResultBatch:
        def request(self, operation, payload):
            assert operation == "poll"
            return {"closed": False, "menu_reply": "not_requested", "input_events": (), "records": [
                frame(1, 1, 500, "match_finished", ("match_finished",), (2, 0)),
                frame(2, 0, 1, "battle", ("round_started",), (0, 0)),
            ]}

    result = run_session(ResultBatch(), RecordingPolicy(), interface(), 0, 1, 1, 10., [].append)
    assert result["termination"] == "matches_completed"
    assert result["last_scores"] == (2, 0) and result["decisions"] == 0
