"""Local play completes original matches and uses the same worker for rematches."""
from pathlib import Path

from hydra import compose, initialize_config_dir

from soku_rl.play.local_session import run_session
from soku_rl.play.match import MatchFrame, MatchState
from test_live_policy import RecordingPolicy, observations
from test_match_policies import ContinuousPolicy, interface


class Connection:
    def __init__(self):
        self.match, self.frame = 0, 0
        self.operations = []

    def request(self, operation, payload):
        self.operations.append((operation, payload))
        if operation in {"start", "reset"}:
            self.match += 1
            self.frame = 0
        elif operation == "step":
            assert set(payload) == {1}
            self.frame += 1
        elif operation == "finish":
            assert self.frame == 4
            return {}
        else:
            raise AssertionError(operation)
        round_id = int(self.frame >= 3)
        scores = (2, 0) if self.frame == 4 else (int(self.frame >= 2), 0)
        hp = (10000, 0) if self.frame in (1, 2, 4) else (10000, 10000)
        state = MatchState(self.match, round_id, self.frame, scores, hp, "battle")
        return {"closed": False, "frame": MatchFrame(state, observations(self.frame, 0))}


def test_complete_matches_preserve_continuous_actor_and_save_before_reset():
    connection, policy, records = Connection(), ContinuousPolicy(), []
    result = run_session(connection, {1: policy}, interface(1, 0), 1732, 2, 30., records.append)
    assert result["termination"] == "matches_completed"
    assert result["matches"] == 2 and result["frames"] == 8
    assert [len(values) for _, values in policy.episodes] == [5, 5]
    operations = [operation for operation, _ in connection.operations]
    assert operations == ["start", *(["step"] * 4), "finish", "reset", *(["step"] * 4), "finish"]
    assert len(records) == 10


def test_closed_game_is_not_counted_as_a_completed_match():
    class ClosedConnection(Connection):
        def request(self, operation, payload):
            if operation == "step":
                return {"closed": True, "replay_saved": False}
            return super().request(operation, payload)
    records = []
    result = run_session(ClosedConnection(), {1: RecordingPolicy()}, interface(1, 0), 1, 1, 30., records.append)
    assert result["termination"] == "game_closed" and result["matches"] == 0
    assert records[-1] == {"kind": "game_closed", "replay_saved": False}


def test_local_play_composes_the_shared_superhuman_configuration():
    root = Path(__file__).parents[1]
    with initialize_config_dir(version_base="1.3", config_dir=str(root / "config")):
        config = compose(config_name="local_match")
    assert config.episode.observation_mode == "privileged_state"
    assert config.episode.decision_frames == 1 and config.episode.latency_frames == 0
    assert config.rules.roster == ["god"]
    assert config.players.player_1 == "human"
