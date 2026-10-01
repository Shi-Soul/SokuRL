"""Stepped play preserves script memory, learner resets and configured latency."""
import pytest

from soku_rl.env import EpisodeConfig
from soku_rl.env.encoding import decode_action
from soku_rl.env.match import LEGACY_MATCH
from soku_rl.env.wrappers.learning import LearningConfig, LearningInterface
from soku_rl.play.match import MatchState
from soku_rl.play.match_policies import MatchPolicies
from soku_rl.policy.base import PlayActor
from test_env_timing import VISIBILITY
from test_live_policy import RecordingPolicy, observations


class ContinuousPolicy(RecordingPolicy):
    def spawn_play(self, seed):
        return PlayActor(self.spawn(seed), False)


def interface(stride, latency):
    return LearningInterface(EpisodeConfig(7200, 1, stride, latency, "state", VISIBILITY, LEGACY_MATCH),
                             LearningConfig("full", False, 0, 0.))


def test_script_continues_through_knockout_while_learner_resets():
    script, learner = ContinuousPolicy(), RecordingPolicy()
    players = MatchPolicies({0: script, 1: learner}, interface(1, 0), 31, 2)
    timeline = ((0, (0, 0), (10000, 10000)), (0, (0, 0), (10000, 0)),
                (0, (1, 0), (10000, 0)), (1, (1, 0), (10000, 10000)),
                (1, (1, 0), (10000, 10000)), (1, (2, 0), (10000, 0)))
    for frame, (round_id, scores, hp) in enumerate(timeline):
        step = players.advance(MatchState(1, round_id, frame, scores, hp, "battle"), observations(frame, 0))
        assert step.inputs[0] == decode_action(41).inputs
        assert step.inputs[1] == decode_action(41 if min(hp) else 256).inputs
    assert step.phase == "match_finished"
    assert [len(records) for _, records in script.episodes] == [6]
    assert [len(records) for _, records in learner.episodes] == [1, 2]
    # A new original match resets both kinds of private actor state.
    players.advance(MatchState(2, 0, 0, (0, 0), (10000, 10000), "battle"), observations(0, 0))
    assert len(script.episodes) == 2 and len(learner.episodes) == 3


def test_single_ai_seat_uses_training_latency_and_discards_prior_round_queue():
    policy = RecordingPolicy()
    players = MatchPolicies({1: policy}, interface(3, 5), 13, 2)
    for frame in range(12):
        round_id = int(frame >= 6)
        hp = (10000, 0) if frame in (4, 5) else (10000, 10000)
        step = players.advance(MatchState(1, round_id, frame, (round_id, 0), hp, "battle"), observations(frame, 0))
        assert set(step.inputs) == {1}  # The human seat is never overwritten.
        assert step.inputs[1] == decode_action(41 if frame == 11 else 256).inputs
    assert [len(records) for _, records in policy.episodes] == [2, 2]


def test_missing_frame_is_rejected_before_policy_state_changes():
    policy = ContinuousPolicy()
    players = MatchPolicies({0: policy}, interface(1, 0), 13, 2)
    players.advance(MatchState(1, 0, 0, (0, 0), (10000, 10000), "battle"), observations(0, 0))
    with pytest.raises(ValueError, match="every frame"):
        players.advance(MatchState(1, 0, 2, (0, 0), (10000, 10000), "battle"), observations(2, 0))
    assert len(policy.episodes[0][1]) == 1
