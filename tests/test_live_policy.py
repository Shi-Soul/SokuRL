"""Live policies must see the exact training features and reset between rounds."""
from types import SimpleNamespace

import numpy as np
import pytest

from test_env_timing import VISIBILITY
from soku_rl.env import EpisodeConfig, HisoutenParallelEnv
from soku_rl.env.encoding import AGENTS
from soku_rl.learning_wrappers import LearningConfig, LearningInterface, LearningParallelEnv
from soku_rl.live_policy import LivePolicy
from soku_rl.pomg import Outcome, TimeStep
from soku_rl.visible_state import StateObservation, STATE_FEATURES


def observations(frame, elapsed):
    values = np.zeros(STATE_FEATURES)
    values[[0, 4, 8, 12]] = 1
    values[1], values[9] = .25+elapsed*.001, .75-elapsed*.001
    values[5], values[13] = .9, .8
    other = values.copy()
    other[:8], other[8:16] = values[8:16], values[:8]
    return tuple(StateObservation(frame, tuple(v)) for v in (values, other))


class PublicBackend:
    def reset_slots(self, seeds):
        self.frame = 0
        return {0: self.current()}

    def current(self):
        return TimeStep(self.frame, observations(self.frame, self.frame), (0., 0.), Outcome.ONGOING, {})

    def step(self, actions):
        self.frame += 1
        return {0: self.current()}

    def close(self):
        pass


class RecordingPolicy:
    def __init__(self):
        self.episodes = []

    def spawn(self, seed):
        records = []
        self.episodes.append((seed, records))

        def act(observation):
            records.append(observation.copy())
            return 41

        return SimpleNamespace(act=act)


@pytest.mark.parametrize("seat", (0, 1))
def test_live_features_equal_training_at_each_decision(seat):
    episode = EpisodeConfig(17, 4, 3, 5, "state", VISIBILITY)
    wrappers = LearningConfig("combat", True, 8, 1.)
    env = LearningParallelEnv(HisoutenParallelEnv(PublicBackend(), episode), wrappers)
    training, _ = env.reset(seed=5)
    policy = RecordingPolicy()
    live = LivePolicy(policy, env.interface, seat)
    live.start_round(1500, observations(1500, 0), 5)
    for decision in range(3):
        command = live.act()
        assert command == env.interface.command(41)
        np.testing.assert_array_equal(policy.episodes[0][1][-1], training[AGENTS[seat]])
        training, *_ = env.step(dict.fromkeys(AGENTS, 41))
        for elapsed in range(decision*3+1, decision*3+4):
            live.observe(1500+elapsed, observations(1500+elapsed, elapsed))
    live.start_round(18000, observations(18000, 0), 9)
    live.act()
    assert [seed for seed, _ in policy.episodes] == [5, 9]
    np.testing.assert_array_equal(policy.episodes[1][1][0], policy.episodes[0][1][0])


def test_live_history_rejects_missing_frames_and_saturates_only_the_clock():
    interface = LearningInterface(EpisodeConfig(3, 4, 3, 5, "state", VISIBILITY),
                                  LearningConfig("combat", True, 8, 1.))
    policy = RecordingPolicy()
    live = LivePolicy(policy, interface, 0)
    live.start_round(100, observations(100, 0), 1)
    with pytest.raises(RuntimeError, match="pending decision"):
        live.observe(101, observations(101, 1))
    live.act()
    with pytest.raises(RuntimeError, match="exactly once"):
        live.act()
    with pytest.raises(RuntimeError, match="consecutive"):
        live.observe(102, observations(102, 2))
    for elapsed in range(1, 7):
        live.observe(100+elapsed, observations(100+elapsed, elapsed))
        if live.decision_due:
            live.act()
    assert policy.episodes[0][1][-1][-65] == 1.
    assert interface.observation_space.contains(policy.episodes[0][1][-1])
    live.stop()
    assert not live.decision_due
    with pytest.raises(RuntimeError, match="exactly once"):
        live.act()
