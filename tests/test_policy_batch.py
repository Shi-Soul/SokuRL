"""Grouped evaluation preserves greedy actions, model identity and stateful order."""
import numpy as np
import pytest
import torch

from soku_rl.policy.batch import episode_actions
from soku_rl.policy.dqn import DQNEpisode
from soku_rl.rl.dqn import DoubleDQN


def test_grouped_actions_do_not_mix_models_or_reorder_stateful_actors():
    class Model:
        def __init__(self, offset):
            self.offset = offset
            self.calls = []

        def predict(self, obs, deterministic):
            assert deterministic
            self.calls.append(len(obs))
            return obs[:, 0].astype(int) + self.offset, None

    order = []

    class Stateful:
        def act(self, obs):
            order.append(int(obs[0]))
            return len(order)

    first, second = Model(10), Model(20)
    stateful = Stateful()
    requests = {
        (3, 0): (DQNEpisode(first), np.array([1])),
        (3, 1): (stateful, np.array([3])),
        (5, 0): (DQNEpisode(second), np.array([2])),
        (5, 1): (stateful, np.array([5])),
        (8, 0): (DQNEpisode(first), np.array([4])),
    }
    assert episode_actions(requests) == {(3, 0): 11, (3, 1): 1, (5, 0): 22, (5, 1): 2, (8, 0): 14}
    assert first.calls == [2] and second.calls == [1]
    assert order == [3, 5]
    assert episode_actions({}) == {}


@pytest.mark.parametrize("dictionary", [False, True])
def test_real_dqn_batch_matches_individual_greedy_predictions(dictionary):
    from stable_baselines3.common.envs import SimpleMultiObsEnv
    from stable_baselines3.common.vec_env import DummyVecEnv
    import gymnasium as gym
    torch.set_num_threads(1)
    if dictionary:
        env = DummyVecEnv([lambda: SimpleMultiObsEnv(random_start=False)])
        policy = "MultiInputPolicy"
    else:
        env = DummyVecEnv([lambda: gym.make("CartPole-v1")])
        policy = "MlpPolicy"
    try:
        model = DoubleDQN(policy, env, seed=81, device="cpu", buffer_size=32)
        actor = DQNEpisode(model)
        env.observation_space.seed(712)
        observations = [env.observation_space.sample() for _ in range(8)]
        expected = {key: actor.act(obs) for key, obs in enumerate(observations)}
        actual = episode_actions({key: (actor, obs) for key, obs in enumerate(observations)})
        assert actual == expected
    finally:
        env.close()
