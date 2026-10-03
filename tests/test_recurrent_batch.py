"""Optional frozen inference groups models but never episode memory or randomness."""
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.envs import SimpleMultiObsEnv
from stable_baselines3.common.vec_env import DummyVecEnv
import torch

from soku_rl.policy.batch import episode_actions, evaluation_inference
from soku_rl.policy.recurrent import RecurrentEpisode
from soku_rl.policy.recurrent_batch import recurrent_episode_actions


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dictionary", [False, True])
def test_real_policies_preserve_private_state_rng_and_asynchronous_replacement(device, dictionary, monkeypatch):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    torch.set_num_threads(1)
    factory = (lambda: SimpleMultiObsEnv(random_start=False)) if dictionary else (lambda: gym.make("CartPole-v1"))
    env = DummyVecEnv([factory])
    try:
        models = [RecurrentPPO("MultiInputLstmPolicy" if dictionary else "MlpLstmPolicy", env,
            n_steps=4, batch_size=4, seed=seed, device=device,
            policy_kwargs={"lstm_hidden_size": 16, "n_lstm_layers": 2, "net_arch": [16]}) for seed in (12, 78)]
        for model in models:
            model.policy.set_training_mode(False)
        separate = {key: RecurrentEpisode(models[key % 2], 90 + key) for key in range(5)}
        grouped = {key: RecurrentEpisode(models[key % 2], 90 + key) for key in range(5)}
        model_parameters = [{key: value.clone() for key, value in model.policy.state_dict().items()} for model in models]
        env.observation_space.seed(719)
        for frame in range(32):
            if frame == 9:
                del separate[1], grouped[1]
            if frame == 14:
                separate[5] = RecurrentEpisode(models[1], 103)
                grouped[5] = RecurrentEpisode(models[1], 103)
            observations = {key: env.observation_space.sample() for key in separate}
            expected = {key: actor.act(observations[key]) for key, actor in separate.items()}
            actual = recurrent_episode_actions({key: (actor, observations[key]) for key, actor in grouped.items()})
            assert actual == expected
            for key in separate:
                assert separate[key].rng.bit_generator.state == grouped[key].rng.bit_generator.state
                assert torch.equal(separate[key].start, grouped[key].start)
                for a, b in zip(separate[key].states, grouped[key].states, strict=True):
                    torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)
                for other in grouped:
                    if key != other:
                        assert grouped[key].states[0].data_ptr() != grouped[other].states[0].data_ptr()
        for model, before in zip(models, model_parameters, strict=True):
            assert all(torch.equal(value, before[key]) for key, value in model.policy.state_dict().items())
    finally:
        env.close()


def test_dispatch_is_explicit_and_duplicate_recurrent_actors_fail_before_other_calls():
    assert evaluation_inference({}) == evaluation_inference({"recurrent_batch": False})
    assert evaluation_inference({})[0] is episode_actions
    assert evaluation_inference({"recurrent_batch": True})[0] is recurrent_episode_actions
    for value in (1, "true", None):
        with pytest.raises(ValueError, match="boolean"):
            evaluation_inference({"recurrent_batch": value})
    assert recurrent_episode_actions({}) == {}
    actor = object.__new__(RecurrentEpisode)
    actor.model = object()
    calls = []
    other = SimpleNamespace(act=lambda observation: calls.append(observation))
    with pytest.raises(ValueError, match="only once"):
        recurrent_episode_actions({0: (other, 7), 1: (actor, 0), 2: (actor, 0)})
    assert not calls


def test_overridden_actor_behavior_remains_sequential():
    calls = []

    class CustomEpisode(RecurrentEpisode):
        def __init__(self):
            pass

        def act(self, observation):
            calls.append(observation)
            return len(calls)

    actor = CustomEpisode()
    assert recurrent_episode_actions({0: (actor, 7), 1: (actor, 12)}) == {0: 1, 1: 2}
    assert calls == [7, 12]


def test_facing_policy_batch_returns_absolute_commands_for_each_actor():
    from gymnasium import spaces
    from soku_rl.rl.behavior_cloning import ObservationContractEnv
    from soku_rl.rl.facing_policy import FacingRecurrentActorCriticPolicy

    interface = SimpleNamespace(observation_space=spaces.Box(-1., 1., shape=(4,), dtype=np.float32),
                                action_space=spaces.Discrete(576))
    env = DummyVecEnv([lambda: ObservationContractEnv(interface)])
    try:
        model = RecurrentPPO(FacingRecurrentActorCriticPolicy, env, seed=8, device="cpu", n_steps=2, batch_size=2,
            policy_kwargs={"facing_index": 0, "lstm_hidden_size": 8, "net_arch": []})
        model.policy.set_training_mode(False)
        with torch.no_grad():
            model.policy.action_net.weight.zero_()
            model.policy.action_net.bias.fill_(-15.)
            model.policy.action_net.bias[448] = 15.
        actors = [RecurrentEpisode(model, seed) for seed in (17, 51)]
        for frame in range(4):
            signs = (1, -1) if frame % 2 else (-1, 1)
            requests = {i: (actor, np.array([0., sign / 65536., .1, 0.], dtype=np.float32))
                for i, (actor, sign) in enumerate(zip(actors, signs, strict=True))}
            assert recurrent_episode_actions(requests) == {i: 448 if sign > 0 else 64 for i, sign in enumerate(signs)}
    finally:
        env.close()
