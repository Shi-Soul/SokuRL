"""Every frame retains a correctly scored choice from the full action space."""
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3.common.vec_env import DummyVecEnv

from soku_rl.env import EpisodeConfig
from soku_rl.env.encoding import decode_action
from soku_rl.env.match import LEGACY_MATCH
from soku_rl.env.wrappers.learning import LearningConfig, LearningInterface
from soku_rl.rl.persistent_policy import RepeatMixtureHead
from soku_rl.rl.ppo import create_ppo
from test_env_timing import VISIBILITY
from test_shared_ppo import fixture_config, save_contract


def test_repeat_mixture_matches_exact_probabilities_for_all_previous_commands():
    torch.set_num_threads(1)
    linear = torch.nn.Linear(12, 576)
    torch.nn.init.zeros_(linear.weight)
    torch.nn.init.zeros_(linear.bias)
    head = RepeatMixtureHead(linear, .8)
    latent = torch.zeros(576, 12)
    latent[:, -8:] = torch.tensor([decode_action(a).inputs for a in range(576)])
    probabilities = head(latent).exp()
    expected = torch.full_like(probabilities, .2 / 576) + torch.eye(576) * .8
    torch.testing.assert_close(probabilities, expected)
    assert torch.all(probabilities > 0)
    # Interrupted commands are scored under the entire mixture, not a hidden gate.
    loss = -head(latent).gather(1, ((torch.arange(576) + 1) % 576)[:, None]).mean()
    loss.backward()
    assert head.gate.bias.grad.abs().item() > 0
    assert all(torch.isfinite(p.grad).all() for p in head.parameters())


def test_shared_factory_updates_and_restores_persistent_policy(tmp_path):
    torch.set_num_threads(1)
    interface = LearningInterface(EpisodeConfig(3, 1, 1, 0, "privileged_state", VISIBILITY, LEGACY_MATCH),
                                  LearningConfig("full", False, 1, 0.))

    class CommandGame(gym.Env):
        observation_space = interface.observation_space
        action_space = interface.action_space

        def reset(self, **kwargs):
            super().reset(**kwargs)
            self.observation = np.zeros(self.observation_space.shape, np.float32)
            self.frame = 0
            return self.observation.copy(), {}

        def step(self, action):
            self.frame += 1
            self.observation[-8:] = decode_action(action).inputs
            return self.observation.copy(), float(action == 256), self.frame == 3, False, {}

    config = fixture_config("mlp") | {"name": "br", "matchups": {"mode": "fixed"}}
    config["ppo"].update(action_persistence={"repeat_probability": .8},
        initial_action_prior={"button_probability": .05})
    config["ppo"]["policy_kwargs"].update(
        features_extractor_class="soku_rl.rl.persistent_policy.ActionContextFeatures",
        features_extractor_kwargs=dict(history_frames=1, object_features=2, player_features=8, features_dim=8))
    env = DummyVecEnv([CommandGame])
    try:
        model, _ = create_ppo(env, interface, config, {"kind": "fresh"}, "cpu", 19)
        observations = torch.from_numpy(env.reset())
        model.policy.set_training_mode(False)
        actions, _, log_prob = model.policy(observations)
        _, evaluated, entropy = model.policy.evaluate_actions(observations, actions)
        torch.testing.assert_close(log_prob, evaluated)
        distribution = model.policy.get_distribution(observations)
        torch.testing.assert_close(log_prob, distribution.log_prob(actions))
        assert torch.isfinite(entropy).all()
        before = model.policy.action_net.gate.bias.detach().clone()
        model.learn(8)
        assert not torch.equal(before, model.policy.action_net.gate.bias)
        assert all(torch.isfinite(p).all() for p in model.policy.parameters())
        checkpoint = tmp_path / "trained.zip"
        model.save(checkpoint)
        contract = save_contract(tmp_path, SimpleNamespace(interface=interface), config)
        for kind in ("weights", "checkpoint"):
            restored, _ = create_ppo(env, interface, config,
                {"kind": kind, "path": str(checkpoint), "training_config": contract}, "cpu", 20)
            for key, value in model.policy.state_dict().items():
                torch.testing.assert_close(restored.policy.state_dict()[key], value, rtol=0, atol=0)
            assert restored.num_timesteps == (8 if kind == "checkpoint" else 0)
        # Saving the standalone policy must also retain its constructor option.
        model.policy.save(tmp_path / "policy.pt")
        restored_policy = type(model.policy).load(tmp_path / "policy.pt", device="cpu")
        torch.testing.assert_close(restored_policy.get_distribution(observations).distribution.probs,
                                   model.policy.get_distribution(observations).distribution.probs)
    finally:
        env.close()


@pytest.mark.parametrize("probability", [0., 1., float("nan"), True])
def test_invalid_repeat_probability_fails_before_network_build(probability):
    from soku_rl.rl.persistent_policy import PersistentActorCriticPolicy
    with pytest.raises(ValueError, match="repeat_probability"):
        PersistentActorCriticPolicy(repeat_probability=probability)
