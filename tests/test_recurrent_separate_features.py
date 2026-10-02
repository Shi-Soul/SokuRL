"""Verify critic-gradient isolation through the shared recurrent PPO factory."""
from pathlib import Path

from hydra import compose, initialize_config_dir
import numpy as np
from omegaconf import OmegaConf
import pytest
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from test_shared_ppo import fixture_config, fixture_env, save_contract
from soku_rl.policy.loader import load_policy
from soku_rl.rl.behavior_cloning import ObservationContractEnv
from soku_rl.rl.ppo import create_ppo, parameter_hash
from soku_rl.rl.recurrent_cloning import zero_states


class TrainableFeatures(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, 8)
        self.network = nn.Sequential(nn.Linear(int(np.prod(observation_space.shape)), 8), nn.Tanh())

    def forward(self, observations):
        return self.network(observations.flatten(1))


def test_offline_and_online_configs_retain_the_same_separate_architecture():
    with initialize_config_dir(config_dir=str(Path(__file__).parents[1] / "config"), version_base="1.3"):
        offline = compose(config_name="pretrain_recurrent_separate_demonstrations")
        online = compose(config_name="train", overrides=["algorithm=br",
            "rl=recurrent_separate_transfer", "track=superhuman_combat", "wrappers=superhuman_learning"])
        assert OmegaConf.to_container(offline.rl.ppo.policy_kwargs) == OmegaConf.to_container(online.rl.ppo.policy_kwargs)
        assert offline.rl.ppo.policy_kwargs.share_features_extractor is False
        assert online.rl.ppo.learning_rate == .0001
        assert offline.rl.ppo.learning_rate == .0003


@pytest.mark.parametrize("shared", [True, False])
def test_critic_only_update_changes_actor_features_only_when_shared(tmp_path, shared):
    torch.set_num_threads(1)
    env = fixture_env()
    config = fixture_config("lstm") | {"name": "br"}
    config["ppo"]["policy_kwargs"].update(share_features_extractor=shared,
        features_extractor_class=f"{__name__}.TrainableFeatures")
    model, _ = create_ppo(ObservationContractEnv(env.interface), env.interface, config,
        {"kind": "fresh"}, "cpu", 19)
    policy = model.policy
    pi, vf = policy.pi_features_extractor, policy.vf_features_extractor
    assert (pi is vf) is shared
    assert bool({p.data_ptr() for p in pi.parameters()} & {p.data_ptr() for p in vf.parameters()}) is shared
    before_pi, before_vf = parameter_hash(pi), parameter_hash(vf)
    observation = torch.ones((2, *env.single_observation_space.shape))
    _, values, _, _ = policy(observation, zero_states(policy, 2), torch.ones(2), deterministic=True)
    policy.optimizer.zero_grad()
    ((values - 1.) ** 2).mean().backward()
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in vf.parameters())
    if not shared:
        assert all(p.grad is None for p in pi.parameters())
        assert all(p.grad is None for p in policy.lstm_actor.parameters())
    policy.optimizer.step()
    assert (parameter_hash(pi) != before_pi) is shared
    assert parameter_hash(vf) != before_vf
    checkpoint = tmp_path / "model.zip"
    model.save(checkpoint)
    contract = save_contract(tmp_path, env, config)
    loaded = load_policy("separate", {"kind": "sb3_recurrent", "path": str(checkpoint),
        "training_config": contract}, env.interface, "cpu")
    assert loaded.model.policy.share_features_extractor is shared
    assert parameter_hash(loaded.model.policy) == parameter_hash(policy)
    restored, source = create_ppo(ObservationContractEnv(env.interface), env.interface, config,
        {"kind": "weights", "path": str(checkpoint), "training_config": contract}, "cpu", 23)
    assert parameter_hash(restored.policy) == parameter_hash(policy)
    assert restored.num_timesteps == 0 and not restored.policy.optimizer.state
    env.close()
