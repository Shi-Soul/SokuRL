from copy import deepcopy

import pytest
import torch

from soku_rl.env.wrappers.learning import LearningConfig, LearningVectorEnv
from soku_rl.policy.loader import load_policy
from soku_rl.policy.population import UniformPolicy
from soku_rl.rl.factorized_policy import DirectionButtonHead
from soku_rl.rl.opponent_env import OpponentMixtureVecEnv
from soku_rl.rl.ppo import create_ppo, parameter_hash
from test_shared_ppo import fixture_config, fixture_env, save_contract


def test_full_distribution_equals_direction_times_button_probabilities():
    torch.set_num_threads(1)
    torch.manual_seed(37)
    head = DirectionButtonHead(5, .05, True).double()
    torch.nn.init.normal_(head.factors.weight, std=.3)
    latent = torch.randn(4, 5, dtype=torch.float64)
    factors = head.factors(latent)
    direction = factors[:, :9].softmax(1)
    buttons = factors[:, 9:].sigmoid()
    expected = []
    for action in range(576):
        probability = direction[:, action // 64].clone()
        for bit in range(6):
            probability *= buttons[:, bit] if action >> bit & 1 else 1 - buttons[:, bit]
        expected.append(probability)
    actual = head(latent).softmax(1)
    torch.testing.assert_close(actual, torch.stack(expected, dim=1), rtol=1e-12, atol=1e-12)
    assert (actual > 0).all()
    torch.testing.assert_close(actual.sum(1), torch.ones(4, dtype=torch.float64))


def test_one_command_updates_shared_direction_and_button_factors():
    head = DirectionButtonHead(5, .05, True).double()
    logits = head(torch.zeros(1, 5, dtype=torch.float64))
    action = 6 * 64 + 0b010101
    torch.nn.functional.cross_entropy(logits, torch.tensor([action])).backward()
    expected_direction = torch.full((9,), 1 / 9, dtype=torch.float64)
    expected_direction[6] -= 1
    expected_buttons = torch.tensor([.05 - ((action >> bit) & 1) for bit in range(6)], dtype=torch.float64)
    torch.testing.assert_close(head.factors.bias.grad, torch.cat((expected_direction, expected_buttons)), rtol=1e-6, atol=1e-8)
    assert sum(p.numel() for p in head.parameters()) == 6 * 15


@pytest.mark.parametrize("probability", [0., 1., float("nan"), float("inf"), True])
def test_invalid_button_probabilities_fail(probability):
    with pytest.raises(ValueError, match="button_probability"):
        DirectionButtonHead(5, probability, True)


def test_shared_ppo_updates_saves_loads_and_rejects_other_heads(tmp_path):
    torch.set_num_threads(1)
    original = fixture_env()
    env = LearningVectorEnv(original.env, LearningConfig("full", False, 0, 0.))
    view = OpponentMixtureVecEnv(env, 0, [UniformPolicy("random", 576)], [1.], 19)
    config = fixture_config("mlp") | {"name": "br", "matchups": {"mode": "fixed"}}
    config["ppo"]["action_factorization"] = {"button_probability": .05}
    try:
        model, _ = create_ppo(view, env.interface, config, {"kind": "fresh"}, "cpu", 19)
        observations = torch.as_tensor(view.reset())
        actions, _, log_prob = model.policy(observations)
        _, scored, entropy = model.policy.evaluate_actions(observations, actions)
        torch.testing.assert_close(log_prob, scored)
        assert torch.isfinite(entropy).all()
        before = parameter_hash(model.policy)
        head_before = model.policy.action_net.factors.weight.detach().clone()
        model.learn(8)
        assert parameter_hash(model.policy) != before
        assert not torch.equal(head_before, model.policy.action_net.factors.weight)
        path = tmp_path / "trained.zip"
        model.save(path)
        contract = save_contract(tmp_path, env, config)
        for kind in ("checkpoint", "weights"):
            restored, _ = create_ppo(view, env.interface, config,
                {"kind": kind, "path": str(path), "training_config": contract}, "cpu", 20)
            assert parameter_hash(restored.policy) == parameter_hash(model.policy)
            assert restored.num_timesteps == (8 if kind == "checkpoint" else 0)
        portable = load_policy("factorized", {"kind": "sb3", "path": str(path),
            "training_config": contract}, env.interface, "cpu")
        actor = portable.spawn(13)
        observation, _ = env.reset({0: 17})
        assert 0 <= actor.act(observation[0]["player_0"]) < 576
        model.policy.save(tmp_path / "policy.pt")
        restored = type(model.policy).load(tmp_path / "policy.pt", device="cpu")
        torch.testing.assert_close(restored.get_distribution(observations).distribution.probs,
            model.policy.get_distribution(observations).distribution.probs)
        incompatible = deepcopy(config)
        del incompatible["ppo"]["action_factorization"]
        with pytest.raises(ValueError, match="architecture"):
            create_ppo(view, env.interface, incompatible,
                {"kind": "weights", "path": str(path), "training_config": contract}, "cpu", 20)
        incompatible = deepcopy(config)
        incompatible["ppo"]["initial_action_prior"] = {"button_probability": .05}
        with pytest.raises(ValueError, match="action-head"):
            create_ppo(view, env.interface, incompatible, {"kind": "fresh"}, "cpu", 20)
    finally:
        view.close()
        env.close()


@pytest.mark.parametrize("method", ["ippo", "nfsp"])
def test_other_marl_methods_use_the_same_factorized_ppo(tmp_path, method):
    from stable_baselines3 import PPO
    from soku_rl.marl.ippo import train_ippo
    from soku_rl.marl.nfsp import train_nfsp
    torch.set_num_threads(1)
    original = fixture_env()
    env = LearningVectorEnv(original.env, LearningConfig("full", False, 0, 0.))
    config = fixture_config("mlp") | {"name": method}
    config["ppo"]["action_factorization"] = {"button_probability": .05}
    try:
        if method == "ippo":
            train_ippo(env, config, "cpu", 7, tmp_path)
        else:
            config.update(iterations=1, timesteps_per_iteration=8, anticipatory_param=.1,
                average={"capacity": 12, "batch_size": 4, "updates": 2}, resume={"kind": "fresh"})
            train_nfsp(env, config, "cpu", 7, tmp_path)
        for player in (0, 1):
            model = PPO.load(tmp_path / f"player_{player}/final.zip", device="cpu")
            assert isinstance(model.policy.action_net, DirectionButtonHead)
    finally:
        env.close()
