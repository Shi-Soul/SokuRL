from copy import deepcopy

import pytest
import torch

from soku_rl.env.wrappers.learning import LearningConfig, LearningVectorEnv
from soku_rl.policy.loader import load_policy
from soku_rl.policy.population import UniformPolicy
from soku_rl.rl.factorized_policy import DirectionButtonHead
from soku_rl.rl.opponent_env import OpponentMixtureVecEnv
from soku_rl.rl.ppo import create_ppo, parameter_hash
from soku_rl.rl.recurrent_storage import SparseRecurrentRolloutBuffer
from test_shared_ppo import fixture_config, fixture_env, save_contract


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("sparse", [False, True])
def test_recurrent_factorized_update_memory_checkpoint_and_portable_policy(tmp_path, device, sparse):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.set_num_threads(1)
    original = fixture_env()
    env = LearningVectorEnv(original.env, LearningConfig("full", False, 0, 0.))
    view = OpponentMixtureVecEnv(env, 0, [UniformPolicy("random", 576)], [1.], 19)
    config = fixture_config("lstm") | {"name": "br", "matchups": {"mode": "fixed"}}
    config["ppo"]["action_factorization"] = {"button_probability": .05}
    config["ppo"]["policy_kwargs"]["n_lstm_layers"] = 2
    if sparse:
        config["ppo"]["recurrent_storage"] = "sparse"
    try:
        model, _ = create_ppo(view, env.interface, config, {"kind": "fresh"}, device, 19)
        observations = torch.as_tensor(view.reset(), device=device)
        states = model._last_lstm_states
        starts = torch.tensor([1., 0.], device=device)
        actions, _, log_prob, memory = model.policy(observations, states, starts)
        _, scored, entropy = model.policy.evaluate_actions(observations, actions, states, starts)
        torch.testing.assert_close(log_prob, scored)
        assert torch.isfinite(entropy).all()
        assert memory.pi[0].shape == (2, 2, 8)
        assert not torch.equal(memory.pi[0], states.pi[0])
        parameters = dict(model.policy.named_parameters())
        selected = ("action_net.factors.weight", "lstm_actor.weight_ih_l0", "lstm_critic.weight_ih_l0")
        before = {name: parameters[name].detach().clone() for name in selected}
        model.learn(16)
        assert all(not torch.equal(parameters[name], before[name]) for name in selected)
        assert set(parameters.values()) == {parameter for group in model.policy.optimizer.param_groups
            for parameter in group["params"]}
        checkpoint = tmp_path / "trained.zip"
        model.save(checkpoint)
        contract = save_contract(tmp_path, env, config)
        for kind in ("checkpoint", "weights"):
            restored, _ = create_ppo(view, env.interface, config,
                {"kind": kind, "path": str(checkpoint), "training_config": contract}, device, 20)
            assert isinstance(restored.policy.action_net, DirectionButtonHead)
            assert isinstance(restored.rollout_buffer, SparseRecurrentRolloutBuffer) == sparse
            assert parameter_hash(restored.policy) == parameter_hash(model.policy)
            assert restored.num_timesteps == (16 if kind == "checkpoint" else 0)
            actual = restored.policy.optimizer.state_dict()
            expected = model.policy.optimizer.state_dict()
            if kind == "weights":
                assert not actual["state"]
            else:
                assert actual["param_groups"] == expected["param_groups"]
                for parameter, values in expected["state"].items():
                    for key, value in values.items():
                        torch.testing.assert_close(actual["state"][parameter][key].cpu(), value.cpu(), rtol=0, atol=0)
        portable = load_policy("factorized-recurrent", {"kind": "sb3_recurrent", "path": str(checkpoint),
            "training_config": contract}, env.interface, device)
        first, second = portable.spawn(13), portable.spawn(13)
        observation, _ = env.reset({0: 17})
        for _ in range(3):
            action = first.act(observation[0]["player_0"])
            assert action == second.act(observation[0]["player_0"]) and 0 <= action < 576
        model.policy.save(tmp_path / "policy.pt")
        standalone = type(model.policy).load(tmp_path / "policy.pt", device=device)
        assert standalone.lstm_actor.num_layers == 2 and standalone.lstm_actor.hidden_size == 8
        expected, expected_state = model.policy.get_distribution(observations, states.pi, starts)
        actual, actual_state = standalone.get_distribution(observations, states.pi, starts)
        torch.testing.assert_close(actual.distribution.probs, expected.distribution.probs, rtol=0, atol=0)
        for left, right in zip(actual_state, expected_state, strict=True):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        incompatible = deepcopy(config)
        del incompatible["ppo"]["action_factorization"]
        with pytest.raises(ValueError, match="architecture"):
            create_ppo(view, env.interface, incompatible,
                {"kind": "weights", "path": str(checkpoint), "training_config": contract}, device, 20)
    finally:
        view.close()
        env.close()
