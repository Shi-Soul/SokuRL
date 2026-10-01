import numpy as np
import pytest

from soku_rl.env import EpisodeConfig, TwoPlayerVectorEnv
from soku_rl.env.match import LEGACY_MATCH
from soku_rl.env.wrappers.learning import LearningConfig, LearningVectorEnv
from soku_rl.policy.population import UniformPolicy
from soku_rl.rl.action_initialization import logical_action_prior
from soku_rl.rl.opponent_env import OpponentMixtureVecEnv
from soku_rl.rl.ppo import create_ppo
from test_env_timing import RecordingBackend, VISIBILITY
from test_shared_ppo import fixture_config, fixture_env, save_contract, torch


@pytest.mark.parametrize("policy_type", ["mlp", "lstm"])
def test_full_action_prior_and_restoration_never_reapply_bias(tmp_path, policy_type):
    torch.set_num_threads(1)
    episode = EpisodeConfig(3, 1, 1, 0, "diagnostic_state", VISIBILITY, LEGACY_MATCH)
    env = LearningVectorEnv(TwoPlayerVectorEnv(RecordingBackend(), 2, episode),
                           LearningConfig("full", False, 0, 0.))
    view = OpponentMixtureVecEnv(env, 0, [UniformPolicy("random", 576)], [1.], 19)
    config = fixture_config(policy_type) | {"name": "br", "matchups": {"mode": "fixed"}}
    config["ppo"]["initial_action_prior"] = {"button_probability": .05}
    try:
        model, _ = create_ppo(view, env.interface, config, {"kind": "fresh"}, "cpu", 19)
        logits = logical_action_prior(env.interface, {"button_probability": .05})
        np.testing.assert_allclose(model.policy.action_net.bias.detach().numpy(), logits)
        probabilities = torch.softmax(model.policy.action_net.bias, dim=0).detach().numpy()
        assert (probabilities > 0).all()
        np.testing.assert_allclose(probabilities.reshape(9, 64).sum(1), np.full(9, 1 / 9), rtol=1e-6)
        for bit in range(6):
            assert probabilities[((np.arange(576) % 64) >> bit & 1) == 1].sum() == pytest.approx(.05)
        # Actual PPO updates remain the upstream algorithm, including LSTM.
        before = model.policy.action_net.bias.detach().clone()
        model.learn(8)
        assert not torch.equal(before, model.policy.action_net.bias)
        path = tmp_path / "trained.zip"
        model.save(path)
        contract = save_contract(tmp_path, env, config)
        for kind in ("checkpoint", "weights"):
            restored, _ = create_ppo(view, env.interface, config,
                {"kind": kind, "path": str(path), "training_config": contract}, "cpu", 20)
            assert torch.equal(restored.policy.action_net.bias, model.policy.action_net.bias)
    finally:
        view.close()
        env.close()


def test_prior_validation_rejects_masks_and_reduced_action_sets():
    env = fixture_env()
    try:
        for probability in (0., 1., float("nan"), True):
            with pytest.raises(ValueError, match="button_probability"):
                logical_action_prior(env.interface, {"button_probability": probability})
        with pytest.raises(ValueError, match="576"):
            logical_action_prior(env.interface, {"button_probability": .05})
        with pytest.raises(ValueError, match="requires"):
            logical_action_prior(env.interface, {"unsupported": .05})
    finally:
        env.close()
