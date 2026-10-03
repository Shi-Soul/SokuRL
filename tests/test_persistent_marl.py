"""Exercise custom action likelihoods through each shared-PPO MARL lifecycle."""
from dataclasses import replace

import pytest
import torch
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO

from soku_rl.env import EpisodeConfig, TwoPlayerVectorEnv
from soku_rl.env.match import LEGACY_MATCH
from soku_rl.env.wrappers.learning import LearningConfig, LearningVectorEnv
from soku_rl.rl.persistent_policy import PersistentActorCriticPolicy
from soku_rl.rl.recurrent_persistent_policy import PersistentRecurrentActorCriticPolicy
from test_env_timing import RecordingBackend, VISIBILITY
from test_privileged_encoding import observation
from test_shared_ppo import fixture_config


class PrivilegedBackend(RecordingBackend):
    def _state(self, slot):
        state = super()._state(slot)
        own = observation(0)
        own = replace(own, world=own.world | {"frame": state.frame})
        opponent = replace(own, players=own.players[::-1])
        return replace(state, observations=(own, opponent))


@pytest.mark.parametrize("method,policy_type", [("ippo", "mlp"), ("nfsp", "mlp"),
                                              ("psro", "mlp"), ("ippo", "lstm"), ("psro", "lstm")])
def test_persistent_policy_survives_marl_training_and_artifact_reload(tmp_path, method, policy_type):
    torch.set_num_threads(1)
    env = LearningVectorEnv(TwoPlayerVectorEnv(PrivilegedBackend(), 2,
        EpisodeConfig(3, 1, 1, 0, "privileged_state", VISIBILITY, LEGACY_MATCH)),
        LearningConfig("full", False, 1, 0.))
    config = fixture_config(policy_type) | {"name": method}
    config["ppo"].update(action_persistence={"repeat_probability": .8},
        initial_action_prior={"button_probability": .05})
    config["ppo"]["policy_kwargs"].update(
        features_extractor_class="soku_rl.rl.persistent_policy.ActionContextFeatures",
        features_extractor_kwargs=dict(history_frames=1, object_features=2, player_features=8, features_dim=8))
    try:
        if method == "ippo":
            from soku_rl.marl.ippo import train_ippo
            report = train_ippo(env, config, "cpu", 19, tmp_path)
            assert report["updates"] == 1
            paths = [tmp_path / f"player_{seat}/final.zip" for seat in (0, 1)]
        elif method == "nfsp":
            from soku_rl.marl.nfsp import train_nfsp
            config.update(iterations=1, timesteps_per_iteration=8, anticipatory_param=.1,
                average=dict(capacity=8, batch_size=4, updates=1), resume={"kind": "fresh"})
            report = train_nfsp(env, config, "cpu", 19, tmp_path)
            assert all(row["updates"] == 1 for row in report["iterations"][0]["learning"])
            paths = [tmp_path / f"player_{seat}/final.zip" for seat in (0, 1)]
        else:
            pytest.importorskip("open_spiel")
            from soku_rl.marl.psro import train_psro
            response = config | {"initialization": "fresh", "timesteps_per_response": 8}
            config = dict(name="psro", iterations=1, simulations_per_entry=2, prd_iterations=20,
                timeout_payoff="zero_at_horizon", response=response, resume={"kind": "fresh"},
                initial_population={f"player_{seat}": {"kind": "uniform"} for seat in (0, 1)})
            report = train_psro(env, config, "cpu", 19, tmp_path)
            assert report["training_state"]["responses"] == 2
            paths = [tmp_path / entry["path"] for role in report["populations"]
                     for entry in role if entry["kind"] != "uniform"]
        assert len(paths) == 2
        observations, _ = env.reset({0: 7})
        for seat, path in enumerate(paths):
            algorithm = PPO if policy_type == "mlp" else RecurrentPPO
            expected = PersistentActorCriticPolicy if policy_type == "mlp" else PersistentRecurrentActorCriticPolicy
            model = algorithm.load(path, device="cpu")
            assert isinstance(model.policy, expected)
            tensor, _ = model.policy.obs_to_tensor(observations[0][f"player_{seat}"])
            if policy_type == "mlp":
                distribution = model.policy.get_distribution(tensor)
            else:
                states = tuple(torch.zeros(model.policy.lstm_hidden_state_shape) for _ in range(2))
                distribution, _ = model.policy.get_distribution(tensor, states, torch.ones(1))
            probabilities = distribution.distribution.probs
            assert torch.isfinite(probabilities).all() and (probabilities > 0).all()
            torch.testing.assert_close(probabilities.sum(1), torch.ones(1))
    finally:
        env.close()
