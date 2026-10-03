from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

from gymnasium import spaces
import numpy as np
import pytest
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from stable_baselines3.common.vec_env import DummyVecEnv
import torch

from soku_rl.env import EpisodeConfig
from soku_rl.env.encoding import decode_action
from soku_rl.env.match import LEGACY_MATCH
from soku_rl.env.observation.memory_schema import FIGHTER_NAMES, PLAYER_WIDTH, PRIVILEGED_FEATURES, WORLD_NAMES
from soku_rl.env.observation.privileged import encode_values
from soku_rl.env.wrappers.learning import LearningConfig, LearningInterface
from soku_rl.policy.loader import load_policy
from soku_rl.rl.behavior_cloning import ObservationContractEnv
from soku_rl.rl.facing_policy import FacingRecurrentActorCriticPolicy
from soku_rl.rl.ppo import create_ppo, parameter_hash
from test_env_timing import VISIBILITY
from test_shared_ppo import fixture_config, save_contract


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_full_command_bijection_likelihood_cache_gradient_and_memory(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.set_num_threads(1)
    kwargs = dict(net_arch=[8], lstm_hidden_size=8, n_lstm_layers=2)
    space = spaces.Box(-np.inf, np.inf, (4,), np.float32)
    policy = FacingRecurrentActorCriticPolicy(space, spaces.Discrete(576), lambda _: .001,
                                             facing_index=2, **kwargs).to(device)
    baseline = RecurrentActorCriticPolicy(space, spaces.Discrete(576), lambda _: .001, **kwargs).to(device)
    baseline.load_state_dict(policy.state_dict())
    for command in range(576):
        mirrored = int(policy.mirrored_commands[command])
        assert int(policy.mirrored_commands[mirrored]) == command
        a, b = decode_action(command).inputs, decode_action(mirrored).inputs
        assert b == (-a[0], *a[1:])
    # Two recurrent sequences, including an interior reset and zero padding.
    observations = torch.zeros((6, 4), device=device)
    observations[:, :2] = torch.arange(12, device=device).reshape(6, 2) / 12
    observations[:, 2:] = torch.tensor(encode_values(np.array([-1., 1., 0., 1., -1., -1.])), device=device)
    starts = torch.tensor([1., 0., 0., 0., 1., 0.], device=device)
    states = RNNStates(*[(torch.ones((2, 2, 8), device=device), torch.ones((2, 2, 8), device=device))
                        for _ in range(2)])
    original_observations = observations.clone()
    relative, base_values, base_log_prob, base_states = baseline(observations, states, starts, deterministic=True)
    base_probabilities = baseline.action_dist.distribution.probs.clone()
    absolute, values, log_prob, memory = policy(observations, states, starts, deterministic=True)
    expected = relative.clone()
    for row in (0, 4, 5):
        expected[row] = policy.mirrored_commands[relative[row]]
    assert torch.equal(absolute, expected)
    assert torch.equal(observations, original_observations)
    torch.testing.assert_close(values, base_values)
    torch.testing.assert_close(log_prob, base_log_prob)
    torch.testing.assert_close(policy.action_dist.log_prob(absolute), log_prob)
    probabilities = policy.action_dist.distribution.probs.clone()
    expected_probabilities = base_probabilities.clone()
    for row in (0, 4, 5):
        expected_probabilities[row] = base_probabilities[row, policy.mirrored_commands]
    torch.testing.assert_close(probabilities, expected_probabilities)
    assert (probabilities > 0).all()
    for actual, reference in zip((*memory.pi, *memory.vf), (*base_states.pi, *base_states.vf), strict=True):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    distribution, _ = policy.get_distribution(observations, states.pi, starts)
    torch.testing.assert_close(distribution.distribution.probs, probabilities)
    _, evaluated, entropy = policy.evaluate_actions(observations, absolute, states, starts)
    torch.testing.assert_close(evaluated, log_prob)
    torch.testing.assert_close(policy.action_dist.log_prob(absolute), evaluated)
    _, reference, base_entropy = baseline.evaluate_actions(observations, relative, states, starts)
    torch.testing.assert_close(entropy, base_entropy)
    # The absolute-label BC cache and PPO likelihood must train the same head.
    loss = -policy.action_dist.log_prob(absolute).mean()
    loss.backward()
    (-reference.mean()).backward()
    for (name, actual), (_, expected) in zip(policy.named_parameters(), baseline.named_parameters(), strict=True):
        if expected.grad is not None:
            torch.testing.assert_close(actual.grad, expected.grad, atol=2e-6, rtol=2e-5, msg=name)


class FacingContractEnv(ObservationContractEnv):
    def __init__(self, interface):
        super().__init__(interface)
        self.position = 2 * (len(WORLD_NAMES) + FIGHTER_NAMES.index("dir"))
        self.count = 0

    def observation(self):
        values = np.zeros(self.observation_space.shape, np.float32)
        values[self.position:self.position + 2] = encode_values(np.array([(-1.) ** self.count]))[0]
        return values

    def reset(self, **kwargs):
        self.count = 0
        return self.observation(), {}

    def step(self, action):
        self.count += 1
        return self.observation(), float(2 * (action % 2) - 1), self.count == 5, False, {}


def configuration():
    episode = EpisodeConfig(5, 1, 1, 0, "privileged_state", VISIBILITY, LEGACY_MATCH)
    interface = LearningInterface(episode, LearningConfig("full", False, 0, 0.))
    config = fixture_config("lstm") | {"name": "br", "matchups": {"mode": "fixed"}}
    config["ppo"].update(action_frame="own_facing", recurrent_storage="sparse")
    config["ppo"]["policy_kwargs"].update(n_lstm_layers=2,
        features_extractor_class="soku_rl.rl.address_invariant_features.AddressInvariantCombatFeatures",
        features_extractor_kwargs=dict(history_frames=1, object_features=2, player_features=4, features_dim=8))
    return interface, config


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_shared_training_resume_weights_and_portable_inference(tmp_path, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.set_num_threads(1)
    interface, config = configuration()
    view = DummyVecEnv([lambda: FacingContractEnv(interface) for _ in range(2)])
    try:
        model, _ = create_ppo(view, interface, config, {"kind": "fresh"}, device, 19)
        assert isinstance(model.policy, FacingRecurrentActorCriticPolicy)
        initial = parameter_hash(model.policy)
        model.learn(16)
        assert initial != parameter_hash(model.policy)
        checkpoint = tmp_path / "trained.zip"
        model.save(checkpoint)
        contract = save_contract(tmp_path, SimpleNamespace(interface=interface), config)
        for kind in ("checkpoint", "weights"):
            restored, _ = create_ppo(view, interface, config,
                {"kind": kind, "path": str(checkpoint), "training_config": contract}, device, 20)
            assert parameter_hash(restored.policy) == parameter_hash(model.policy)
            assert restored.num_timesteps == (16 if kind == "checkpoint" else 0)
            if kind == "weights":
                assert not restored.policy.optimizer.state
            else:
                for actual, expected in zip(restored.policy.optimizer.state.values(), model.policy.optimizer.state.values(), strict=True):
                    for key in expected:
                        torch.testing.assert_close(actual[key].cpu(), expected[key].cpu(), rtol=0, atol=0)
        portable = load_policy("facing", {"kind": "sb3_recurrent", "path": str(checkpoint),
            "training_config": contract}, interface, device)
        first, second = portable.spawn(13), portable.spawn(13)
        for observation in view.reset():
            assert first.act(observation) == second.act(observation)
        model.policy.save(tmp_path / "policy.pt")
        standalone = FacingRecurrentActorCriticPolicy.load(tmp_path / "policy.pt", device=device)
        assert standalone.facing_index == model.policy.facing_index
        assert standalone.lstm_actor.num_layers == 2 and standalone.lstm_actor.hidden_size == 8
        observations = torch.as_tensor(view.reset(), device=device)
        starts = torch.ones(2, device=device)
        expected, _ = model.policy.get_distribution(observations, model._last_lstm_states.pi, starts)
        actual, _ = standalone.get_distribution(observations, model._last_lstm_states.pi, starts)
        torch.testing.assert_close(actual.distribution.probs, expected.distribution.probs, rtol=0, atol=0)
        incompatible = deepcopy(config)
        del incompatible["ppo"]["action_frame"]
        with pytest.raises(ValueError, match="architecture"):
            create_ppo(view, interface, incompatible,
                {"kind": "weights", "path": str(checkpoint), "training_config": contract}, device, 20)
    finally:
        view.close()


@pytest.mark.parametrize("option", ["world", False, {"action_frame": "own_facing"}])
def test_invalid_action_frame_is_rejected(option):
    interface, config = configuration()
    config["ppo"]["action_frame"] = option
    view = DummyVecEnv([lambda: FacingContractEnv(interface)])
    try:
        with pytest.raises(ValueError, match="facing actions require"):
            create_ppo(view, interface, config, {"kind": "fresh"}, "cpu", 19)
    finally:
        view.close()


def test_mapping_uses_current_own_facing_not_old_history_or_opponent():
    interface, config = configuration()
    interface = LearningInterface(replace(interface.episode, history_frames=2), interface.config)
    config["ppo"]["policy_kwargs"]["features_extractor_kwargs"]["history_frames"] = 2
    view = DummyVecEnv([lambda: ObservationContractEnv(interface)])
    try:
        model, _ = create_ppo(view, interface, config, {"kind": "fresh"}, "cpu", 19)
        offset = 2 * (len(WORLD_NAMES) + FIGHTER_NAMES.index("dir"))
        observations = torch.zeros((1, interface.observation_space.shape[0]))
        observations[:, offset:offset + 2] = torch.tensor(encode_values(np.array([-1.]))[0])
        current = PRIVILEGED_FEATURES + offset
        opponent = current + 2 * PLAYER_WIDTH
        observations[:, current:current + 2] = torch.tensor(encode_values(np.array([1.]))[0])
        observations[:, opponent:opponent + 2] = torch.tensor(encode_values(np.array([-1.]))[0])
        with torch.no_grad():
            model.policy.action_net.weight.zero_()
            model.policy.action_net.bias.fill_(-10.)
            model.policy.action_net.bias[448] = 10.
        actions, _, _, _ = model.policy(observations, model._last_lstm_states, torch.ones(1), deterministic=True)
        assert int(actions[0]) == 448
        observations[:, current:current + 2] = torch.tensor(encode_values(np.array([-1.]))[0])
        observations[:, opponent:opponent + 2] = torch.tensor(encode_values(np.array([1.]))[0])
        actions, _, _, _ = model.policy(observations, model._last_lstm_states, torch.ones(1), deterministic=True)
        assert int(actions[0]) == 64
    finally:
        view.close()
