from copy import deepcopy
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
from soku_rl.env.wrappers.learning import LearningConfig, LearningInterface
from soku_rl.policy.export_actor import export_actor
from soku_rl.policy.loader import load_policy
from soku_rl.rl.behavior_cloning import ObservationContractEnv
from soku_rl.rl.ppo import create_ppo, parameter_hash
from soku_rl.rl.recurrent_cloning import sequence_epoch, zero_states
from soku_rl.rl.recurrent_persistent_policy import PersistentRecurrentActorCriticPolicy
from soku_rl.rl.storage import PackedObservation
from test_env_timing import VISIBILITY
from test_shared_ppo import fixture_config, save_contract


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_all_previous_commands_exact_mixture_and_gradients(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.set_num_threads(1)
    policy = PersistentRecurrentActorCriticPolicy(spaces.Box(-1, 1, (8,), np.float32),
        spaces.Discrete(576), lambda _: .001, net_arch=[8], lstm_hidden_size=8,
        repeat_probability=.8).to(device)
    with torch.no_grad():
        policy.action_net.weight.zero_()
        policy.action_net.bias.zero_()
    observations = torch.tensor([decode_action(a).inputs for a in range(576)], dtype=torch.float32, device=device)
    distribution = policy._mixed_distribution(observations, torch.zeros(576, 8, device=device))
    expected = torch.full((576, 576), .2 / 576, device=device) + .8 * torch.eye(576, device=device)
    torch.testing.assert_close(distribution.distribution.probs, expected)
    assert (distribution.distribution.probs > 0).all()
    interrupted = (torch.arange(576, device=device) + 1) % 576
    loss = -distribution.log_prob(interrupted).mean()
    loss.backward()
    torch.testing.assert_close(policy.repeat_gate.bias.grad, torch.tensor([.8], device=device))
    assert torch.isfinite(policy.action_net.bias.grad).all()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("critic,shared_features", [("separate", True), ("separate", False),
                                                    ("shared", True), ("feedforward", True)])
def test_sequence_resets_values_memory_likelihood_and_bc_cache(device, critic, shared_features):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.set_num_threads(1)
    kwargs = dict(net_arch=[8], lstm_hidden_size=8, n_lstm_layers=2,
        shared_lstm=critic == "shared", enable_critic_lstm=critic == "separate",
        share_features_extractor=shared_features)
    space = spaces.Box(-np.inf, np.inf, (12,), np.float32)
    policy = PersistentRecurrentActorCriticPolicy(space, spaces.Discrete(576), lambda _: .001,
                                                 repeat_probability=.8, **kwargs).to(device)
    baseline = RecurrentActorCriticPolicy(space, spaces.Discrete(576), lambda _: .001, **kwargs).to(device)
    baseline.load_state_dict({k:v for k,v in policy.state_dict().items() if not k.startswith("repeat_gate.")})
    observations = torch.zeros(6, 12, device=device)
    observations[:, :4] = torch.arange(24, device=device).reshape(6, 4) / 24
    previous = torch.tensor([0, 575, 256, 511, 64, 448], device=device)
    observations[:, -8:] = torch.tensor([decode_action(int(a)).inputs for a in previous], device=device)
    starts = torch.tensor([1., 0., 0., 0., 1., 0.], device=device)
    states = RNNStates(*[(torch.ones(2, 2, 8, device=device), torch.ones(2, 2, 8, device=device)) for _ in range(2)])
    untouched = observations.clone()
    _, base_values, _, base_memory = baseline(observations, states, starts, deterministic=True)
    fresh = baseline.action_dist.distribution.probs
    expected = .2 * fresh + .8 * torch.nn.functional.one_hot(previous, 576)
    actions, values, log_prob, memory = policy(observations, states, starts, True)
    torch.testing.assert_close(policy.action_dist.distribution.probs, expected)
    assert torch.equal(actions, expected.argmax(1)) and torch.equal(observations, untouched)
    torch.testing.assert_close(values, base_values)
    torch.testing.assert_close(policy.predict_values(observations, states.vf, starts), base_values)
    for actual, reference in zip((*memory.pi, *memory.vf), (*base_memory.pi, *base_memory.vf), strict=True):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    distribution, _ = policy.get_distribution(observations, states.pi, starts)
    torch.testing.assert_close(distribution.distribution.probs, expected)
    _, evaluated, entropy = policy.evaluate_actions(observations, actions, states, starts)
    torch.testing.assert_close(evaluated, log_prob)
    torch.testing.assert_close(policy.action_dist.log_prob(actions), evaluated)
    torch.testing.assert_close(entropy, -(expected * expected.log()).sum(1))
    # Compare gradients with an independently assembled probability mixture.
    labels = torch.tensor([1, 575, 255, 511, 65, 447], device=device)
    _, scored, _ = policy.evaluate_actions(observations, labels, states, starts)
    (-scored.mean()).backward()
    (-expected.gather(1, labels[:, None]).log().mean()).backward()
    parameters = dict(policy.named_parameters())
    for name, reference in baseline.named_parameters():
        if reference.grad is not None:
            torch.testing.assert_close(parameters[name].grad, reference.grad, atol=2e-6, rtol=2e-4, msg=name)
    assert policy.repeat_gate.bias.grad.abs().item() > 0


class CommandHistoryEnv(ObservationContractEnv):
    def reset(self, **kwargs):
        self.frame = 0
        self.observation = np.zeros(self.observation_space.shape, np.float32)
        return self.observation.copy(), {}

    def step(self, action):
        self.frame += 1
        self.observation[-8:] = decode_action(int(action)).inputs
        return self.observation.copy(), float(action == 256), self.frame == 5, False, {}


def configuration():
    interface = LearningInterface(EpisodeConfig(5, 1, 1, 0, "privileged_state", VISIBILITY, LEGACY_MATCH),
                                  LearningConfig("full", False, 1, 0.))
    config = fixture_config("lstm") | {"name": "br", "matchups": {"mode": "fixed"}}
    config["ppo"].update(action_persistence={"repeat_probability": .8}, recurrent_storage="sparse")
    config["ppo"]["policy_kwargs"].update(n_lstm_layers=2,
        features_extractor_class="soku_rl.rl.address_invariant_features.AddressInvariantCombatFeatures",
        features_extractor_kwargs=dict(history_frames=1, object_features=2, player_features=4, features_dim=8))
    return interface, config


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_shared_ppo_update_checkpoint_weights_and_private_inference(tmp_path, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.set_num_threads(1)
    interface, config = configuration()
    env = DummyVecEnv([lambda: CommandHistoryEnv(interface) for _ in range(2)])
    try:
        model, _ = create_ppo(env, interface, config, {"kind": "fresh"}, device, 19)
        assert isinstance(model.policy, PersistentRecurrentActorCriticPolicy)
        assert set(model.policy.parameters()) == {p for g in model.policy.optimizer.param_groups for p in g["params"]}
        initial = model.policy.repeat_gate.bias.detach().clone()
        model.learn(32)
        assert not torch.equal(initial, model.policy.repeat_gate.bias)
        path = tmp_path / "trained.zip"
        model.save(path)
        contract = save_contract(tmp_path, SimpleNamespace(interface=interface), config)
        for kind in ("checkpoint", "weights"):
            restored, _ = create_ppo(env, interface, config,
                {"kind": kind, "path": str(path), "training_config": contract}, device, 20)
            assert parameter_hash(restored.policy) == parameter_hash(model.policy)
            assert restored.num_timesteps == (32 if kind == "checkpoint" else 0)
            if kind == "weights":
                assert not restored.policy.optimizer.state
            else:
                for actual, expected in zip(restored.policy.optimizer.state.values(), model.policy.optimizer.state.values(), strict=True):
                    for key in expected:
                        torch.testing.assert_close(actual[key].cpu(), expected[key].cpu(), rtol=0, atol=0)
        portable = load_policy("persistent", {"kind": "sb3_recurrent", "path": str(path),
            "training_config": contract}, interface, device)
        first, second = portable.spawn(13), portable.spawn(13)
        for observation in env.reset():
            assert first.act(observation) == second.act(observation)
        model.policy.save(tmp_path / "policy.pt")
        standalone = PersistentRecurrentActorCriticPolicy.load(tmp_path / "policy.pt", device=device)
        assert standalone.lstm_actor.num_layers == 2
        observations = torch.as_tensor(env.reset(), device=device)
        starts = torch.ones(2, device=device)
        expected, _ = model.policy.get_distribution(observations, model._last_lstm_states.pi, starts)
        actual, _ = standalone.get_distribution(observations, model._last_lstm_states.pi, starts)
        torch.testing.assert_close(actual.distribution.probs, expected.distribution.probs, rtol=0, atol=0)
        incompatible = deepcopy(config)
        del incompatible["ppo"]["action_persistence"]
        with pytest.raises(ValueError, match="architecture"):
            create_ppo(env, interface, incompatible,
                {"kind": "weights", "path": str(path), "training_config": contract}, device, 20)
    finally:
        env.close()


def test_fresh_gate_preserves_backbone_initialization_and_rng():
    torch.set_num_threads(1)
    space = spaces.Box(-1, 1, (8,), np.float32)
    kwargs = dict(net_arch=[8], lstm_hidden_size=8, n_lstm_layers=2)
    torch.manual_seed(71)
    baseline = RecurrentActorCriticPolicy(space, spaces.Discrete(576), lambda _: .001, **kwargs)
    expected_rng = torch.random.get_rng_state()
    torch.manual_seed(71)
    policy = PersistentRecurrentActorCriticPolicy(space, spaces.Discrete(576), lambda _: .001,
                                                 repeat_probability=.8, **kwargs)
    assert torch.equal(torch.random.get_rng_state(), expected_rng)
    for key, value in baseline.state_dict().items():
        torch.testing.assert_close(policy.state_dict()[key], value, rtol=0, atol=0)


def test_bc_sequence_padding_matches_online_scoring_and_updates_gate():
    torch.set_num_threads(1)
    interface, config = configuration()
    model, _ = create_ppo(ObservationContractEnv(interface), interface, config, {"kind": "fresh"}, "cpu", 17)
    episodes = []
    for size in (3, 7, 5):
        rows = []
        for frame in range(size):
            observation = np.zeros(interface.observation_space.shape, np.float32)
            observation[-8:] = decode_action((frame * 61) % 576).inputs
            rows.append((PackedObservation.pack(observation), (frame * 61 + 1) % 576, 1., -1 if frame == 0 else 1))
        episodes.append(rows)
    reference = []
    with torch.no_grad():
        for episode in episodes:
            states = zero_states(model.policy, 1)
            for packed, action, _, _ in episode:
                distribution, next_pi = model.policy.get_distribution(torch.from_numpy(packed.unpack())[None], states.pi, torch.zeros(1))
                states = RNNStates(next_pi, states.vf)
                reference.append(-float(distribution.log_prob(torch.tensor([action]))))
    for length, batch in ((1, 1), (2, 4), (4, 12), (16, 32)):
        metrics, _, updates = sequence_epoch(model, episodes, [2, 0, 1], batch, length, .5, False, 1.)
        assert metrics["nll"] == pytest.approx(np.mean(reference), abs=1e-6) and updates == 0
    before = model.policy.repeat_gate.bias.detach().clone()
    _, _, updates = sequence_epoch(model, episodes, [0, 1, 2], 4, 2, .5, True, 1.)
    assert updates > 0 and not torch.equal(before, model.policy.repeat_gate.bias)


def test_onnx_keeps_gate_for_changing_commands_and_recurrent_states(tmp_path):
    torch.set_num_threads(1)
    interface, config = configuration()
    model, _ = create_ppo(ObservationContractEnv(interface), interface, config, {"kind": "fresh"}, "cpu", 19)
    with torch.no_grad():
        model.policy.repeat_gate.weight.copy_(torch.linspace(-.5, .5, 8).reshape(1, 8))
    path = tmp_path / "best.zip"
    model.save(path)
    contract = save_contract(tmp_path, SimpleNamespace(interface=interface), config)
    report = export_actor({"candidate": {"name": "persistent", "policy": {
        "kind": "checkpoint", "path": str(path), "training_config": contract}},
        "verification_steps": 512, "seed": 13, "output": str(tmp_path / "portable")})
    assert report["verification"]["steps"] == 512
    assert report["verification"]["maximum_absolute_error"] < 2e-6


@pytest.mark.parametrize("probability", [0., 1., float("nan"), True])
def test_invalid_repeat_probability_fails_before_network_build(probability):
    with pytest.raises(ValueError, match="repeat_probability"):
        PersistentRecurrentActorCriticPolicy(repeat_probability=probability)


def test_shared_factory_rejects_missing_command_history():
    interface, config = configuration()
    interface = LearningInterface(interface.episode, LearningConfig("full", False, 0, 0.))
    with pytest.raises(ValueError, match="action history"):
        create_ppo(ObservationContractEnv(interface), interface, config, {"kind": "fresh"}, "cpu", 19)
