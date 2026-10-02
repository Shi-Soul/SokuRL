import copy
from pathlib import Path
import random

from hydra import compose, initialize_config_dir
import numpy as np
from omegaconf import OmegaConf
import pytest
import torch
from stable_baselines3.common.logger import configure

from test_shared_ppo import fixture_config, fixture_env, save_contract
from soku_rl.rl import ppo_settings
from soku_rl.rl.actor_windows import actor_window_logits, categorical_distance
from soku_rl.rl.behavior_cloning import ObservationContractEnv
from soku_rl.rl.online_anchor import attach_anchor, validate_anchor
from soku_rl.rl.ppo import algorithm_type, create_ppo, parameter_hash
from soku_rl.rl.recurrent_cloning import zero_states
from soku_rl.rl.storage import PackedObservation


def reference_settings(directory, env, config, device):
    directory.mkdir()
    reference, _ = create_ppo(ObservationContractEnv(env.interface), env.interface,
        config, {"kind": "fresh"}, device, 7)
    path = directory / "reference.zip"
    reference.save(path)
    return {"reference": {"kind": "sb3_recurrent" if config["policy_type"] == "lstm" else "sb3",
        "path": str(path), "training_config": save_contract(directory, env, config | {"name": "ppo"})},
        "updates_per_rollout": 2, "sequences": 2, "sequence_length": 2,
        "learning_rate": .01, "seed": 23}


@pytest.mark.parametrize("algorithm", ["ppo", "br", "ippo", "nfsp", "psro"])
def test_shared_anchor_configuration(algorithm):
    with initialize_config_dir(config_dir=str(Path(__file__).parents[1] / "config"), version_base="1.3"):
        config = OmegaConf.to_container(compose(config_name="train", overrides=[
            f"algorithm={algorithm}", "rl=recurrent_online_anchor"]), resolve=True)
    shared = ppo_settings(config)
    learner = config["algorithm"]["response"] if algorithm == "psro" else config["algorithm"]
    assert learner["online_anchor"] == shared["online_anchor"]
    learner["online_anchor"] = dict(shared["online_anchor"], seed=0)
    with pytest.raises(ValueError, match="online_anchor"):
        ppo_settings(config)


@pytest.mark.parametrize("kind", ["mlp", "lstm"])
def test_br_updates_and_portable_resume(tmp_path, kind):
    from soku_rl.marl.br import OpponentEntry
    from soku_rl.rl.opponent_env import OpponentMixtureVecEnv
    from test_demonstrations import ConstantPolicy
    torch.set_num_threads(1)
    env = fixture_env()
    config = fixture_config(kind) | {"name": "br"}
    config["online_anchor"] = reference_settings(tmp_path / "reference", env, config, "cpu")
    view = OpponentMixtureVecEnv(env, 0, [OpponentEntry("constant", ConstantPolicy(8))], [1.], 17)
    model, _ = create_ppo(view, env.interface, config, {"kind": "fresh"}, "cpu", 7)
    frozen = parameter_hash(model._anchor.reference.policy)
    model.learn(16)
    assert model.num_timesteps == 16 and model._n_updates == 2
    assert model.anchor_state["updates"] == 4
    assert model.anchor_state["collected_frames"] == 16
    assert model.anchor_state["last_update"]["rollout_frames"] == 8
    assert model.anchor_state["frames"] > 0
    assert parameter_hash(model._anchor.reference.policy) == frozen
    assert all(p.grad is None for p in model._anchor.reference.policy.parameters())
    path = tmp_path / "anchor.zip"
    model.save(path)
    inference = algorithm_type(kind).load(path, device="cpu")
    assert not hasattr(inference, "_anchor")
    assert parameter_hash(inference.policy) == parameter_hash(model.policy)
    contract = save_contract(tmp_path, env, config)
    data = OmegaConf.load(contract)
    data.rl.online_anchor = config["online_anchor"]
    OmegaConf.save(data, contract)
    source = {"kind": "checkpoint", "path": str(path), "training_config": contract}
    resumed, _ = create_ppo(view, env.interface, config, source, "cpu", 19)
    assert resumed.anchor_state == model.anchor_state
    assert not resumed._anchor.recent and not any(resumed._anchor.active)
    resumed.learn(8, reset_num_timesteps=False)
    assert resumed.num_timesteps == 24
    assert resumed.anchor_state["collected_frames"] == 24
    reset, _ = create_ppo(view, env.interface, config, source | {"kind": "weights"}, "cpu", 19)
    assert reset.num_timesteps == reset.anchor_state["updates"] == 0
    assert not reset.policy.optimizer.state
    changed = copy.deepcopy(config)
    changed["online_anchor"]["seed"] += 1
    with pytest.raises(ValueError, match="online_anchor"):
        create_ppo(view, env.interface, changed, source, "cpu", 19)
    # Same path with changed bytes must also invalidate continuation.
    changed_reference = model._anchor.reference
    with torch.no_grad():
        changed_reference.policy.action_net.bias[0] += .1
    changed_reference.save(config["online_anchor"]["reference"]["path"])
    with pytest.raises(ValueError, match="reference identity"):
        create_ppo(view, env.interface, config, source, "cpu", 19)
    view.close()
    env.close()


@pytest.mark.parametrize("kind", ["mlp", "lstm"])
def test_joint_collector_uses_online_states(tmp_path, kind):
    from soku_rl.marl.ippo import train_ippo
    env = fixture_env()
    config = fixture_config(kind) | {"name": "ippo"}
    config["online_anchor"] = reference_settings(tmp_path / "reference", env, config, "cpu")
    output = tmp_path / "joint"
    output.mkdir()
    report = train_ippo(env, config, "cpu", 13, output)
    assert report["updates"] == 1
    for player in (0, 1):
        model = algorithm_type(kind).load(output / f"player_{player}/final.zip", device="cpu")
        assert model.anchor_state["updates"] == 2
        assert model.anchor_state["collected_frames"] == model.num_timesteps == 8
    env.close()


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_frozen_reference_rng_and_private_critic(tmp_path, device):
    if device == "cuda:0" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.set_num_threads(1)
    env = fixture_env()
    config = fixture_config("lstm")
    config["ppo"]["policy_kwargs"].update(share_features_extractor=False,
        features_extractor_class="test_recurrent_separate_features.TrainableFeatures")
    settings = reference_settings(tmp_path / "reference", env, config, device)
    model, _ = create_ppo(ObservationContractEnv(env.interface), env.interface,
        config, {"kind": "fresh"}, device, 13)
    model.set_logger(configure(folder=None, format_strings=[]))
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone() if device == "cuda:0" else None
    attach_anchor(model, env.interface, settings, False)
    assert python_rng == random.getstate()
    assert numpy_rng[0] == np.random.get_state()[0]
    assert np.array_equal(numpy_rng[1], np.random.get_state()[1])
    assert numpy_rng[2:] == np.random.get_state()[2:]
    assert torch.equal(cpu_rng, torch.get_rng_state())
    if device == "cuda:0":
        assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
    observations = torch.ones((2, *env.single_observation_space.shape), device=device)
    _, values, _, _ = model.policy(observations, zero_states(model.policy, 2),
        torch.ones(2, device=device), deterministic=True)
    values.square().mean().backward()
    model.policy.optimizer.step()
    with torch.no_grad():
        model.policy.action_net.bias[0] += 1.
    critic = [model.policy.vf_features_extractor, model.policy.lstm_critic, model.policy.value_net]
    before = list(map(parameter_hash, critic))
    optimizer = copy.deepcopy(model.policy.optimizer.state_dict())
    frozen = parameter_hash(model._anchor.reference.policy)
    for step in range(3):
        model._anchor.record(observations[:1].cpu().numpy(), [step == 0])
    metrics = model._anchor.update()
    assert metrics["kl_before"] > metrics["kl_after"] >= 0
    assert metrics["total_variation_before"] > metrics["total_variation_after"]
    assert list(map(parameter_hash, critic)) == before
    assert parameter_hash(model._anchor.reference.policy) == frozen
    actual = model.policy.optimizer.state_dict()
    assert actual["param_groups"] == optimizer["param_groups"]
    for key, states in optimizer["state"].items():
        for name, value in states.items():
            assert torch.equal(actual["state"][key][name], value)
    assert model.num_timesteps == model._n_updates == 0
    assert torch.backends.cudnn.enabled
    env.close()


def test_prefix_retention_and_episode_boundaries(tmp_path):
    env = fixture_env()
    config = fixture_config("lstm")
    config["online_anchor"] = reference_settings(tmp_path / "reference", env, config, "cpu")
    model, _ = create_ppo(ObservationContractEnv(env.interface), env.interface,
        config, {"kind": "fresh"}, "cpu", 7)
    anchor = model._anchor
    observations = np.zeros((1, *model.observation_space.shape), dtype=np.float32)
    with pytest.raises(RuntimeError, match="episode start"):
        anchor.record(observations, [False])
    anchor.record(observations, [True])
    previous = anchor.active[0]
    anchor.reset()
    assert not anchor.recent and anchor.active[0] is previous
    anchor.record(observations + 1, [False])
    assert anchor.recent[0][1] == 1
    anchor.record(observations + 2, [True])
    assert anchor.recent[0][0] is previous
    assert anchor.recent[1][0] is anchor.active[0] and anchor.active[0] is not previous
    assert [len(row[0]) for row in anchor.recent] == [2, 1]
    anchor.record(observations, [False])
    anchor.record(observations, [False])
    with pytest.raises(RuntimeError, match="horizon"):
        anchor.record(observations, [False])
    env.close()


def test_replayed_memory_matches_online_actor_and_returns_independent_logits():
    torch.set_num_threads(1)
    env = fixture_env()
    model, _ = create_ppo(ObservationContractEnv(env.interface), env.interface,
        fixture_config("lstm"), {"kind": "fresh"}, "cpu", 7)
    rng = np.random.default_rng(13)
    episode = [PackedObservation.pack(rng.normal(size=model.observation_space.shape).astype(np.float32))
        for _ in range(263)]
    model.policy.set_training_mode(False)
    with torch.no_grad():
        states, expected = zero_states(model.policy, 1).pi, []
        for observation in episode:
            distribution, states = model.policy.get_distribution(torch.tensor(observation.unpack())[None],
                states, torch.zeros(1))
            expected.append(distribution.distribution.logits.clone())
    actual = actor_window_logits(model, episode, 259, 4, True)
    assert torch.allclose(actual, torch.cat(expected[259:]), atol=1e-6)
    saved = actual.detach().clone()
    actor_window_logits(model, episode, 0, 2, False)
    assert torch.equal(saved, actual.detach())
    (-actual[:, 0].mean()).backward()
    assert model.policy.lstm_actor.weight_hh_l0.grad.abs().sum() > 0
    equal_kl, equal_tv = categorical_distance(actual.detach(), actual.detach())
    assert torch.equal(equal_kl, torch.zeros_like(equal_kl)) and not equal_tv.any()
    env.close()


@pytest.mark.parametrize("key,value", [("seed", -1), ("sequences", True), ("sequence_length", 0),
    ("updates_per_rollout", 0), ("learning_rate", float("nan")), ("learning_rate", True),
    ("reference", {"kind": "rule"})])
def test_invalid_anchor_settings(key, value):
    config = {"reference": {"kind": "sb3", "path": "a", "training_config": "b"},
        "updates_per_rollout": 1, "sequences": 1, "sequence_length": 1, "learning_rate": .01, "seed": 0}
    config[key] = value
    with pytest.raises(ValueError):
        validate_anchor(config)
