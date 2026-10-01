"""DQN targets, lossless n-step replay, all MARL schedulers and continuation."""
from dataclasses import asdict
import json

import numpy as np
import pytest
from gymnasium import spaces
from omegaconf import OmegaConf
import torch
from stable_baselines3.common.buffers import NStepReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples

from soku_rl.rl.dqn import DoubleDQN, double_q_target
from soku_rl.rl.replay import PackedNStepReplayBuffer
from soku_rl.rl.learner import create_learner
from soku_rl.policy.loader import load_policy
from test_shared_ppo import fixture_env, fixture_config


def dqn_config():
    return fixture_config("mlp") | {"learner": "dqn", "dqn": {
        "learning_rate": .001, "buffer_size": 64, "learning_starts": 0,
        "batch_size": 4, "gamma": 1., "n_steps": 3, "train_freq": 4,
        "gradient_steps": 2, "target_update_interval": 8,
        "exploration_decay_steps": 8, "tau": 1., "max_grad_norm": 10., "exploration_initial_eps": 1., "exploration_final_eps": .05,
        "policy_kwargs": {"net_arch": [8]}, "verbose": 0}}


def contract(directory, env, config):
    path = directory / "config.yaml"
    shared = config["response"] if config["name"] == "psro" else config
    OmegaConf.save(OmegaConf.create({"episode": asdict(env.interface.episode),
        "wrappers": asdict(env.interface.config), "algorithm": config,
        "rl": {key: shared[key] for key in ("learner", "policy_type", "timeout_payoff", "ppo", "dqn")}}), path)
    return str(path)


def test_double_target_selects_online_evaluates_target_and_stops_at_terminal():
    data = ReplayBufferSamples(torch.zeros(2, 1), torch.zeros(2, 1), torch.zeros(2, 1),
        torch.tensor([[0.], [1.]]), torch.tensor([[2.], [7.]]), torch.tensor([[.5], [.5]]))
    online = lambda obs: torch.tensor([[5., 1.], [5., 1.]])
    target = lambda obs: torch.tensor([[3., 9.], [3., 9.]])
    assert torch.equal(double_q_target(online, target, data, .9), torch.tensor([[3.5], [7.]]))


@pytest.mark.parametrize("wrap", [False, True])
@pytest.mark.parametrize("n_steps", [1, 3, 5])
def test_packed_replay_matches_upstream_across_terminals_and_ring_boundary(wrap, n_steps):
    obs_space = spaces.Box(-1, 1, (30,), dtype=np.float32)
    args = dict(buffer_size=16, observation_space=obs_space, action_space=spaces.Discrete(4),
        device="cpu", n_envs=2, optimize_memory_usage=False, n_steps=n_steps, gamma=.9,
        handle_timeout_termination=False)
    reference, packed = NStepReplayBuffer(**args), PackedNStepReplayBuffer(**args)
    rng = np.random.default_rng(123)
    for index in range(20 if wrap else 6):
        obs = rng.normal(size=(2, 30)).astype(np.float32)
        obs[:, 10:] = 0
        for buffer in (reference, packed):
            buffer.add(obs, obs + .1, np.array([1, 2]), np.array([index, -index]),
                       np.array([index % 3 == 0, index % 5 == 0]), [{}, {}])
    for seed in range(4):
        np.random.seed(seed)
        expected = reference.sample(20)
        np.random.seed(seed)
        actual = packed.sample(20)
        for first, second in zip(expected, actual, strict=True):
            torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert not packed.timeouts.any()


def test_replay_cut_bootstraps_without_joining_resumed_game():
    replay = PackedNStepReplayBuffer(16, spaces.Box(-1, 1, (1,), dtype=np.float32),
        spaces.Discrete(2), "cpu", 1, False, 3, .5, False)
    for value in (1, 2):
        replay.add(np.array([[value]], dtype=np.float32), np.array([[value + 1]], dtype=np.float32),
                   np.array([0]), np.array([value]), np.array([False]), [{}])
    replay.cut_trajectories()
    replay.add(np.array([[9]], dtype=np.float32), np.array([[10]], dtype=np.float32),
               np.array([0]), np.array([100]), np.array([False]), [{}])
    result = replay._get_samples(np.array([0]), None)
    assert result.rewards.item() == 2
    assert result.next_observations.item() == 3
    assert result.discounts.item() == .25
    assert result.dones.item() == 0


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_br_updates_target_replay_save_load_and_resume(tmp_path, device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from soku_rl.marl.br import train_br
    torch.set_num_threads(1)
    env = fixture_env()
    config = dqn_config() | {"name": "br", "player": 1, "matchups": {"mode": "fixed"},
        "timesteps": 16, "checkpoint_every": 8, "initial_policy": {"kind": "fresh"},
        "opponents": [{"name": "random", "probability": 1., "policy": {"kind": "uniform"}}]}
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir(); second.mkdir()
    try:
        result = train_br(env, config, device, 13, first)
        source = contract(first, env, config)
        model = DoubleDQN.load(result["checkpoint"], device=device)
        assert model._n_updates == 4
        assert (first / "final.replay.pkl").is_file()
        old_optimizer = next(iter(model.policy.optimizer.state.values()))["step"].item()
        policy = load_policy("saved", {"kind": "sb3_dqn", "path": result["checkpoint"],
            "training_config": source}, env.interface, device)
        obs, _ = env.reset({0: 19})
        assert policy.spawn(1).act(obs[0]["player_1"]) == policy.spawn(2).act(obs[0]["player_1"])
        config["initial_policy"] = {"kind": "checkpoint", "path": result["checkpoint"], "training_config": source}
        resumed = train_br(env, config, device, 15, second)
        loaded = DoubleDQN.load(resumed["checkpoint"], device=device)
        loaded.load_replay_buffer(second / "final.replay.pkl")
        assert resumed["start_steps"] == 16 and resumed["steps"] == 32
        assert loaded._n_updates == 8 and loaded.replay_buffer.size() == 16
        assert next(iter(loaded.policy.optimizer.state.values()))["step"].item() > old_optimizer
        assert not loaded.q_net_target.training
    finally:
        env.close()


def test_independent_dqn_joint_collection_and_resume(tmp_path):
    from soku_rl.marl.ippo import train_ippo
    torch.set_num_threads(1)
    env = fixture_env()
    config = dqn_config() | {"name": "ippo"}
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir(); second.mkdir()
    try:
        report = train_ippo(env, config, "cpu", 8, first)
        source = contract(first, env, config)
        assert report["updates"] == 1 and report["games"]
        for player in (0, 1):
            path = first / f"player_{player}" / "final.zip"
            model = DoubleDQN.load(path)
            model.load_replay_buffer(path.with_suffix(".replay.pkl"))
            assert model.replay_buffer.dones[:4].any()
            assert model._n_updates == 2
            config["initial_policies"][f"player_{player}"] = {"kind": "checkpoint", "path": str(path),
                "training_config": source}
        resumed = train_ippo(env, config, "cpu", 9, second)
        assert all(resumed[f"player_{p}"]["steps"] == 16 for p in (0, 1))
    finally:
        env.close()


def test_dqn_nfsp_keeps_categorical_average_and_resumes_replay(tmp_path):
    from soku_rl.marl.nfsp import train_nfsp
    from stable_baselines3 import PPO
    torch.set_num_threads(1)
    env = fixture_env()
    config = dqn_config() | {"name": "nfsp", "iterations": 1, "timesteps_per_iteration": 8,
        "anticipatory_param": .1, "average": {"capacity": 12, "batch_size": 4, "updates": 2},
        "resume": {"kind": "fresh"}}
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir(); second.mkdir()
    try:
        report = train_nfsp(env, config, "cpu", 8, first)
        source = contract(first, env, config)
        assert PPO.load(first / "player_0/final.zip", device="cpu").policy.get_distribution is not None
        config["resume"] = {"kind": "checkpoint", "path": report["resume_checkpoint"], "training_config": source}
        resumed = train_nfsp(env, config, "cpu", 9, second)
        saved = torch.load(resumed["resume_checkpoint"], weights_only=False)
        assert saved["iteration"] == 2
        assert all(s["seen"] == 16 for s in saved["reservoirs"])
        model = DoubleDQN.load(second / "checkpoint-2/response-p0.zip")
        model.load_replay_buffer(second / "checkpoint-2/response-p0.replay.pkl")
        assert model.replay_buffer.size() == 8 and model._n_updates == 4
    finally:
        env.close()


def test_psro_dqn_population_load_and_continue(tmp_path):
    pytest.importorskip("open_spiel")
    from soku_rl.marl.psro import train_psro
    torch.set_num_threads(1)
    env = fixture_env()
    response = dqn_config() | {"initialization": "parent_weights", "timesteps_per_response": 8}
    config = {"name": "psro", "iterations": 1, "simulations_per_entry": 2,
        "prd_iterations": 20, "timeout_payoff": "zero_at_horizon", "response": response,
        "initial_population": {a: {"kind": "uniform"} for a in ("player_0", "player_1")},
        "resume": {"kind": "fresh"}}
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir(); second.mkdir()
    try:
        report = train_psro(env, config, "cpu", 23, first)
        source = contract(first, env, config)
        assert report["populations"][0][1]["kind"] == "sb3_dqn"
        config["resume"] = {"kind": "checkpoint", "path": str(first / "population.json"), "training_config": source}
        resumed = train_psro(env, config, "cpu", 99, second)
        assert [len(p) for p in resumed["populations"]] == [3, 3]
        for old, new in zip(report["meta_game"], resumed["meta_game"], strict=True):
            assert np.array_equal(old, np.array(new)[:2, :2])
    finally:
        env.close()


def test_dictionary_nstep_replay_keeps_fields_and_boundaries():
    space = spaces.Dict({"pixels": spaces.Box(0, 255, (2, 3, 3), dtype=np.uint8),
                         "commands": spaces.Box(-10, 10, (3,), dtype=np.float32)})
    replay = PackedNStepReplayBuffer(8, space, spaces.Discrete(2), "cpu", 1, False, 3, .5, False)
    for value in (1, 2, 3):
        obs = {"pixels": np.full((1, 2, 3, 3), value, dtype=np.uint8),
               "commands": np.full((1, 3), value, dtype=np.float32)}
        nxt = {key: array + 1 for key, array in obs.items()}
        replay.add(obs, nxt, np.array([1]), np.array([value]), np.array([value == 2]), [{}])
    result = replay._get_samples(np.array([0]), None)
    assert result.rewards.item() == 2 and result.dones.item() == 1
    assert result.observations["pixels"].dtype == torch.uint8
    assert (result.next_observations["commands"] == 3).all()
    assert (result.next_observations["pixels"] == 3).all()


def test_replay_ring_overwrite_clears_resume_boundary():
    replay = PackedNStepReplayBuffer(3, spaces.Box(-1, 1, (1,), dtype=np.float32),
        spaces.Discrete(2), "cpu", 1, False, 3, .9, False)
    for _ in range(3):
        replay.add(np.zeros((1, 1), dtype=np.float32), np.zeros((1, 1), dtype=np.float32),
                   np.array([0]), np.array([1]), np.array([False]), [{}])
    replay.cut_trajectories()
    assert replay.timeouts[2, 0] == 1
    for _ in range(3):
        replay.add(np.zeros((1, 1), dtype=np.float32), np.zeros((1, 1), dtype=np.float32),
                   np.array([0]), np.array([1]), np.array([False]), [{}])
    assert not replay.timeouts.any()


@pytest.mark.parametrize("algorithm", ["br", "ppo", "ippo", "nfsp", "psro"])
def test_dqn_hydra_plugs_into_every_scheduler(algorithm):
    from pathlib import Path
    from hydra import compose, initialize_config_dir
    from soku_rl.rl import ppo_settings
    with initialize_config_dir(config_dir=str(Path(__file__).parents[1] / "config"), version_base="1.3"):
        composed = compose(config_name="train", overrides=[f"algorithm={algorithm}", "rl=dqn",
            "track=superhuman_combat", "wrappers=superhuman_learning", "rl.dqn.learning_rate=0.0002"])
        config = {key: OmegaConf.to_container(composed[key], resolve=True) for key in ("rl", "algorithm")}
    assert ppo_settings(config) is config["rl"]
    settings = config["algorithm"]["response"] if algorithm == "psro" else config["algorithm"]
    assert settings["dqn"]["policy_kwargs"] == config["rl"]["ppo"]["policy_kwargs"]
    settings["dqn"]["learning_rate"] = .001
    with pytest.raises(ValueError, match="configure rl.dqn"):
        ppo_settings(config)


def test_image_and_command_history_dqn_updates(tmp_path):
    from soku_rl.env import TwoPlayerVectorEnv
    from soku_rl.env.wrappers.learning import LearningVectorEnv
    from soku_rl.rl.opponent_env import OpponentMixtureVecEnv
    from soku_rl.policy.population import UniformPolicy
    from test_image_learning import ImageBackend, episode, learning
    torch.set_num_threads(1)
    env = LearningVectorEnv(TwoPlayerVectorEnv(ImageBackend(), 1, episode()), learning())
    view = OpponentMixtureVecEnv(env, 0, [UniformPolicy("random", 90)], [1.], 1)
    try:
        config = dqn_config()
        model, _ = create_learner(view, env.interface, config, {"kind": "fresh"}, "cpu", 1)
        model.learn(4)
        assert model._n_updates == 2
        samples = model.replay_buffer.sample(2)
        assert set(samples.observations) == {"image", "commands"}
        assert samples.observations["image"].dtype == torch.uint8
        assert samples.dones.all()
    finally:
        view.close()
        env.close()
