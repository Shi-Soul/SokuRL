"""Exercise shared PPO collection, supervised reservoirs and portable continuation."""
from dataclasses import asdict
import json

import numpy as np
from omegaconf import OmegaConf
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")

from soku_rl.env import EpisodeConfig, TwoPlayerVectorEnv
from soku_rl.env.match import LEGACY_MATCH
from soku_rl.env.wrappers.learning import LearningConfig, LearningVectorEnv
from soku_rl.marl.ippo import train_ippo
from soku_rl.marl.nfsp import train_nfsp
from soku_rl.policy.base import RLPolicy
from soku_rl.policy.loader import load_policy
from test_env_timing import RecordingBackend, VISIBILITY


def fixture_config(policy_type):
    architecture = {"net_arch": [8]}
    if policy_type == "lstm":
        architecture["lstm_hidden_size"] = 8
    return {"policy_type": policy_type, "timeout_payoff": "zero_at_horizon",
        "ppo": {"n_steps": 4, "batch_size": 4, "n_epochs": 1, "gamma": 1.,
                "policy_kwargs": architecture},
        "timesteps_per_player": 8, "checkpoint_every": 1,
        "initial_policies": {f"player_{p}": {"kind": "fresh"} for p in (0, 1)}}


def fixture_env():
    config = EpisodeConfig(3, 1, 1, 0, "diagnostic_state", VISIBILITY, LEGACY_MATCH)
    return LearningVectorEnv(TwoPlayerVectorEnv(RecordingBackend(), 2, config),
                            LearningConfig("combat", False, 0, 0.))


def save_contract(directory, env, config):
    path = directory / "config.yaml"
    OmegaConf.save(OmegaConf.create({"episode": asdict(env.interface.episode),
        "wrappers": asdict(env.interface.config), "algorithm": config,
        "rl": {key: config[key] for key in ("policy_type", "timeout_payoff", "ppo")}}), path)
    return str(path)


@pytest.mark.parametrize("policy_type", ["mlp", "lstm"])
def test_joint_collection_save_load_and_continue(tmp_path, policy_type):
    torch.set_num_threads(1)
    env = fixture_env()
    config = fixture_config(policy_type) | {"name": "ippo"}
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir(); second.mkdir()
    report = train_ippo(env, config, "cpu", 7, first)
    assert report["updates"] == 1
    assert report["games"] and all(game["outcome"] == "time_limit" for game in report["games"])
    contract = save_contract(first, env, config)
    kind = "sb3" if policy_type == "mlp" else "sb3_recurrent"
    for player in (0, 1):
        path = first / f"player_{player}" / "final.zip"
        policy = load_policy("saved", {"kind": kind, "path": str(path),
            "training_config": contract}, env.interface, "cpu")
        assert isinstance(policy, RLPolicy)
        first_actor, second_actor = policy.spawn(19), policy.spawn(19)
        observations, _ = env.reset({0: 8})
        observation = observations[0][f"player_{player}"]
        assert first_actor.act(observation) == second_actor.act(observation)
        config["initial_policies"][f"player_{player}"] = {
            "kind": "checkpoint", "path": str(path), "training_config": contract}
    resumed = train_ippo(env, config, "cpu", 8, second)
    assert all(resumed[f"player_{p}"]["steps"] == 16 for p in (0, 1))
    env.close()


def test_nfsp_averages_share_loader_and_resume_reservoirs(tmp_path):
    torch.set_num_threads(1)
    env = fixture_env()
    config = fixture_config("mlp") | {"name": "nfsp", "iterations": 1,
        "timesteps_per_iteration": 8, "anticipatory_param": .1,
        "average": {"capacity": 12, "batch_size": 4, "updates": 2}, "resume": {"kind": "fresh"}}
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir(); second.mkdir()
    result = train_nfsp(env, config, "cpu", 10, first)
    contract = save_contract(first, env, config)
    state = torch.load(result["resume_checkpoint"], weights_only=False)
    assert [reservoir["seen"] for reservoir in state["reservoirs"]] == [8, 8]
    config["resume"] = {"kind": "checkpoint", "path": result["resume_checkpoint"],
                        "training_config": contract}
    resumed = train_nfsp(env, config, "cpu", 12, second)
    state = torch.load(resumed["resume_checkpoint"], weights_only=False)
    assert state["iteration"] == 2
    assert all(reservoir["seen"] == 16 and len(reservoir["samples"]) == 12 for reservoir in state["reservoirs"])
    for player in (0, 1):
        policy = load_policy("average", {"kind": "sb3",
            "path": str(second / f"player_{player}" / "final.zip"), "training_config": contract},
            env.interface, "cpu")
        assert isinstance(policy, RLPolicy)
    env.close()
