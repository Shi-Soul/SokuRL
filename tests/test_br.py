"""The BR scheduler uses the common learner, mixture, and continuation contract."""
import json

import pytest

from test_shared_ppo import fixture_config, fixture_env, save_contract, torch
from soku_rl.marl.br import train_br
from soku_rl.rl.ppo import algorithm_type, parameter_hash


@pytest.mark.parametrize("policy_type", ["mlp", "lstm"])
def test_br_mixture_and_continuation(tmp_path, policy_type):
    torch.set_num_threads(1)
    env = fixture_env()
    config = fixture_config(policy_type) | {"name": "br", "player": 1,
        "matchups": {"mode": "fixed"},
        "timesteps": 8, "checkpoint_every": 4, "initial_policy": {"kind": "fresh"},
        "opponents": [
            {"name": "excluded", "probability": 0., "policy": {"kind": "uniform"}},
            {"name": "selected", "probability": 1., "policy": {"kind": "uniform"}}]}
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    try:
        report = train_br(env, config, "cpu", 13, first)
        assert report["additional_steps"] == 8
        assert report["initial_policy_hash"] != report["final_policy_hash"]
        timing = json.loads((first / "timing.json").read_text())
        assert timing["phase"] == "finished"
        assert timing["rollouts"][0]["ppo_n_updates"] >= 1
        updated = algorithm_type(policy_type).load(timing["rollouts"][0]["updated_checkpoint"], device="cpu")
        assert parameter_hash(updated.policy) == report["final_policy_hash"]
        progress = json.loads((first / "progress.json").read_text())
        records = progress["episodes"]
        assert records
        assert all(record["training_context"]["opponent"] == "selected" for record in records)
        assert all(record["training_context"]["player"] == 1 for record in records)
        assert all(record["outcome"] == "time_limit" for record in records)
        assert all(0 < record["end_steps"] <= 8 for record in records)
        assert progress["episode_summary"]["overall"]["counts"]["time_limit"] == len(records)
        assert progress["rollout_episode_summary"]["episodes"] == len(records)
        contract = save_contract(first, env, config)
        config["initial_policy"] = {"kind": "checkpoint", "path": report["checkpoint"],
                                    "training_config": contract}
        resumed = train_br(env, config, "cpu", 14, second)
        assert resumed["start_steps"] == 8
        assert resumed["steps"] == 16
        assert resumed["initial_policy_hash"] == report["final_policy_hash"]
    finally:
        env.close()


@pytest.mark.parametrize("probabilities", [[.2, .2], [-1., 2.], [float("nan"), 1.]])
def test_invalid_br_distribution_fails_before_loading(tmp_path, probabilities):
    env = fixture_env()
    try:
        with pytest.raises(ValueError, match="probabilities"):
            train_br(env, {"opponents": [
                {"name": str(index), "probability": probability, "policy": {"kind": "invalid"}}
                for index, probability in enumerate(probabilities)]}, "cpu", 1, tmp_path)
    finally:
        env.close()


def test_updated_checkpoints_follow_global_step_boundaries_after_resuming(tmp_path):
    torch.set_num_threads(1)
    env = fixture_env()
    config = fixture_config("mlp") | {"name": "br", "player": 0,
        "matchups": {"mode": "fixed"}, "timesteps": 32, "checkpoint_every": 16,
        "initial_policy": {"kind": "fresh"},
        "opponents": [{"name": "random", "probability": 1., "policy": {"kind": "uniform"}}]}
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    try:
        train_br(env, config, "cpu", 13, first)
        contract = save_contract(first, env, config)
        config["timesteps"] = 24
        config["initial_policy"] = {"kind": "checkpoint", "training_config": contract,
            "path": str(first / "checkpoints" / "updated_16_steps.zip")}
        train_br(env, config, "cpu", 14, second)
        for directory, expected in ((first, [8, 16, 32]), (second, [24, 32])):
            timing = json.loads((directory / "timing.json").read_text())
            saved = [row for row in timing["rollouts"] if "updated_checkpoint" in row]
            assert [row["steps"] for row in saved] == expected
            for row in saved:
                model = algorithm_type("mlp").load(row["updated_checkpoint"], device="cpu")
                assert model.num_timesteps == row["steps"]
                assert model._n_updates == row["ppo_n_updates"]
    finally:
        env.close()
