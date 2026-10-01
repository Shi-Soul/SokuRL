"""The BR scheduler uses the common learner, mixture, and continuation contract."""
import json

import pytest

from test_shared_ppo import fixture_config, fixture_env, save_contract, torch
from soku_rl.marl.br import train_br


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
        records = json.loads((first / "progress.json").read_text())["episodes"]
        assert records
        assert all(record["training_context"]["opponent"] == "selected" for record in records)
        assert all(record["training_context"]["player"] == 1 for record in records)
        assert all(record["outcome"] == "time_limit" for record in records)
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
