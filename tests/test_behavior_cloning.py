"""Offline initialization preserves whole-game splits and shared PPO portability."""
import hashlib
import json

from omegaconf import OmegaConf
import pytest
import torch

from soku_rl.rl.behavior_cloning import ObservationContractEnv, fit_demonstrations, load_demonstrations, score_samples
from soku_rl.rl.demonstrations import collect_demonstrations, demonstration_plan
from soku_rl.rl.ppo import create_ppo, parameter_hash
from test_demonstrations import ConstantPolicy, population
from test_shared_ppo import fixture_config, fixture_env, save_contract


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    env = fixture_env()
    backend = env.env.backend
    monkeypatch.setattr(backend, "reset_matchups", lambda seeds, matches: backend.reset_slots(seeds), raising=False)
    config = fixture_config("mlp") | {"name": "br"}
    config["ppo"]["learning_rate"] = .02
    plan = demonstration_plan({"episodes_per_seat": 2, "validation_per_seat": 1}, population(), 42, {8, 9})
    manifest = collect_demonstrations(env, plan, {"character": 1, "palette": 0, "deck": 0},
        population(), ConstantPolicy(3), [ConstantPolicy(8)], tmp_path / "episodes")
    path = save_contract(tmp_path, env, config)
    contract = OmegaConf.load(path)
    contract.excluded = {"validation": {"world_seeds": [8]}, "test": {"world_seeds": [9]}}
    OmegaConf.save(contract, path)
    (tmp_path / "result.json").write_text(json.dumps({"success": True,
        "method": "rule_demonstrations", "result": manifest}))
    yield tmp_path, env.interface, config
    env.close()


def rewrite_manifest(directory, manifest):
    (directory / "episodes/manifest.json").write_text(json.dumps(manifest))
    report = json.loads((directory / "result.json").read_text())
    report["result"] = manifest
    (directory / "result.json").write_text(json.dumps(report))


def test_fit_uses_shared_ppo_and_weights_reload_with_fresh_optimizer(dataset):
    directory, interface, config = dataset
    torch.set_num_threads(1)
    samples, manifest, _, digest = load_demonstrations(directory, interface)
    assert {key: len(rows) for key, rows in samples.items()} == {"train": 6, "validation": 6}
    assert manifest["successful_env_steps"] == 12
    assert digest == hashlib.sha256((directory / "episodes/manifest.json").read_bytes()).hexdigest()
    validation_before = [(row[0].unpack().copy(), *row[1:]) for row in samples["validation"]]
    output = directory / "fit"
    output.mkdir()
    result = fit_demonstrations(interface, config, samples,
        {"epochs": 20, "batch_size": 4, "value_coef": .5}, "cpu", 7, output)
    assert result["ppo_steps"] == 0 and result["supervised_updates"] == 40
    assert result["best_epoch"] > 0
    assert result["history"][-1]["validation"]["nll"] < result["history"][0]["validation"]["nll"] * .5
    assert result["history"][-1]["validation"]["accuracy"] == 1.
    for before, after in zip(validation_before, samples["validation"], strict=True):
        assert (before[0] == after[0].unpack()).all() and before[1:] == after[1:]
    model, source = create_ppo(ObservationContractEnv(interface), interface, config,
        {"kind": "weights", "path": result["final_checkpoint"],
         "training_config": str(directory / "config.yaml")}, "cpu", 71)
    assert parameter_hash(model.policy) == result["final_policy_hash"]
    assert not model.policy.optimizer.state
    assert model.num_timesteps == source["source_steps"] == 0


def test_change_accuracy_exposes_a_policy_that_only_copies_previous_commands(dataset):
    directory, interface, config = dataset
    manifest = json.loads((directory / "episodes/manifest.json").read_text())
    for row in manifest["episodes"]:
        path = directory / "episodes" / row["path"]
        data = torch.load(path, weights_only=False)
        data["actions"][:] = [3, 3, 8]
        torch.save(data, path)
        row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    rewrite_manifest(directory, manifest)
    samples, _, _, _ = load_demonstrations(directory, interface)
    assert [int(row[3]) for row in samples["validation"]] == [-1, 0, 1, -1, 0, 1]
    model, _ = create_ppo(ObservationContractEnv(interface), interface, config, {"kind": "fresh"}, "cpu", 7)
    with torch.no_grad():
        model.policy.action_net.weight.zero_()
        model.policy.action_net.bias.zero_()
        model.policy.action_net.bias[3] = 10.
    scored = score_samples(model, samples["validation"], 4)
    assert scored["accuracy"] == pytest.approx(2 / 3)
    assert scored["changed_samples"] == 2
    assert scored["changed_accuracy"] == 0.
    assert scored["changed_nll"] > scored["nll"]


@pytest.mark.parametrize("damage", ["partial", "hash", "return", "reserved", "split", "accounting"])
def test_loader_rejects_corruption_and_data_leakage(dataset, damage):
    directory, interface, _ = dataset
    manifest = json.loads((directory / "episodes/manifest.json").read_text())
    row = manifest["episodes"][0]
    if damage == "partial":
        manifest["complete"] = False
    elif damage == "hash":
        row["sha256"] = "incorrect"
    elif damage == "return":
        path = directory / "episodes" / row["path"]
        data = torch.load(path, weights_only=False)
        data["returns"][0] += 1
        torch.save(data, path)
        row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    elif damage == "reserved":
        row["world_seed"] = 8
    elif damage == "split":
        row["split"] = "train" if row["split"] == "validation" else "validation"
    elif damage == "accounting":
        manifest["successful_env_steps"] += 1
    rewrite_manifest(directory, manifest)
    with pytest.raises(ValueError):
        load_demonstrations(directory, interface)


@pytest.mark.parametrize("key,value", [("epochs", 0), ("batch_size", True), ("value_coef", float("nan"))])
def test_bad_optimization_settings_fail_before_training(dataset, key, value):
    directory, interface, algorithm = dataset
    samples, _, _, _ = load_demonstrations(directory, interface)
    config = {"epochs": 1, "batch_size": 4, "value_coef": .5} | {key: value}
    with pytest.raises(ValueError):
        fit_demonstrations(interface, algorithm, samples, config, "cpu", 7, directory)
