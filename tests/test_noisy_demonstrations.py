"""Teacher perturbations retain labels, actual input history and replayable evidence."""
import hashlib
import json
from pathlib import Path

from hydra import compose, initialize_config_dir
import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

from soku_rl.env.encoding import decode_action
from soku_rl.env.wrappers.learning import LearningConfig, LearningVectorEnv
from soku_rl.policy.action_noise import ActionNoisePolicy
from soku_rl.rl.behavior_cloning import load_demonstrations, fit_demonstrations, ObservationContractEnv
from soku_rl.rl.demonstration_evaluation import score_validation
from soku_rl.rl.demonstration_sets import load_demonstration_sets
from soku_rl.rl.demonstrations import collect_demonstrations, demonstration_plan
from soku_rl.rl.ppo import create_ppo
from test_behavior_cloning import rewrite_manifest
from test_demonstrations import ConstantPolicy, population
from test_recovery_demonstrations import CountingTeacher
from test_shared_ppo import fixture_config, fixture_env, save_contract


@pytest.fixture
def noisy_dataset(tmp_path, monkeypatch):
    env = fixture_env()
    env = LearningVectorEnv(env.env, LearningConfig("combat", False, 2, 0.))
    backend = env.env.backend
    monkeypatch.setattr(backend, "reset_matchups", lambda seeds, matches: backend.reset_slots(seeds), raising=False)
    teacher = CountingTeacher()
    behavior = ActionNoisePolicy("noisy", teacher, 90, .5)
    plan = demonstration_plan({"episodes_per_seat": 2, "validation_per_seat": 1}, population(), 42, {8, 9})
    manifest = collect_demonstrations(env, plan, {"character": 1, "palette": 0, "deck": 0},
        population(), teacher, [ConstantPolicy(8)], behavior, tmp_path / "episodes")
    config = fixture_config("lstm") | {"name": "br", "matchups": {"mode": "sampled",
        "learner": {"character": 1, "palette": 0, "deck": 0}}, "opponents": population()}
    path = save_contract(tmp_path, env, config)
    contract = OmegaConf.load(path)
    contract.excluded = {"validation": {"world_seeds": [8]}, "test": {"world_seeds": [9]}}
    contract.teacher = {"kind": "rule", "name": "fixture_teacher"}
    contract.behavior = {"kind": "action_noise", "random_probability": .5, "policy": contract.teacher}
    OmegaConf.save(contract, path)
    (tmp_path / "result.json").write_text(json.dumps({"success": True,
        "method": "rule_demonstrations", "result": manifest}))
    yield tmp_path, env, teacher, manifest
    env.close()


def test_noise_keeps_teacher_clock_labels_and_actual_history(noisy_dataset):
    directory, env, teacher, manifest = noisy_dataset
    assert manifest["schema"] == 4 and manifest["control"] == "teacher_noise"
    assert len(teacher.actors) == 4 and all(actor.calls == 3 for actor in teacher.actors)
    samples, loaded, _, _ = load_demonstrations(directory, env.interface)
    assert loaded == manifest
    replacements = 0
    for row in manifest["episodes"]:
        data = torch.load(directory / "episodes" / row["path"], weights_only=False)
        assert data["actions"].tolist() == [(row["teacher_seed"] + frame) % 90 for frame in range(3)]
        gate, actions = np.random.SeedSequence(row["behavior_seed"]).spawn(2)
        gate_rng, action_rng = np.random.default_rng(gate), np.random.default_rng(actions)
        for frame in range(3):
            selected = gate_rng.random() < .5
            action = action_rng.integers(90) if selected else data["actions"][frame]
            assert data["noise_selected"][frame] == selected
            assert data["executed_actions"][frame] == action
            if frame < 2:
                np.testing.assert_array_equal(data["observations"][frame + 1].unpack()[-8:],
                    np.asarray(decode_action(env.interface.command(int(action))).inputs, dtype=np.float32))
        assert row["noise_decisions"] == int(data["noise_selected"].sum())
        replacements += row["noise_decisions"]
    assert 0 < replacements < 12
    for rows in samples.values():
        assert len(rows) == 6 and all(len(row) == 4 for row in rows)
        assert [row[3] == -1 for row in rows] == [True, False, False] * 2
    with pytest.raises(ValueError, match="value_coef=0"):
        load_demonstration_sets([str(directory)], env.interface, .5)
    _, _, identities = load_demonstration_sets([str(directory)], env.interface, 0.)
    assert identities[0]["control"] == "teacher_noise"
    assert identities[0]["frames"] == {"train": 6, "validation": 6}


@pytest.mark.parametrize("damage", ["mask", "dtype", "executed", "count", "fingerprint", "probability", "contract", "seed"])
def test_noise_loader_rejects_tampered_semantics_after_hash_update(noisy_dataset, damage):
    directory, env, _, manifest = noisy_dataset
    row = manifest["episodes"][0]
    path = directory / "episodes" / row["path"]
    if damage in {"mask", "dtype", "executed"}:
        data = torch.load(path, weights_only=False)
        if damage == "mask":
            data["noise_selected"][0] = not data["noise_selected"][0]
            row["noise_decisions"] = int(data["noise_selected"].sum())
        elif damage == "dtype":
            data["noise_selected"] = data["noise_selected"].astype(np.int64)
        else:
            data["executed_actions"][0] = (data["executed_actions"][0] + 1) % 90
            row["teacher_behavior_disagreements"] = int(np.count_nonzero(data["actions"] != data["executed_actions"]))
        torch.save(data, path)
        row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    elif damage == "count":
        row["noise_decisions"] += 1
    elif damage == "fingerprint":
        manifest["behavior_fingerprint"] = "wrong"
    elif damage == "probability":
        manifest["random_probability"] = True
    elif damage == "seed":
        row["behavior_seed"] = 999
        plan_path = directory / "episodes/plan.json"
        plan = json.loads(plan_path.read_text())
        next(item for item in plan if item["id"] == row["id"])["behavior_seed"] = 999
        plan_path.write_text(json.dumps(plan))
    else:
        contract = OmegaConf.load(directory / "config.yaml")
        contract.behavior.policy.name = "wrong"
        OmegaConf.save(contract, directory / "config.yaml")
    rewrite_manifest(directory, manifest)
    with pytest.raises(ValueError, match="noisy-teacher"):
        load_demonstrations(directory, env.interface)


@pytest.mark.parametrize("policy_type", ["mlp", "lstm"])
def test_noisy_labels_train_all_frames_and_omit_teacher_value_score(noisy_dataset, policy_type):
    directory, env, _, manifest = noisy_dataset
    torch.set_num_threads(1)
    samples, _, _ = load_demonstration_sets([str(directory)], env.interface, 0.)
    algorithm = fixture_config(policy_type)
    config = {"epochs": 2, "batch_size": 4, "value_coef": 0., "action_change_weight": 1.,
        "initial_policy": {"kind": "fresh"}}
    if policy_type == "lstm":
        config["sequence_length"] = 2
    output = directory / "fit"
    output.mkdir()
    report = fit_demonstrations(env.interface, algorithm, samples, config, "cpu", 7, output)
    assert report["supervised_frames"] == {"train": 6, "validation": 6}
    assert report["initial_policy_hash"] != report["final_policy_hash"]
    model, _ = create_ppo(ObservationContractEnv(env.interface), env.interface, algorithm,
        {"kind": "fresh"}, "cpu", 7)
    score = score_validation(model, samples, manifest, 4, 2)
    assert score["groups"]["overall"]["frames"] == 6
    assert "value_mse" not in score["groups"]["overall"]["metrics"]


def test_noise_teacher_mismatch_fails_before_creating_output(tmp_path):
    env = fixture_env()
    try:
        with pytest.raises(ValueError, match="declared teacher"):
            collect_demonstrations(env, [], {}, [], ConstantPolicy(3), [],
                ActionNoisePolicy("bad", ConstantPolicy(4), 90, .02), tmp_path / "episodes")
        assert not (tmp_path / "episodes").exists()
    finally:
        env.close()


def test_noisy_collection_preset_keeps_both_seats_and_fixed_god_opponents():
    with initialize_config_dir(config_dir=str(Path(__file__).parents[1] / "config"), version_base="1.3"):
        config = OmegaConf.to_container(compose(config_name="collect_address_noisy_demonstrations"), resolve=True)
    assert config["behavior"] == {"kind": "action_noise", "policy": config["teacher"], "random_probability": .02}
    assert config["collection"] == {"episodes_per_seat": 8, "validation_per_seat": 2}
    assert config["seed"] == 2901383 and config["num_envs"] == 8
    assert config["episode"]["decision_frames"] == 1 and config["episode"]["latency_frames"] == 0
    assert config["wrappers"]["action_set"] == "full"
    assert "curriculum" not in config["algorithm"]
    assert len(config["excluded_datasets"]) == 9
