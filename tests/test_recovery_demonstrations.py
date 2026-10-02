"""Recovery records actual teacher suffixes while retaining unsupervised history."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir
import pytest
import torch

from soku_rl.env.encoding import decode_action
from soku_rl.env.wrappers.learning import LearningConfig, LearningVectorEnv
from soku_rl.policy.takeover import TeacherTakeoverPolicy
from soku_rl.rl.behavior_cloning import load_demonstrations, fit_demonstrations, ObservationContractEnv
from soku_rl.rl.demonstrations import collect_demonstrations, demonstration_plan
from soku_rl.rl.demonstration_evaluation import score_validation
from soku_rl.rl.demonstration_sets import load_demonstration_sets
from soku_rl.rl.ppo import create_ppo
from test_behavior_cloning import rewrite_manifest
from test_demonstrations import ConstantPolicy, population
from test_shared_ppo import fixture_config, fixture_env, save_contract


@dataclass
class CountingActor:
    seed: int
    calls: int

    def act(self, observation):
        action = (self.seed + self.calls) % 90
        self.calls += 1
        return action


class CountingTeacher:
    fingerprint = "counting-teacher-v1"

    def __init__(self):
        self.actors = []

    def spawn(self, seed):
        actor = CountingActor(seed, 0)
        self.actors.append(actor)
        return actor


@pytest.fixture
def recovery_dataset(tmp_path, monkeypatch):
    env = fixture_env()
    env = LearningVectorEnv(env.env, LearningConfig("combat", False, 2, 0.))
    backend = env.env.backend
    monkeypatch.setattr(backend, "reset_matchups", lambda seeds, matches: backend.reset_slots(seeds), raising=False)
    teacher = CountingTeacher()
    behavior = TeacherTakeoverPolicy("recovery", ConstantPolicy(5), teacher, 2)
    plan = demonstration_plan({"episodes_per_seat": 2, "validation_per_seat": 1}, population(), 42, {8, 9})
    manifest = collect_demonstrations(env, plan, {"character": 1, "palette": 0, "deck": 0},
        population(), teacher, [ConstantPolicy(8)], behavior, tmp_path / "episodes")
    config = fixture_config("lstm") | {"name": "br", "matchups": {"mode": "sampled",
        "learner": {"character": 1, "palette": 0, "deck": 0}}, "opponents": population()}
    path = save_contract(tmp_path, env, config)
    contract = OmegaConf.load(path)
    contract.excluded = {"validation": {"world_seeds": [8]}, "test": {"world_seeds": [9]}}
    contract.teacher = {"kind": "rule", "name": "fixture_teacher"}
    contract.behavior = {"kind": "teacher_takeover", "after_frames": 2,
        "teacher": contract.teacher, "learner": {"kind": "fixture_learner"}}
    OmegaConf.save(contract, path)
    (tmp_path / "result.json").write_text(json.dumps({"success": True,
        "method": "rule_demonstrations", "result": manifest}))
    yield tmp_path, env, teacher, manifest, config
    env.close()


def test_collection_uses_one_teacher_actor_once_per_frame_and_executes_its_suffix(recovery_dataset):
    directory, env, teacher, manifest, _ = recovery_dataset
    assert manifest["schema"] == 3 and manifest["control"] == "teacher_takeover"
    assert len(teacher.actors) == 4 and all(actor.calls == 3 for actor in teacher.actors)
    for row in manifest["episodes"]:
        data = torch.load(directory / "episodes" / row["path"], weights_only=False)
        expected = [(row["teacher_seed"] + frame) % 90 for frame in range(3)]
        assert data["actions"].tolist() == expected
        assert data["executed_actions"].tolist() == [5, 5, expected[2]]
        assert data["supervised"].tolist() == [False, False, True]
        assert row["supervised_steps"] == 1
        np.testing.assert_array_equal(data["observations"][2].unpack()[-8:],
            np.asarray(decode_action(env.interface.command(5)).inputs, dtype=np.float32))
    samples, loaded, _, _ = load_demonstrations(directory, env.interface)
    assert loaded == manifest
    for rows in samples.values():
        assert len(rows) == 6
        assert [bool(row[4]) for row in rows] == [False, False, True] * 2
        assert [row[3] == -1 for row in rows] == [True, False, False] * 2
    with pytest.raises(ValueError, match="value_coef=0"):
        load_demonstration_sets([str(directory)], env.interface, .5)
    aggregated, _, identities = load_demonstration_sets([str(directory)], env.interface, 0.)
    assert len(aggregated["train"]) == 6 and identities[0]["control"] == "teacher_takeover"
    assert identities[0]["supervised_frames"] == {"train": 2, "validation": 2}


@pytest.mark.parametrize("damage", ["mask", "executed", "count", "fingerprint", "boundary", "contract"])
def test_loader_rejects_inconsistent_recovery_evidence(recovery_dataset, damage):
    directory, env, _, manifest, _ = recovery_dataset
    row = manifest["episodes"][0]
    path = directory / "episodes" / row["path"]
    if damage in {"mask", "executed"}:
        data = torch.load(path, weights_only=False)
        if damage == "mask":
            data["supervised"][0] = True
        else:
            data["executed_actions"][-1] = (data["actions"][-1] + 1) % 90
            row["teacher_behavior_disagreements"] = int(np.count_nonzero(data["actions"] != data["executed_actions"]))
        torch.save(data, path)
        row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    elif damage == "count":
        row["supervised_steps"] = 2
    elif damage == "fingerprint":
        manifest["behavior_fingerprint"] = "wrong"
    elif damage == "boundary":
        manifest["after_frames"] = True
    else:
        contract = OmegaConf.load(directory / "config.yaml")
        contract.behavior.after_frames = 1
        OmegaConf.save(contract, directory / "config.yaml")
    rewrite_manifest(directory, manifest)
    with pytest.raises(ValueError, match="recovery"):
        load_demonstrations(directory, env.interface)


@pytest.mark.parametrize("policy_type", ["mlp", "lstm"])
def test_fit_and_validation_count_only_executed_teacher_labels(recovery_dataset, policy_type):
    directory, env, _, manifest, _ = recovery_dataset
    torch.set_num_threads(1)
    samples, _, _, _ = load_demonstrations(directory, env.interface)
    algorithm = fixture_config(policy_type)
    config = {"epochs": 2, "batch_size": 4, "value_coef": 0., "action_change_weight": 2.,
        "initial_policy": {"kind": "fresh"}}
    if policy_type == "lstm":
        config["sequence_length"] = 2
    output = directory / "fit"
    output.mkdir()
    report = fit_demonstrations(env.interface, algorithm, samples, config, "cpu", 7, output)
    assert report["supervised_frames"] == {"train": 2, "validation": 2}
    assert report["train_frames"] == report["validation_frames"] == 6
    assert sum(report["constant_action_baseline"]["label_counts"]["train"]) == 2
    assert report["supervised_updates"] == 2
    model, _ = create_ppo(ObservationContractEnv(env.interface), env.interface, algorithm,
        {"kind": "fresh"}, "cpu", 7)
    score = score_validation(model, samples, manifest, 4, 2)
    assert score["groups"]["overall"]["frames"] == 6
    assert score["groups"]["overall"]["supervised_frames"] == 2
    assert "value_mse" not in score["groups"]["overall"]["metrics"]
    with pytest.raises(ValueError, match="value_coef=0"):
        fit_demonstrations(env.interface, algorithm, samples, config | {"value_coef": .5}, "cpu", 7, output)


@pytest.mark.parametrize("boundary", [0, 10])
def test_pure_teacher_and_no_recovery_frames_keep_complete_collection(tmp_path, monkeypatch, boundary):
    env = fixture_env()
    backend = env.env.backend
    monkeypatch.setattr(backend, "reset_matchups", lambda seeds, matches: backend.reset_slots(seeds), raising=False)
    teacher = CountingTeacher()
    behavior = TeacherTakeoverPolicy("recovery", ConstantPolicy(5), teacher, boundary)
    plan = demonstration_plan({"episodes_per_seat": 2, "validation_per_seat": 1}, population(), 42, set())
    try:
        manifest = collect_demonstrations(env, plan, {"character": 1, "palette": 0, "deck": 0},
            population(), teacher, [ConstantPolicy(8)], behavior, tmp_path / "episodes")
        assert manifest["complete"] and manifest["successful_env_steps"] == 12
        for row in manifest["episodes"]:
            data = torch.load(tmp_path / "episodes" / row["path"], weights_only=False)
            assert row["supervised_steps"] == (3 if boundary == 0 else 0)
            assert data["supervised"].tolist() == [boundary == 0] * 3
            assert data["executed_actions"].tolist() == (data["actions"].tolist() if boundary == 0 else [5] * 3)
    finally:
        env.close()


def test_mismatched_teacher_is_rejected_before_creating_output(tmp_path):
    env = fixture_env()
    try:
        with pytest.raises(ValueError, match="declared teacher"):
            collect_demonstrations(env, [], {}, [], ConstantPolicy(3), [],
                TeacherTakeoverPolicy("bad", ConstantPolicy(5), ConstantPolicy(4), 2), tmp_path / "episodes")
        assert not (tmp_path / "episodes").exists()
    finally:
        env.close()


def test_recovery_presets_keep_explicit_seed_exclusions_and_shared_bc_contract():
    with initialize_config_dir(config_dir=str(Path(__file__).parents[1] / "config"), version_base="1.3"):
        collection = OmegaConf.to_container(compose(config_name="collect_address_recovery_demonstrations"), resolve=True)
        fitting = OmegaConf.to_container(compose(config_name="pretrain_address_recovery_demonstrations"), resolve=True)
    assert collection["behavior"]["teacher"] == collection["teacher"]
    assert collection["behavior"]["after_frames"] == 1024
    assert collection["seed"] == 1901279 and collection["num_envs"] == 8
    assert collection["collection"] == {"episodes_per_seat": 8, "validation_per_seat": 2}
    assert collection["wrappers"]["action_set"] == "full"
    assert collection["episode"]["decision_frames"] == 1 and collection["episode"]["latency_frames"] == 0
    assert fitting["pretraining"]["initial_policy"]["path"] == collection["behavior"]["learner"]["path"]
    assert fitting["pretraining"]["value_coef"] == 0.
    assert fitting["pretraining"]["epochs"] == 20 and fitting["pretraining"]["sequence_length"] == 64
    assert fitting["rl"]["ppo"] == fitting["algorithm"]["ppo"]
    assert "curriculum" not in fitting["algorithm"] and "curriculum" not in collection["algorithm"]
