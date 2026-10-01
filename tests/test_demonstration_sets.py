import hashlib
import json
import shutil

from omegaconf import OmegaConf
import pytest
import torch

from soku_rl.rl.demonstration_sets import load_demonstration_sets
from test_behavior_cloning import dataset, rewrite_manifest
from test_demonstrations import population


@pytest.fixture
def pair(dataset):
    first, interface, _ = dataset
    config = OmegaConf.load(first / "config.yaml")
    config.algorithm.matchups = {"mode": "sampled", "learner": {"character": 1, "palette": 0, "deck": 0}}
    config.algorithm.opponents = population()
    config.teacher = {"kind": "constant", "value": 3}
    OmegaConf.save(config, first / "config.yaml")
    second = first / "second"
    second.mkdir()
    for name in ("config.yaml", "result.json"):
        shutil.copyfile(first / name, second / name)
    shutil.copytree(first / "episodes", second / "episodes")
    manifest = json.loads((second / "episodes/manifest.json").read_text())
    manifest.update(control="learner", behavior_fingerprint="constant:5")
    plan = json.loads((second / "episodes/plan.json").read_text())
    for row in plan:
        row["world_seed"] = 1000 + row["id"]
    for row in manifest["episodes"]:
        row["world_seed"] = 1000 + row["id"]
        row["teacher_behavior_disagreements"] = row["steps"]
        path = second / "episodes" / row["path"]
        data = torch.load(path, weights_only=False)
        data["executed_actions"][:] = 5
        torch.save(data, path)
        row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (second / "episodes/plan.json").write_text(json.dumps(plan))
    rewrite_manifest(second, manifest)
    return first, second, interface


def test_aggregation_preserves_splits_and_uses_actual_previous_inputs(pair):
    first, second, interface = pair
    samples, contract, identities = load_demonstration_sets([first, second], interface, 0.)
    assert {split: len(rows) for split, rows in samples.items()} == {"train": 12, "validation": 12}
    assert sum(identity["environment_steps"] for identity in identities) == 24
    assert [identity["control"] for identity in identities] == ["teacher", "learner"]
    assert contract["teacher"]["value"] == 3
    # Teacher labels stay constant, but copying the actual preceding learner input is wrong.
    assert [int(row[3]) for row in samples["validation"][6:]] == [-1, 1, 1, -1, 1, 1]
    train_worlds = {world for identity in identities for world in identity["worlds"]["train"]}
    validation_worlds = {world for identity in identities for world in identity["worlds"]["validation"]}
    assert not train_worlds & validation_worlds


@pytest.mark.parametrize("problem", ["duplicate_path", "world_overlap", "teacher", "reserved", "critic"])
def test_aggregation_rejects_incompatible_or_leaking_data(pair, problem):
    first, second, interface = pair
    sources, coefficient = [first, second], 0.
    if problem == "duplicate_path":
        sources = [first, first]
    elif problem == "critic":
        coefficient = .5
    elif problem == "reserved":
        config = OmegaConf.load(second / "config.yaml")
        config.excluded.test.world_seeds = [99]
        OmegaConf.save(config, second / "config.yaml")
    else:
        manifest = json.loads((second / "episodes/manifest.json").read_text())
        if problem == "teacher":
            manifest["teacher_fingerprint"] = "different_teacher"
        else:
            previous = json.loads((first / "episodes/manifest.json").read_text())["episodes"]
            held_out = next(row["world_seed"] for row in previous if row["split"] == "validation")
            row = next(row for row in manifest["episodes"] if row["split"] == "train")
            row["world_seed"] = held_out
            plan = json.loads((second / "episodes/plan.json").read_text())
            next(p for p in plan if p["id"] == row["id"])["world_seed"] = held_out
            (second / "episodes/plan.json").write_text(json.dumps(plan))
        rewrite_manifest(second, manifest)
    with pytest.raises(ValueError):
        load_demonstration_sets(sources, interface, coefficient)
