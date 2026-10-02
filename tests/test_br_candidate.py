"""Reference controllers use the BR matchup evaluator without pretending to load a model."""
import hashlib
from pathlib import Path
import runpy

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest


specification = runpy.run_path(str(Path(__file__).parents[1] / "tools/benchmark_br.py"))["candidate_specification"]


def test_god_reference_does_not_require_or_hash_a_checkpoint(tmp_path):
    with initialize_config_dir(config_dir=str(Path(__file__).parents[1] / "config"), version_base="1.3"):
        cfg = compose(config_name="benchmark_br", overrides=["+br_candidate=god"])
        candidate = OmegaConf.to_container(cfg.candidate, resolve=True)
    name, spec, metadata = specification({"candidate": candidate}, tmp_path, {})
    assert name == "rule-br:god"
    assert spec == candidate
    assert spec["rules"]["god"]["script"] == "character"
    assert metadata == {}


@pytest.mark.parametrize("policy_type,kind", [("mlp", "sb3"), ("lstm", "sb3_recurrent")])
def test_default_candidate_retains_model_source_and_hash(tmp_path, policy_type, kind):
    model = tmp_path / "final.zip"
    model.write_bytes(b"model-for-source-selection-test")
    name, spec, metadata = specification({"candidate": {"kind": "checkpoint"},
        "checkpoint": "final.zip"}, tmp_path, {"policy_type": policy_type})
    assert name == "learned-br"
    assert spec == {"kind": kind, "path": str(model), "training_config": str(tmp_path / "config.yaml")}
    assert metadata["checkpoint_sha256"] == hashlib.sha256(model.read_bytes()).hexdigest()


def test_checkpoint_outside_training_directory_is_rejected(tmp_path):
    source = tmp_path / "run"
    source.mkdir()
    (tmp_path / "outside.zip").write_bytes(b"unrelated")
    with pytest.raises(ValueError, match="belong"):
        specification({"candidate": {"kind": "checkpoint"}, "checkpoint": "../outside.zip"}, source, {})


@pytest.mark.parametrize("policy_type,kind", [("mlp", "sb3"), ("lstm", "sb3_recurrent")])
def test_greedy_candidate_preserves_checkpoint_identity_and_labels_inference(tmp_path, policy_type, kind):
    with initialize_config_dir(config_dir=str(Path(__file__).parents[1] / "config"), version_base="1.3"):
        cfg = compose(config_name="benchmark_br", overrides=["+br_candidate=greedy"])
        candidate = OmegaConf.to_container(cfg.candidate, resolve=True)
    model = tmp_path / "best.zip"
    model.write_bytes(b"greedy-source-selection")
    name, spec, metadata = specification({"candidate": candidate, "checkpoint": "best.zip"},
        tmp_path, {"policy_type": policy_type})
    assert name == "learned-br:greedy"
    assert spec == {"kind": "greedy", "policy": {"kind": kind, "path": str(model),
        "training_config": str(tmp_path / "config.yaml")}}
    assert metadata["checkpoint_sha256"] == hashlib.sha256(model.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="belong"):
        nested = tmp_path / "run"
        nested.mkdir()
        specification({"candidate": candidate, "checkpoint": "../best.zip"}, nested, {"policy_type": policy_type})


@pytest.mark.parametrize("candidate", [{"kind": "uniform"}, {"kind": "checkpoint", "extra": 1},
    {"kind": "checkpoint", "inference": "unknown"},
    {"kind": "rule", "name": "" , "rules": {}}, {"kind": "rule", "name": "god"}])
def test_invalid_candidate_is_rejected(candidate, tmp_path):
    with pytest.raises(ValueError):
        specification({"candidate": candidate}, tmp_path, {})


@pytest.mark.parametrize("candidate", [{"kind": "checkpoint"}, {"kind": "checkpoint", "inference": "greedy"}])
def test_dqn_candidate_uses_native_greedy_policy(candidate, tmp_path):
    model = tmp_path / "final.zip"
    model.write_bytes(b"dqn-source-selection")
    name, spec, metadata = specification({"candidate": candidate, "checkpoint": "final.zip"},
        tmp_path, {"learner": "dqn", "policy_type": "mlp"})
    assert name == "learned-br"
    assert spec == {"kind": "sb3_dqn", "path": str(model), "training_config": str(tmp_path / "config.yaml")}
    assert metadata["checkpoint_sha256"] == hashlib.sha256(model.read_bytes()).hexdigest()
