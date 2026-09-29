"""Reject incomplete training and select final policies independently of moved paths."""
import json
from pathlib import Path
import sys

from omegaconf import OmegaConf
import pytest

from soku_rl.policy.artifacts import completed_policies


def save_report(directory, name, result):
    (directory / "result.json").write_text(json.dumps({"success": True, "algorithm": name, "result": result}))
    OmegaConf.save(OmegaConf.create({"algorithm": {"name": name, "policy_type": "lstm"},
                                    "episode": {"latency_frames": 5}}), directory / "config.yaml")


def test_intermediate_checkpoint_cannot_replace_missing_final_policy(tmp_path):
    save_report(tmp_path, "ppo", {})
    for seat in ("player_0", "player_1"):
        (tmp_path / seat).mkdir()
        (tmp_path / seat / "checkpoint.zip").write_bytes(b"intermediate")
    with pytest.raises(FileNotFoundError, match="final.zip"):
        completed_policies(tmp_path)
    for seat in ("player_0", "player_1"):
        (tmp_path / seat / "final.zip").write_bytes(b"final")
    training, candidate = completed_policies(tmp_path)
    assert training["episode"]["latency_frames"] == 5
    assert candidate["player_0"]["kind"] == "sb3_recurrent"
    assert candidate["player_0"]["path"] != candidate["player_1"]["path"]


def test_failed_run_with_checkpoint_is_rejected(tmp_path):
    (tmp_path / "result.json").write_text(json.dumps({"success": False, "error": "worker failed"}))
    with pytest.raises(ValueError, match="successful completed"):
        completed_policies(tmp_path)


def test_ippo_final_checkpoint_survives_moving_training_directory(tmp_path):
    save_report(tmp_path, "ippo", {"frames": 262144, "experiment_directory": "/old/location/experiment"})
    checkpoints = tmp_path / "experiment" / "checkpoints"
    checkpoints.mkdir(parents=True)
    for steps in (253952, 262144):
        (checkpoints / f"checkpoint_{steps}.pt").write_bytes(b"checkpoint")
    (tmp_path / "benchmarl-config.json").write_text("{}")
    _, candidate = completed_policies(tmp_path)
    for seat in ("player_0", "player_1"):
        assert candidate[seat]["path"] == str(checkpoints / "checkpoint_262144.pt")
        assert candidate[seat]["player"] == seat


@pytest.mark.parametrize("algorithm,kind", [("nfsp", "nfsp_average"), ("psro", "psro_mixture")])
def test_both_average_strategy_seats_are_exported(tmp_path, algorithm, kind):
    save_report(tmp_path, algorithm, {})
    (tmp_path / "final").mkdir()
    for seat in ("player_0", "player_1"):
        (tmp_path / "final" / f"{seat}.pt").write_bytes(b"checkpoint")
    (tmp_path / "population.json").write_text("{}")
    _, candidate = completed_policies(tmp_path)
    for seat in ("player_0", "player_1"):
        assert candidate[seat]["kind"] == kind
        assert candidate[seat]["player"] == seat


def test_evaluation_uses_exact_saved_contract_and_independent_runtime(tmp_path, monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    import benchmark_training

    save_report(tmp_path, "ppo", {})
    training = OmegaConf.load(tmp_path / "config.yaml")
    training.wrappers = {"action_set": "combat"}
    training.track = "human"
    OmegaConf.save(training, tmp_path / "config.yaml")
    for seat in ("player_0", "player_1"):
        (tmp_path / seat).mkdir()
        (tmp_path / seat / "final.zip").write_bytes(b"final")
    captured = []
    monkeypatch.setattr(benchmark_training, "run", captured.append)
    cfg = OmegaConf.create({"training_directory": str(tmp_path), "algorithm": {"name": "nfsp", "agent": {}},
        "episode": {"latency_frames": 12, "old_field": 1}, "candidate": {"player_0": "???"},
        "runtime": {"command": ["evaluation-worker"]}, "evaluation": {"world_seeds": [12, 13]}})
    benchmark_training.main.__wrapped__(cfg)
    actual = OmegaConf.to_container(captured[0], resolve=True, throw_on_missing=True)
    assert actual["episode"] == {"latency_frames": 5}
    assert actual["algorithm"] == {"name": "ppo", "policy_type": "lstm"}
    assert actual["runtime"] == {"command": ["evaluation-worker"]}
    assert actual["evaluation"]["world_seeds"] == [12, 13]
    assert len(actual["training_result_sha256"]) == 64
