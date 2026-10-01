"""Final evaluation gates reject failed, incomplete or inconsistent training."""
import json

import pytest

from queue_br_evaluations import digest, finished_training, live_job


def checkpoint(directory):
    (directory / "final.zip").write_bytes(b"model fixture")
    (directory / "final.replay.pkl").write_bytes(b"replay fixture")
    manifest = {"steps": 262144, "updates": 32256,
                "model_sha256": digest(directory / "final.zip"),
                "replay_sha256": digest(directory / "final.replay.pkl")}
    (directory / "final.replay.json").write_text(json.dumps(manifest))
    result = {"success": True, "algorithm": "br", "result": {"steps": 262144}}
    (directory / "result.json").write_text(json.dumps(result))
    return manifest, result


def test_queue_accepts_complete_pair_and_rejects_modified_checkpoint(tmp_path):
    manifest, _ = checkpoint(tmp_path)
    assert finished_training(tmp_path, 262144) == manifest
    (tmp_path / "final.zip").write_bytes(b"changed")
    with pytest.raises(ValueError, match="integrity"):
        finished_training(tmp_path, 262144)


def test_queue_rejects_failed_training_even_with_final_files(tmp_path):
    _, result = checkpoint(tmp_path)
    result["success"] = False
    (tmp_path / "result.json").write_text(json.dumps(result))
    with pytest.raises(RuntimeError, match="did not succeed"):
        finished_training(tmp_path, 262144)


def test_queue_rejects_smaller_budget(tmp_path):
    checkpoint(tmp_path)
    with pytest.raises(ValueError, match="budget"):
        finished_training(tmp_path, 524288)


def test_reused_pid_does_not_count_as_owned_job(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import queue_br_evaluations as queue
    (tmp_path / "launch.json").write_text(json.dumps({"trainer_pid": 123}))
    monkeypatch.setattr(queue.psutil, "Process", lambda pid:
        SimpleNamespace(cmdline=lambda: ["python", "output=another-task"]))
    assert not live_job(tmp_path, "trainer")
    monkeypatch.setattr(queue.psutil, "Process", lambda pid:
        SimpleNamespace(cmdline=lambda: ["python", "output=" + str(tmp_path)]))
    assert live_job(tmp_path, "trainer")
