"""Only the private worker Wine server may be stopped during cleanup."""
import json
import errno
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from linux_runtime import worker_session


@pytest.mark.parametrize("cleanup_error", [0, errno.ENOTEMPTY, errno.EACCES])
def test_worker_owns_distinct_prefix_and_server_for_entire_lifecycle(tmp_path, monkeypatch, cleanup_error):
    source = tmp_path / "source"
    source.mkdir()
    (source / "system.reg").write_text("original registry")
    config = {"root": str(tmp_path), "state": str(tmp_path / "state"), "prefix_source": str(source),
              "wine": str(tmp_path / "wine/bin/wine"), "windows_python": "python.exe"}
    environments, commands = [], []

    def environment(settings):
        assert Path(settings["prefix"]) != source
        assert (Path(settings["prefix"]) / "system.reg").read_text() == "original registry"
        return {"WINEPREFIX": settings["prefix"], "SOKURL_WINE_SERVER_ROOT": "shared-forbidden"}

    def popen(command, env):
        environments.append(dict(env))
        commands.append(command)
        return SimpleNamespace(pid=123, wait=lambda: 0)

    def run(command, env, timeout, check):
        environments.append(dict(env))
        commands.append(command)
        assert timeout == 30 and check is False
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(worker_session, "wine_environment", environment)
    monkeypatch.setattr(worker_session.subprocess, "Popen", popen)
    monkeypatch.setattr(worker_session.subprocess, "run", run)
    original = worker_session.shutil.rmtree
    removals = []

    def remove(path):
        removals.append(path)
        if cleanup_error and len(removals) == 1:
            raise OSError(cleanup_error, "simulated removal error")
        return original(path)

    monkeypatch.setattr(worker_session.shutil, "rmtree", remove)
    monkeypatch.setattr(worker_session.time, "sleep", lambda seconds: None)
    if cleanup_error == errno.EACCES:
        with pytest.raises(OSError):
            worker_session.run_worker(config, ["tools/rollout_worker.py"])
    else:
        assert worker_session.run_worker(config, ["tools/rollout_worker.py"]) == 0
    assert environments[0] == environments[1] == environments[2]
    assert environments[0]["SOKURL_WINE_SERVER_ROOT"] != "shared-forbidden"
    assert commands[1][-1] == "-k" and commands[2][-1] == "-w"
    prefix = Path(environments[0]["WINEPREFIX"])
    assert prefix.exists() == (cleanup_error == errno.EACCES)
    assert source.exists()
    record = json.loads((prefix.parent / "session.json").read_text())
    assert record["exit_code"] == record["server_wait_exit"] == 0
    if cleanup_error == errno.EACCES:
        assert "cleanup_error" in record and len(removals) == 1
    else:
        assert record["cleanup_attempts"]["prefix"] == (2 if cleanup_error else 1)


def test_cleanup_retry_is_bounded_and_still_reports_failure(tmp_path, monkeypatch):
    attempts = []

    def busy(path):
        attempts.append(path)
        raise OSError(errno.EBUSY, "still held open")

    monkeypatch.setattr(worker_session.shutil, "rmtree", busy)
    monkeypatch.setattr(worker_session.time, "sleep", lambda seconds: None)
    with pytest.raises(OSError, match="still held open"):
        worker_session.remove_exited_worker_tree(tmp_path, 3)
    assert attempts == [tmp_path] * 3


@pytest.mark.parametrize("attempts", [0, -1, True])
def test_invalid_cleanup_budget_is_rejected(tmp_path, attempts):
    with pytest.raises(ValueError, match="positive integer"):
        worker_session.remove_exited_worker_tree(tmp_path, attempts)
