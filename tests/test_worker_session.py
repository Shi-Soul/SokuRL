"""Only the private worker Wine server may be stopped during cleanup."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from linux_runtime import worker_session


def test_worker_owns_distinct_prefix_and_server_for_entire_lifecycle(tmp_path, monkeypatch):
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
    assert worker_session.run_worker(config, ["tools/rollout_worker.py"]) == 0
    assert environments[0] == environments[1] == environments[2]
    assert environments[0]["SOKURL_WINE_SERVER_ROOT"] != "shared-forbidden"
    assert commands[1][-1] == "-k" and commands[2][-1] == "-w"
    prefix = Path(environments[0]["WINEPREFIX"])
    assert not prefix.exists()
    assert source.exists()
    record = json.loads((prefix.parent / "session.json").read_text())
    assert record["exit_code"] == record["server_wait_exit"] == 0
