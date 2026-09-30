"""Runtime identity includes implementations after moving them into packages."""
import sys
from unittest.mock import Mock

import pytest

if sys.platform != "win32":
    pytest.skip("Windows runtime identity", allow_module_level=True)

import runtime_identity


def test_nested_runtime_change_changes_implementation_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime_identity, "ROOT", tmp_path)
    sources = ("src/soku_rl/play/match.py", "tools/network_worker.py",
               "tools/network_runtime/game.py", "tools/game_runtime/privileged.py")
    for name in sources:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# original\n", encoding="utf-8")
    game = tmp_path / "game"
    game.mkdir()
    for name in ("th123.exe", "th123a.dat", "th123b.dat", "th123c.dat", "d3d9.dll", "SWRSToys.ini"):
        (game / name).write_bytes(b"fixture")
    monkeypatch.setattr(runtime_identity.sokurl, "GAME_DIR", game)
    kernel = Mock()
    kernel.CreateMutexW.return_value = 1
    kernel.WaitForSingleObject.return_value = runtime_identity.sokurl.WAIT_OBJECT_0
    monkeypatch.setattr(runtime_identity.sokurl, "kernel32", kernel)
    before = runtime_identity.fingerprints()
    assert set(before["source_hashes"]) == set(sources)
    (tmp_path / "tools/network_runtime/game.py").write_text("# changed\n", encoding="utf-8")
    after = runtime_identity.fingerprints()
    assert before["implementation"] != after["implementation"]
    assert before["game_id"] == after["game_id"]
