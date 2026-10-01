"""A failed network launch restores configuration and closes only owned resources."""
from dataclasses import asdict
import sys
from unittest.mock import Mock

import pytest

if sys.platform != "win32":
    pytest.skip("Windows process adapter", allow_module_level=True)

from game_runtime import startup
from network_runtime import game
from test_replay_rollout import config


@pytest.mark.parametrize("mutex_status", (0, 0x80))
@pytest.mark.parametrize("render", (False, True))
def test_mapping_failure_restores_launch_configuration_and_closes_game(tmp_path, monkeypatch, mutex_status, render):
    path = tmp_path / "SkipIntro.ini"
    original = b"[SkipIntro]\r\nscene_id = 3\r\n"
    path.write_bytes(original)
    monkeypatch.setattr(game.sokurl, "SKIPINTRO_INI", path)
    kernel = Mock()
    kernel.CreateMutexW.return_value = 17
    kernel.WaitForSingleObject.return_value = mutex_status
    monkeypatch.setattr(startup, "kernel32", kernel)
    process = Mock(pid=29)
    process.poll.return_value = None

    def launch(*args, **kwargs):
        assert path.read_bytes() == original.replace(b"3", b"2")
        assert kwargs["env"]["SOKURL_HEADLESS_RENDER"] == ("0" if render else "1")
        assert kwargs["env"]["SOKURL_UNLIMITED_PACING"] == "0"
        return process

    monkeypatch.setattr(game.psutil, "Popen", launch)
    monkeypatch.setattr(game.sokurl, "_read_process_values", lambda pid: (2,))
    close = Mock()
    monkeypatch.setattr(game.sokurl, "_post_close", close)
    state = Mock()
    monkeypatch.setattr(game, "NetworkStateClient", lambda pid: state)
    monkeypatch.setattr(game, "NetworkHistoryClient", Mock(side_effect=OSError("mapping unavailable")))
    with pytest.raises(OSError, match="mapping unavailable"):
        game.NetworkGame({"role": "host", "address": "127.0.0.1", "port": 10800,
                          "automate_menu": False}, asdict(config().visibility), 2., render, False)
    assert path.read_bytes() == original
    kernel.ReleaseMutex.assert_called_once_with(17)
    kernel.CloseHandle.assert_called_once_with(17)
    close.assert_called_once_with(29)
    process.wait.assert_called_once_with(timeout=10)
    state.close.assert_called_once_with()
