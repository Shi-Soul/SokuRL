"""Check reset ownership for image restarts and native state-only resets."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))


@pytest.fixture
def batch(monkeypatch):
    game = SimpleNamespace(shutdown=Mock())
    monkeypatch.setitem(sys.modules, "sokurl", game)
    spec = importlib.util.spec_from_file_location("reset_batch_under_test", TOOLS / "game_batch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.SokuGameBatch(180.0)
    for slot in (0, 1):
        result.processes[slot] = SimpleNamespace(pid=100 + slot, is_running=lambda: True)
        result.clients[slot] = Mock()
        result.clients[slot].snapshot.return_value.latest.segmentId = 3
        result.clients[slot].reset_episode.return_value = 10 + slot
        result.image_clients[slot] = Mock()
        result.frames[slot] = 20
        result.buffers[slot] = object()
    result.active = {0, 1}
    result._observe = Mock(return_value="scene-reset")

    def launch(seeds):
        for slot in seeds:
            result.processes[slot] = SimpleNamespace(pid=200 + slot, is_running=lambda: True)
        return {slot: "new-process" for slot in seeds}

    result._launch_slots = Mock(side_effect=launch)
    return result, game


@pytest.mark.parametrize("mode", ["image", "state", "diagnostic_state"])
def test_reset_only_recreates_selected_image_slots(batch, mode):
    backend, game = batch
    backend.observation_mode = mode
    selected, peer = backend.clients[0], backend.clients[1]
    selected_image, peer_image = backend.image_clients[0], backend.image_clients[1]
    peer_buffer = backend.buffers[1]
    states = backend.reset_slots({0: 1732, 2: 291038774})
    assert backend.active == {0, 1, 2}
    assert backend.processes[1].pid == 101
    assert backend.frames[1] == 20
    assert backend.buffers[1] is peer_buffer
    peer.close.assert_not_called()
    peer.reset_episode.assert_not_called()
    peer_image.close.assert_not_called()
    if mode == "image":
        selected.close.assert_called_once()
        selected_image.close.assert_called_once()
        game.shutdown.assert_called_once_with(5.0, 100)
        selected.reset_episode.assert_not_called()
        backend._launch_slots.assert_called_once_with({0: 1732, 2: 291038774})
        assert states == {0: "new-process", 2: "new-process"}
    else:
        selected.close.assert_not_called()
        selected_image.close.assert_not_called()
        game.shutdown.assert_not_called()
        selected.reset_episode.assert_called_once_with(1732)
        selected.wait_for_reset.assert_called_once_with(10, 4, 1732, 180.0)
        backend._launch_slots.assert_called_once_with({2: 291038774})
        assert states == {0: "scene-reset", 2: "new-process"}


def test_invalid_seed_does_not_close_any_image_slot(batch):
    backend, game = batch
    backend.observation_mode = "image"
    with pytest.raises(ValueError, match="reserved"):
        backend.reset_slots({0: 0xFFFFFFFF})
    game.shutdown.assert_not_called()
    backend.clients[0].close.assert_not_called()
    backend._launch_slots.assert_not_called()
