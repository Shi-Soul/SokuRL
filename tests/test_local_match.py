"""Local match adapters use paused scores and leave human seat control untouched."""
import struct
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

if sys.platform != "win32":
    pytest.skip("Windows game adapter", allow_module_level=True)

from bridge_shared import RawFrameState
from game_runtime import local_match
from test_replay_rollout import config


@pytest.fixture
def game(monkeypatch):
    raw = RawFrameState()
    raw.frameId, raw.segmentId, raw.roundId = 9, 4, 1
    raw.p1.hp, raw.p2.hp = 10000, 0
    snapshot = SimpleNamespace(latest=raw, game_frame=9, in_gameplay=True, run_state_name="PAUSED")
    process = Mock(pid=317)
    process.poll.return_value = None
    monkeypatch.setattr(local_match.sokurl, "_launch_vs_group_from_title", Mock(return_value=[process]))
    monkeypatch.setattr(local_match.sokurl, "shutdown", Mock())
    client = Mock()
    client.snapshot.return_value = snapshot
    monkeypatch.setattr(local_match, "BridgeClient", lambda pid: client)
    monkeypatch.setattr(local_match, "wait_for_frame_zero", Mock())
    reader = Mock()
    reader.read.return_value = SimpleNamespace(frame=9, observations=("left", "right"))
    monkeypatch.setattr(local_match, "ObservationReader", lambda *args: reader)
    memory = Mock()
    data = {0x8985E4: struct.pack("<I", 100), 112: struct.pack("<2I", 1000, 2000),
            1000 + 0x573: b"\x01", 2000 + 0x573: b"\x00"}
    memory.read.side_effect = lambda address, size: data[address]
    monkeypatch.setattr(local_match, "ProcessMemory", lambda pid: memory)
    wait = Mock()
    monkeypatch.setattr(local_match, "wait_for_steps", wait)
    result = local_match.LocalMatch(config(), 1732, 2.)
    yield result, client, wait, reader, memory
    result.close()
    client.close.assert_called_once_with()
    reader.close.assert_called_once_with()
    memory.close.assert_called_once_with()
    local_match.sokurl.shutdown.assert_called_once_with(5., 317)


def test_knockout_uses_original_scores_and_does_not_end_match(game):
    result, client, wait, _, _ = game
    frame = result.read()
    assert (frame.match.match, frame.match.round, frame.match.frame) == (5, 1, 9)
    assert frame.match.scores == (1, 0)
    assert frame.match.hp == (10000, 0)
    assert frame.observations == ("left", "right")
    keys = {1: (-1, 0, 1, 0, 0, 0, 0, 0)}
    result.step(keys)
    client.step_controlled.assert_called_once_with(keys)
    wait.assert_called_once_with([client], [client.step_controlled.return_value], [10], .05)
    client.drain_frames.assert_called_once_with()


def test_score_read_refuses_running_game_before_reading_memory(game):
    result, client, _, reader, memory = game
    client.snapshot.return_value.run_state_name = "RUNNING"
    with pytest.raises(RuntimeError, match="paused"):
        result.read()
    reader.read.assert_not_called()
    memory.read.assert_not_called()


def test_window_close_is_detected_before_submitting_inputs(game):
    result, client, _, _, _ = game
    result.process.poll.return_value = 0
    with pytest.raises(EOFError, match="closed"):
        result.step({0: (0,) * 8})
    client.step_controlled.assert_not_called()


def test_window_close_during_step_does_not_resubmit_input(game):
    result, client, wait, _, _ = game

    def closing(*args):
        result.process.poll.return_value = 0
        raise TimeoutError("observation interval expired")

    wait.side_effect = closing
    with pytest.raises(EOFError, match="closed"):
        result.step({0: (0,) * 8})
    client.step_controlled.assert_called_once_with({0: (0,) * 8})
    client.drain_frames.assert_not_called()


def test_crashed_game_is_not_reported_as_a_normal_window_close(game):
    result, client, _, _, _ = game
    result.process.poll.return_value = 7
    with pytest.raises(RuntimeError, match="exit code 7"):
        result.step({0: (0,) * 8})
    client.step_controlled.assert_not_called()
