"""Verify ordered copies and acknowledgements at the native frame-stream interface."""
import ctypes
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from bridge_shared import BridgeMapping, BridgeUnavailable, FRAME_RING_CAPACITY, RawFrameState, calculate_state_hash
from game_runtime.frames import FRAME_SIZE, drain_frames_into, wait_for_frame_zero


def ring_client(read, count):
    mapping = BridgeMapping()
    mapping.control.ringReadSeq = read
    mapping.control.ringWriteSeq = (read + count) & 0xFFFFFFFF
    for offset in range(min(count, FRAME_RING_CAPACITY)):
        frame = mapping.frames[(read + offset) % FRAME_RING_CAPACITY]
        frame.frameId = offset + 700
        frame.p1.spirit = -88 + offset
    return SimpleNamespace(block=mapping.control, mapping=mapping)


@pytest.mark.parametrize("read", [0, FRAME_RING_CAPACITY - 2, 0xFFFFFFFE])
def test_drain_preserves_order_across_ring_and_sequence_wrap(read):
    client = ring_client(read, 4)
    buffer = (RawFrameState * 4)()
    assert drain_frames_into(client, buffer) == 4
    assert [frame.frameId for frame in buffer] == [700, 701, 702, 703]
    assert [frame.p1.spirit for frame in buffer] == [-88, -87, -86, -85]
    assert client.block.ringReadSeq == client.block.ringWriteSeq
    client.mapping.frames[read % FRAME_RING_CAPACITY].p1.spirit = 999
    assert buffer[0].p1.spirit == -88
    assert drain_frames_into(client, buffer) == 0


def test_undersized_buffer_does_not_acknowledge_or_copy_records():
    client = ring_client(5, 2)
    buffer = (ctypes.c_ubyte * (FRAME_SIZE - 1))()
    before = bytes(buffer)
    with pytest.raises(ValueError, match="buffer"):
        drain_frames_into(client, buffer)
    assert bytes(buffer) == before
    assert client.block.ringReadSeq == 5


def test_invalid_backlog_does_not_acknowledge_records():
    client = ring_client(0, FRAME_RING_CAPACITY + 1)
    with pytest.raises(BridgeUnavailable, match="sequence"):
        drain_frames_into(client, (RawFrameState * FRAME_RING_CAPACITY)())
    assert client.block.ringReadSeq == 0


def initial_client():
    state = RawFrameState()
    state.sceneId = 5
    state.p1.hp = 9000
    state.stateHash = calculate_state_hash(state)
    snapshot = SimpleNamespace(in_gameplay=True, checkpoint_valid=True, game_frame=0,
                               run_state_name="PAUSED", latest=state, result_code=1)
    return Mock(snapshot=Mock(return_value=snapshot), drain_frames=Mock(return_value=[]))


def test_initial_state_is_independent_of_later_native_updates():
    client = initial_client()
    state = wait_for_frame_zero(client, 123, 1.0)
    client.snapshot.return_value.latest.p1.hp = 100
    assert state.p1.hp == 9000
    client.drain_frames.assert_called_once_with()


def test_initial_hash_failure_keeps_unverified_records_unacknowledged():
    client = initial_client()
    client.snapshot.return_value.latest.stateHash ^= 1
    with pytest.raises(RuntimeError, match="hash mismatch"):
        wait_for_frame_zero(client, 123, 1.0)
    client.drain_frames.assert_not_called()


def test_initial_timeout_reports_last_native_state():
    client = initial_client()
    client.snapshot.return_value.game_frame = 8
    with patch("game_runtime.frames.time.monotonic", side_effect=[0.0, 1.0]):
        with pytest.raises(RuntimeError, match="frame=8.*scene=5"):
            wait_for_frame_zero(client, 123, 0.5)
    client.drain_frames.assert_not_called()


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), True])
def test_invalid_timeout_is_rejected_before_accessing_native_state(timeout):
    client = Mock()
    with pytest.raises(ValueError, match="positive and finite"):
        wait_for_frame_zero(client, 123, timeout)
    client.snapshot.assert_not_called()
