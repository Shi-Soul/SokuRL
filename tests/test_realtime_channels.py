"""Verify native transport layouts, stable frame copies and nonblocking writes."""
import ctypes as C
import mmap
import struct
import sys
import threading
from types import SimpleNamespace

import pytest

if sys.platform != "win32":
    pytest.skip("Windows shared-memory transport", allow_module_level=True)

import bridge_shared
from play_runtime import channels
from soku_rl.play.match import MatchState


@pytest.fixture
def history():
    mapping = mmap.mmap(-1, channels.HISTORY_SIZE)
    address = C.addressof(C.c_char.from_buffer(mapping))
    reader = object.__new__(channels.RealtimeHistory)
    reader.view, reader.frequency = address, 1000000
    C.memmove(address, channels.HISTORY_HEADER.pack(
        0x48524B53, 1, channels.HISTORY_SIZE, channels.CAPACITY, 1000000, 0, 1), 32)
    yield reader
    reader.view = 0
    mapping.close()


def publish(history, sequence, frame, guard):
    address = history.view + 32 + ((sequence-1) % channels.CAPACITY)*channels.FRAME_SIZE
    prefix = channels.FRAME_HEADER.pack(guard, 3, frame, 0, 1, 20)
    raw = bridge_shared.RawFrameState()
    raw.segmentId, raw.frameId, raw.roundId = 3, frame, 1
    raw.p1.hp = raw.p2.hp = 10000
    render = bytearray(channels.RENDER_STATE_SIZE)
    struct.pack_into("<3f", render, 0, 0, 0, 1)
    content = prefix + bytes(raw) + render + struct.pack("<3I", 1, 4, 0)
    C.memmove(address, content, len(content))
    C.memmove(address + channels.MEMORY_OFFSET + 12, struct.pack("<3I", 100, 4, 0), 12)
    C.memmove(address + channels.DATA_OFFSET, b"abcd", 4)
    C.c_uint32.from_address(history.view + 24).value = sequence


def test_history_copies_only_complete_frames_and_keeps_immutable_bytes(history):
    publish(history, 1, 0, 2)
    cursor, frames = history.read_after(0)
    assert cursor == 1 and len(frames) == 1
    assert frames[0].match == MatchState(3, 1, 0, (0, 1), (10000, 10000), "battle")
    assert frames[0].memory.read(100, 4) == b"abcd"
    assert frames[0].capture_seconds == .00002
    publish(history, 2, 1, 1)
    assert history.read_after(cursor) == (1, ())
    publish(history, 2, 1, 2)
    assert history.read_after(cursor)[0] == 2
    C.memmove(history.view + 32 + channels.DATA_OFFSET, b"xxxx", 4)
    assert frames[0].memory.read(100, 4) == b"abcd"


def test_history_reports_overrun_and_native_capture_errors(history):
    publish(history, 17, 16, 4)
    with pytest.raises(BufferError, match="overwritten"):
        history.read_after(0)
    offset = history.view + 32 + channels.MEMORY_OFFSET + 8
    C.c_uint32.from_address(offset).value = 3
    with pytest.raises(RuntimeError, match="capture failed"):
        history.read_after(16)


def test_history_cursor_wraps_without_losing_order(history):
    publish(history, 0, 200, 4)
    cursor, frames = history.read_after(0xFFFFFFFF)
    assert cursor == 0 and frames[0].match.frame == 200


def test_snapshot_transport_preserves_offline_segment_zero(history):
    publish(history, 1, 0, 2)
    address = history.view + channels.HISTORY_HEADER.size
    C.c_uint32.from_address(address + 4).value = 0
    raw = bridge_shared.RawFrameState.from_address(address + channels.FRAME_HEADER.size)
    raw.segmentId = 0
    cursor, frames = channels.SnapshotHistory.read_after(history, 0)
    assert cursor == 1 and frames[0].raw.segmentId == frames[0].raw.frameId == 0
    assert frames[0].scores == (0, 1) and frames[0].memory.read(100, 4) == b'abcd'
    with pytest.raises(ValueError, match='positive match'):
        history.read_after(0)


def test_live_history_recovers_overrun_at_latest_complete_frame(history):
    publish(history, 21, 20, 4)
    cursor, frames, dropped = history.read_latest(0)
    assert (cursor, dropped) == (21, 20)
    assert [frame.match.frame for frame in frames] == [20]
    publish(history, 22, 21, 5)
    assert history.read_latest(cursor) == (cursor, (), 0)
    publish(history, 22, 21, 6)
    cursor, frames, dropped = history.read_latest(cursor)
    assert (cursor, dropped, frames[0].match.frame) == (22, 0, 21)


def test_live_history_retries_copy_races_without_advancing_cursor(history, monkeypatch):
    publish(history, 1, 0, 2)

    def overwritten(cursor):
        raise BufferError("AI observation was overwritten during copying")

    monkeypatch.setattr(history, "read_after", overwritten)
    assert history.read_latest(0) == (0, (), 0)


def test_late_live_commands_are_scheduled_after_the_current_game_frame():
    buffer = C.create_string_buffer(channels.INPUT_SIZE)
    client = object.__new__(channels.RealtimeInput)
    client.view, client.seat, client.sequence = C.addressof(buffer), 1, 0
    client.writer = threading.get_ident()
    channels.STATUS.pack_into(buffer, 92, 2, 0, 0, 7, 2, 80, 0, 0, 0, *([0] * 8))
    state = MatchState(7, 2, 30, (0, 0), (10000, 10000), "battle")
    keys = (-1, 0, 1, 0, 0, 0, 0, 0)
    reply = client.submit_current(state, keys, 5, 3)
    assert reply == {"submitted": True, "request": 1, "target": 81, "expires": 84}
    assert channels.COMMAND.unpack_from(buffer, 20) == (1, 7, 2, 1, 30, 81, 84, *keys)
    changed_round = MatchState(7, 3, 81, (1, 0), (10000, 10000), "battle")
    assert not client.submit_current(changed_round, keys, 5, 3)["submitted"]


def test_temporary_snapshot_timeout_keeps_the_play_session_open():
    from play_runtime.session import RealtimeSession
    session = object.__new__(RealtimeSession)

    def busy():
        raise TimeoutError("snapshot writer is busy")

    session._poll = busy
    batch = session.poll()
    assert not batch["closed"] and not batch["records"]
    assert batch["events"][0]["kind"] == "observation_wait"


def test_input_has_native_offsets_and_never_waits_for_acknowledgement():
    assert channels.COMMAND.size == 72 and channels.STATUS.size == 76
    buffer = C.create_string_buffer(channels.INPUT_SIZE)
    client = object.__new__(channels.RealtimeInput)
    client.view, client.seat, client.sequence = C.addressof(buffer), 1, 0

    client.writer = threading.get_ident()
    state = MatchState(7, 2, 30, (0, 0), (10000, 10000), "battle")
    keys = (-1, 0, 1, 0, 0, 0, 0, 0)
    assert client.submit(state, keys, 5, 8)
    assert struct.unpack_from("<I", buffer, 16)[0] == 2
    assert channels.COMMAND.unpack_from(buffer, 20) == (1, 7, 2, 1, 30, 35, 43, *keys)
    before = bytes(buffer)
    for _ in range(10000):
        assert not client.submit(state, keys, 5, 8)
    assert bytes(buffer) == before
    struct.pack_into("<3I", buffer, 92, 2, 1, 1)
    assert client.status()["result"] == "accepted"
    assert client.submit(state, keys, 5, 8)
    struct.pack_into("<I", buffer, 92, 3)
    assert client.status() is None


def test_live_memory_cannot_enter_snapshot_decode():
    from game_runtime.privileged import PrivilegedReader
    with pytest.raises(TypeError, match="immutable"):
        PrivilegedReader(object()).observe_snapshot(object())
