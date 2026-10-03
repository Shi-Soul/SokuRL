"""An offline consumer must keep every native frame and the original reader state."""
import ctypes as C
from types import SimpleNamespace

import pytest

from game_runtime.offline_snapshot import OfflineSnapshotReader, offline_snapshot_requested
from game_runtime.privileged import PrivilegedReader
from game_runtime.snapshot_memory import SnapshotMemory
from god_reference.memory import game_memory
from test_privileged_snapshot import capture, snapshot


class Raw(C.Structure):
    _fields_ = [('frameId', C.c_uint64), ('segmentId', C.c_uint32)]


class History:
    def __init__(self):
        self.frames = []
        self.closed = False

    def read_after(self, cursor):
        return len(self.frames), tuple(self.frames[cursor:])

    def close(self):
        self.closed = True


def client(raw):
    return SimpleNamespace(snapshot=lambda: SimpleNamespace(
        run_state_name='PAUSED', game_frame=raw.frameId, latest=raw))


def test_native_frames_keep_full_observations_and_reset_persistent_fields(capture):
    memory = game_memory(1)
    direct = PrivilegedReader(memory)
    history = History()
    reader = OfflineSnapshotReader(history)
    for segment, frame in ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1)):
        raw = Raw(frame, segment)
        copied = snapshot(capture, memory)
        assert copied.error == 0
        regions = [(r.address, r.size, r.offset) for r in copied.index[:copied.regions]]
        history.frames.append(SimpleNamespace(raw=raw,
            memory=SnapshotMemory(regions, bytes(copied.data[:copied.used]))))
        assert reader.observe(raw, client(raw)) == direct.observe(raw, client(raw))
        assert reader.cursor == len(history.frames)
        # The original reader retains a previous float on this invalid tiny value.
        memory.write(0x101000 + 0xEC, 'f', 1e-30 if frame == 0 else 500.5)
    reader.close()
    assert history.closed


@pytest.mark.parametrize('count', [0, 2, 17])
def test_missing_or_multiple_frames_cannot_be_dropped_or_replayed(count):
    raw = Raw(3, 7)
    history = History()
    history.frames = [SimpleNamespace(raw=raw)] * count
    reader = OfflineSnapshotReader(history)
    with pytest.raises(RuntimeError, match='exactly one'):
        reader.observe(raw, client(raw))
    assert reader.cursor == 0


@pytest.mark.parametrize('wrong', [Raw(2, 7), Raw(3, 8)])
def test_frame_and_segment_must_match_the_requested_native_state(wrong):
    raw = Raw(3, 7)
    history = History()
    history.frames = [SimpleNamespace(raw=wrong)]
    reader = OfflineSnapshotReader(history)
    with pytest.raises(RuntimeError, match='differs'):
        reader.observe(raw, client(raw))
    assert reader.cursor == 0


def test_advancing_game_is_rejected_before_reading_history():
    raw = Raw(3, 7)
    reader = OfflineSnapshotReader(History())
    bridge = SimpleNamespace(snapshot=lambda: SimpleNamespace(run_state_name='RUNNING', game_frame=3))
    with pytest.raises(RuntimeError, match='paused'):
        reader.observe(raw, bridge)


@pytest.mark.parametrize('value', ['', '0', 'true', '2'])
def test_opt_in_rejects_ambiguous_values(monkeypatch, value):
    monkeypatch.setenv('SOKURL_OFFLINE_SNAPSHOT', value)
    with pytest.raises(ValueError, match='exactly 1'):
        offline_snapshot_requested()


def test_opt_in_is_explicit_and_cannot_take_over_realtime_input(monkeypatch):
    for key in ('SOKURL_OFFLINE_SNAPSHOT', 'SOKURL_REALTIME_SEAT', 'SOKURL_NETWORK_ROLE'):
        monkeypatch.delenv(key, raising=False)
    assert not offline_snapshot_requested()
    monkeypatch.setenv('SOKURL_OFFLINE_SNAPSHOT', '1')
    assert offline_snapshot_requested()
    for key in ('SOKURL_REALTIME_SEAT', 'SOKURL_NETWORK_ROLE'):
        monkeypatch.setenv(key, '0')
        with pytest.raises(ValueError, match='realtime or network'):
            offline_snapshot_requested()
        monkeypatch.delenv(key)


def test_snapshot_mode_rejects_other_observation_contracts(monkeypatch):
    from game_runtime.observation import ObservationReader
    monkeypatch.setenv('SOKURL_OFFLINE_SNAPSHOT', '1')
    with pytest.raises(ValueError, match='complete privileged'):
        ObservationReader(123, 'state', object())
