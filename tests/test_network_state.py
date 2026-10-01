"""Check network snapshot identity and distinguish rounds from matches."""
import ctypes
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from network_runtime import state as network_state
from network_runtime.state import HEADER, MAGIC, MAPPING_SIZE, decode_network_state
from bridge_shared import RawFrameState, calculate_state_hash
from soku_rl.env.observation.render_state import RENDER_STATE_SIZE


def snapshot(scene, match, frame, round_id, scores):
    raw = RawFrameState()
    raw.frameId, raw.segmentId, raw.sceneId, raw.roundId = frame, match, scene, round_id
    raw.stateHash = calculate_state_hash(raw)
    header = HEADER.pack(MAGIC, 2, MAPPING_SIZE, 2, 1, scene, match, 0, frame, *scores)
    return header + bytes(raw) + bytes(RENDER_STATE_SIZE)


def test_network_mapping_layout():
    assert HEADER.size == 48
    assert ctypes.sizeof(RawFrameState) == 10596
    assert MAPPING_SIZE == 51672


def test_round_result_does_not_end_network_match():
    first = decode_network_state(snapshot(13, 4, 1000, 0, (1, 0)))
    assert first.match_state.phase == "battle"
    assert (first.match_state.match, first.match_state.round, first.match_state.frame) == (4, 0, 1000)
    assert first.match_state.scores == (1, 0)
    second = decode_network_state(snapshot(13, 4, 1100, 1, (1, 0)))
    assert first.in_battle and second.in_battle
    assert first.match == second.match == 4
    assert first.scores == second.scores == (1, 0)
    assert first.raw.roundId != second.raw.roundId
    menu = decode_network_state(snapshot(8, 4, 1100, 1, (2, 0)))
    assert not menu.in_battle


def test_rejects_mixed_frame_and_corrupt_payload():
    data = bytearray(snapshot(13, 1, 42, 0, (0, 0)))
    data[HEADER.size] ^= 1
    with pytest.raises(ValueError, match="identity"):
        decode_network_state(data)
    data = bytearray(snapshot(13, 1, 42, 0, (0, 0)))
    data[HEADER.size + RawFrameState.stateHash.offset] ^= 1
    with pytest.raises(ValueError, match="checksum"):
        decode_network_state(data)


@pytest.mark.parametrize("scene", (2, 8, 9, 10, 11, 13, 14))
def test_menu_status_matches_full_snapshot_without_scanning_battle_objects(scene, monkeypatch):
    data = snapshot(scene, 7, 321, 2, (2, 1))
    expected = decode_network_state(data)
    memory = ctypes.create_string_buffer(data)
    client = network_state.NetworkStateClient.__new__(network_state.NetworkStateClient)
    client.view = ctypes.addressof(memory)

    def forbidden(*args):
        raise AssertionError("menu status scanned the complete battle payload")

    monkeypatch.setattr(network_state.bridge_shared, "calculate_state_hash", forbidden)
    monkeypatch.setattr(network_state.RenderSnapshot, "decode", forbidden)
    actual = client.read_status(.2)
    assert actual.match_state == expected.match_state
    assert actual.in_battle == expected.in_battle
    assert (actual.scene, actual.local_seat) == (expected.scene, expected.local_seat)
    assert ctypes.sizeof(actual.raw) < 512


def test_status_reader_rejects_mixed_frame_and_unfinished_write():
    data = bytearray(snapshot(13, 1, 42, 0, (0, 0)))
    data[HEADER.size] ^= 1
    with pytest.raises(ValueError, match="identity"):
        network_state.decode_network_status(data[:network_state.STATUS_SIZE])
    data = bytearray(snapshot(13, 1, 42, 0, (0, 0)))
    data[12] = 3
    with pytest.raises(ValueError, match="header"):
        network_state.decode_network_status(data[:network_state.STATUS_SIZE])
