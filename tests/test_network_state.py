"""Check network snapshot identity and distinguish rounds from matches."""
import ctypes
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from network_state import HEADER, MAGIC, MAPPING_SIZE, decode_network_state
from bridge_shared import RawFrameState, calculate_state_hash
from soku_rl.render_state import RENDER_STATE_SIZE


def snapshot(scene, match, frame, round_id, scores):
    raw = RawFrameState()
    raw.frameId, raw.segmentId, raw.sceneId, raw.roundId = frame, match, scene, round_id
    raw.stateHash = calculate_state_hash(raw)
    header = HEADER.pack(MAGIC, 1, MAPPING_SIZE, 2, 1, scene, match, 0, frame, *scores)
    return header + bytes(raw) + bytes(RENDER_STATE_SIZE)


def test_network_mapping_layout():
    assert HEADER.size == 48
    assert ctypes.sizeof(RawFrameState) == 10596
    assert MAPPING_SIZE == 13272


def test_round_result_does_not_end_network_match():
    first = decode_network_state(snapshot(13, 4, 1000, 0, (1, 0)))
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
