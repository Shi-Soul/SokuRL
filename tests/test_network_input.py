import ctypes
from pathlib import Path
import struct
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from network_input import BODY, SIZE, NetworkInputClient, input_payload, result_confirm_payload


def test_request_binds_match_round_and_latency():
    state = SimpleNamespace(in_battle=True, match=7, updates=321, scores=(1, 1),
        raw=SimpleNamespace(roundId=2, p1=SimpleNamespace(hp=10000), p2=SimpleNamespace(hp=10000)))
    assert BODY.unpack(input_payload(state, (1, 0, 1, 0, 0, 0, 0, 0), 3)) == (
        1, 7, 2, 3, 321, 326, 1, 0, 1, 0, 0, 0, 0, 0)
    with pytest.raises(ValueError, match="duration"):
        input_payload(state, (0,) * 8, 0)
    with pytest.raises(ValueError, match="integer"):
        input_payload(state, (True,) + (0,) * 7, 3)
    state.in_battle = False
    with pytest.raises(ValueError, match="active battle"):
        input_payload(state, (0,) * 8, 3)


def test_result_confirmation_cannot_control_an_unfinished_match():
    state = SimpleNamespace(in_battle=True, match=7, updates=400, scores=(1, 0),
        raw=SimpleNamespace(roundId=1, p1=SimpleNamespace(hp=10000), p2=SimpleNamespace(hp=0)))
    with pytest.raises(ValueError, match="completed match"):
        result_confirm_payload(state)
    with pytest.raises(ValueError, match="active battle"):
        input_payload(state, (0,) * 8, 3)
    state.scores = (2, 0)
    assert BODY.unpack(result_confirm_payload(state)) == (
        3, 7, 1, 1, 400, 405, 0, 0, 1, 0, 0, 0, 0, 0)
    with pytest.raises(ValueError, match="active battle"):
        input_payload(state, (0,) * 8, 3)
    state.in_battle = False
    with pytest.raises(ValueError, match="completed match"):
        result_confirm_payload(state)


def test_mailbox_does_not_overwrite_unacknowledged_command():
    memory = ctypes.create_string_buffer(SIZE)
    client = NetworkInputClient.__new__(NetworkInputClient)
    client.view = ctypes.addressof(memory)
    assert client.release() == 1
    assert struct.unpack_from("<I", memory, 24)[0] == 2
    with pytest.raises(RuntimeError, match="previous request"):
        client.release()
    ctypes.c_uint32.from_address(client.view + 16).value = 1
    assert client.release() == 2
