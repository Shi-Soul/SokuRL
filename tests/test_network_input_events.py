"""Input acknowledgments and terminal outcomes must remain separate records."""
import ctypes
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from network_history import CAPACITY, HEADER
from network_input_events import EVENT, NetworkInputEventsClient, decode_input_event


def test_ordered_acceptance_injection_expiry_and_round_cancellation():
    events = [
        (1, 1, 1, 3, 0, 3, 100, 105, 100),
        (2, 1, 1, 3, 0, 3, 103, 108, 103),
        (1, 8, 1, 3, 0, 3, 100, 105, 105),
        (2, 9, 1, 3, 0, 3, 103, 108, 109),
        (3, 1, 1, 3, 0, 3, 106, 111, 109),
        (3, 10, 1, 3, 0, 3, 106, 111, 110),
    ]
    size = HEADER.size+CAPACITY*EVENT.size
    memory = ctypes.create_string_buffer(size)
    HEADER.pack_into(memory, 0, NetworkInputEventsClient.magic, 1, size, CAPACITY, 2, 1, 258)
    for index, event in enumerate(events, 252):
        EVENT.pack_into(memory, HEADER.size+index % CAPACITY*EVENT.size, *event)
    client = NetworkInputEventsClient.__new__(NetworkInputEventsClient)
    client.view = ctypes.addressof(memory)
    cursor, actual = client.read_after(252, 1.)
    assert cursor == 258
    assert [x["result"] for x in actual] == ["accepted", "accepted", "injected", "expired", "accepted", "cancelled"]
    assert [(x["request"], x["at"]) for x in actual if x["result"] == "injected"] == [(1, 105)]
    assert client.read_after(cursor, 1.) == (cursor, ())
    with pytest.raises(BufferError, match="lost"):
        client.read_after(0, 1.)
    ctypes.c_uint32.from_address(client.view+20).value = 0
    with pytest.raises(EOFError, match="closed"):
        client.read_after(cursor, 1.)


def test_invalid_event_is_rejected():
    with pytest.raises(ValueError, match="length"):
        decode_input_event(b"")
    for request, kind in ((0, 1), (1, 0), (1, 12)):
        with pytest.raises(ValueError, match="identity"):
            decode_input_event(EVENT.pack(request, kind, 1, 1, 0, 3, 1, 6, 1))


def test_superseded_intent_is_distinct_from_injection():
    event = decode_input_event(EVENT.pack(1, 11, 1, 1, 0, 3, 208, 213, 216))
    assert event["result"] == "superseded" and event["at"] == 216
