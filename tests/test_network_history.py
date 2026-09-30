"""Verify ordered network records, ring wrap, loss reporting and producer close."""
import ctypes
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from network_runtime.history import CAPACITY, HEADER, MAGIC, SIZE, NetworkHistoryClient, retained_indices
from network_runtime.state import MAPPING_SIZE
from test_network_state import snapshot


def test_history_wrap_and_overrun():
    assert retained_indices(258, 254) == (254, 255, 0, 1)
    assert retained_indices(256, 0) == tuple(range(CAPACITY))
    assert retained_indices(256, 256) == ()
    with pytest.raises(BufferError, match="lost 1 entries"):
        retained_indices(257, 0)
    for cursor in (-1, 10, True):
        with pytest.raises(ValueError, match="cursor"):
            retained_indices(3, cursor)


def test_reads_round_and_match_records_in_order():
    memory = ctypes.create_string_buffer(SIZE)
    HEADER.pack_into(memory, 0, MAGIC, 2, SIZE, CAPACITY, 4, 1, 258)
    states = [snapshot(13, 1, 1000, 0, (1, 0)), snapshot(13, 1, 1100, 1, (1, 0)),
              snapshot(8, 1, 1200, 1, (2, 0)), snapshot(13, 2, 1, 0, (0, 0))]
    for index, state in zip((254, 255, 0, 1), states):
        ctypes.memmove(ctypes.addressof(memory)+HEADER.size+index*MAPPING_SIZE, state, MAPPING_SIZE)
    client = NetworkHistoryClient.__new__(NetworkHistoryClient)
    client.view = ctypes.addressof(memory)
    cursor, records = client.read_after(254, 1)
    assert cursor == 258
    assert [(r.match, r.raw.roundId, r.scene) for r in records] == [
        (1, 0, 13), (1, 1, 13), (1, 1, 8), (2, 0, 13)]
    assert client.read_after(cursor, 1) == (cursor, ())
    with pytest.raises(BufferError, match="lost 2"):
        client.read_after(0, 1)
    ctypes.c_uint32.from_address(client.view+20).value = 0
    with pytest.raises(EOFError, match="closed"):
        client.read_after(cursor, 1)
