"""Effective deck traversal retains exact reads across chunk and frame boundaries."""
import struct

import pytest

from game_runtime.privileged import PrivilegedReader
from game_runtime.snapshot_memory import SnapshotMemory


def deck_memory(seat, chunks, counter, cards):
    table = 0x100000
    regions, data = [], bytearray()
    fields = [(0x899D10 + seat * 0x20 + 8, struct.pack('<5I', 0, table, chunks, counter, 20))]
    for index, values in enumerate(cards):
        address = 0x200000 + index * 0x1000
        fields.append((table + index * 4, struct.pack('<I', address)))
        # Match native capture's individual two-byte ranges; no over-reading.
        fields.extend((address + offset * 2, struct.pack('<H', value)) for offset, value in enumerate(values))
    for address, block in fields:
        regions.append((address, len(block), len(data)))
        data.extend(block)
    return SnapshotMemory(regions, bytes(data))


@pytest.mark.parametrize('seat', [0, 1])
@pytest.mark.parametrize('chunks', [1, 2, 3, 4, 8])
@pytest.mark.parametrize('counter', [0, 1, 7, 8, 15, 21, 63, 0xFFFFFFFE])
def test_deck_chunk_ring_order_and_unsigned_values(seat, chunks, counter):
    cards = [tuple((index * 7919 + offset * 9973) % 65536 for offset in range(8))
             for index in range(chunks)]
    memory = deck_memory(seat, chunks, counter, cards)
    actual = PrivilegedReader(memory).deck(seat)
    # Build the logical ring independently of the reader's pointer traversal.
    ring = [value for chunk in cards for value in chunk]
    expected = tuple((ring * (20 // len(ring) + 2))[counter % len(ring):counter % len(ring) + 20])
    assert actual == expected


def test_deck_refreshes_pointers_and_values_in_next_frame():
    first = [tuple(range(8)), tuple(range(8, 16)), tuple(range(16, 24))]
    second = [tuple(range(65528, 65536)), (0,) * 8, (65535,) * 8]
    reader = PrivilegedReader(deck_memory(0, 3, 7, first))
    before = reader.deck(0)
    reader.memory = deck_memory(0, 3, 15, second)
    # Swap pointer slots as well as card values; no per-reader cache survives.
    blocks = dict(reader.memory.blocks)
    blocks[0x100000], blocks[0x100008] = blocks[0x100008], blocks[0x100000]
    data, regions = bytearray(), []
    for address, block in blocks.items():
        regions.append((address, len(block), len(data)))
        data.extend(block)
    reader.memory = SnapshotMemory(regions, bytes(data))
    ring = [value for chunk in (second[2], second[1], second[0]) for value in chunk]
    assert reader.deck(0) == tuple((ring * 2)[15:35])
    assert before == tuple((list(range(24)) * 2)[7:27])


def test_deck_does_not_hide_an_uncaptured_card():
    memory = deck_memory(1, 3, 0, [tuple(range(8))] * 3)
    del memory.blocks[0x202006]
    memory.addresses = sorted(memory.blocks)
    with pytest.raises(ValueError, match='uncaptured game memory'):
        PrivilegedReader(memory).deck(1)


@pytest.mark.parametrize('chunks,count', [(0, 20), (3, 19), (3, 21)])
def test_invalid_deck_header_still_fails(chunks, count):
    data = struct.pack('<5I', 0, 0x100000, chunks, 0, count)
    memory = SnapshotMemory([(0x899D18, len(data), 0)], data)
    with pytest.raises(RuntimeError, match='must contain twenty cards'):
        PrivilegedReader(memory).deck(0)
