"""Compare optimized decks with the original scalar traversal in real paused frames."""
import json
import os
from pathlib import Path

from game_runtime import batch
from game_runtime.privileged import PrivilegedReader
from validation.snapshot_worker import VerifiedObservationReader
import rollout_worker


optimized_deck = PrivilegedReader.deck
verified_decks = 0


def checked_deck(reader, seat):
    global verified_decks
    actual = optimized_deck(reader, seat)
    address = 0x899D10 + seat * 0x20 + 8
    _, table, chunks, counter, count = reader.value(address, '5I')
    if count != 20 or chunks == 0:
        raise RuntimeError('invalid original deck header during verification')
    expected = tuple(reader.value(reader.value(table + (((counter + i) >> 3) % chunks) * 4, 'I')
                                  + ((counter + i) & 7) * 2, 'H') for i in range(20))
    if actual != expected:
        raise AssertionError('optimized deck differs from original scalar traversal')
    verified_decks += 1
    return actual


if __name__ == '__main__':
    output = Path(os.environ['SOKURL_SNAPSHOT_AUDIT'])
    output.mkdir(parents=True, exist_ok=True)
    PrivilegedReader.deck = checked_deck
    batch.ObservationReader = VerifiedObservationReader
    completed = False
    try:
        rollout_worker.main()
        completed = True
    finally:
        (output / 'deck-comparisons.json').write_text(json.dumps({
            'verified_deck_reads': verified_decks,
            'worker_returned_successfully': completed}, indent=2), encoding='utf-8')
