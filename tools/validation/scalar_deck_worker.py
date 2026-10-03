"""Performance control retaining the deck traversal used before pointer reuse."""
from game_runtime.privileged import PrivilegedReader
import rollout_worker


def scalar_deck(reader, seat):
    address = 0x899D10 + seat * 0x20 + 8
    _, table, chunks, counter, count = reader.value(address, '5I')
    if count != 20 or chunks == 0:
        raise RuntimeError('the selected effective deck must contain twenty cards')
    return tuple(reader.value(reader.value(table + (((counter + i) >> 3) % chunks) * 4, 'I')
                              + ((counter + i) & 7) * 2, 'H') for i in range(20))


if __name__ == '__main__':
    PrivilegedReader.deck = scalar_deck
    rollout_worker.main()
