"""Keep the complete observation byte-compatible with existing trained policies."""
import hashlib

import numpy as np
import pytest

from soku_rl.env.observation.memory_schema import FIGHTER_NAMES, MAX_BOXES, MAX_OBJECTS, WORLD_NAMES
from soku_rl.env.observation.privileged import PrivilegedObservation, decode_privileged, encode_privileged


def observation(object_count):
    players = []
    for seat in (0, 1):
        fighter = dict.fromkeys(FIGHTER_NAMES, 0)
        boxes = tuple((i+.25, -i-.5, i+1., -i-2.) for i in range(MAX_BOXES))
        fighter.update(x=float(np.float32(-.000125)), y=480.5, dir=1-2*seat, hp=10000,
            char=seat, fflags=0xFFFFFFFF, aflags=0x81234567, address=0xABCDEF01+seat,
            hitarea_n=MAX_BOXES, attackarea_n=MAX_BOXES, hitarea=boxes, attackarea=boxes,
            cards=(-1, 0, 1, 2, 3, 4, 5, 6, 7, 8), skills=tuple(range(16)),
            special=tuple(range(28)), keys=tuple(range(10)), deck=tuple(range(20)), obj_n=object_count)
        fighter["objects"] = tuple(fighter | {"address": i, "x": i+.125} for i in range(object_count))
        players.append(fighter)
    world = dict(zip(WORLD_NAMES, range(len(WORLD_NAMES)), strict=True))
    return PrivilegedObservation(world, tuple(players))


@pytest.mark.parametrize("count", (0, 1, MAX_OBJECTS))
def test_complete_state_keeps_every_value_and_original_tensor_bytes(count):
    original = observation(count)
    encoded = encode_privileged(original)
    restored = decode_privileged(encoded)
    assert restored.world == original.world
    for actual, expected in zip(restored.players, original.players, strict=True):
        for name, value in actual.items():
            if name != "objects":
                assert value == expected[name]
        for obj, source in zip(actual["objects"], expected["objects"], strict=True):
            assert obj == {name: source[name] for name in obj}
    assert hashlib.sha256(encoded.tobytes()).hexdigest() == ORIGINAL_DIGESTS[count]


# Computed from the original dense codec before optimizing unused object slots.
ORIGINAL_DIGESTS = {
    0: "5681fbd29ec8344d26c50a67e4b9dc1c82718bf871165e088e013042049e2f11",
    1: "9654a8908d456ac173b3e7e789719d150127607ec9f90b4e2e6394365e53bb16",
    1024: "11c97fe6dc4187430103d1b2763d78563849176c1e3b2ee03749c1df832f03bb",
}
