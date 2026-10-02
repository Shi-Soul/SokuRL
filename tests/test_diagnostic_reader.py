"""Effects may fill bridge records without filling compact projectile features."""
from types import SimpleNamespace

import pytest

from game_runtime.diagnostic import DiagnosticReader
from god_reference.memory import game_memory
from soku_rl.env.observation.diagnostic import observe
from test_spirit_observation import recorded_frame


def scene(count, active):
    memory = game_memory(0)
    memory.regions[0x200000] = bytearray(0x100000)
    raw, _ = recorded_frame(-88)
    raw.segmentId = 7
    for seat in (0, 1):
        base = 0x101000 + seat * 0x10000
        sentinel = base + 0x1700
        nodes = 0x200000 + seat * 0x80000
        objects = 0x210000 + seat * 0x80000
        memory.write(base + 0x1658, "3I", 0, sentinel, count)
        memory.write(sentinel, "I", nodes if count else sentinel)
        bridge_objects = []
        for index in range(count):
            address = objects + index * 0x400
            node = nodes + index * 12
            memory.write(node, "3I", node + 12 if index + 1 < count else sentinel, 0, address)
            values = (100. + index, 50. + seat, -2., 3.)
            memory.write(address + 0xEC, "4f", *values)
            memory.write(address + 0x34C, "i", int(index in active))
            memory.write(address + 0x1CB, "B", 1)
            bridge_objects.append(SimpleNamespace(x=values[0], y=values[1],
                speedX=values[2], speedY=values[3], isActive=int(index in active), hitBoxCount=1))
        prefix = f"p{seat + 1}"
        setattr(raw, prefix + "Objects", tuple(bridge_objects[:64]))
        setattr(raw, prefix + "ObjectCount", min(count, 64))
        setattr(raw, prefix + "ObjectOverflow", count > 64)
    bridge = SimpleNamespace(snapshot=lambda: SimpleNamespace(
        run_state_name="PAUSED", game_frame=raw.frameId, latest=raw))
    return memory, raw, bridge


def test_normal_frames_keep_exact_existing_observations_without_memory_reads():
    _, raw, bridge = scene(5, {0, 3})
    assert DiagnosticReader(object()).observe(raw, bridge) == tuple(observe(raw, seat) for seat in (0, 1))


def test_complete_reader_matches_native_fields_order_and_filtering():
    memory, raw, _ = scene(5, {0, 3})
    reader = DiagnosticReader(memory)
    for seat in (0, 1):
        assert reader.projectiles(seat) == observe(raw, 1 - seat).enemy_projectiles


def test_effect_overflow_reads_projectile_after_the_bridge_capacity():
    memory, raw, bridge = scene(66, {0, 65})
    with pytest.raises(RuntimeError, match="overflow"):
        observe(raw, 0)
    observations = DiagnosticReader(memory).observe(raw, bridge)
    for seat, result in enumerate(observations):
        assert [p.x for p in result.enemy_projectiles] == [100., 165.]
        assert [p.y for p in result.enemy_projectiles] == [51. - seat] * 2
        assert result.player.hp == (raw.p1, raw.p2)[seat].hp


def test_real_projectile_capacity_overflow_is_not_truncated():
    memory, raw, bridge = scene(65, set(range(65)))
    with pytest.raises(RuntimeError, match="active projectile observation exceeds"):
        DiagnosticReader(memory).observe(raw, bridge)


def test_corrupt_linked_list_still_fails():
    memory, raw, bridge = scene(66, {65})
    memory.write(0x200000, "I", 0x200000)
    with pytest.raises(RuntimeError, match="invalid diagnostic object list"):
        DiagnosticReader(memory).observe(raw, bridge)


@pytest.mark.parametrize("advanced", [False, True])
def test_live_or_advancing_game_is_rejected(advanced):
    memory, raw, _ = scene(66, {65})
    before = SimpleNamespace(run_state_name="PAUSED", game_frame=raw.frameId, latest=raw)
    after = SimpleNamespace(run_state_name="RUNNING", game_frame=raw.frameId + 1, latest=raw)
    snapshots = iter((before, after) if advanced else (after, after))
    with pytest.raises(RuntimeError, match="advanced|exact paused"):
        DiagnosticReader(memory).observe(raw, SimpleNamespace(snapshot=lambda: next(snapshots)))
