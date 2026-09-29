"""Compare audited package machine code without launching the legacy program."""
from pathlib import Path
import sys

import pytest

pytest.importorskip("unicorn")
pytest.importorskip("lupa.lua51")
from lupa.lua51 import LuaRuntime
from god_reference.binary import BinaryReference
from god_reference.memory import game_memory
from soku_rl.policy.god.api import ScriptAPI
from test_god_scripts import observation

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from privileged_reader import PrivilegedReader


@pytest.fixture
def reference():
    package = ROOT / "third_party/th123_ai/package/th123ai"
    header = ROOT / "third_party/th123_ai/source/th123_ai/address.h"
    if not package.is_dir() or not header.is_file():
        pytest.skip("the exact external package and source header are required")
    memory = game_memory(1)
    return BinaryReference(package / "th123_ai.exe", header, package / "SWRSAddr.ini", memory)


@pytest.mark.parametrize("x,y,flag", [(400.25, 20.5, 0), (-0.25, 20.5, 1),
                                     (1e-5, 1e-5, 1), (1000.25, 300.75, 1)])
def test_box_coordinates_match_packaged_x87_arithmetic(reference, x, y, flag):
    memory = reference.memory
    memory.write(0x101000 + 0xEC, "2f", x, y)
    memory.write(0x101000 + 0x334, "I", flag)
    original = reference.entity(0x101000)
    migrated = PrivilegedReader(memory).entity(0x101000, None)
    assert original == {name: migrated[name] for name in original}


def test_object_query_returns_the_packaged_eleven_fields(reference):
    memory = reference.memory
    value = observation(1)
    entity = PrivilegedReader(memory).entity(0x103000, None)
    value.players[0].update(objects=(entity,), obj_n=1)
    api = ScriptAPI(LuaRuntime(encoding=None, unpack_returned_tuples=True), 1)
    api.observe(value)
    expected = reference.object_values(0x103000)
    assert len(expected) == 11
    assert api.get_obj_data(0, 0) == expected


@pytest.mark.parametrize("character", (0, 3, 5, 12))
def test_option_query_preserves_all_matching_objects(reference, character):
    value = observation(character)
    objects = tuple({"act": act, "x": 100.5 + i, "y": 20.25 + i, "img": 1,
                     "attackarea_n": 0, "attackarea": (), "frame": 0, "hp": 0}
                    for i, act in enumerate((888, 805, 805, 899, 801, 852, 855)))
    value.players[0].update(objects=objects, obj_n=len(objects))
    api = ScriptAPI(LuaRuntime(encoding=None, unpack_returned_tuples=True), 1)
    api.observe(value)
    for index in range(-1, 9):
        assert api.get_opt_xy(0, index) == reference.option_xy(character, objects, index)


@pytest.mark.parametrize("character", range(20))
@pytest.mark.parametrize("state", (-2, -1, 0, 3, 4, 5, 100))
def test_character_special_values_match_the_packaged_reader(reference, character, state):
    from soku_rl.env.observation.special import special_values
    from soku_rl.env.observation.memory_schema import SPECIAL_FIELDS
    memory = reference.memory
    for _, offset, kind in SPECIAL_FIELDS.values():
        memory.write(0x101000 + offset, kind, state & 0xFFFF if kind == "H" else state)
    reader = PrivilegedReader(memory)
    read = lambda offset, kind: reader.value(0x101000 + offset, kind)
    assert special_values(character, read, ()) == reference.special(0x101000, character, ())


@pytest.mark.parametrize("boxes", (0, 1, 15, 16))
def test_projectile_box_limit_matches_the_packaged_program(reference, boxes):
    value = observation(1)
    objects = ({"act": 1, "x": 450., "y": 0., "img": 0, "frame": 0, "hp": 0,
                "attackarea_n": boxes, "attackarea": ((390, 0, 410, 0),) * boxes},)
    value.players[0]["x"] = 400.
    value.players[1].update(objects=objects, obj_n=1)
    api = ScriptAPI(LuaRuntime(encoding=None, unpack_returned_tuples=True), 1)
    api.observe(value)
    assert (api.globals[b"obj_dis"], api.globals[b"obj_dis2"]) == reference.projectile_distance(400., objects)


@pytest.mark.parametrize("character,act,img", [(4, 0x358, 0x154), (4, 0x358, 0x1B3),
    (4, 0x358, 0x1B4), (4, 1, 0x154),
    (4, 0x358, 1), (4, 1, 1), (10, 0x35A, 0xF0), (10, 1, 0xF0), (10, 0x35A, 1)])
@pytest.mark.parametrize("frame,hp", [(100, 1), (600, 0), (601, 65535), (900, 32768)])
def test_object_based_special_values_match_the_packaged_reader(reference, character, act, img, frame, hp):
    from soku_rl.env.observation.special import special_values
    reader = PrivilegedReader(reference.memory)
    read = lambda offset, kind: reader.value(0x101000 + offset, kind)
    objects = ({"act": act, "img": img, "frame": frame, "hp": hp, "x": 400., "y": 0.,
                "attackarea_n": 0, "attackarea": ()},)
    assert special_values(character, read, objects) == reference.special(0x101000, character, objects)
