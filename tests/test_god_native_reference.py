"""Compare migration behavior with functions compiled from the upstream C++ files."""
import ctypes as C
from pathlib import Path
import random
import shutil
import subprocess
import sys

import pytest

pytest.importorskip("lupa.lua51")
from lupa.lua51 import LuaRuntime
from god_reference.build import generate
from god_reference.memory import game_memory
from god_reference.reader import Reference
from god_reference.compare import players_equal
from soku_rl.policy.god.api import ScriptAPI
from test_god_scripts import observation

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from privileged_reader import PrivilegedReader


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    upstream = ROOT / "third_party/th123_ai/source/th123_ai"
    if not upstream.is_dir():
        pytest.skip("original th123_ai source is required for differential tests")
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("a host C++ compiler is required for the standalone reference")
    output = tmp_path_factory.mktemp("god-reference")
    generate(upstream, output)
    library = output / ("god_reference.dll" if sys.platform == "win32" else "libgod_reference.so")
    platform_flags = ["-static"] if sys.platform == "win32" else ["-fPIC"]
    subprocess.run([compiler, "-std=c++17", "-O2", "-shared", *platform_flags, "-static-libgcc",
                    "-static-libstdc++", str(output / "reference.cpp"), "-o", str(library)],
                   check=True, capture_output=True)
    return Reference(library)


def test_all_key_operations_match_upstream_event_queue(reference):
    api = ScriptAPI(LuaRuntime(encoding=None, unpack_returned_tuples=True), 1)
    native = reference.dll
    native.reference_keys(5, 0)
    sequence = [(operation, key) for operation in (0, 1) for key in range(14)]
    rng = random.Random(41)
    sequence += [(rng.randrange(3), rng.randrange(14)) for _ in range(400)]
    for index, (operation, key) in enumerate(sequence):
        if index % 19 == 0:
            delay = index % 7
            api.set_delay(b"key_delay", delay)
            native.reference_keys(3, delay)
        native.reference_keys(operation, key)
        (api.key_on(key) if operation == 0 else api.key_off(key)
         if operation == 1 else api.key_reset())
        assert api.get_key_map() == tuple(native.reference_key(k, 1) for k in range(10))
        api.inputs(); native.reference_keys(4, 0)
        assert api.applied == [native.reference_key(k, 0) for k in range(10)]


def test_delayed_enemy_globals_match_upstream_buffer_updates(reference):
    api = ScriptAPI(LuaRuntime(encoding=None, unpack_returned_tuples=True), 1)
    for frame in range(36):
        delay = (0, 1, 3, 2, 0, 3)[frame // 6]
        api.globals[b"data_delay"] = delay
        value = observation(1)
        value.players[1].update(act=300+frame, act_block=10+frame, frame=frame+20)
        api.observe(value)
        reference.dll.reference_delayed_action(delay, 300+frame, 10+frame, frame+20)
        for name in ("enemy_act", "enemy_act_block", "enemy_frame"):
            assert api.globals[name.encode()] == reference.value(name), (frame, name)


def test_projectile_distance_preserves_upstream_traversal_order(reference):
    api = ScriptAPI(LuaRuntime(encoding=None, unpack_returned_tuples=True), 1)
    value = observation(1)
    value.players[0]["x"] = 400.
    for boxes in (((400, 450), (390, 410)), ((390, 410), (400, 450))):
        value.players[1]["objects"] = tuple({"x": 450., "attackarea": ((l, 0, r, 0),)} for l, r in boxes)
        reference.dll.reference_projectiles(400., 2, (C.c_float * 2)(450., 450.),
                                              (C.c_int * 4)(*(n for b in boxes for n in b)))
        api.observe(value)
        for name in ("obj_dis", "obj_dis2"):
            assert api.globals[name.encode()] == reference.value(name)


def test_distance_thresholds_use_the_same_float_precision(reference):
    api = ScriptAPI(LuaRuntime(encoding=None, unpack_returned_tuples=True), 1)
    for x, y, ex, ey in ((1e-5, 0, 1000, 0), (100.0001, 0, 1100, 30),
                         (400.5, 200.2, 600.5, 30.6)):
        x,y,ex,ey = (C.c_float(v).value for v in (x,y,ex,ey))
        value = observation(1)
        value.players[0].update(x=x,y=y)
        value.players[1].update(x=ex,y=ey)
        api.observe(value)
        reference.dll.reference_positions(x,y,ex,ey)
        for name in ("dis", "dis_y", "dis2"):
            assert api.globals[name.encode()] == reference.value(name), name


@pytest.mark.parametrize("character", range(20))
def test_all_character_fields_and_objects_match_original_reader(reference, character):
    memory = game_memory(character)
    reference.initialize(memory)
    reader = PrivilegedReader(memory)
    for weather in (0, 2, 11, 0):
        reference.dll.reference_reload(0x100000, 1, weather)
        players = tuple(reader.fighter(0x101000 + seat * 0x10000, seat, weather) for seat in (0, 1))
        reader.previous = players
        players_equal(reference, players)


def test_float_filter_keeps_the_same_persistent_fighter_value(reference):
    memory = game_memory(1)
    reference.initialize(memory)
    reader = PrivilegedReader(memory)
    for value in (1.25, -0., 1e-8, -1e-8, 0., -2.5, 2e-6):
        memory.write(0x101000 + 0xF4, "f", value)
        reference.dll.reference_reload(0x100000, 1, 0)
        own = reader.fighter(0x101000, 0, 0)
        reader.previous[0] = own
        assert own["xspeed"] == reference.value("my_xspeed")


@pytest.mark.skipif(sys.platform != "win32", reason="the original Windows CRT is required")
def test_random_sequence_matches_the_original_windows_crt():
    crt = C.CDLL("msvcrt")
    crt.srand.argtypes, crt.srand.restype = [C.c_uint], None
    crt.rand.argtypes, crt.rand.restype = [], C.c_int
    lua = LuaRuntime(encoding=None, unpack_returned_tuples=True)
    api = ScriptAPI(lua, 1)
    for seed in (1, 37, 2**31, 2**32-1):
        crt.srand(seed)
        api.randomseed(seed)
        for _ in range(1000):
            assert api.random() == (crt.rand() % 32767) / 32767.
