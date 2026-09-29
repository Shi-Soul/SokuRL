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
