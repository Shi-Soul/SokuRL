"""The source scheduler must match the actual shipped executable's Lua code."""
from pathlib import Path

import pytest

pytest.importorskip("lupa.lua51")
from lupa.lua51 import LuaRuntime
from god_reference.package import embedded_api, lua_program


def test_scheduler_instructions_match_the_packaged_executable():
    root = Path(__file__).parents[1]
    executable = root / "third_party/th123_ai/package/th123ai/th123_ai.exe"
    source = root / "third_party/th123_ai/source/th123_ai/api.ai"
    if not executable.is_file() or not source.is_file():
        pytest.skip("original package and source are unavailable")
    lua = LuaRuntime(encoding=None)
    compile_code = lua.eval(b"function(s) return string.dump(assert(loadstring(s, '@api.ai'))) end")
    original = lua_program(embedded_api(executable))
    migrated = lua_program(compile_code(source.read_bytes()))
    assert original == migrated
    assert len(original[3]) == 11
