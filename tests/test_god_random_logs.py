"""The unchanged random helper must replay privately without changing its RNG."""
from pathlib import Path

import pytest

LuaRuntime = pytest.importorskip("lupa.lua51").LuaRuntime
from soku_rl.policy.god.api import ScriptAPI
from soku_rl.policy.god.random_logs import install_random_logs


def test_original_helper_keeps_actor_logs_and_random_draws_isolated(tmp_path, monkeypatch):
    source = (Path(__file__).parents[1] /
              "third_party/th123_ai/package/th123ai/script/my_rand.ai").read_bytes()
    monkeypatch.chdir(tmp_path)
    actors = []
    for seed in (17, 29):
        lua = LuaRuntime(encoding=None, unpack_returned_tuples=True)
        files = install_random_logs(lua)
        api = ScriptAPI(lua, seed)
        lua.execute(b"my_dir=1; is_dir_front=true")
        lua.execute(source)
        lua.execute(b"my_rand.set_rand(false)")
        draws = [lua.eval(b"my_rand.random(1, 100)") for _ in range(20)]
        actors.append((lua, files, draws, api))
    assert actors[0][2] != actors[1][2]
    for lua, files, draws, api in actors:
        assert [int(v) for v in files[b"311000851"].splitlines()] == draws
        lua.execute(b"my_rand.set_rand(true)")
        assert [lua.eval(b"my_rand.random(1, 100)") for _ in draws] == draws
        assert lua.eval(b"my_rand.random(1, 100)") == -1
    reference = LuaRuntime(encoding=None, unpack_returned_tuples=True)
    api = ScriptAPI(reference, 17)
    assert [reference.eval(b"math.random(1, 100)") for _ in range(20)] == actors[0][2]
    assert not list(tmp_path.iterdir())
