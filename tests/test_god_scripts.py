"""Check original source loading, exact observation values and per-actor state."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("lupa.lua51")

from soku_rl.env import EpisodeConfig
from soku_rl.env.match import LEGACY_MATCH
from soku_rl.env.observation.memory_schema import FIGHTER_NAMES, WORLD_NAMES
from soku_rl.env.observation.privileged import PrivilegedObservation, encode_privileged, decode_privileged
from soku_rl.policy.god.package import ScriptPackage
from soku_rl.policy.god.runtime import GodPolicy
from test_env_timing import VISIBILITY

ROOT = Path(__file__).parents[1]
SCRIPTS = ROOT / "third_party/th123_ai/package/th123ai/script"
NAMES = sorted(p.name for p in SCRIPTS.glob("*.ai") if p.name[:2].isdigit() and "_main" in p.name)


def observation(character):
    def fighter(character, x, direction):
        value = dict.fromkeys(FIGHTER_NAMES, 0)
        value.update(char=character, x=x, y=0., dir=direction, hp=10000, rei=1000, rmax=1000,
                     rate=1., attack_power=1., defense_power=1., speed_power=1., card=-1, fflags=1)
        value.update(hitarea=(), attackarea=(), cards=(-1,) * 10, skills=(0,) * 4 + (-1,) * 12,
                     special=(0,) * 28, keys=(0,) * 10, deck=tuple(range(20)), objects=())
        return value
    world = dict.fromkeys(WORLD_NAMES, 0)
    return PrivilegedObservation(world, (fighter(character, 400., 1), fighter(0, 800., -1)))


def test_flag_and_address_low_bits_survive_the_learning_observation():
    original = observation(1)
    original.players[0].update(fflags=0xFFFFFFFF, aflags=0x81234567, address=0xABCDEF01,
                               xspeed=float(np.float32(-0.000125)))
    restored = decode_privileged(encode_privileged(original))
    assert restored == original


@pytest.mark.skipif(not NAMES, reason="external original community package is not installed")
@pytest.mark.parametrize("name", NAMES)
def test_every_published_character_strategy_executes_without_rewriting(name):
    package = ScriptPackage(SCRIPTS, ROOT / "third_party/th123_ai/source/th123_ai/api.ai")
    episode = EpisodeConfig(7200, 1, 1, 0, "privileged_state", VISIBILITY, LEGACY_MATCH)
    policy = GodPolicy(name, package, name, episode)
    first, second = policy.spawn(37), policy.spawn(37)
    current = observation(int(name[:2]))
    encoded = encode_privileged(current)
    if name == "01_marisa_main_新版厨远A.ai":
        # The supplied variant uses undefined wait_frame. Preserve its source,
        # surface the failure, and never replace the absent value with a guess.
        from lupa.lua51 import LuaError
        with pytest.raises(LuaError, match="script stopped"):
            for frame in range(12):
                encoded[1], encoded[3] = frame / 65536., frame / 65536.
                first.act(encoded)
        assert first.failures
        return
    for frame in range(12):
        # Update only the two clock fields; the source sees an unchanged stance.
        encoded[1], encoded[3] = frame / 65536., frame / 65536.
        assert first.act(encoded) == second.act(encoded)
    assert first.lua is not second.lua
    assert first.api.requested is not second.api.requested


@pytest.mark.skipif(not NAMES, reason="external original api.ai is not installed")
def test_frame_boundary_scheduler_matches_the_unmodified_original_api():
    import queue
    import threading
    from lupa.lua51 import LuaRuntime
    from soku_rl.policy.god.api import ScriptAPI
    from soku_rl.policy.god.runtime import SCHEDULER

    original = (ROOT / "third_party/th123_ai/source/th123_ai/api.ai").read_bytes()
    program = b"""
      function auxiliary() while true do key_on(ACT_C); yield(); key_off(ACT_C); wait(2) end end
      function main()
        create_thread(auxiliary)
        while true do
          key_on(ACT_RIGHT); wait(3); key_off(ACT_RIGHT)
          key_on(ACT_DLEFT); key_on(ACT_A); wait(2); key_reset(); yield()
        end
      end
    """
    output, requests = queue.Queue(), queue.Queue()

    def reference():
        lua = LuaRuntime(encoding=None, unpack_returned_tuples=True)
        api = ScriptAPI(lua, 7)
        def boundary():
            output.put(api.inputs())
            if requests.get(timeout=10) == "stop":
                raise RuntimeError("reference stopped")
        lua.globals()[b"_yield"] = boundary
        lua.execute(original)
        lua.execute(program)
        try:
            lua.globals()[b"api_main"]()
        except RuntimeError:
            pass
    thread = threading.Thread(target=reference)
    thread.start()
    lua = LuaRuntime(encoding=None, unpack_returned_tuples=True)
    api = ScriptAPI(lua, 7)
    failures = []
    lua.globals()[b"_script_error"] = failures.append
    lua.execute(original); lua.execute(SCHEDULER); lua.execute(program)
    step = lua.globals()[b"make_step"]()
    try:
        for frame in range(24):
            step()
            assert api.inputs() == output.get(timeout=10)
            requests.put("continue" if frame < 23 else "stop")
    finally:
        requests.put("stop")
        thread.join(timeout=10)
    assert not thread.is_alive() and not failures
