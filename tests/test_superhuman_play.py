"""Original scripts must receive the same data through live and offline adapters."""
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("lupa.lua51")

from soku_rl.env import EpisodeConfig
from soku_rl.env.match import LEGACY_MATCH
from soku_rl.env.observation_history import ObservationHistory
from soku_rl.env.wrappers.learning import LearningConfig, LearningEpisode, LearningInterface
from soku_rl.play.live_policy import LivePolicy
from soku_rl.policy.god.package import ScriptPackage
from soku_rl.policy.god.runtime import GodPolicy
from soku_rl.policy.rules.observed_rules import LearningRulePolicy
from test_env_timing import VISIBILITY
from test_god_scripts import observation

ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize("seat", (0, 1))
def test_original_strategy_matches_offline_with_absolute_live_frame_numbers(seat):
    scripts = ROOT / "third_party/th123_ai/package/th123ai/script"
    if not scripts.is_dir():
        pytest.skip("original strategy package is unavailable")
    package = ScriptPackage(scripts, ROOT / "third_party/th123_ai/source/th123_ai/api.ai")
    episode = EpisodeConfig(7200, 2, 1, 0, "privileged_state", VISIBILITY, LEGACY_MATCH)
    interface = LearningInterface(episode, LearningConfig("full", False, 2, 0.))
    policy = LearningRulePolicy(GodPolicy("god", package, "character", episode), interface)
    live = LivePolicy(policy, interface, seat)
    for origin, seed in ((1500, 37), (9000, 52)):
        actor = policy.spawn(seed)
        history, features = ObservationHistory(episode), LearningEpisode(interface)
        for frame in range(12):
            first = observation(1)
            first.world.update(frame=frame, battle_time=500 + frame)
            pair = (first, replace(first, players=first.players[::-1]))
            absolute = tuple(replace(value, world=value.world | {"frame": origin + frame}) for value in pair)
            if frame == 0:
                history.reset(frame, pair)
                features.reset_agent(live.agent, history.observations()[live.agent])
                live.start_round(origin, absolute, seed)
                with pytest.raises(RuntimeError, match="cannot skip"):
                    live.skip_decision()
            else:
                history.append(frame, pair)
                live.observe(origin + frame, absolute)
            encoded = features.observation(live.agent, history.observations()[live.agent], frame)
            command = interface.command(actor.act(encoded))
            assert live.act() == command
            features.record_command(live.agent, command)
            assert all(value.world["frame"] == origin + frame for value in absolute)
