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
from soku_rl.play.match import MatchFrame, MatchState
from soku_rl.play.match_policies import MatchPolicies
from soku_rl.play.realtime_session import RealtimePolicy
from soku_rl.env.encoding import AGENTS, decode_action
from soku_rl.policy.god.package import ScriptPackage
from soku_rl.policy.god.runtime import GodPolicy
from soku_rl.policy.rules.observed_rules import LearningRulePolicy
from test_env_timing import VISIBILITY
from test_god_scripts import NAMES, SCRIPTS, observation

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
        history, features = ObservationHistory(episode, AGENTS), LearningEpisode(interface)
        for frame in range(12):
            first = observation(1)
            first.world.update(frame=frame, battle_time=500 + frame)
            pair = (first, replace(first, players=first.players[::-1]))
            absolute = tuple(replace(value, world=value.world | {"frame": origin + frame}) for value in pair)
            if frame == 0:
                history.reset(frame, pair)
                features.reset_agent(live.agent, history.observations()[live.agent])
                live.start_round(origin, absolute, seed)
                assert live.reset_each_round is False
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


@pytest.mark.parametrize("seat", (0, 1))
def test_match_controller_preserves_original_lua_state_across_knockout(seat):
    scripts = ROOT / "third_party/th123_ai/package/th123ai/script"
    if not scripts.is_dir():
        pytest.skip("original strategy package is unavailable")
    package = ScriptPackage(scripts, ROOT / "third_party/th123_ai/source/th123_ai/api.ai")
    episode = EpisodeConfig(7200, 1, 1, 0, "privileged_state", VISIBILITY, LEGACY_MATCH)
    interface = LearningInterface(episode, LearningConfig("full", False, 0, 0.))
    policy = LearningRulePolicy(GodPolicy("god", package, "character", episode), interface)
    baseline = policy.spawn(37 + seat)
    controller = MatchPolicies({seat: policy}, interface, 37, 2)
    for frame in range(8):
        first = observation(1)
        first.world.update(frame=frame, battle_time=500 + frame)
        first.players[1]["hp"] = 0 if frame in (2, 3) else 10000
        first.players[0]["win_count"] = int(frame >= 3)
        pair = (first, replace(first, players=first.players[::-1]))
        state = MatchState(1, int(frame >= 4), frame, (int(frame >= 3), 0),
                           (10000, first.players[1]["hp"]), "battle")
        result = controller.advance(state, pair)
        expected = interface.command(baseline.act(episode.encode(pair[seat])))
        assert result.inputs[seat] == decode_action(expected).inputs
    assert controller.instances[seat] == 1


@pytest.mark.skipif(not NAMES, reason="external original strategy package is unavailable")
@pytest.mark.parametrize("script", NAMES)
@pytest.mark.parametrize("seat", (0, 1))
def test_network_controller_matches_original_scheduler_through_knockout(script, seat):
    from god_reference.scheduler import OriginalScheduler
    package = ScriptPackage(SCRIPTS, ROOT / "third_party/th123_ai/source/th123_ai/api.ai")
    episode = EpisodeConfig(7200, 1, 1, 0, "privileged_state", VISIBILITY, LEGACY_MATCH)
    interface = LearningInterface(episode, LearningConfig("full", False, 0, 0.))
    policy = LearningRulePolicy(GodPolicy("god", package, script, episode), interface)
    controller = RealtimePolicy(policy, interface, seat, 37)
    current = observation(int(script[:2]))
    current.world["battle_time"] = 500
    reference = OriginalScheduler(package, script, 37+seat, current)
    try:
        for frame in range(12):
            current = observation(int(script[:2]))
            current.world.update(frame=frame, battle_time=500+frame)
            current.players[1]["hp"] = 0 if frame in (2, 3) else 10000
            current.players[0]["win_count"] = int(frame >= 3)
            if frame:
                reference.advance(current)
            live = replace(current, world=current.world | {"frame": frame+1})
            pair = (live, replace(live, players=live.players[::-1]))
            if seat:
                pair = pair[::-1]
            state = MatchState(1, int(frame >= 4), frame+1,
                tuple(value.players[0]["win_count"] for value in pair),
                tuple(value.players[0]["hp"] for value in pair), "battle")
            result = controller.advance(MatchFrame(state, pair))
            assert result.inputs[seat] == reference.inputs()
        assert controller.instances == 1
    finally:
        reference.close()
        controller.stop()


@pytest.mark.parametrize("seat", (0, 1))
def test_realtime_god_consumes_complete_observations_without_dense_tensor_roundtrip(seat, monkeypatch):
    if not SCRIPTS.is_dir():
        pytest.skip("original strategy package is unavailable")
    package = ScriptPackage(SCRIPTS, ROOT / "third_party/th123_ai/source/th123_ai/api.ai")
    episode = EpisodeConfig(7200, 1, 1, 0, "privileged_state", VISIBILITY, LEGACY_MATCH)
    interface = LearningInterface(episode, LearningConfig("full", False, 0, 0.))
    policy = LearningRulePolicy(GodPolicy("god", package, "00_reimuEX1_main.ai", episode), interface)
    baseline = policy.spawn(1732 + seat)
    frames, expected = [], []
    for index in range(24):
        current = observation(0)
        current.world.update(frame=index, battle_time=500 + index)
        for player in current.players:
            player["objects"] = tuple(player | {"address": 0x100000 + i * 0x1000,
                "x": 450. + i} for i in range(128))
            player["obj_n"] = 128
        expected.append(interface.command(baseline.act(episode.encode(current))))
        absolute = replace(current, world=current.world | {"frame": index + 1})
        pair = (absolute, replace(absolute, players=absolute.players[::-1]))
        if seat:
            pair = pair[::-1]
        frames.append(MatchFrame(MatchState(1, 0, index + 1, (0, 0), (10000, 10000), "battle"), pair))

    def reject_dense_encoding(*args):
        raise AssertionError("realtime god rebuilt a dense learning tensor")

    monkeypatch.setattr(EpisodeConfig, "encode", reject_dense_encoding)
    controller = RealtimePolicy(policy, interface, seat, 1732)
    try:
        for frame, command in zip(frames, expected, strict=True):
            assert controller.advance(frame).inputs[seat] == decode_action(command).inputs
        with pytest.raises(RuntimeError, match="lost a simulation frame"):
            controller.advance(replace(frames[-1], match=replace(frames[-1].match, frame=26)))
    finally:
        controller.stop()
