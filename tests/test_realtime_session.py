"""Live network control preserves cadence, round memory and physical seat ownership."""
from copy import deepcopy
from dataclasses import asdict, replace
from types import SimpleNamespace
import sys

import pytest
from omegaconf import OmegaConf

from soku_rl.env import EpisodeConfig
from soku_rl.env.match import LEGACY_MATCH, PlayerSetup
from soku_rl.env.wrappers.learning import LearningConfig, LearningInterface
from soku_rl.play.loader import checkpoint_interface, load_play_policy
from soku_rl.play.match import MatchFrame, MatchState
from soku_rl.play.realtime_session import RealtimePolicy
from soku_rl.play.settings import client_plan
from soku_rl.policy.contract import read_training_contract
from test_env_timing import VISIBILITY
from test_live_policy import observations, RecordingPolicy


def settings():
    return {"connection": "local", "ai": {"character": 19, "render": False, "mute_audio": True},
            "human": {"seat": 2, "character": 6, "render": True, "mute_audio": False, "automate_menu": False},
            "network": {"address": "127.0.0.1", "port": 10811}}


@pytest.mark.parametrize("seat", (1, 2))
@pytest.mark.parametrize("character", range(20))
def test_local_plan_preserves_every_human_character_and_seat(seat, character):
    config = settings()
    config["human"].update(seat=seat, character=character)
    plan = client_plan(config)
    assert [client["role"] for client in plan] == ["host", "join"]
    human = next(client for client in plan if client["name"] == "human")
    assert (human["seat"], human["character"], human["realtime"], human["automate_menu"]) == (seat-1, character, False, False)
    assert next(client for client in plan if client["name"] == "ai")["seat"] == 2-seat


@pytest.mark.parametrize("role, seat", (("host", 0), ("join", 1)))
def test_external_network_plan_starts_only_the_ai(role, seat):
    config = settings() | {"connection": role}
    plan = client_plan(config)
    assert len(plan) == 1 and plan[0]["seat"] == seat and plan[0]["realtime"]


@pytest.mark.parametrize("seat", (0, 1))
def test_network_rounds_reuse_training_cadence_and_never_hold_old_decisions(seat):
    interface = LearningInterface(EpisodeConfig(7200, 4, 3, 5, "state", VISIBILITY, LEGACY_MATCH),
                                  LearningConfig("combat", True, 8, 1.))
    policy = RecordingPolicy()
    controller = RealtimePolicy(policy, interface, seat, 20)
    assert [seed for seed, _ in policy.episodes] == [20+seat]
    assert policy.episodes[0][1] == []
    decisions = []
    for frame in range(1, 13):
        state = MatchState(1, int(frame >= 7), frame, (int(frame >= 6), 0),
                           (10000, 0 if 5 <= frame <= 6 else 10000), "battle")
        step = controller.advance(MatchFrame(state, observations(frame, frame)))
        if step.inputs:
            assert set(step.inputs) == {seat}
            decisions.append(frame)
    assert decisions == [1, 4, 7, 10]
    assert [seed for seed, _ in policy.episodes] == [20+seat, 22+seat]
    assert [len(records) for _, records in policy.episodes] == [2, 2]
    with pytest.raises(RuntimeError, match="lost a simulation frame"):
        controller.advance(MatchFrame(replace(state, frame=14), observations(14, 14)))


def test_play_changes_match_setup_but_keeps_strict_artifact_contract(tmp_path, monkeypatch):
    training = EpisodeConfig(7200, 4, 3, 5, "state", VISIBILITY, LEGACY_MATCH)
    wrappers = LearningConfig("full", False, 0, 0.)
    path = tmp_path / "training.yaml"
    OmegaConf.save(OmegaConf.create({"episode": asdict(training), "wrappers": asdict(wrappers)}), path)
    live = replace(training, match=replace(LEGACY_MATCH, player_0=PlayerSetup(19, 2, 3)))
    interface = LearningInterface(live, wrappers)
    spec = {"kind": "nfsp_average", "training_config": str(path), "player": "player_0"}
    checked = checkpoint_interface(spec, interface)
    assert checked.episode == training
    with pytest.raises(ValueError, match="episode"):
        read_training_contract(path, interface)
    with pytest.raises(ValueError, match="episode"):
        read_training_contract(path, checkpoint_interface(spec,
            LearningInterface(replace(live, latency_frames=6), wrappers)))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(set_num_threads=lambda count: None))
    monkeypatch.setattr("soku_rl.play.loader.load_policy", lambda name, data, contract, device: (data, contract))
    data, contract = load_play_policy({"name": "saved", "policy": spec}, interface, {}, "cpu", 1)
    assert data["player"] == "player_0" and contract.episode == training
