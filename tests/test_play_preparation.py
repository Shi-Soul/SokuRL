"""Preparation must finish before connection without changing match memory."""
from dataclasses import asdict
import json

from hydra import compose, initialize_config_dir
from pathlib import Path
import numpy as np
from omegaconf import OmegaConf
import pytest

from soku_rl.env import EpisodeConfig
from soku_rl.env.match import LEGACY_MATCH
from soku_rl.env.wrappers.learning import LearningConfig, LearningInterface
from soku_rl.play.loader import play_interface, warm_play_policy
from soku_rl.play.menu import play_menu
from soku_rl.play.opponents import Opponent
from soku_rl.play.settings import client_plan
from soku_rl.policy.base import RLPolicy
from soku_rl.policy.population import MixturePolicy, UniformPolicy
from test_env_timing import VISIBILITY


def test_artifact_supplies_complete_observation_and_timing_contract(tmp_path):
    episode = EpisodeConfig(7200, 4, 3, 5, "state", VISIBILITY, LEGACY_MATCH)
    wrappers = LearningConfig("combat", True, 8, 1.)
    OmegaConf.save(OmegaConf.create({"episode": asdict(episode), "wrappers": asdict(wrappers)}),
                   tmp_path / "training.yaml")
    (tmp_path / "policy.json").write_text(json.dumps({"training_config": "training.yaml"}), encoding="utf-8")
    candidate = {"policy": {"kind": "onnx_recurrent", "path": str(tmp_path / "policy.json")}}
    interface = play_interface(candidate, {}, {}, "human", ["play.ai.character=19"])
    assert interface.episode == episode and interface.config == wrappers
    assert interface.observation_space.shape == (1701,)
    for override in ("episode.latency_frames=0", "wrappers=raw", "+wrappers.action_history=0"):
        with pytest.raises(ValueError, match="remove overrides"):
            play_interface(candidate, {}, {}, "human", [override])
    with pytest.raises(ValueError, match="selected track"):
        play_interface(candidate, {}, {}, "superhuman", [])


class LearnedPolicy(RLPolicy):
    def __init__(self):
        self.actors = []

    def spawn(self, seed):
        rng, actions = np.random.default_rng(seed), []
        self.actors.append(actions)

        class Actor:
            def act(self, observation):
                assert observation.shape == (400,) and observation.dtype == np.float32
                actions.append(int(rng.integers(90)))
                return actions[-1]

        return Actor()


def test_warmup_visits_all_members_and_discards_actor_memory():
    interface = LearningInterface(EpisodeConfig(7200, 1, 3, 5, "state", VISIBILITY, LEGACY_MATCH),
                                  LearningConfig("combat", False, 0, 0.))
    first, second = LearnedPolicy(), LearnedPolicy()
    nested = MixturePolicy("nested", (first, UniformPolicy("uniform", 90)), (.5, .5), "nested")
    policy = MixturePolicy("all", (nested, second), (1., 0.), "all")
    assert warm_play_policy(policy, interface, 19) == 2
    assert len(first.actors[0]) == len(second.actors[0]) == 8
    for member in (first, second):
        observation = np.zeros(400, np.float32)
        fresh = member.spawn(19)
        assert [fresh.act(observation) for _ in range(8)] == member.actors[0]
        assert member.actors[0] is not member.actors[1]


@pytest.mark.parametrize("connection", ("local", "host", "join"))
@pytest.mark.parametrize("track", ("human", "superhuman"))
def test_menu_composes_the_same_hydra_config_for_all_connections(connection, track):
    catalog = {
        "rush": Opponent("rush", "rush", (0, 1), ("human", "superhuman"), {"kind": "rule"}, ""),
        "god:variant": Opponent("god:variant", "新版策略", (19,), ("superhuman",), {"kind": "rule"}, "script")}
    answers = [str(("local", "host", "join").index(connection)+1), "1" if track == "human" else "2",
               "1" if track == "human" else "2"]
    if track == "human":
        answers.append("2")  # Marisa, independently of the physical seat.
    if connection == "local":
        answers.append("1")
    else:
        if connection == "join":
            answers.append("192.0.2.7")
        answers.append("10812")
    values, printed = iter(answers), []
    overrides = play_menu(catalog, {"network": {"address": "127.0.0.1", "port": 10811}},
                          lambda prompt: next(values), printed.append)
    with initialize_config_dir(version_base="1.3", config_dir=str(Path(__file__).parents[1] / "config")):
        config = compose(config_name="play", overrides=overrides)
    clients = client_plan(OmegaConf.to_container(config.play))
    ai = next(client for client in clients if client["realtime"])
    assert ai["character"] == (1 if track == "human" else 19)
    assert config.episode.observation_mode == ("state" if track == "human" else "privileged_state")
    assert ai["role"] == ("host" if connection == "host" else "join")
    assert len(clients) == (2 if connection == "local" else 1)
    if connection != "local":
        assert config.play.network.port == 10812


def test_original_opponent_owns_its_character_selection():
    god = Opponent("god:suwako", "suwako", (19,), ("superhuman",), {}, "script")
    rule = Opponent("rush", "rush", (0, 1), ("human",), {}, "")
    assert god.character("opponent") == 19
    with pytest.raises(ValueError, match="explicit"):
        rule.character("opponent")
    for invalid in (0, True, "19"):
        with pytest.raises(ValueError, match="AI characters"):
            god.character(invalid)
