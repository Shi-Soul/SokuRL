"""All public rule opponents must use the live learner action interface."""
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import pytest

from soku_rl.env import EpisodeConfig
from soku_rl.env.wrappers.learning import LearningConfig, LearningInterface
from soku_rl.play.loader import load_play_policy


def test_all_fifteen_rules_run_without_a_checkpoint():
    root = Path(__file__).resolve().parents[1]
    config = OmegaConf.to_container(OmegaConf.load(root / "config/rules/default.yaml"))
    training = OmegaConf.to_container(OmegaConf.load(root / "config/train.yaml"))
    episode = training["episode"] | {"decision_frames": 3, "latency_frames": 5, "observation_mode": "state"}
    interface = LearningInterface(EpisodeConfig.from_dict(episode), LearningConfig("combat", True, 8, 1.))
    assert len(config["roster"]) == 15
    for seat in (0, 1):
        for name in config["roster"]:
            policy = load_play_policy({"name": name, "policy": {"kind": "rule", "name": name}},
                                      interface, config, "cpu", seat)
            actor = policy.spawn(1732)
            for _ in range(4):
                action = actor.act(np.zeros(interface.observation_space.shape, np.float32))
                assert interface.action_space.contains(action)


@pytest.mark.parametrize("enemy_character", range(20))
def test_rules_can_face_every_playable_human_character(enemy_character):
    root = Path(__file__).resolve().parents[1]
    rules = OmegaConf.to_container(OmegaConf.load(root / "config/rules/default.yaml"))
    training = OmegaConf.to_container(OmegaConf.load(root / "config/train.yaml"))
    episode = training["episode"] | {"decision_frames": 3, "latency_frames": 5, "observation_mode": "state"}
    interface = LearningInterface(EpisodeConfig.from_dict(episode), LearningConfig("combat", True, 8, 1.))
    for seat, own_character in ((0, 1), (1, 0)):
        observation = np.zeros(interface.observation_space.shape, np.float32)
        history = observation[:1600].reshape(4, 400)
        history[:, :8] = [1, .3, .7, 1, 1, 1, 1, own_character / 19]
        history[:, 8:16] = [1, .5, .7, -1, 1, 1, 1, enemy_character / 19]
        for name in rules["roster"]:
            policy = load_play_policy({"name": name, "policy": {"kind": "rule", "name": name}},
                                      interface, rules, "cpu", seat)
            actor = policy.spawn(1732)
            for _ in range(8):
                assert interface.action_space.contains(actor.act(observation))
