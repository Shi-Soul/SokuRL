"""All public rule opponents must use the live learner action interface."""
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import pytest

from soku_rl.env import EpisodeConfig
from soku_rl.learning_wrappers import LearningConfig, LearningInterface
from soku_rl.play_policy import load_play_policy


def test_all_fifteen_rules_run_without_a_checkpoint():
    root = Path(__file__).resolve().parents[1]
    config = OmegaConf.to_container(OmegaConf.load(root / "config/rules/default.yaml"))
    training = OmegaConf.to_container(OmegaConf.load(root / "config/train.yaml"))
    episode = training["episode"] | {"decision_frames": 3, "latency_frames": 5, "observation_mode": "state"}
    interface = LearningInterface(EpisodeConfig(**episode), LearningConfig("combat", True, 8, 1.))
    assert len(config["roster"]) == 15
    for seat in (0, 1):
        for name in config["roster"]:
            policy = load_play_policy({"name": name, "policy": {"kind": "rule", "name": name}},
                                      interface, config, "cpu", seat)
            actor = policy.spawn(1732)
            for _ in range(4):
                action = actor.act(np.zeros(interface.observation_space.shape, np.float32))
                assert interface.action_space.contains(action)
    for kind in ("nfsp_average", "psro_mixture"):
        with pytest.raises(ValueError, match="seat"):
            load_play_policy({"name": kind, "policy": {"kind": kind, "player": "player_0"}},
                             interface, config, "cpu", 1)
