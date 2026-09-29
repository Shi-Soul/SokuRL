"""Validate the observation and action contract shared by saved policies."""
from dataclasses import asdict

from omegaconf import OmegaConf

from soku_rl.env.wrappers.learning import LearningConfig
from soku_rl.env.hisouten_env import EpisodeConfig


def read_training_contract(path, interface):
    training = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if EpisodeConfig.from_dict(training["episode"]) != interface.episode:
        raise ValueError("checkpoint and evaluation episode configurations differ")
    # Artifacts from before learning wrappers use the original identity contract.
    expected = (LearningConfig(**training["wrappers"]) if "wrappers" in training else
                LearningConfig("full", False, 0, 0.))
    if expected != interface.config:
        raise ValueError("checkpoint and evaluation learning wrappers differ")
    return training
