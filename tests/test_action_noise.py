"""Noisy opponents preserve controller time and reproducible independent actors."""
from dataclasses import dataclass
from pathlib import Path

from hydra import compose, initialize_config_dir
import numpy as np
from omegaconf import OmegaConf
import pytest

from soku_rl.policy.action_noise import ActionNoisePolicy
from soku_rl.policy.base import PlayActor
from soku_rl.policy.loader import load_policy
from soku_rl.policy.matchups import opponent_interface
from test_shared_ppo import fixture_env


@dataclass
class CounterActor:
    count: int

    def act(self, observation):
        self.count += 1
        return (self.count + observation) % 4


class CounterPolicy:
    fingerprint = "counter-v1"

    def spawn(self, seed):
        return CounterActor(seed % 4)

    def spawn_play(self, seed):
        return PlayActor(self.spawn(seed), False)


def test_zero_noise_preserves_sequence_and_play_memory_boundary():
    base = CounterPolicy()
    plain = base.spawn(31)
    noisy = ActionNoisePolicy("noisy", base, 4, 0.).spawn_play(31)
    assert noisy.reset_each_round is False
    for observation in range(100):
        assert noisy.act(observation) == plain.act(observation)


def test_full_noise_is_uniform_and_still_advances_original_controller():
    actor = ActionNoisePolicy("random", CounterPolicy(), 4, 1.).spawn(31)
    draws = np.array([actor.act(0) for _ in range(20000)])
    assert actor.actor.count == 20003
    assert np.all(np.abs(np.bincount(draws, minlength=4) / len(draws) - .25) < .015)


def test_interleaved_actors_have_independent_reproducible_randomness():
    policy = ActionNoisePolicy("noisy", CounterPolicy(), 4, .7)
    first, replay, unrelated = policy.spawn(17), policy.spawn(17), policy.spawn(29)
    different = False
    for observation in range(100):
        action = first.act(observation)
        different |= action != unrelated.act(observation)
        unrelated.act(observation + 1)
        assert action == replay.act(observation)
    assert different
    assert first.actor.count == replay.actor.count == 101
    assert policy.fingerprint != ActionNoisePolicy("noisy", CounterPolicy(), 4, .6).fingerprint


@pytest.mark.parametrize("probability", [-.1, 1.1, float("nan"), float("inf"), True, "0.5"])
def test_invalid_probability_is_rejected(probability):
    with pytest.raises(ValueError, match="probability"):
        ActionNoisePolicy("bad", CounterPolicy(), 4, probability)


def test_loader_composes_noise_and_rejects_mismatched_wrapped_god():
    env = fixture_env()
    try:
        spec = {"kind": "action_noise", "random_probability": .5, "policy": {"kind": "uniform"}}
        policy = load_policy("noisy", spec, env.interface, "cpu")
        assert 0 <= policy.spawn(13).act(0) < env.interface.action_space.n
        with pytest.raises(ValueError, match="action_noise"):
            load_policy("bad", spec | {"typo": 1}, env.interface, "cpu")
        spec["policy"] = {"kind": "rule", "name": "god", "rules": {"god": {"script": "00_reimu_main.ai"}}}
        with pytest.raises(ValueError, match="script and opponent character"):
            opponent_interface(env.interface, {"character": 1, "palette": 0, "deck": 0},
                {"policy": spec, "setup": {"character": 6, "palette": 0, "deck": 0}})
    finally:
        env.close()


def test_noisy_target_preset_preserves_ppo_and_exposes_probability():
    with initialize_config_dir(config_dir=str(Path(__file__).parents[1] / "config"), version_base="1.3"):
        cfg = compose(config_name="train", overrides=["algorithm=br", "rules=god",
            "wrappers=superhuman_learning", "track=superhuman_combat",
            "+br_opponents=god_noisy_target", "algorithm.target.random_probability=0.75"])
        opponent, = OmegaConf.to_container(cfg.algorithm.opponents, resolve=True)
        assert OmegaConf.to_container(cfg.algorithm.ppo, resolve=True) == OmegaConf.to_container(cfg.rl.ppo, resolve=True)
    assert opponent["policy"]["random_probability"] == .75
    assert opponent["policy"]["policy"]["rules"]["god"]["script"] == "character"
