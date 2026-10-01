"""Mixtures and wrappers preserve the selected member's play memory boundary."""
from types import SimpleNamespace

import pytest

from soku_rl.policy.base import PlayActor, RulePolicy
from soku_rl.policy.population import MixturePolicy


class SeedPolicy(RulePolicy):
    def __init__(self, continuous):
        self.continuous = continuous

    def spawn(self, seed):
        return SimpleNamespace(act=lambda observation: seed)

    def spawn_play(self, seed):
        return PlayActor(self.spawn(seed), not self.continuous)


@pytest.mark.parametrize("continuous", (False, True))
def test_mixture_keeps_selected_memory_boundary_and_episode_randomness(continuous):
    selected = SeedPolicy(continuous)
    unused = SeedPolicy(not continuous)
    policy = MixturePolicy("mixed", (selected, unused), (.999, .001), "fixture")
    for seed in range(20):
        member, _ = policy._select(seed)
        actor = policy.spawn_play(seed)
        assert actor.reset_each_round is not member.continuous
        assert actor.act(()) == policy.spawn(seed).act(())
