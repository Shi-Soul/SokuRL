"""Takeover preserves constituent seeds, episode memory and exact switching time."""
from dataclasses import dataclass, replace

import numpy as np
import pytest

from soku_rl.env.wrappers.learning import LearningInterface
from soku_rl.policy.loader import load_policy
from soku_rl.policy.takeover import TeacherTakeoverPolicy
from test_shared_ppo import fixture_env


@dataclass
class StatefulActor:
    rng: object
    observations: list
    offset: int

    def act(self, observation):
        self.observations.append(observation)
        return int(self.rng.integers(100)) + self.offset + sum(self.observations)


@dataclass(frozen=True)
class StatefulPolicy:
    offset: int

    @property
    def fingerprint(self):
        return str(self.offset)

    def spawn(self, seed):
        return StatefulActor(np.random.default_rng(seed), [], self.offset)


@pytest.mark.parametrize("boundary", [0, 1, 7, 100])
def test_exact_prefix_and_continuously_advanced_teacher(boundary):
    learner, teacher = StatefulPolicy(0), StatefulPolicy(1000)
    policy = TeacherTakeoverPolicy("diagnostic", learner, teacher, boundary)
    actor = policy.spawn(123)
    plain_learner, plain_teacher = learner.spawn(123), teacher.spawn(123)
    for frame in range(20):
        expected_learner = plain_learner.act(frame)
        expected_teacher = plain_teacher.act(frame)
        assert actor.act(frame) == (expected_learner if frame < boundary else expected_teacher)
        assert actor.teacher.observations == list(range(frame + 1))
    assert actor.learner.observations == list(range(min(boundary, 20)))
    assert actor.frame == 20


def test_interleaving_does_not_change_private_memory_or_randomness():
    policy = TeacherTakeoverPolicy("diagnostic", StatefulPolicy(0), StatefulPolicy(1000), 7)
    first, replay, unrelated = policy.spawn(23), policy.spawn(23), policy.spawn(53)
    assert first.learner is not first.teacher
    for observation in range(30):
        action = first.act(observation)
        unrelated.act(observation + 100)
        unrelated.act(observation + 200)
        assert replay.act(observation) == action
    assert first.teacher.observations == replay.teacher.observations
    assert policy.spawn(23).frame == 0
    assert policy.fingerprint == TeacherTakeoverPolicy("renamed", StatefulPolicy(0), StatefulPolicy(1000), 7).fingerprint
    for learner, teacher, boundary in [(StatefulPolicy(1), StatefulPolicy(1000), 7),
            (StatefulPolicy(0), StatefulPolicy(1001), 7), (StatefulPolicy(0), StatefulPolicy(1000), 8)]:
        assert policy.fingerprint != TeacherTakeoverPolicy("diagnostic", learner, teacher, boundary).fingerprint


@pytest.mark.parametrize("boundary", [-1, True, 1.0, "1", None, float("inf")])
def test_invalid_boundary_rejected(boundary):
    with pytest.raises(ValueError, match="nonnegative integer"):
        TeacherTakeoverPolicy("bad", StatefulPolicy(0), StatefulPolicy(1000), boundary)


def test_loader_preserves_uniform_constituent_rng_and_rejects_invalid_contracts():
    env = fixture_env()
    spec = {"kind": "teacher_takeover", "learner": {"kind": "uniform"},
            "teacher": {"kind": "uniform"}, "after_frames": 2}
    try:
        policy = load_policy("diagnostic", spec, env.interface, "cpu")
        actor, plain = policy.spawn(7), load_policy("plain", {"kind": "uniform"}, env.interface, "cpu").spawn(7)
        for _ in range(20):
            assert actor.act(0) == plain.act(0)
        for change in ({"extra": 1}, {"learner": "invalid"}, {"after_frames": True}):
            with pytest.raises(ValueError):
                load_policy("bad", spec | change, env.interface, "cpu")
        for change in ({"decision_frames": 2}, {"latency_frames": 1}):
            interface = LearningInterface(replace(env.interface.episode, **change), env.interface.config)
            with pytest.raises(ValueError, match="every-frame"):
                load_policy("bad", spec, interface, "cpu")
    finally:
        env.close()
