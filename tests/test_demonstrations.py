from dataclasses import dataclass
import hashlib
import json

import numpy as np
import pytest
import torch

from soku_rl.env.encoding import decode_action
from soku_rl.env.wrappers.learning import LearningConfig, LearningVectorEnv
from soku_rl.rl.demonstrations import collect_demonstrations, demonstration_plan
from test_shared_ppo import fixture_env


@dataclass
class ConstantPolicy:
    value: int

    @property
    def fingerprint(self):
        return f"constant:{self.value}"

    def spawn(self, seed):
        return self

    def act(self, observation):
        return self.value


def population():
    return [{"name": "opponent", "probability": 1.,
             "setup": {"character": 0, "palette": 0, "deck": 0}}]


def test_plan_balances_seats_and_splits_without_reusing_reserved_worlds():
    config = {"episodes_per_seat": 8, "validation_per_seat": 2}
    first = demonstration_plan(config, population(), 42, set())
    excluded = {row["world_seed"] for row in first[:5]}
    second = demonstration_plan(config, population(), 42, excluded)
    assert second == demonstration_plan(config, population(), 42, excluded)
    assert len({row["world_seed"] for row in second}) == 16
    assert not excluded & {row["world_seed"] for row in second}
    assert len({row["id"] for row in second}) == 16
    for seat in (0, 1):
        assert sum(row["learner_seat"] == seat and row["split"] == "validation" for row in second) == 2
        assert sum(row["learner_seat"] == seat and row["split"] == "train" for row in second) == 6


@pytest.mark.parametrize("total,validation", [(1, 1), (2, 0), (2, True), (2.5, 1)])
def test_invalid_split_is_rejected(total, validation):
    with pytest.raises(ValueError):
        demonstration_plan({"episodes_per_seat": total, "validation_per_seat": validation}, population(), 42, set())


@pytest.mark.parametrize("learner_controls", [False, True])
def test_collection_preserves_own_history_rewards_and_complete_game_boundaries(tmp_path, monkeypatch, learner_controls):
    env = fixture_env()
    env = LearningVectorEnv(env.env, LearningConfig("combat", False, 2, 0.))
    backend = env.env.backend
    monkeypatch.setattr(backend, "reset_matchups", lambda seeds, matches: backend.reset_slots(seeds), raising=False)
    step = env.step

    def rewarded(actions):
        values = step(actions)
        for pair in values[1].values():
            pair.update(player_0=1., player_1=-1.)
        return values

    monkeypatch.setattr(env, "step", rewarded)
    plan = demonstration_plan({"episodes_per_seat": 2, "validation_per_seat": 1}, population(), 42, set())
    teacher = ConstantPolicy(3)
    behavior = ConstantPolicy(5) if learner_controls else teacher
    try:
        metadata = collect_demonstrations(env, plan, {"character": 1, "palette": 0, "deck": 0},
            population(), teacher, [ConstantPolicy(8)], behavior, tmp_path / "episodes")
        assert metadata["complete"] is True
        assert metadata["successful_env_steps"] == 12
        assert metadata["incomplete_episodes"] == []
        assert len(metadata["episodes"]) == 4
        assert metadata["control"] == ("learner" if learner_controls else "teacher")
        assert metadata["behavior_fingerprint"] == behavior.fingerprint
        for row in metadata["episodes"]:
            path = tmp_path / "episodes" / row["path"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
            data = torch.load(path, weights_only=False)
            assert data["actions"].tolist() == [3, 3, 3]
            assert data["executed_actions"].tolist() == [behavior.value] * 3
            assert row["teacher_behavior_disagreements"] == (3 if learner_controls else 0)
            sign = 1 if row["learner_seat"] == 0 else -1
            assert data["rewards"].tolist() == [sign] * 3
            assert data["returns"].tolist() == [3 * sign, 2 * sign, sign]
            assert row["match"][f"player_{row['learner_seat']}"]["character"] == 1
            assert row["outcome"] == "time_limit"
            assert row["frame"] == 3
            assert row["return"] == 3 * sign
            # Each saved observation precedes its label; never use the reset state.
            assert np.array_equal(data["observations"][1].unpack()[-8:],
                np.asarray(decode_action(env.interface.command(behavior.value)).inputs, dtype=np.float32))
    finally:
        env.close()


def test_failed_collection_does_not_mark_partial_episodes_complete(tmp_path, monkeypatch):
    env = fixture_env()
    backend = env.env.backend
    monkeypatch.setattr(backend, "reset_matchups", lambda seeds, matches: backend.reset_slots(seeds), raising=False)
    step = env.step
    calls = []

    def failed(actions):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("test worker interruption")
        return step(actions)

    monkeypatch.setattr(env, "step", failed)
    plan = demonstration_plan({"episodes_per_seat": 2, "validation_per_seat": 1}, population(), 42, set())
    teacher = ConstantPolicy(3)
    try:
        with pytest.raises(RuntimeError, match="interruption"):
            collect_demonstrations(env, plan, {"character": 1, "palette": 0, "deck": 0},
                population(), teacher, [ConstantPolicy(8)], teacher, tmp_path / "episodes")
        manifest = json.loads((tmp_path / "episodes/manifest.json").read_text())
        assert manifest["complete"] is False
        assert manifest["episodes"] == []
        assert manifest["successful_env_steps"] == 2
        assert all(row["observed_steps"] == 1 and row["attempted_actions"] == 2
                   for row in manifest["incomplete_episodes"])
    finally:
        env.close()
