"""Ignored labels never train; their observations still determine recurrent memory."""
import copy

import numpy as np
import pytest
import torch

from soku_rl.rl.behavior_cloning import ObservationContractEnv
from soku_rl.rl.demonstration_supervision import supervision_mask
from soku_rl.rl.ppo import create_ppo, parameter_hash
from soku_rl.rl.recurrent_cloning import sequence_epoch, zero_states
from soku_rl.rl.rehearsal import DemonstrationRehearsal
from soku_rl.rl.storage import PackedObservation
from test_shared_ppo import fixture_config, fixture_env


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_masked_scoring_matches_online_memory_and_training_ignores_prefix_targets(device):
    torch.set_num_threads(1)
    env = fixture_env()
    try:
        model, _ = create_ppo(ObservationContractEnv(env.interface), env.interface, fixture_config("lstm"),
            {"kind": "fresh"}, device, 17)
        rng = np.random.default_rng(19)
        episodes = [[(PackedObservation.pack(rng.normal(size=env.single_observation_space.shape).astype(np.float32)),
            int(rng.integers(env.single_action_space.n)), float(rng.normal()), -1 if j == 0 else 1, j >= 3)
            for j in range(size)] for size in (5, 8)]
        metrics = []
        model.policy.set_training_mode(False)
        with torch.no_grad():
            for episode in episodes:
                states = zero_states(model.policy, 1)
                for packed, action, target, _, supervised in episode:
                    tensor = torch.from_numpy(packed.unpack()).unsqueeze(0).to(device)
                    predicted, values, _, states = model.policy(tensor, states,
                        torch.zeros(1, device=device), deterministic=True)
                    distribution = model.policy.action_dist
                    if supervised:
                        metrics.append((-float(distribution.log_prob(torch.tensor([action], device=device))),
                            float(predicted[0] == action), float((values[0, 0] - target) ** 2),
                            float(distribution.entropy())))
        expected = dict(zip(("nll", "accuracy", "value_mse", "entropy"), np.mean(metrics, axis=0)))
        for length, batch in ((1, 2), (2, 4), (4, 8)):
            actual, loss, updates = sequence_epoch(model, episodes, [0, 1], batch, length, 0., False, 1.)
            for key, value in expected.items():
                assert actual[key] == pytest.approx(value, abs=2e-6)
            assert actual["changed_samples"] == 7
            assert loss == pytest.approx(expected["nll"], abs=2e-6) and updates == 0
        other = copy.deepcopy(model)
        modified = [[row if row[4] else (row[0], (row[1] + 1) % env.single_action_space.n,
            row[2] + 1000, row[3], False) for row in episode] for episode in episodes]
        before = parameter_hash(model.policy)
        _, _, updates = sequence_epoch(model, episodes, [0, 1], 4, 2, 0., True, 1.)
        _, _, other_updates = sequence_epoch(other, modified, [0, 1], 4, 2, 0., True, 1.)
        assert updates == other_updates == 3
        assert parameter_hash(model.policy) == parameter_hash(other.policy) != before
        assert {int(state["step"].item()) for state in model.policy.optimizer.state.values()} == {3}
    finally:
        env.close()


def test_rehearsal_samples_only_teacher_suffix_and_burns_in_the_actual_prefix(monkeypatch, tmp_path):
    import soku_rl.rl.rehearsal as module
    torch.set_num_threads(1)
    env = fixture_env()
    try:
        model, _ = create_ppo(ObservationContractEnv(env.interface), env.interface, fixture_config("lstm"),
            {"kind": "fresh"}, "cpu", 17)
        packed = PackedObservation.pack(np.zeros(env.single_observation_space.shape, dtype=np.float32))
        samples = [(packed, 3, 0., -1 if index == 0 else 0, index >= 3) for index in range(6)]
        config = {"updates_per_rollout": 2, "sequences": 3, "sequence_length": 2, "learning_rate": .001, "seed": 13}
        model.rehearsal_state = {"rng": np.random.default_rng(13).bit_generator.state,
            "updates": 0, "frames": 0, "burn_in_frames": 0}
        # Match the regular training attachment of a logger.
        from stable_baselines3.common.logger import configure
        model.set_logger(configure(folder=str(tmp_path), format_strings=[]))
        original = module.window_distribution
        windows = []

        def checked(model, episode, offset, length):
            assert offset >= 3 and all(row[4] for row in episode[offset:offset + length])
            assert len(episode) == 6 and all(not row[4] for row in episode[:3])
            windows.append((offset, length))
            return original(model, episode, offset, length)

        monkeypatch.setattr(module, "window_distribution", checked)
        sampler = DemonstrationRehearsal(samples, config)
        sampler.update(model)
        assert len(windows) == 6
        assert model.rehearsal_state["burn_in_frames"] == sum(offset for offset, _ in windows)
        assert model.rehearsal_state["frames"] == sum(length for _, length in windows)
    finally:
        env.close()


@pytest.mark.parametrize("rows", [[(0, 0, 0)], [(0, 0, 0, -1, 1)], [(0, 0, 0, -1, None)]])
def test_invalid_supervision_flags_are_rejected(rows):
    with pytest.raises(ValueError, match="supervision"):
        supervision_mask(rows)


def test_no_supervised_frames_and_discontinuous_rehearsal_masks_fail():
    with pytest.raises(ValueError, match="supervised"):
        DemonstrationRehearsal([(None, 0, 0., -1, False)], {})
    with pytest.raises(ValueError, match="contiguous suffix"):
        DemonstrationRehearsal([(None, 0, 0., -1, True), (None, 0, 0., 0, False)], {})
