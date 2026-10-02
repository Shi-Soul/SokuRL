import numpy as np
import pytest
import torch

from test_shared_ppo import fixture_config, fixture_env
from soku_rl.rl.anchor_diagnostics import reference_windows, score_memory_transfer, score_windows, select_windows
from soku_rl.rl.behavior_cloning import ObservationContractEnv
from soku_rl.rl.ppo import create_ppo, parameter_hash
from soku_rl.rl.storage import PackedObservation


def test_read_only_distances_and_memory_controls():
    torch.set_num_threads(1)
    env = fixture_env()
    reference, _ = create_ppo(ObservationContractEnv(env.interface), env.interface,
        fixture_config('lstm'), {'kind': 'fresh'}, 'cpu', 7)
    current, _ = create_ppo(ObservationContractEnv(env.interface), env.interface,
        fixture_config('lstm'), {'kind': 'fresh'}, 'cpu', 13)
    rng = np.random.default_rng(3)
    episodes = [[PackedObservation.pack(rng.normal(size=current.observation_space.shape).astype(np.float32))
        for _ in range(length)] for length in (7, 9)]
    windows = select_windows(episodes, [0, 3, 8], 3)
    assert windows == [(0, 0, 3), (0, 3, 3), (1, 0, 3), (1, 3, 3)]
    before = [parameter_hash(model.policy) for model in (reference, current)]
    targets = reference_windows(reference, episodes, windows)
    assert all(not value.requires_grad for value in targets)
    identical = score_windows(reference, episodes, windows, targets)
    assert identical['mean_kl'] == identical['mean_tv'] == 0
    assert identical['argmax_agreement'] == 1
    measured = score_windows(current, episodes, windows, targets)
    assert measured['mean_kl'] > 0 and measured['mean_tv'] > 0
    assert measured['frames'] == 12
    assert measured['mean_kl'] == pytest.approx(sum(sum(row['kl_by_frame']) for row in measured['windows']) / 12)
    memory = score_memory_transfer(current, reference, episodes, windows)
    assert all(not any(row['tv_by_frame']) for row in memory if row['offset'] == 0)
    assert any(any(row['tv_by_frame']) for row in memory if row['offset'] > 0)
    assert before == [parameter_hash(model.policy) for model in (reference, current)]
    assert all(parameter.grad is None for model in (reference, current) for parameter in model.policy.parameters())
    assert current.num_timesteps == reference.num_timesteps == 0
    env.close()


@pytest.mark.parametrize('episodes,offsets,length', [([], [0], 1), ([[]], [0], 1),
    ([[1]], [0, 0], 1), ([[1]], [-1], 1), ([[1]], [True], 1), ([[1]], [1], 1),
    ([[1]], [0], 0)])
def test_invalid_diagnostic_windows(episodes, offsets, length):
    with pytest.raises(ValueError):
        select_windows(episodes, offsets, length)
