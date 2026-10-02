"""Storage changes must preserve complete recurrent batches and actual updates."""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pytest
import torch
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.buffers import RecurrentRolloutBuffer
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from stable_baselines3.common.vec_env import DummyVecEnv

from soku_rl.rl.recurrent_storage import SparseRecurrentRolloutBuffer, attach_sparse_recurrent_buffer
from soku_rl.rl.storage import PackedArray
from test_recurrent_state_diagnostics import MemoryEnv, assert_tree_equal


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_complete_recurrent_minibatches_preserve_words_masks_states_and_rng(device):
    if device.startswith('cuda') and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    arguments = (5, spaces.Box(-100, 100, (128,), np.float32), spaces.Discrete(3), (5, 2, 3, 4))
    buffers = [kind(*arguments, device=device, n_envs=3, gamma=1., gae_lambda=.95)
        for kind in (RecurrentRolloutBuffer, SparseRecurrentRolloutBuffer)]
    for step in range(5):
        observations = np.zeros((3, 128), np.float32)
        observations[0, step] = step + 1
        observations[1] = np.arange(128, dtype=np.float32) + step + 1
        observations[2].view(np.uint32)[[0, 50, 127]] = [0x80000000, 0x7fc01234, 1]
        state = torch.arange(24, dtype=torch.float32, device=device).reshape(2, 3, 4) + step
        for buffer in buffers:
            buffer.add(observations, np.array([0, 1, 2]), np.array([.1, -.2, .3]),
                np.array([step == 0, step in (0, 2), step in (0, 3)]),
                torch.tensor([.1, .2, .3]), torch.tensor([-.1, -.2, -.3]),
                lstm_states=RNNStates((state, state + 1), (state + 2, state + 3)))
    for buffer in buffers:
        buffer.compute_returns_and_advantage(torch.tensor([.3, .2, .1]), np.array([False, True, False]))
    for batch_size in (4, 7, 15):
        results, rng_states = [], []
        for buffer in buffers:
            np.random.seed(972)
            before = torch.get_rng_state()
            results.append(list(buffer.get(batch_size)))
            assert torch.equal(before, torch.get_rng_state())
            rng_states.append(np.random.get_state())
        assert rng_states[0][0] == rng_states[1][0]
        assert np.array_equal(rng_states[0][1], rng_states[1][1])
        assert rng_states[0][2:] == rng_states[1][2:]
        for expected, actual in zip(*results, strict=True):
            assert torch.equal(expected.observations.view(torch.int32), actual.observations.view(torch.int32))
            assert_tree_equal(expected[1:], actual[1:])
        assert any((row.mask == 0).any() for row in results[0])
    buffers[1].reset()
    assert isinstance(buffers[1].observations, PackedArray)
    assert buffers[1].observations.shape == (5, 3, 128) and buffers[1].pos == 0


class SparseMemoryEnv(MemoryEnv):
    def __init__(self, length):
        super().__init__(length)
        self.observation_space = spaces.Box(-2, 2, (512,), np.float32)

    def observation(self):
        result = np.zeros(512, np.float32)
        result[:4] = super().observation()
        result[100] = -0.
        return result


def storage_model(device):
    env = DummyVecEnv([lambda: SparseMemoryEnv(3), lambda: SparseMemoryEnv(7)])
    return RecurrentPPO('MlpLstmPolicy', env, device=device, seed=972, n_steps=4,
        batch_size=8, n_epochs=2, learning_rate=.003,
        policy_kwargs={'lstm_hidden_size': 8, 'net_arch': [8]})


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_actual_ppo_matches_dense_and_checkpoint_stays_portable(tmp_path, device):
    if device.startswith('cuda') and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    torch.set_num_threads(1)
    dense = storage_model(device)
    dense.learn(32)
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(device) if device.startswith('cuda') else []
    sparse = storage_model(device)
    attach_sparse_recurrent_buffer(sparse)
    sparse.learn(32)
    assert type(sparse) is RecurrentPPO and isinstance(sparse.rollout_buffer.observations, PackedArray)
    assert torch.equal(cpu_rng, torch.get_rng_state())
    if device.startswith('cuda'):
        assert torch.equal(cuda_rng, torch.cuda.get_rng_state(device))
    assert_tree_equal(dense.policy.state_dict(), sparse.policy.state_dict())
    assert_tree_equal(dense.policy.optimizer.state_dict(), sparse.policy.optimizer.state_dict())
    assert_tree_equal(dense._last_lstm_states, sparse._last_lstm_states)
    assert np.array_equal(dense._last_obs, sparse._last_obs)
    for a, b in zip(dense.env.envs, sparse.env.envs, strict=True):
        assert a.actions == b.actions
    checkpoint = tmp_path / 'model.zip'
    sparse.save(checkpoint)
    loaded = RecurrentPPO.load(checkpoint, device=device)
    assert type(loaded.rollout_buffer) is RecurrentRolloutBuffer
    assert_tree_equal(loaded.policy.state_dict(), sparse.policy.state_dict())
    assert_tree_equal(loaded.policy.optimizer.state_dict(), sparse.policy.optimizer.state_dict())
    attach_sparse_recurrent_buffer(loaded)
    with pytest.raises(ValueError, match='empty original'):
        attach_sparse_recurrent_buffer(loaded)
    dense.env.close()
    sparse.env.close()


def test_reject_nonempty_and_non_float_storage():
    model = storage_model('cpu')
    model.rollout_buffer.pos = 1
    with pytest.raises(ValueError, match='empty original'):
        attach_sparse_recurrent_buffer(model)
    model.env.close()
    with pytest.raises(ValueError, match='float32 Box'):
        SparseRecurrentRolloutBuffer(2, spaces.Box(0, 1, (4,), np.float64), spaces.Discrete(2),
            (2, 1, 1, 4), device='cpu')


@pytest.mark.parametrize('policy_type,mode,override', [('mlp', 'sparse', False),
    ('lstm', 'packed', False), ('lstm', None, False), ('lstm', 'sparse', True)])
def test_factory_rejects_incompatible_storage(policy_type, mode, override):
    from soku_rl.rl.ppo import create_ppo
    from test_policy_artifacts import interface
    contract = interface()
    env = gym.Env()
    env.observation_space, env.action_space = contract.observation_space, contract.action_space
    config = {'policy_type': policy_type, 'timeout_payoff': 'zero_at_horizon',
        'ppo': {'gamma': 1., 'policy_kwargs': {'net_arch': [8]}, 'recurrent_storage': mode}}
    if override:
        config['ppo']['rollout_buffer_class'] = 'soku_rl.rl.buffers.PackedRolloutBuffer'
    with pytest.raises(ValueError, match='recurrent_storage requires'):
        create_ppo(env, contract, config, {'kind': 'fresh'}, 'cpu', 73)
