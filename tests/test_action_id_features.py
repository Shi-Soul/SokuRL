"""Categorical action IDs must preserve the raw contract and work in shared PPO."""
from pathlib import Path

from gymnasium import Env, spaces
from hydra import compose, initialize_config_dir
import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

from soku_rl.env import EpisodeConfig
from soku_rl.env.observation.memory_schema import (
    FIGHTER_NAMES, FIGHTER_WIDTH, MAX_OBJECTS, OBJECT_WIDTH, PLAYER_WIDTH,
    PRIVILEGED_FEATURES, WORLD_NAMES)
from soku_rl.env.observation.privileged import encode_values
from soku_rl.env.wrappers.learning import LearningConfig, LearningInterface
from soku_rl.rl.action_id_features import ActionIdCombatFeatures
from soku_rl.rl.address_invariant_features import AddressInvariantCombatFeatures
from soku_rl.rl.ppo import algorithm_type, create_ppo, parameter_hash


def make_encoder(history):
    torch.set_num_threads(1)
    torch.manual_seed(31)
    return ActionIdCombatFeatures(spaces.Box(-np.inf, np.inf,
        (history * PRIVILEGED_FEATURES + 8,), np.float32), history, 4, 16, 8, 4)


@pytest.mark.parametrize('width', [FIGHTER_WIDTH, OBJECT_WIDTH])
def test_full_uint16_ids_and_original_numeric_channels(width):
    encoder = make_encoder(1)
    raw = np.zeros((3, width))
    raw[:, FIGHTER_NAMES.index('act')] = [0, 300, 65535]
    raw[:, FIGHTER_NAMES.index('address')] = [0x10000000, 0x20000000, 0xF0000123]
    records = torch.from_numpy(encode_values(raw.reshape(-1)).reshape(3, -1))
    before = records.clone()
    actual = encoder.numeric_features(records)
    expected = AddressInvariantCombatFeatures.numeric_features(encoder, records)
    assert torch.equal(actual[:, :3 * width], expected)
    assert torch.equal(actual[:, 3 * width:], encoder.action_embedding.weight[[0, 300, 65535]])
    assert not torch.equal(actual[0, 3 * width:], actual[2, 3 * width:])
    actual.sum().backward()
    nonzero = encoder.action_embedding.weight.grad.abs().sum(1).nonzero().flatten().tolist()
    assert nonzero == [0, 300, 65535]
    assert torch.equal(records, before)
    assert encoder.numeric_features(records[:0]).shape == (0, 3 * width + 4)
    world = torch.zeros(2, len(WORLD_NAMES) * 2)
    assert encoder.numeric_features(world).shape == (2, len(WORLD_NAMES) * 3)


@pytest.mark.parametrize('history', [1, 2])
def test_all_slots_relocation_padding_and_state_roundtrip(history, tmp_path):
    encoder = make_encoder(history)
    values = torch.zeros(1, history * PRIVILEGED_FEATURES + 8)
    pointers, objects = [], []
    for frame in range(history):
        for seat in (0, 1):
            base = frame * PRIVILEGED_FEATURES + (len(WORLD_NAMES) + seat * PLAYER_WIDTH) * 2
            count = base + FIGHTER_NAMES.index('obj_n') * 2
            values[0, count + 1] = MAX_OBJECTS / 65536.
            pointers.append(base + FIGHTER_NAMES.index('address') * 2)
            start = base + FIGHTER_WIDTH * 2
            objects.append(start)
            pointers.extend(start + i * OBJECT_WIDTH * 2 + FIGHTER_NAMES.index('address') * 2
                            for i in range(MAX_OBJECTS))
    values.requires_grad_()
    output = encoder(values)
    output.sum().backward()
    for start in objects:
        gradient = values.grad[0, start:start + MAX_OBJECTS * OBJECT_WIDTH * 2].reshape(MAX_OBJECTS, -1)
        assert torch.all(gradient.abs().sum(1) > 0)
    assert torch.count_nonzero(values.grad[0, pointers]) == 0
    assert torch.count_nonzero(values.grad[0, [p + 1 for p in pointers]]) == 0
    relocated = values.detach().clone()
    relocated[0, pointers] = .125
    relocated[0, [p + 1 for p in pointers]] = .75
    assert torch.equal(output, encoder(relocated))
    padding = torch.zeros_like(values)
    changed = padding.clone()
    for start in objects:
        changed[:, start:start + MAX_OBJECTS * OBJECT_WIDTH * 2] = 123.
    assert torch.equal(encoder(padding), encoder(changed))
    encoder.zero_grad(set_to_none=True)
    encoder(padding).sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in encoder.parameters() if p.grad is not None)
    path = tmp_path / 'encoder.pt'
    torch.save(encoder.state_dict(), path)
    other = make_encoder(history)
    other.load_state_dict(torch.load(path, weights_only=True))
    assert torch.equal(encoder(values), other(values))


@pytest.mark.parametrize('dimension', [0, -1, True, 1.5])
def test_bad_dimensions_fail(dimension):
    with pytest.raises(ValueError, match='action_embedding_dim'):
        ActionIdCombatFeatures(spaces.Box(-np.inf, np.inf, (PRIVILEGED_FEATURES,), np.float32),
                               1, 4, 16, 8, dimension)


class ContractProbe(Env):
    def __init__(self, interface):
        self.observation_space, self.action_space = interface.observation_space, interface.action_space
        self.observation = np.zeros(self.observation_space.shape, np.float32)
        self.steps = 0

    def reset(self, **kwargs):
        self.steps = 0
        return self.observation.copy(), {}

    def step(self, action):
        self.steps += 1
        return self.observation.copy(), float(action == 256), self.steps == 3, False, {}


@pytest.mark.parametrize('policy_type', ['mlp', 'lstm'])
def test_shared_ppo_updates_embedding_and_restores_checkpoint(tmp_path, policy_type):
    torch.set_num_threads(1)
    with initialize_config_dir(config_dir=str(Path(__file__).parents[1] / 'config'), version_base='1.3'):
        offline = compose(config_name='pretrain_recurrent_action_id_demonstrations')
        online = compose(config_name='train', overrides=['algorithm=br', 'rl=recurrent_demonstration_transfer',
            'track=superhuman_action_id', 'wrappers=superhuman_learning'])
        assert OmegaConf.to_container(offline.rl.ppo.policy_kwargs) == OmegaConf.to_container(online.rl.ppo.policy_kwargs)
        cfg = OmegaConf.to_container(online, resolve=True)
    interface = LearningInterface(EpisodeConfig.from_dict(cfg['episode']), LearningConfig(**cfg['wrappers']))
    settings = cfg['rl']
    settings['policy_type'] = policy_type
    settings['ppo'].update(n_steps=4, batch_size=4, n_epochs=1)
    arch = settings['ppo']['policy_kwargs']
    arch.update(net_arch=[8], features_extractor_kwargs={'history_frames': 1, 'object_features': 4,
        'player_features': 8, 'features_dim': 16, 'action_embedding_dim': 4})
    if policy_type == 'mlp':
        for field in ('lstm_hidden_size', 'n_lstm_layers', 'shared_lstm', 'enable_critic_lstm'):
            arch.pop(field)
    else:
        arch['lstm_hidden_size'] = 8
    model, _ = create_ppo(ContractProbe(interface), interface, settings, {'kind': 'fresh'}, 'cpu', 17)
    before = model.policy.features_extractor.action_embedding.weight.detach().clone()
    model.learn(8)
    assert model.num_timesteps == 8
    assert not torch.equal(before, model.policy.features_extractor.action_embedding.weight)
    path = tmp_path / 'policy.zip'
    model.save(path)
    restored = algorithm_type(policy_type).load(path, device='cpu')
    assert parameter_hash(restored.policy) == parameter_hash(model.policy)
    assert restored.action_space.n == 576
