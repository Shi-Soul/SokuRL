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
from soku_rl.rl.ppo import algorithm_type, create_ppo, parameter_hash
from soku_rl.rl.relational_features import RelationalCombatFeatures


def make_encoder(history):
    torch.set_num_threads(1)
    torch.manual_seed(23)
    return RelationalCombatFeatures(spaces.Box(-np.inf, np.inf,
        (history * PRIVILEGED_FEATURES + 8,), np.float32), history, 4, 16, 8, 2, 2)


def set_field(values, seat, field, value):
    index = (len(WORLD_NAMES) + seat * PLAYER_WIDTH + FIGHTER_NAMES.index(field)) * 2
    values[..., index:index + 2] = torch.from_numpy(encode_values(np.array([value]))[0])


def test_all_object_slots_have_gradients_and_padding_is_ignored():
    encoder = make_encoder(1)
    values = torch.zeros(1, PRIVILEGED_FEATURES + 8)
    for seat in (0, 1):
        set_field(values, seat, 'obj_n', MAX_OBJECTS)
        set_field(values, seat, 'dir', 1)
    values.requires_grad_()
    encoder(values).sum().backward()
    for seat in (0, 1):
        start = (len(WORLD_NAMES) + seat * PLAYER_WIDTH + FIGHTER_WIDTH) * 2
        gradients = values.grad[0, start:start + MAX_OBJECTS * OBJECT_WIDTH * 2].reshape(MAX_OBJECTS, -1)
        assert torch.all(gradients.abs().sum(1) > 0)
        address = FIGHTER_NAMES.index('address') * 2
        assert torch.count_nonzero(gradients[:, address:address + 2]) == 0
    assert torch.all(values.grad[0, -8:].abs() > 0)
    empty = torch.zeros_like(values)
    changed = empty.clone()
    for seat in (0, 1):
        start = (len(WORLD_NAMES) + seat * PLAYER_WIDTH + FIGHTER_WIDTH) * 2
        changed[:, start:start + MAX_OBJECTS * OBJECT_WIDTH * 2] = 123.
    assert torch.equal(encoder(empty), encoder(changed))
    encoder.zero_grad(set_to_none=True)
    encoder(empty).sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in encoder.parameters() if p.grad is not None)


def test_relative_geometry_uses_observer_for_both_owners_and_respects_facing():
    encoder = make_encoder(1)
    fighters = torch.zeros(2, FIGHTER_WIDTH * 2)
    objects = torch.zeros(2, MAX_OBJECTS, OBJECT_WIDTH * 2)
    for field, value in {'x': 300., 'y': 20., 'xspeed': 2., 'yspeed': 1., 'dir': -1.}.items():
        index = FIGHTER_NAMES.index(field) * 2
        fighters[0, index:index + 2] = torch.from_numpy(encode_values(np.array([value]))[0])
    # The opponent fighter is elsewhere; both object lists must still use observer coordinates.
    fighters[1, :2] = torch.from_numpy(encode_values(np.array([1100.]))[0])
    for field, value in {'x': 428., 'y': 148., 'xspeed': 4., 'yspeed': -1.}.items():
        index = FIGHTER_NAMES.index(field) * 2
        objects[:, 0, index:index + 2] = torch.from_numpy(encode_values(np.array([value]))[0])
    result = encoder.relative_object_geometry(objects, fighters)
    torch.testing.assert_close(result[:, 0], torch.tensor([[-.1, .1, -.1, -.1]]).expand(2, -1))


def test_relocation_invariance_and_order_are_both_preserved():
    encoder = make_encoder(1)
    values = torch.zeros(1, PRIVILEGED_FEATURES + 8)
    set_field(values, 0, 'obj_n', 2)
    set_field(values, 0, 'dir', 1)
    start = (len(WORLD_NAMES) + FIGHTER_WIDTH) * 2
    values[0, start:start + 2] = torch.from_numpy(encode_values(np.array([100.]))[0])
    values[0, start + OBJECT_WIDTH * 2:start + OBJECT_WIDTH * 2 + 2] = torch.from_numpy(encode_values(np.array([500.]))[0])
    before = values.clone()
    relocated = values.clone()
    address = FIGHTER_NAMES.index('address') * 2
    for seat in (0, 1):
        base = (len(WORLD_NAMES) + seat * PLAYER_WIDTH) * 2
        relocated[0, base + address:base + address + 2] = torch.tensor([.5, .25])
    for obj in (0, 1):
        index = start + obj * OBJECT_WIDTH * 2 + address
        relocated[0, index:index + 2] = torch.tensor([.125, .875])
    assert torch.equal(encoder(values), encoder(relocated))
    swapped = values.clone()
    swapped[:, start:start + 4 * OBJECT_WIDTH] = values[:, start:start + 4 * OBJECT_WIDTH].reshape(1, 2, -1).flip(1).flatten(1)
    assert not torch.equal(encoder(values), encoder(swapped))
    assert torch.equal(values, before)


@pytest.mark.parametrize('history', [1, 2])
def test_history_and_state_dict_roundtrip(tmp_path, history):
    encoder = make_encoder(history)
    values = torch.zeros(2, history * PRIVILEGED_FEATURES + 8)
    output = encoder(values)
    assert output.shape == (2, 8) and torch.isfinite(output).all()
    path = tmp_path / 'features.pt'
    torch.save(encoder.state_dict(), path)
    other = make_encoder(history)
    other.load_state_dict(torch.load(path, weights_only=True), strict=True)
    torch.testing.assert_close(output, other(values), rtol=0, atol=0)


@pytest.mark.parametrize('heads,queries', [(0, 2), (3, 2), (True, 2), (2, 0), (2, 1.5)])
def test_invalid_attention_dimensions_fail(heads, queries):
    with pytest.raises(ValueError, match='attention_heads'):
        RelationalCombatFeatures(spaces.Box(-np.inf, np.inf, (PRIVILEGED_FEATURES,), np.float32),
                                 1, 4, 16, 8, heads, queries)


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
def test_shared_ppo_updates_and_checkpoint_roundtrip(tmp_path, policy_type):
    torch.set_num_threads(1)
    with initialize_config_dir(config_dir=str(Path(__file__).parents[1] / 'config'), version_base='1.3'):
        offline = compose(config_name='pretrain_recurrent_relational_demonstrations')
        online = compose(config_name='train', overrides=['algorithm=br', 'rl=recurrent_demonstration_transfer',
            'track=superhuman_relational', 'wrappers=superhuman_learning'])
        assert OmegaConf.to_container(offline.rl.ppo.policy_kwargs) == OmegaConf.to_container(online.rl.ppo.policy_kwargs)
        cfg = OmegaConf.to_container(online, resolve=True)
    interface = LearningInterface(EpisodeConfig.from_dict(cfg['episode']), LearningConfig(**cfg['wrappers']))
    settings = cfg['rl']
    settings['policy_type'] = policy_type
    settings['ppo'].update(n_steps=4, batch_size=4, n_epochs=1)
    arch = settings['ppo']['policy_kwargs']
    arch.update(net_arch=[8], features_extractor_kwargs={'history_frames': 1, 'object_features': 4,
        'player_features': 8, 'features_dim': 16, 'attention_heads': 2, 'attention_queries': 2})
    for field in ('lstm_hidden_size', 'n_lstm_layers', 'shared_lstm', 'enable_critic_lstm'):
        if policy_type == 'mlp':
            arch.pop(field)
    if policy_type == 'lstm':
        arch['lstm_hidden_size'] = 8
    env = ContractProbe(interface)
    model, _ = create_ppo(env, interface, settings, {'kind': 'fresh'}, 'cpu', 17)
    before = parameter_hash(model.policy)
    model.learn(8)
    assert model.num_timesteps == 8 and parameter_hash(model.policy) != before
    path = tmp_path / 'policy.zip'
    model.save(path)
    restored = algorithm_type(policy_type).load(path, device='cpu')
    assert parameter_hash(restored.policy) == parameter_hash(model.policy)
    assert restored.action_space.n == 576
