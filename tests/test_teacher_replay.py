"""Mixed teacher windows preserve held-out boundaries and the shared PPO contract."""
import copy
import random

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch
from stable_baselines3.common.logger import configure

from test_behavior_cloning import dataset
from test_demonstration_sets import pair
from test_online_teacher import teacher, settings, RecordingActor
from test_shared_ppo import fixture_config, fixture_env, save_contract
from soku_rl.rl.behavior_cloning import ObservationContractEnv
from soku_rl.rl.online_teacher import validate_teacher
from soku_rl.rl.ppo import algorithm_type, create_ppo, parameter_hash
from soku_rl.rl.recurrent_cloning import zero_states
from soku_rl.rl.teacher_replay import load_teacher_replay


def replay_settings(path):
    return settings() | {'sequences': 4, 'demonstration_replay': {
        'datasets': [str(path)], 'sequences': 2, 'seed': 419}}


@pytest.fixture
def matching_teacher(teacher, monkeypatch):
    teacher.fingerprint = 'constant:3'
    monkeypatch.setattr(RecordingActor, 'act', lambda self, observation: 3)
    return teacher


@pytest.mark.parametrize('kind', ['mlp', 'lstm'])
def test_shared_br_mixed_windows_resume_and_inference(pair, matching_teacher, kind):
    from soku_rl.marl.br import OpponentEntry
    from soku_rl.rl.opponent_env import OpponentMixtureVecEnv
    from test_demonstrations import ConstantPolicy
    first, _, interface = pair
    torch.set_num_threads(1)
    env = fixture_env()
    view = OpponentMixtureVecEnv(env, 'balanced', [OpponentEntry('constant', ConstantPolicy(8))], [1.], 17)
    config = fixture_config(kind) | {'name': 'br', 'online_teacher': replay_settings(first)}
    model, _ = create_ppo(view, interface, config, {'kind': 'fresh'}, 'cpu', 7)
    assert model._online_teacher.replay.ends[-1] == 6
    assert len(model._online_teacher.replay.episodes) == 2
    model.learn(16)
    state = model.teacher_state
    assert model.num_timesteps == state['collected_frames'] == 16
    assert state['updates'] == state['demonstration_replay']['updates'] == 4
    assert 0 < state['demonstration_replay']['frames'] < state['frames'] <= 4 * 4 * 2
    assert sum(state['label_counts']) == 16  # Replay labels are not new game queries.
    path = first / 'mixed.zip'
    model.save(path)
    inference = algorithm_type(kind).load(path, device='cpu')
    assert not hasattr(inference, '_online_teacher')
    assert parameter_hash(inference.policy) == parameter_hash(model.policy)
    (first / 'mixed-contract').mkdir()
    contract = save_contract(first / 'mixed-contract', env, config)
    raw = OmegaConf.load(contract)
    raw.rl.online_teacher = config['online_teacher']
    OmegaConf.save(raw, contract)
    source = {'kind': 'checkpoint', 'path': str(path), 'training_config': contract}
    resumed, _ = create_ppo(view, interface, config, source, 'cpu', 19)
    assert resumed.teacher_state == state and not resumed._online_teacher.recent
    rngs = [np.random.default_rng(), np.random.default_rng()]
    for rng in rngs:
        rng.bit_generator.state = state['demonstration_replay']['rng']
    windows = [learner._online_teacher.replay.sample(rng, 12, 2, 'cpu')
               for learner, rng in zip((model, resumed), rngs, strict=True)]
    for a, b in zip(*windows, strict=True):
        assert a[1:3] == b[1:3] and torch.equal(a[3], b[3])
        assert a[4] == b[4] == 'replay'
        for x, y in zip(a[0], b[0], strict=True):
            np.testing.assert_array_equal(x.unpack(), y.unpack())
    resumed.learn(8, reset_num_timesteps=False)
    assert resumed.num_timesteps == resumed.teacher_state['collected_frames'] == 24
    fresh, _ = create_ppo(view, interface, config, source | {'kind': 'weights'}, 'cpu', 19)
    assert fresh.teacher_state['demonstration_replay']['updates'] == 0
    assert not fresh.policy.optimizer.state and not fresh._online_teacher.optimizer.state
    changed = copy.deepcopy(config)
    changed['online_teacher']['demonstration_replay']['seed'] += 1
    with pytest.raises(ValueError, match='online_teacher'):
        create_ppo(view, interface, changed, source, 'cpu', 19)
    matching_teacher.fingerprint = 'changed'
    with pytest.raises(ValueError, match='same rule teacher'):
        create_ppo(view, interface, config, source, 'cpu', 19)
    view.close()
    env.close()


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_mixed_loss_accounting_rng_and_optimizer_isolation(pair, matching_teacher, device):
    if device == 'cuda:0' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    first, _, interface = pair
    torch.set_num_threads(1)
    config = fixture_config('lstm') | {'online_teacher': replay_settings(first)}
    config['ppo']['policy_kwargs'].update(share_features_extractor=False,
        features_extractor_class='test_recurrent_separate_features.TrainableFeatures')
    model, _ = create_ppo(ObservationContractEnv(interface), interface, config, {'kind': 'fresh'}, device, 13)
    model.set_logger(configure(folder=None, format_strings=[]))
    obs = torch.ones((1, *model.observation_space.shape), device=device)
    _, values, _, _ = model.policy(obs, zero_states(model.policy, 1), torch.ones(1, device=device))
    values.square().mean().backward()
    model.policy.optimizer.step()
    critics = [model.policy.vf_features_extractor, model.policy.lstm_critic, model.policy.value_net]
    hashes = list(map(parameter_hash, critics))
    adam = copy.deepcopy(model.policy.optimizer.state_dict())
    for frame in range(3):
        model._online_teacher.record(obs.cpu().numpy(), [frame == 0], [1])
    py_rng, np_rng, cpu_rng = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone() if device == 'cuda:0' else None
    teacher_rng = copy.deepcopy(model.teacher_state['teacher_rng'])
    metrics = model._online_teacher.update()
    assert metrics['frames'] == metrics['online_frames'] + metrics['replay_frames'] <= 16
    assert metrics['online_frames'] > 0 and metrics['replay_frames'] > 0
    assert metrics['nll_after'] < metrics['nll_before']
    for key in ('nll_before', 'nll_after', 'accuracy_before', 'accuracy_after'):
        weighted = sum(metrics[f'{source}_{key}'] * metrics[f'{source}_frames'] for source in ('online', 'replay'))
        assert weighted / metrics['frames'] == pytest.approx(metrics[key], abs=1e-7)
    assert metrics['current_burn_in_frames'] == metrics['online_burn_in_frames'] + metrics['replay_burn_in_frames']
    assert model.teacher_state['demonstration_replay']['frames'] == metrics['replay_frames']
    assert model.teacher_state['teacher_rng'] == teacher_rng
    assert hashes == list(map(parameter_hash, critics))
    actual = model.policy.optimizer.state_dict()
    assert actual['param_groups'] == adam['param_groups']
    for key, states in adam['state'].items():
        for name, value in states.items():
            assert torch.equal(actual['state'][key][name], value)
    assert py_rng == random.getstate() and np_rng[0] == np.random.get_state()[0]
    np.testing.assert_array_equal(np_rng[1], np.random.get_state()[1])
    assert np_rng[2:] == np.random.get_state()[2:] and torch.equal(cpu_rng, torch.get_rng_state())
    if device == 'cuda:0':
        assert torch.equal(cuda_rng, torch.cuda.get_rng_state())


def test_replay_rejects_other_teacher_or_learner_controlled_data(pair):
    first, second, interface = pair
    config = replay_settings(first)['demonstration_replay']
    with pytest.raises(ValueError, match='same rule teacher'):
        load_teacher_replay(config, interface, 'other')
    with pytest.raises(ValueError, match='original trajectories'):
        load_teacher_replay(config | {'datasets': [str(second)]}, interface, 'constant:3')


@pytest.mark.parametrize('kind', ['mlp', 'lstm'])
def test_joint_marl_learners_receive_the_same_mixed_update(pair, matching_teacher, kind):
    from soku_rl.marl.ippo import train_ippo
    first, _, _ = pair
    torch.set_num_threads(1)
    env = fixture_env()
    config = fixture_config(kind) | {'name': 'ippo', 'online_teacher': replay_settings(first)}
    output = first / 'mixed-ippo'
    output.mkdir()
    report = train_ippo(env, config, 'cpu', 13, output)
    assert report['updates'] == 1
    for seat in (0, 1):
        model = algorithm_type(kind).load(output / f'player_{seat}/final.zip', device='cpu')
        assert model.teacher_state['updates'] == model.teacher_state['demonstration_replay']['updates'] == 2
        assert model.teacher_state['collected_frames'] == model.num_timesteps == 8
        assert model.teacher_state['demonstration_replay']['frames'] > 0
    env.close()


@pytest.mark.parametrize('key,value', [('sequences', 0), ('sequences', 4), ('sequences', True),
    ('datasets', []), ('datasets', ['']), ('seed', -1), ('seed', True)])
def test_invalid_mixing_budget_and_sources(key, value):
    config = replay_settings('fixture')
    config['demonstration_replay'][key] = value
    with pytest.raises(ValueError, match='teacher replay'):
        validate_teacher(config)
