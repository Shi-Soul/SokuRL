"""Online rule labels stay private, chronological and separate from executed actions."""
import copy
from pathlib import Path
import random

from hydra import compose, initialize_config_dir
import numpy as np
from omegaconf import OmegaConf
import pytest
import torch
from stable_baselines3.common.logger import configure

from test_shared_ppo import fixture_config, fixture_env, save_contract
from soku_rl.rl import ppo_settings
from soku_rl.rl.behavior_cloning import ObservationContractEnv
from soku_rl.rl.online_teacher import attach_teacher, validate_teacher
from soku_rl.rl.ppo import algorithm_type, create_ppo, parameter_hash
from soku_rl.rl.recurrent_cloning import zero_states


def settings():
    return {'teacher': {'kind': 'rule', 'name': 'diagnostic', 'rules': {}},
        'updates_per_rollout': 2, 'sequences': 2, 'sequence_length': 2, 'learning_rate': .01, 'seed': 23}


class RecordingTeacher:
    fingerprint = 'recording-teacher-v1'

    def __init__(self):
        self.actors = []

    def spawn(self, seed):
        actor = RecordingActor(seed)
        self.actors.append(actor)
        return actor


class RecordingActor:
    def __init__(self, seed):
        self.seed, self.observations = seed, []

    def act(self, observation):
        assert not observation.flags.writeable
        self.observations.append(observation.copy())
        return 0


@pytest.fixture
def teacher(monkeypatch):
    value = RecordingTeacher()
    monkeypatch.setattr('soku_rl.rl.online_teacher.load_policy', lambda *args: value)
    return value


@pytest.mark.parametrize('algorithm', ['ppo', 'br', 'ippo', 'nfsp', 'psro'])
def test_shared_online_teacher_configuration(algorithm):
    with initialize_config_dir(config_dir=str(Path(__file__).parents[1] / 'config'), version_base='1.3'):
        config = OmegaConf.to_container(compose(config_name='train', overrides=[
            f'algorithm={algorithm}', 'rl=recurrent_online_teacher']), resolve=True)
    shared = ppo_settings(config)
    learner = config['algorithm']['response'] if algorithm == 'psro' else config['algorithm']
    assert learner['online_teacher'] == shared['online_teacher']
    learner['online_teacher'] = dict(shared['online_teacher'], seed=0)
    with pytest.raises(ValueError, match='online_teacher'):
        ppo_settings(config)


@pytest.mark.parametrize('kind', ['mlp', 'lstm'])
def test_br_update_resume_and_teacher_free_inference(tmp_path, teacher, kind):
    from soku_rl.marl.br import OpponentEntry
    from soku_rl.rl.opponent_env import OpponentMixtureVecEnv
    from test_demonstrations import ConstantPolicy
    torch.set_num_threads(1)
    env = fixture_env()
    config = fixture_config(kind) | {'name': 'br', 'online_teacher': settings()}
    view = OpponentMixtureVecEnv(env, 'balanced', [OpponentEntry('constant', ConstantPolicy(8))], [1.], 17)
    model, _ = create_ppo(view, env.interface, config, {'kind': 'fresh'}, 'cpu', 7)
    model.learn(16)
    assert model.num_timesteps == 16 and model._n_updates == 2
    state = model.teacher_state
    assert state['updates'] == 4 and state['collected_frames'] == sum(state['label_counts']) == 16
    assert state['teacher_episodes'] == len(teacher.actors) == 6
    assert sum(row['frames'] for row in state['episodes']) == 16
    assert [row['teacher_seed'] for row in state['episodes']] == [actor.seed for actor in teacher.actors]
    assert state['collection_seconds'] >= state['query_seconds'] >= 0
    assert sorted(len(actor.observations) for actor in teacher.actors) == [2, 2, 3, 3, 3, 3]
    assert state['last_update']['rollout_frames'] == 8
    assert 0 <= state['behavior_matches'] < state['collected_frames']
    assert state['frames'] > 0
    path = tmp_path / 'teacher.zip'
    model.save(path)
    inference = algorithm_type(kind).load(path, device='cpu')
    assert not hasattr(inference, '_online_teacher')
    assert parameter_hash(inference.policy) == parameter_hash(model.policy)
    observation = view.reset()
    actions, _ = inference.predict(observation, episode_start=np.ones(2, dtype=bool))
    assert actions.shape == (2,)
    contract = save_contract(tmp_path, env, config)
    data = OmegaConf.load(contract)
    data.rl.online_teacher = config['online_teacher']
    OmegaConf.save(data, contract)
    source = {'kind': 'checkpoint', 'path': str(path), 'training_config': contract}
    resumed, _ = create_ppo(view, env.interface, config, source, 'cpu', 19)
    assert resumed.teacher_state == state
    assert not resumed._online_teacher.recent and all(x is None for x in resumed._online_teacher.active)
    resumed.learn(8, reset_num_timesteps=False)
    assert resumed.num_timesteps == resumed.teacher_state['collected_frames'] == 24
    fresh, _ = create_ppo(view, env.interface, config, source | {'kind': 'weights'}, 'cpu', 19)
    assert fresh.num_timesteps == fresh.teacher_state['updates'] == 0
    assert not fresh.policy.optimizer.state
    changed = copy.deepcopy(config)
    changed['online_teacher']['seed'] += 1
    with pytest.raises(ValueError, match='online_teacher'):
        create_ppo(view, env.interface, changed, source, 'cpu', 19)
    teacher.fingerprint = 'changed-rule-bytes'
    with pytest.raises(ValueError, match='teacher identity'):
        create_ppo(view, env.interface, config, source, 'cpu', 19)
    view.close()
    env.close()


@pytest.mark.parametrize('kind', ['mlp', 'lstm'])
def test_joint_collection_queries_each_learners_observation(tmp_path, teacher, kind):
    from soku_rl.marl.ippo import train_ippo
    env = fixture_env()
    config = fixture_config(kind) | {'name': 'ippo', 'online_teacher': settings()}
    report = train_ippo(env, config, 'cpu', 13, tmp_path)
    assert report['updates'] == 1
    for player in (0, 1):
        model = algorithm_type(kind).load(tmp_path / f'player_{player}/final.zip', device='cpu')
        assert model.teacher_state['updates'] == 2
        assert model.teacher_state['collected_frames'] == model.num_timesteps == 8
    assert sum(len(actor.observations) for actor in teacher.actors) == 16
    env.close()


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_auxiliary_ce_preserves_ppo_adam_private_critic_and_global_rng(teacher, device):
    if device == 'cuda:0' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    torch.set_num_threads(1)
    env = fixture_env()
    config = fixture_config('lstm')
    config['ppo']['policy_kwargs'].update(share_features_extractor=False,
        features_extractor_class='test_recurrent_separate_features.TrainableFeatures')
    model, _ = create_ppo(ObservationContractEnv(env.interface), env.interface,
        config, {'kind': 'fresh'}, device, 13)
    model.set_logger(configure(folder=None, format_strings=[]))
    python_rng, numpy_rng, cpu_rng = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone() if device == 'cuda:0' else None
    attach_teacher(model, env.interface, settings(), False)
    with pytest.raises(ValueError, match='already attached'):
        attach_teacher(model, env.interface, settings(), False)
    observations = torch.ones((1, *env.single_observation_space.shape), device=device)
    _, values, _, _ = model.policy(observations, zero_states(model.policy, 1),
        torch.ones(1, device=device), deterministic=True)
    values.square().mean().backward()
    model.policy.optimizer.step()
    critic = [model.policy.vf_features_extractor, model.policy.lstm_critic, model.policy.value_net]
    before = list(map(parameter_hash, critic))
    optimizer = copy.deepcopy(model.policy.optimizer.state_dict())
    for frame in range(3):
        model._online_teacher.record(observations.cpu().numpy(), [frame == 0], [1])
    seed_state = copy.deepcopy(model.teacher_state['teacher_rng'])
    metrics = model._online_teacher.update()
    assert metrics['nll_after'] < metrics['nll_before']
    assert metrics['behavior_agreement'] == 0
    assert model.teacher_state['teacher_rng'] == seed_state
    assert list(map(parameter_hash, critic)) == before
    actual = model.policy.optimizer.state_dict()
    assert actual['param_groups'] == optimizer['param_groups']
    for key, states in optimizer['state'].items():
        for name, value in states.items():
            assert torch.equal(actual['state'][key][name], value)
    assert python_rng == random.getstate()
    assert numpy_rng[0] == np.random.get_state()[0]
    assert np.array_equal(numpy_rng[1], np.random.get_state()[1])
    assert numpy_rng[2:] == np.random.get_state()[2:]
    assert torch.equal(cpu_rng, torch.get_rng_state())
    if device == 'cuda:0':
        assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
    assert model.num_timesteps == model._n_updates == 0
    env.close()


def test_teacher_and_complete_prefix_survive_rollout_but_reset_at_new_episode(teacher):
    env = fixture_env()
    config = fixture_config('lstm') | {'online_teacher': settings()}
    model, _ = create_ppo(ObservationContractEnv(env.interface), env.interface,
        config, {'kind': 'fresh'}, 'cpu', 7)
    helper = model._online_teacher
    obs = np.zeros((1, *model.observation_space.shape), dtype=np.float32)
    with pytest.raises(RuntimeError, match='episode start'):
        helper.record(obs, [False], [1])
    helper.record(obs, [True], [1])
    episode, actor = helper.active[0], helper.actors[0]
    helper.reset()
    helper.record(obs + 1, [False], [2])
    assert helper.active[0] is episode and helper.actors[0] is actor
    assert len(actor.observations) == 2 and helper.recent[0][1] == 1
    helper.record(obs + 2, [True], [3])
    assert helper.active[0] is not episode and helper.actors[0] is not actor
    assert helper.recent[0][0] is episode and helper.recent[1][0] is helper.active[0]
    assert episode['labels'] == [0, 0]
    helper.record(obs, [False], [1])
    helper.record(obs, [False], [1])
    with pytest.raises(RuntimeError, match='horizon'):
        helper.record(obs, [False], [1])
    np.testing.assert_array_equal(obs, np.zeros_like(obs))
    env.close()


@pytest.mark.parametrize('key,value', [('seed', -1), ('sequences', True), ('sequence_length', 0),
    ('updates_per_rollout', 0), ('learning_rate', float('nan')), ('learning_rate', True),
    ('teacher', {'kind': 'uniform'})])
def test_invalid_teacher_settings(key, value):
    config = settings()
    config[key] = value
    with pytest.raises(ValueError):
        validate_teacher(config)


@pytest.mark.parametrize('other', ['online_anchor', 'rehearsal'])
def test_auxiliary_objectives_cannot_silently_combine(teacher, other):
    env = fixture_env()
    config = fixture_config('lstm') | {'online_teacher': settings(), other: {}}
    with pytest.raises(ValueError, match='combining auxiliary'):
        create_ppo(ObservationContractEnv(env.interface), env.interface, config, {'kind': 'fresh'}, 'cpu', 1)
    env.close()


@pytest.mark.parametrize('label', [-1, 9999, True, 0.5])
def test_invalid_rule_label_is_rejected_before_ppo_buffer_write(monkeypatch, teacher, label):
    env = fixture_env()
    config = fixture_config('mlp') | {'online_teacher': settings()}
    model, _ = create_ppo(ObservationContractEnv(env.interface), env.interface,
        config, {'kind': 'fresh'}, 'cpu', 7)
    monkeypatch.setattr(RecordingActor, 'act', lambda self, observation: label)
    with pytest.raises(ValueError, match='outside the learning vocabulary'):
        model._online_teacher.record(np.zeros((1, *model.observation_space.shape), np.float32), [True], [0])
    assert model.rollout_buffer.pos == model.teacher_state['collected_frames'] == 0
    env.close()


def test_original_god_labels_match_independent_actor_across_rollouts_and_characters():
    from dataclasses import replace
    from test_god_scripts import NAMES, observation
    from soku_rl.env.observation.privileged import encode_privileged
    from soku_rl.env.wrappers.learning import LearningConfig, LearningInterface
    from soku_rl.policy.loader import load_policy
    if not NAMES:
        pytest.skip('external original community package is not installed')
    env = fixture_env()
    interface = LearningInterface(replace(env.interface.episode, max_frames=20,
        observation_mode='privileged_state', history_frames=1), LearningConfig('full', False, 8, 1.))
    with initialize_config_dir(config_dir=str(Path(__file__).parents[1] / 'config'), version_base='1.3'):
        config = OmegaConf.to_container(compose(config_name='train', overrides=['rules=god']), resolve=True)
    setting = settings()
    setting['teacher'] = {'kind': 'rule', 'name': 'god', 'rules': config['rules']}
    ppo = fixture_config('lstm') | {'online_teacher': setting}
    ppo['ppo']['recurrent_storage'] = 'sparse'
    ppo['ppo']['policy_kwargs']['features_extractor_class'] = (
        'soku_rl.rl.address_invariant_features.AddressInvariantCombatFeatures')
    ppo['ppo']['policy_kwargs']['features_extractor_kwargs'] = {
        'history_frames': 1, 'object_features': 8, 'player_features': 8, 'features_dim': 16}
    model, _ = create_ppo(ObservationContractEnv(interface), interface, ppo, {'kind': 'fresh'}, 'cpu', 7)
    reference = load_policy('reference', setting['teacher'], interface, 'cpu')
    helper = model._online_teacher
    for character in (1, 6):
        current = observation(character)
        for frame in range(4):
            current.world.update(frame=frame, battle_time=frame)
            obs = np.concatenate((encode_privileged(current), np.full(64, frame / 4, np.float32)))
            before = obs.copy().view(np.uint32)
            if frame == 2:
                helper.reset()
            helper.record(obs[None], [frame == 0], [256])
            if frame == 0:
                actor = reference.spawn(helper.active[0]['teacher_seed'])
            assert helper.active[0]['labels'][-1] == actor.act(obs)
            np.testing.assert_array_equal(before, obs.view(np.uint32))
        assert len(helper.active[0]['observations']) == len(helper.active[0]['labels']) == 4
    assert model.teacher_state['collected_frames'] == 8
    assert model.teacher_state['teacher_episodes'] == 2
    env.close()
