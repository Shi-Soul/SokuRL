"""Frozen actor features/memory survive PPO, auxiliary updates and continuation."""
import copy
from pathlib import Path

from hydra import compose, initialize_config_dir
import pytest
import torch

from test_shared_ppo import fixture_config, fixture_env, save_contract
from test_feature_split import predictions
from soku_rl.rl.behavior_cloning import ObservationContractEnv
from soku_rl.rl import ppo_settings
from soku_rl.rl.frozen_representation import freeze_actor_representation
from soku_rl.rl.ppo import algorithm_type, create_ppo, parameter_hash
from soku_rl.rl.learner import learner_kind
from soku_rl.rl.opponent_env import OpponentMixtureVecEnv
from soku_rl.marl.br import OpponentEntry
from test_demonstrations import ConstantPolicy


def settings(kind, shared):
    config = fixture_config(kind) | {'name': 'br'}
    config['ppo']['freeze_actor_representation'] = True
    config['ppo']['policy_kwargs'].update(
        features_extractor_class='test_feature_split.SplitFeatureFixture', share_features_extractor=shared)
    return config


def frozen_state(model):
    modules = [model.policy.pi_features_extractor]
    if hasattr(model.policy, 'lstm_actor'):
        modules.append(model.policy.lstm_actor)
    return {f'{index}.{name}': value.detach().clone() for index, module in enumerate(modules)
        for name, value in module.state_dict().items()}


def assert_frozen(model, before):
    assert all(torch.equal(before[name], value) for name, value in frozen_state(model).items())
    for name, parameter in model.policy.named_parameters():
        if name in model.frozen_actor_representation['parameters']:
            assert not parameter.requires_grad and parameter.grad is None
            assert parameter not in model.policy.optimizer.state
    assert not model.policy.pi_features_extractor.training
    if hasattr(model.policy, 'lstm_actor'):
        assert not model.policy.lstm_actor.training


@pytest.mark.parametrize('kind', ['mlp', 'lstm'])
@pytest.mark.parametrize('shared', [False, True])
@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_update_resume_and_weights_keep_representation_fixed(tmp_path, kind, shared, device):
    if device == 'cuda:0' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    torch.set_num_threads(1)
    env = fixture_env()
    view = OpponentMixtureVecEnv(env, 'balanced', [OpponentEntry('constant', ConstantPolicy(8))], [1.], 17)
    config = settings(kind, shared)
    model, _ = create_ppo(view, env.interface, config, {'kind': 'fresh'}, device, 7)
    initial = parameter_hash(model.policy)
    before = frozen_state(model)
    critic = {key: value.clone() for key, value in model.policy.mlp_extractor.value_net.state_dict().items()}
    action = model.policy.action_net.weight.detach().clone()
    model.policy.train(True)
    if kind == 'lstm':
        model.policy.lstm_actor.train(True)
    assert_frozen(model, before)
    model.learn(16)
    assert_frozen(model, before)
    assert parameter_hash(model.policy) != initial
    assert not torch.equal(action, model.policy.action_net.weight)
    assert any(not torch.equal(critic[key], value) for key, value in model.policy.mlp_extractor.value_net.state_dict().items())
    if not shared:
        assert model.policy.vf_features_extractor.network[1].num_batches_tracked > 0
    path = tmp_path / 'frozen.zip'
    model.save(path)
    contract = save_contract(tmp_path, env, config)
    source = {'kind': 'checkpoint', 'path': str(path), 'training_config': contract}
    resumed, _ = create_ppo(view, env.interface, config, source, device, 19)
    assert parameter_hash(resumed.policy) == parameter_hash(model.policy)
    assert resumed.num_timesteps == 16 and resumed.policy.optimizer.state
    assert_frozen(resumed, before)
    resumed.learn(8, reset_num_timesteps=False)
    assert resumed.num_timesteps == 24
    assert_frozen(resumed, before)
    fresh, _ = create_ppo(view, env.interface, config, source | {'kind': 'weights'}, device, 19)
    assert fresh.num_timesteps == 0 and not fresh.policy.optimizer.state
    assert_frozen(fresh, before)
    ordinary = copy.deepcopy(config)
    del ordinary['ppo']['freeze_actor_representation']
    with pytest.raises(ValueError, match='continued PPO'):
        create_ppo(view, env.interface, ordinary, source, device, 19)
    unfrozen, _ = create_ppo(view, env.interface, ordinary, source | {'kind': 'weights'}, device, 19)
    assert all(parameter.requires_grad for parameter in unfrozen.policy.parameters())
    inference = algorithm_type(kind).load(path, device=device)
    assert parameter_hash(inference.policy) == parameter_hash(model.policy)
    assert 'train' not in inference.policy.pi_features_extractor.__dict__
    # Inference uses ordinary SB3 policy classes, and is independent of the training hook.
    observation = view.reset()
    left, _ = model.predict(observation, deterministic=True)
    right, _ = inference.predict(observation, deterministic=True)
    assert (left == right).all()
    tensor = torch.as_tensor(observation, device=device)
    starts = torch.ones(len(observation), device=device)
    assert all(torch.equal(a, b) for a, b in zip(
        predictions(model, tensor, starts), predictions(inference, tensor, starts), strict=True))
    view.close()
    env.close()


@pytest.mark.parametrize('algorithm', ['ppo', 'br', 'ippo', 'nfsp', 'psro'])
def test_all_marl_schedulers_receive_shared_freeze_setting(algorithm):
    with initialize_config_dir(config_dir=str(Path(__file__).parents[1] / 'config'), version_base='1.3'):
        config = compose(config_name='train', overrides=[f'algorithm={algorithm}', '++rl.ppo.freeze_actor_representation=true'])
        from omegaconf import OmegaConf
        config = OmegaConf.to_container(config, resolve=True)
    ppo_settings(config)
    learner = config['algorithm']['response'] if algorithm == 'psro' else config['algorithm']
    assert learner['ppo']['freeze_actor_representation'] is True


@pytest.mark.parametrize('kind', ['mlp', 'lstm'])
def test_real_joint_ippo_training_preserves_frozen_parameters(tmp_path, monkeypatch, kind):
    from soku_rl.marl import ippo
    torch.set_num_threads(1)
    env = fixture_env()
    config = settings(kind, True) | {'name': 'ippo'}
    originals = []
    initialize = ippo.create_learner
    def capture(view, interface, config, source, device, seed):
        model, metadata = initialize(view, interface, config, source, device, seed)
        originals.append({'parameters': parameter_hash(model.policy), 'frozen': frozen_state(model)})
        return model, metadata
    monkeypatch.setattr(ippo, 'create_learner', capture)
    result = ippo.train_ippo(env, config, 'cpu', 13, tmp_path)
    assert result['updates'] == 1
    for player in (0, 1):
        final = algorithm_type(kind).load(tmp_path / f'player_{player}/final.zip', device='cpu')
        assert all(torch.equal(value, originals[player]['frozen'][name]) for name, value in frozen_state(final).items())
        assert originals[player]['parameters'] != parameter_hash(final.policy)
    env.close()


@pytest.mark.parametrize('invalid', [0, 1, 'true', None])
def test_invalid_flag_fails_before_training(invalid):
    env = fixture_env()
    config = settings('mlp', True)
    config['ppo']['freeze_actor_representation'] = invalid
    with pytest.raises(ValueError, match='boolean'):
        create_ppo(ObservationContractEnv(env.interface), env.interface, config, {'kind': 'fresh'}, 'cpu', 1)
    env.close()


def test_reject_dqn_and_repeated_hooks():
    with pytest.raises(ValueError, match='PPO learner'):
        learner_kind({'learner': 'dqn', 'ppo': {'freeze_actor_representation': True}})
    env = fixture_env()
    config = settings('mlp', True)
    model, _ = create_ppo(ObservationContractEnv(env.interface), env.interface, config, {'kind': 'fresh'}, 'cpu', 1)
    with pytest.raises(ValueError, match='existing'):
        freeze_actor_representation(model)
    env.close()


@pytest.mark.parametrize('kind', ['mlp', 'lstm'])
def test_freezing_existing_weights_preserves_initial_policy(tmp_path, kind):
    env = fixture_env()
    view = ObservationContractEnv(env.interface)
    config = settings(kind, True)
    ordinary = copy.deepcopy(config)
    del ordinary['ppo']['freeze_actor_representation']
    baseline, _ = create_ppo(view, env.interface, ordinary, {'kind': 'fresh'}, 'cpu', 13)
    path = tmp_path / 'ordinary.zip'
    baseline.save(path)
    contract = save_contract(tmp_path, env, ordinary)
    frozen, _ = create_ppo(view, env.interface, config,
        {'kind': 'weights', 'path': str(path), 'training_config': contract}, 'cpu', 99)
    assert parameter_hash(frozen.policy) == parameter_hash(baseline.policy)
    assert not frozen.policy.optimizer.state and frozen.num_timesteps == 0
    config['ppo']['freeze_actor_representation'] = False
    disabled, _ = create_ppo(view, env.interface, config, {'kind': 'fresh'}, 'cpu', 13)
    assert parameter_hash(disabled.policy) == parameter_hash(baseline.policy)
    assert all(parameter.requires_grad for parameter in disabled.policy.parameters())
    assert 'train' not in disabled.policy.pi_features_extractor.__dict__
    env.close()


@pytest.mark.parametrize('kind', ['mlp', 'lstm'])
def test_reject_missing_frozen_parameters_or_shared_recurrence(kind):
    env = fixture_env()
    config = fixture_config(kind)
    config['ppo']['freeze_actor_representation'] = True
    if kind == 'lstm':
        config['ppo']['policy_kwargs'].update(shared_lstm=True, enable_critic_lstm=False)
    with pytest.raises(ValueError, match='frozen and trainable|independent critic'):
        create_ppo(ObservationContractEnv(env.interface), env.interface, config, {'kind': 'fresh'}, 'cpu', 1)
    env.close()


@pytest.mark.parametrize('kind', ['mlp', 'lstm'])
@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_online_teacher_cannot_unfreeze_actor_representation(monkeypatch, kind, device):
    from test_online_teacher import RecordingTeacher, settings as teacher_settings
    if device == 'cuda:0' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    torch.set_num_threads(1)
    teacher = RecordingTeacher()
    monkeypatch.setattr('soku_rl.rl.online_teacher.load_policy', lambda *args: teacher)
    env = fixture_env()
    view = OpponentMixtureVecEnv(env, 'balanced', [OpponentEntry('constant', ConstantPolicy(8))], [1.], 17)
    config = settings(kind, True) | {'online_teacher': teacher_settings()}
    model, _ = create_ppo(view, env.interface, config, {'kind': 'fresh'}, device, 7)
    before = frozen_state(model)
    model.learn(16)
    assert model.teacher_state['updates'] == 4
    assert_frozen(model, before)
    view.close()
    env.close()
