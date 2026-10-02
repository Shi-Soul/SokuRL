import copy

import numpy as np
import pytest
import torch
from stable_baselines3.common.policies import BaseModel
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from test_shared_ppo import fixture_config, fixture_env, save_contract
from soku_rl.rl.behavior_cloning import ObservationContractEnv
from soku_rl.rl.feature_split import copy_split_feature_weights, split_feature_config
from soku_rl.rl.ppo import create_ppo, parameter_hash
from soku_rl.rl.recurrent_cloning import zero_states


class SplitFeatureFixture(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, 8)
        self.network = torch.nn.Sequential(torch.nn.Linear(int(np.prod(observation_space.shape)), 8),
            torch.nn.BatchNorm1d(8), torch.nn.Tanh())
        self.register_buffer('scale', torch.tensor(1.), persistent=False)

    def forward(self, observations):
        return self.network(observations.flatten(1)) * self.scale


def predictions(model, observations, starts):
    model.policy.set_training_mode(False)
    with torch.no_grad():
        if hasattr(model.policy, 'lstm_actor'):
            _, values, _, _ = model.policy(observations, zero_states(model.policy, 1), starts, deterministic=True)
            probabilities = model.policy.action_dist.distribution.probs.clone()
        else:
            probabilities = model.policy.get_distribution(observations).distribution.probs.clone()
            values = model.policy.predict_values(observations)
    return probabilities, values.clone()


def value_update(model, observations, starts):
    policy = model.policy
    policy.set_training_mode(False)
    features = BaseModel.extract_features(policy, observations, policy.vf_features_extractor)
    if hasattr(policy, 'lstm_critic'):
        policy.lstm_critic.train(True)  # cuDNN backward requires training mode.
        features, _ = policy._process_sequence(features, zero_states(policy, 1).vf, starts, policy.lstm_critic)
    values = policy.value_net(policy.mlp_extractor.forward_critic(features))
    loss = (values + 1.).square().mean()
    policy.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    policy.optimizer.step()


@pytest.mark.parametrize('policy_type', ['mlp', 'lstm'])
@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_exact_conversion_value_gradient_isolation_and_shared_ppo_loading(tmp_path, policy_type, device):
    if device == 'cuda:0' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    torch.set_num_threads(1)
    env = fixture_env()
    config = fixture_config(policy_type) | {'name': 'br'}
    config['ppo']['policy_kwargs']['features_extractor_class'] = 'test_feature_split.SplitFeatureFixture'
    training = {'algorithm': config, 'rl': {key: copy.deepcopy(config[key]) for key in
        ('policy_type', 'timeout_payoff', 'ppo')}}
    converted = split_feature_config(training)
    assert 'share_features_extractor' not in config['ppo']['policy_kwargs']
    assert converted['rl']['ppo']['policy_kwargs']['share_features_extractor'] is False
    view = ObservationContractEnv(env.interface)
    source, _ = create_ppo(view, env.interface, config, {'kind': 'fresh'}, device, 31)
    destination, _ = create_ppo(view, env.interface, converted['algorithm'], {'kind': 'fresh'}, device, 49)
    obs = torch.as_tensor(np.random.default_rng(27).normal(size=(8, *env.single_observation_space.shape)),
        device=device, dtype=torch.float32)
    starts = torch.tensor([1, 0, 0, 1, 0, 0, 0, 0], device=device, dtype=torch.float32)
    source.policy.pi_features_extractor.network[1].running_mean.fill_(.25)
    value_update(source, obs, starts)
    source.num_timesteps, source._n_updates = 37, 11
    original = {key: value.clone() for key, value in source.policy.state_dict().items()}
    report = copy_split_feature_weights(source, destination)
    assert report['source_steps'] == 37 and report['source_ppo_n_updates'] == 11
    assert report['destination_parameters'] > report['source_parameters']
    assert destination.num_timesteps == destination._n_updates == 0 and not destination.policy.optimizer.state
    assert source.num_timesteps == 37 and source._n_updates == 11 and source.policy.optimizer.state
    assert all(torch.equal(value, original[key]) for key, value in source.policy.state_dict().items())
    before = predictions(source, obs, starts)
    copied = predictions(destination, obs, starts)
    assert all(torch.equal(a, b) for a, b in zip(before, copied, strict=True))
    for model in (source, destination):
        value_update(model, obs, starts)
    assert not torch.equal(predictions(source, obs, starts)[0], before[0])
    after = predictions(destination, obs, starts)
    assert torch.equal(after[0], before[0]) and not torch.equal(after[1], before[1])
    assert all(p.grad is None for p in destination.policy.pi_features_extractor.parameters())
    assert any(p.grad is not None for p in destination.policy.vf_features_extractor.parameters())
    path = tmp_path / 'split.zip'
    destination.save(path)
    contract = save_contract(tmp_path, env, converted['algorithm'])
    from soku_rl.rl.opponent_env import OpponentMixtureVecEnv
    from soku_rl.marl.br import OpponentEntry
    from test_demonstrations import ConstantPolicy
    opponent = OpponentMixtureVecEnv(env, 0, [OpponentEntry('constant', ConstantPolicy(8))], [1.], 17)
    for kind in ('weights', 'checkpoint'):
        loaded, _ = create_ppo(opponent, env.interface, converted['algorithm'],
            {'kind': kind, 'path': str(path), 'training_config': contract}, device, 13)
        assert parameter_hash(loaded.policy) == parameter_hash(destination.policy)
        assert all(torch.equal(a, b) for a, b in zip(predictions(loaded, obs, starts), after, strict=True))
        assert loaded.policy.pi_features_extractor is not loaded.policy.vf_features_extractor
        if kind == 'weights':
            assert not loaded.policy.optimizer.state
        else:
            assert loaded.policy.optimizer.state
        loaded.learn(8)
        assert loaded.num_timesteps == 8 and parameter_hash(loaded.policy) != parameter_hash(destination.policy)
    opponent.close()
    env.close()


@pytest.mark.parametrize('change', ['already_split', 'shared_lstm', 'auxiliary', 'dqn', 'inconsistent'])
def test_invalid_feature_conversion_config_rejected(change):
    config = fixture_config('lstm') | {'name': 'br'}
    shared = {key: copy.deepcopy(config[key]) for key in ('policy_type', 'timeout_payoff', 'ppo')}
    if change == 'already_split':
        shared['ppo']['policy_kwargs']['share_features_extractor'] = False
    elif change == 'shared_lstm':
        shared['ppo']['policy_kwargs']['shared_lstm'] = True
    elif change == 'auxiliary':
        shared['online_anchor'] = {}
    elif change == 'dqn':
        shared['learner'] = 'dqn'
    else:
        shared['ppo']['policy_kwargs']['net_arch'] = [16]
    if change != 'inconsistent':
        config.update(copy.deepcopy(shared))
    with pytest.raises(ValueError):
        split_feature_config({'rl': shared, 'algorithm': config})


@pytest.mark.parametrize('change', ['destination_steps', 'destination_updates', 'architecture',
    'aliased_storage', 'nonpersistent_buffer'])
def test_unsafe_weight_copy_rejected(change):
    env = fixture_env()
    config = fixture_config('mlp')
    config['ppo']['policy_kwargs']['features_extractor_class'] = 'test_feature_split.SplitFeatureFixture'
    other = copy.deepcopy(config)
    other['ppo']['policy_kwargs']['share_features_extractor'] = False
    if change == 'architecture':
        other['ppo']['policy_kwargs']['net_arch'] = [16]
    source, _ = create_ppo(ObservationContractEnv(env.interface), env.interface, config, {'kind': 'fresh'}, 'cpu', 31)
    destination, _ = create_ppo(ObservationContractEnv(env.interface), env.interface, other, {'kind': 'fresh'}, 'cpu', 49)
    if change == 'destination_steps':
        destination.num_timesteps = 1
    elif change == 'destination_updates':
        destination._n_updates = 1
    elif change == 'aliased_storage':
        a = destination.policy.pi_features_extractor.network[0]
        b = destination.policy.vf_features_extractor.network[0]
        b.weight.data = a.weight.data
    elif change == 'nonpersistent_buffer':
        source.policy.pi_features_extractor.scale.fill_(2.)
    with pytest.raises(ValueError):
        copy_split_feature_weights(source, destination)
    env.close()
