import numpy as np
import pytest
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from test_shared_ppo import fixture_config, fixture_env, save_contract
from soku_rl.rl.behavior_cloning import ObservationContractEnv
from soku_rl.rl.critic_calibration import critic_epoch, private_critic_parameters, validate_calibration
from soku_rl.rl.ppo import create_ppo, parameter_hash
from soku_rl.rl.recurrent_cloning import zero_states
from soku_rl.rl.storage import PackedObservation


class FrozenFeatureFixture(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, 8)
        self.network = torch.nn.Sequential(torch.nn.Linear(int(np.prod(observation_space.shape)), 8),
            torch.nn.BatchNorm1d(8), torch.nn.Tanh())

    def forward(self, observations):
        return self.network(observations.flatten(1))


@pytest.mark.parametrize('shared', [True, False])
@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_private_fit_preserves_actor_features_buffers_and_shared_ppo_resume(tmp_path, shared, device):
    if device == 'cuda:0' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    torch.set_num_threads(1)
    env = fixture_env()
    config = fixture_config('lstm') | {'name': 'br'}
    config['ppo']['policy_kwargs'].update(share_features_extractor=shared,
        features_extractor_class='test_critic_calibration.FrozenFeatureFixture')
    model, _ = create_ppo(ObservationContractEnv(env.interface), env.interface, config, {'kind': 'fresh'}, device, 23)
    rng = np.random.default_rng(37)
    episodes = [[(PackedObservation.pack(rng.normal(size=env.single_observation_space.shape).astype(np.float32)),
        0, -.7, -1 if frame == 0 else 1) for frame in range(size)] for size in (3, 7, 5)]
    private = private_critic_parameters(model)
    private_ids = {id(p) for p in private}
    frozen = {name: parameter.detach().clone() for name, parameter in model.policy.named_parameters()
        if id(parameter) not in private_ids}
    buffers = {name: value.clone() for name, value in model.policy.named_buffers()}
    optimizer = torch.optim.Adam(private, lr=.02)
    model.policy.set_training_mode(False)
    errors, actor_before = [], []
    with torch.no_grad():
        for episode in episodes:
            states = zero_states(model.policy, 1)
            for obs, _, target, _ in episode:
                _, values, _, states = model.policy(torch.from_numpy(obs.unpack()).unsqueeze(0).to(device),
                    states, torch.zeros(1, device=device), deterministic=True)
                errors.append(float((values.item() - target) ** 2))
                actor_before.append(model.policy.action_dist.distribution.probs.clone())
    for length, batch in ((1, 1), (2, 4), (8, 16)):
        score = critic_epoch(model, episodes, [2, 0, 1], batch, length, optimizer, False)
        assert score['mse'] == pytest.approx(np.mean(errors), abs=1e-7)
        assert score['frames'] == 15 and score['updates'] == 0
    initial_private = [p.clone() for p in private]
    for _ in range(12):
        trained = critic_epoch(model, episodes, [0, 1, 2], 4, 2, optimizer, True)
        assert trained['frames'] == 15 and trained['updates'] == 7
    score = critic_epoch(model, episodes, [0, 1, 2], 4, 2, optimizer, False)
    assert score['mse'] < np.mean(errors) / 10
    assert any(not torch.equal(p, previous) for p, previous in zip(private, initial_private))
    for name, parameter in model.policy.named_parameters():
        if name in frozen:
            assert torch.equal(parameter, frozen[name]) and parameter.grad is None
    assert all(torch.equal(value, buffers[name]) for name, value in model.policy.named_buffers())
    assert not model.policy.optimizer.state and model.num_timesteps == model._n_updates == 0
    actor_after = []
    with torch.no_grad():
        for episode in episodes:
            states = zero_states(model.policy, 1)
            for obs, _, _, _ in episode:
                _, _, _, states = model.policy(torch.from_numpy(obs.unpack()).unsqueeze(0).to(device), states,
                    torch.zeros(1, device=device), deterministic=True)
                actor_after.append(model.policy.action_dist.distribution.probs.clone())
    assert all(torch.equal(a, b) for a, b in zip(actor_before, actor_after))
    path = tmp_path / 'calibrated.zip'
    model.save(path)
    contract = save_contract(tmp_path, env, config)
    from soku_rl.rl.opponent_env import OpponentMixtureVecEnv
    from soku_rl.marl.br import OpponentEntry
    from test_demonstrations import ConstantPolicy
    view = OpponentMixtureVecEnv(env, 0, [OpponentEntry('constant', ConstantPolicy(8))], [1.], 17)
    resumed, _ = create_ppo(view, env.interface, config,
        {'kind': 'weights', 'path': str(path), 'training_config': contract}, device, 31)
    assert parameter_hash(resumed.policy) == parameter_hash(model.policy)
    resumed.learn(8)
    assert resumed.num_timesteps == 8 and parameter_hash(resumed.policy) != parameter_hash(model.policy)
    view.close()
    env.close()


@pytest.mark.parametrize('change', [{'epochs': 0}, {'learning_rate': float('nan')},
    {'sequence_length': 3}, {'batch_size': True}])
def test_invalid_calibration_rejected(change):
    manifest = {'schema': 2, 'control': 'learner', 'behavior_fingerprint': 'correct'}
    config = {'epochs': 2, 'batch_size': 4, 'sequence_length': 2, 'learning_rate': .001} | change
    with pytest.raises(ValueError):
        validate_calibration(config, manifest, 'correct')


@pytest.mark.parametrize('control,digest', [('teacher', 'correct'), ('learner', 'wrong')])
def test_off_policy_returns_rejected(control, digest):
    with pytest.raises(ValueError, match='exact frozen checkpoint'):
        validate_calibration({'epochs': 2, 'batch_size': 4, 'sequence_length': 2, 'learning_rate': .001},
            {'schema': 2, 'control': control, 'behavior_fingerprint': digest}, 'correct')


def test_optimizer_cannot_include_actor_or_shared_features():
    env = fixture_env()
    model, _ = create_ppo(ObservationContractEnv(env.interface), env.interface,
        fixture_config('lstm'), {'kind': 'fresh'}, 'cpu', 11)
    with pytest.raises(ValueError, match='exactly the private critic'):
        critic_epoch(model, [], [], 4, 2, model.policy.optimizer, True)
    env.close()
