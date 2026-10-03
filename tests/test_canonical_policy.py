from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from hydra import compose, initialize_config_dir
import numpy as np
from omegaconf import OmegaConf
import pytest
import torch
from stable_baselines3.common.vec_env import DummyVecEnv

from soku_rl.env.observation.privileged import encode_privileged
from soku_rl.env.wrappers.learning import LearningInterface
from soku_rl.policy.export_actor import export_actor
from soku_rl.policy.loader import load_policy
from soku_rl.rl.canonical_features import CanonicalCombatFeatures
from soku_rl.rl.ppo import create_ppo, parameter_hash
from soku_rl.rl.recurrent_cloning import sequence_epoch, zero_states
from soku_rl.rl.storage import PackedObservation
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from test_canonical_features import reflected, scene
from test_facing_policy import FacingContractEnv, configuration
from test_shared_ppo import save_contract


def canonical_configuration():
    interface, config = configuration()
    interface = LearningInterface(interface.episode, replace(interface.config, action_history=1))
    config['ppo']['policy_kwargs']['features_extractor_class'] = 'soku_rl.rl.canonical_features.CanonicalCombatFeatures'
    config['ppo']['policy_kwargs']['features_extractor_kwargs']['arena_width'] = 1280.
    return interface, config


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_shared_policy_mirror_probabilities_values_memory_and_gradients(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    torch.set_num_threads(1)
    interface, config = canonical_configuration()
    env = DummyVecEnv([lambda: FacingContractEnv(interface)])
    try:
        model, _ = create_ppo(env, interface, config, {'kind':'fresh'}, device, 47)
        frames = [scene(2, facing) for facing in (1, -1, 1)]
        commands = np.array([[-1,0,1,0,0,0,0,0],[1,0,0,1,0,0,0,0],[0,0,0,0,0,0,0,1]], np.float32)
        left = torch.from_numpy(np.stack([np.r_[encode_privileged(frame), command]
            for frame,command in zip(frames,commands,strict=True)])).to(device)
        commands[:,0] *= -1
        right = torch.from_numpy(np.stack([np.r_[encode_privileged(reflected(frame)), command]
            for frame,command in zip(frames,commands,strict=True)])).to(device)
        starts = torch.tensor([1.,0.,1.],device=device)
        states = model._last_lstm_states
        policy = model.policy
        permutation = policy.mirrored_commands
        labels = torch.tensor([65,448,575],device=device)
        gradients, outputs = [], []
        for observations, actions in ((left,labels),(right,permutation[labels])):
            policy.zero_grad(set_to_none=True)
            values, log_prob, entropy = policy.evaluate_actions(observations,actions,states,starts)
            torch.testing.assert_close(policy.action_dist.log_prob(actions),log_prob)
            distribution, memory = policy.get_distribution(observations,states.pi,starts)
            outputs.append((values.detach(),entropy.detach(),distribution.distribution.probs.detach(),memory))
            (-log_prob.mean()+values.square().mean()).backward()
            gradients.append([p.grad.clone() for p in policy.parameters() if p.grad is not None])
        torch.testing.assert_close(outputs[0][0],outputs[1][0],rtol=0,atol=0)
        torch.testing.assert_close(outputs[0][1],outputs[1][1],rtol=1e-6,atol=1e-6)
        torch.testing.assert_close(outputs[0][2],outputs[1][2][:,permutation],rtol=1e-6,atol=1e-7)
        for a,b in zip(outputs[0][3],outputs[1][3],strict=True):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
        for a,b in zip(*gradients,strict=True):
            torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-6)
    finally:
        env.close()


def test_update_resume_weights_export_and_factory_contract(tmp_path):
    torch.set_num_threads(1)
    interface, config = canonical_configuration()
    env = DummyVecEnv([lambda: FacingContractEnv(interface) for _ in range(2)])
    try:
        model, _ = create_ppo(env, interface, config, {'kind':'fresh'}, 'cpu', 47)
        initial = parameter_hash(model.policy)
        original = deepcopy(config)
        original['ppo']['policy_kwargs']['features_extractor_class'] = 'soku_rl.rl.address_invariant_features.AddressInvariantCombatFeatures'
        del original['ppo']['policy_kwargs']['features_extractor_kwargs']['arena_width']
        baseline, _ = create_ppo(env,interface,original,{'kind':'fresh'},'cpu',47)
        assert parameter_hash(baseline.policy) == initial
        model.learn(16)
        assert parameter_hash(model.policy) != initial
        path = tmp_path/'final.zip'
        model.save(path)
        contract = save_contract(tmp_path,SimpleNamespace(interface=interface),config)
        for kind in ('weights','checkpoint'):
            restored, _ = create_ppo(env,interface,config,{'kind':kind,'path':str(path),'training_config':contract},'cpu',48)
            assert parameter_hash(restored.policy) == parameter_hash(model.policy)
            assert isinstance(restored.policy.features_extractor,CanonicalCombatFeatures)
            assert restored.num_timesteps == (16 if kind=='checkpoint' else 0)
        with pytest.raises(ValueError,match='architecture'):
            create_ppo(env,interface,original,{'kind':'weights','path':str(path),'training_config':contract},'cpu',48)
        invalid = deepcopy(config)
        del invalid['ppo']['action_frame']
        with pytest.raises(ValueError,match='own_facing'):
            create_ppo(env,interface,invalid,{'kind':'fresh'},'cpu',48)
        loaded = load_policy('canonical',{'kind':'sb3_recurrent','path':str(path),'training_config':contract},interface,'cpu')
        first, second = loaded.spawn(7), loaded.spawn(7)
        observation = np.r_[encode_privileged(scene(2,-1)),np.zeros(8,np.float32)]
        assert first.act(observation) == second.act(observation)
        result = export_actor({'candidate':{'name':'canonical','policy':{'kind':'sb3_recurrent',
            'path':str(path),'training_config':contract}},'verification_steps':512,'seed':13,
            'output':str(tmp_path/'exported')})
        assert result['verification']['maximum_absolute_error'] < 2e-6
    finally:
        env.close()


def test_offline_and_br_configuration_share_the_canonical_contract():
    with initialize_config_dir(config_dir=str(Path(__file__).parents[1]/'config'),version_base='1.3'):
        offline = compose(config_name='pretrain_recurrent_canonical_demonstrations')
        online = compose(config_name='train',overrides=['algorithm=br','rl=recurrent_demonstration_transfer',
            'track=superhuman_canonical','wrappers=superhuman_learning'])
        assert OmegaConf.to_container(offline.rl.ppo.policy_kwargs) == OmegaConf.to_container(online.rl.ppo.policy_kwargs)
        assert offline.rl.ppo.action_frame == online.rl.ppo.action_frame == 'own_facing'
        assert offline.pretraining.initial_policy.kind == 'fresh'


def test_padded_bc_scores_absolute_labels_like_sequential_inference_and_updates():
    torch.set_num_threads(1)
    interface, config = canonical_configuration()
    env = FacingContractEnv(interface)
    model, _ = create_ppo(env,interface,config,{'kind':'fresh'},'cpu',37)
    episodes, reference = [], []
    for size in (1,5,7):
        episode = []
        states = zero_states(model.policy,1)
        for step in range(size):
            value = np.r_[encode_privileged(scene(1,(-1)**step)),np.zeros(8,np.float32)]
            action = (65+step*71)%576
            episode.append((PackedObservation.pack(value),action,1.,-1 if step==0 else 256))
            with torch.no_grad():
                distribution, memory = model.policy.get_distribution(torch.from_numpy(value)[None],states.pi,torch.zeros(1))
                reference.append(-float(distribution.log_prob(torch.tensor([action]))))
                states = RNNStates(memory,states.vf)
        episodes.append(episode)
    for length,batch in ((1,1),(2,4),(4,12)):
        metrics, _, updates = sequence_epoch(model,episodes,[2,0,1],batch,length,.5,False,1.)
        assert updates == 0 and metrics['nll'] == pytest.approx(np.mean(reference),abs=1e-6)
    before = parameter_hash(model.policy)
    _, _, updates = sequence_epoch(model,episodes,[0,1,2],4,2,.5,True,1.)
    assert updates > 0 and parameter_hash(model.policy) != before
    env.close()
