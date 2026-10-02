"""Calibrate only a frozen behavior policy's private critic before shared PPO training."""
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import time

import hydra
from omegaconf import OmegaConf


@hydra.main(version_base='1.3', config_path='../config', config_name='calibrate_critic')
def main(cfg):
    import numpy as np
    import torch
    from soku_rl.env import EpisodeConfig
    from soku_rl.env.wrappers.learning import LearningConfig, LearningInterface
    from soku_rl.rl import configure_runtime, ppo_settings
    from soku_rl.rl.behavior_cloning import load_demonstrations, ObservationContractEnv
    from soku_rl.rl.critic_calibration import critic_epoch, private_critic_parameters, validate_calibration
    from soku_rl.rl.learner import create_learner, learner_kind, parameter_hash
    from soku_rl.rl.recurrent_cloning import demonstration_episodes

    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    configure_runtime({'cpu_threads': config['cpu_threads']})
    if str(config['device']).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    training_path = Path(config['training_config'])
    training = OmegaConf.to_container(OmegaConf.load(training_path), resolve=True)
    shared = ppo_settings(training)
    if (learner_kind(shared) != 'ppo' or training['algorithm']['name'] != 'br'
            or any(key in shared for key in ('rehearsal', 'online_anchor'))
            or 'curriculum' in training['algorithm'] or shared['ppo']['gamma'] != 1.):
        raise ValueError('calibration requires a frozen undiscounted BR PPO without auxiliary objectives or curriculum')
    interface = LearningInterface(EpisodeConfig.from_dict(training['episode']), LearningConfig(**training['wrappers']))
    samples, manifest, behavior, manifest_sha = load_demonstrations(config['dataset'], interface)
    digest = hashlib.sha256(Path(config['checkpoint']).read_bytes()).hexdigest()
    validate_calibration(config['calibration'], manifest, digest)
    if any(behavior['algorithm'][key] != training['algorithm'][key] for key in ('matchups', 'opponents')):
        raise ValueError('critic trajectories must retain the original matchups and opponents')
    source = {'kind': 'weights', 'path': config['checkpoint'], 'training_config': config['training_config']}
    model, initialization = create_learner(ObservationContractEnv(interface), interface,
        training['algorithm'], source, config['device'], config['seed'])
    parameters = private_critic_parameters(model)
    private_ids = {id(parameter) for parameter in parameters}
    frozen = {name: parameter.detach().clone() for name, parameter in model.policy.named_parameters()
        if id(parameter) not in private_ids}
    buffers = {name: value.detach().clone() for name, value in model.policy.named_buffers()}
    optimizer = torch.optim.Adam(parameters, lr=config['calibration']['learning_rate'])
    episodes = {split: demonstration_episodes(rows) for split, rows in samples.items()}
    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=False)
    training.update(training_method='private_critic_calibration', critic_calibration=config, output=str(output))
    (output / 'config.yaml').write_text(OmegaConf.to_yaml(OmegaConf.create(training)), encoding='utf-8')
    root = Path(__file__).resolve().parents[1]
    identity = {'source_checkpoint_sha256': digest, 'dataset_manifest_sha256': manifest_sha,
        'training_config_sha256': hashlib.sha256(training_path.read_bytes()).hexdigest(),
        'dataset_episodes': manifest['episodes'], 'initialization': initialization,
        'source_hashes': {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in (root / 'src/soku_rl', root / 'tools') for p in sorted(folder.rglob('*.py'))},
        'packages': {name: version(name) for name in ('torch', 'numpy', 'stable-baselines3', 'sb3-contrib')},
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'device': config['device']}
    (output / 'identity.json').write_text(json.dumps(identity, indent=2))
    settings = config['calibration']
    report = {'success': False, 'method': 'private_critic_calibration', 'ppo_steps': 0,
        'initial_parameter_hash': parameter_hash(model.policy), 'history': []}
    started = time.perf_counter()
    rng = np.random.default_rng(config['seed'])
    best, updates = float('inf'), 0
    try:
        model.save(output / 'initial.zip')
        for epoch in range(settings['epochs'] + 1):
            if epoch:
                fitted = critic_epoch(model, episodes['train'], rng.permutation(len(episodes['train'])),
                    settings['batch_size'], settings['sequence_length'], optimizer, True)
                updates += fitted['updates']
            else:
                fitted = critic_epoch(model, episodes['train'], np.arange(len(episodes['train'])),
                    settings['batch_size'], settings['sequence_length'], optimizer, False)
            validation = critic_epoch(model, episodes['validation'], np.arange(len(episodes['validation'])),
                settings['batch_size'], settings['sequence_length'], optimizer, False)
            for name, parameter in model.policy.named_parameters():
                if name in frozen and not torch.equal(parameter, frozen[name]):
                    raise RuntimeError(f'critic calibration changed frozen parameter {name}')
            if any(not torch.equal(value, buffers[name]) for name, value in model.policy.named_buffers()):
                raise RuntimeError('critic calibration changed frozen buffers')
            if model.num_timesteps or model._n_updates or model.policy.optimizer.state:
                raise RuntimeError('critic calibration altered the PPO optimizer or step counters')
            if validation['mse'] < best:
                best = validation['mse']
                report['best_epoch'] = epoch
                model.save(output / 'best.zip')
            row = {'epoch': epoch, 'train': fitted, 'validation': validation,
                'updates': updates, 'seconds': time.perf_counter() - started}
            report['history'].append(row)
            print(json.dumps(row), flush=True)
        model.save(output / 'final.zip')
        torch.save(optimizer.state_dict(), output / 'critic-optimizer.pt')
        report.update(success=True, frozen_actor_features_and_buffers_verified=True,
            final_parameter_hash=parameter_hash(model.policy), supervised_updates=updates)
    except BaseException as error:
        report['error'] = repr(error)
        raise
    finally:
        report['seconds'] = time.perf_counter() - started
        (output / 'result.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
