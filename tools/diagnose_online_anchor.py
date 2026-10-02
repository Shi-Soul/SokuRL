"""Calibrate bounded anchor updates on learner trajectories; never optimize held-out games."""
import copy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import time

import hydra
from omegaconf import OmegaConf


@hydra.main(version_base='1.3', config_path='../config', config_name='diagnose_online_anchor')
def main(cfg):
    import torch
    from stable_baselines3.common.logger import configure
    from soku_rl.env import EpisodeConfig
    from soku_rl.env.wrappers.learning import LearningConfig, LearningInterface
    from soku_rl.policy.contract import read_training_contract
    from soku_rl.rl.anchor_diagnostics import (
        reference_windows, score_memory_transfer, score_windows, select_windows)
    from soku_rl.rl.behavior_cloning import load_demonstrations
    from soku_rl.rl.online_anchor import attach_anchor
    from soku_rl.rl.ppo import algorithm_type, parameter_hash
    from soku_rl.rl.recurrent_cloning import demonstration_episodes

    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    if type(config['cpu_threads']) is not int or config['cpu_threads'] < 1:
        raise ValueError('cpu_threads must be positive')
    rates, counts = config['learning_rates'], config['update_counts']
    if (not isinstance(rates, list) or not rates or len(set(rates)) != len(rates)
            or any(type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0 for rate in rates)
            or not isinstance(counts, list) or not counts
            or any(type(count) is not int or count < 1 for count in counts)
            or counts != sorted(set(counts))):
        raise ValueError('explicit positive distinct learning rates and increasing update counts are required')
    torch.set_num_threads(config['cpu_threads'])
    device = torch.device(config['device'])
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    contract_path = Path(config['training_config'])
    contract_bytes = contract_path.read_bytes()
    training = OmegaConf.to_container(OmegaConf.load(io.StringIO(contract_bytes.decode())), resolve=True)
    interface = LearningInterface(EpisodeConfig.from_dict(training['episode']), LearningConfig(**training['wrappers']))
    read_training_contract(contract_path, interface)
    if contract_path.read_bytes() != contract_bytes or training['rl']['policy_type'] != 'lstm':
        raise ValueError('diagnostic requires a stable recurrent training contract')
    checkpoint_bytes = Path(config['checkpoint']).read_bytes()
    checkpoint_sha = hashlib.sha256(checkpoint_bytes).hexdigest()
    samples, manifest, _, manifest_sha = load_demonstrations(config['dataset'], interface)
    if manifest['schema'] != 2 or manifest['control'] != 'learner' or manifest['behavior_fingerprint'] != checkpoint_sha:
        raise ValueError('diagnostic requires trajectories controlled by this exact checkpoint')
    episodes = {split: [[row[0] for row in episode] for episode in demonstration_episodes(rows)]
        for split, rows in samples.items()}
    windows = {split: select_windows(rows, config['offsets'], config['sequence_length'])
        for split, rows in episodes.items()}
    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'config.yaml').open('x') as stream:
        stream.write(OmegaConf.to_yaml(cfg, resolve=True))
    root = Path(__file__).resolve().parents[1]
    report = {'success': False, 'measurement': 'offline auxiliary update calibration; no PPO updates or game control',
        'checkpoint_sha256': checkpoint_sha, 'training_config_sha256': hashlib.sha256(contract_bytes).hexdigest(),
        'dataset_manifest_sha256': manifest_sha, 'dataset_episodes': manifest['episodes'],
        'device': str(device), 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES', ''),
        'source_hashes': {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for folder in (root / 'src/soku_rl', root / 'tools') for path in sorted(folder.rglob('*.py'))},
        'runs': []}
    started = time.perf_counter()
    try:
        for rate in rates:
            model = algorithm_type('lstm').load(io.BytesIO(checkpoint_bytes), device=device)
            if model.observation_space != interface.observation_space or model.action_space != interface.action_space:
                raise ValueError('checkpoint observation/action spaces disagree')
            anchor_config = copy.deepcopy(training['rl']['online_anchor'])
            anchor_config.update(learning_rate=rate, updates_per_rollout=1)
            attach_anchor(model, interface, anchor_config, False)
            model.set_logger(configure(folder=None, format_strings=[]))
            anchor = model._anchor
            # Supply only validated training trajectories. This is an offline
            # calibration store, not a claim that these frames came from a new PPO rollout.
            anchor.recent = [(episode, offset) for episode in episodes['train'] for offset in range(len(episode))]
            model.anchor_state['collected_frames'] = len(anchor.recent)
            frozen_hash, initial_hash = parameter_hash(anchor.reference.policy), parameter_hash(model.policy)
            initial_steps, initial_updates = model.num_timesteps, model._n_updates
            critic_hashes = [parameter_hash(module) for module in (model.policy.lstm_critic, model.policy.value_net)]
            targets = {split: reference_windows(anchor.reference, rows, windows[split])
                for split, rows in episodes.items()}
            before = {split: score_windows(model, rows, windows[split], targets[split])
                for split, rows in episodes.items()}
            if not report['runs']:
                report['memory_transfer_before'] = {split: score_memory_transfer(model, anchor.reference, rows, windows[split])
                    for split, rows in episodes.items()}
            run = {'learning_rate': rate, 'initial_parameter_hash': initial_hash,
                'reference_identity': model.anchor_state['identity'], 'before': before, 'updates': [], 'checkpoints': []}
            for update in range(1, max(counts) + 1):
                run['updates'].append(anchor.update())
                if update in counts:
                    run['checkpoints'].append({'updates': update, 'parameter_hash': parameter_hash(model.policy),
                        'state': copy.deepcopy(model.anchor_state),
                        'scores': {split: score_windows(model, rows, windows[split], targets[split])
                            for split, rows in episodes.items()}})
            assert model.num_timesteps == initial_steps and model._n_updates == initial_updates
            assert parameter_hash(anchor.reference.policy) == frozen_hash
            assert [parameter_hash(module) for module in (model.policy.lstm_critic, model.policy.value_net)] == critic_hashes
            run.update(ppo_steps_unchanged=initial_steps, ppo_updates_unchanged=initial_updates,
                reference_unchanged=True, private_critic_unchanged=True)
            report['runs'].append(run)
            print(rate, {row['updates']: {split: values['mean_kl'] for split, values in row['scores'].items()}
                for row in run['checkpoints']}, flush=True)
            del model, anchor, targets
        report['success'] = True
    except BaseException as error:
        report['error'] = repr(error)
        raise
    finally:
        report['seconds'] = time.perf_counter() - started
        (output / 'result.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
