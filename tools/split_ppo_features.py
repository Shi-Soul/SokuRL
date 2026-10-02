"""Export equivalent PPO weights with independent actor and critic encoders."""
import hashlib
from importlib.metadata import version
import io
import json
import os
from pathlib import Path
import time

import hydra
from omegaconf import OmegaConf


@hydra.main(version_base='1.3', config_path='../config', config_name='split_ppo_features')
def main(cfg):
    import torch
    from soku_rl.env import EpisodeConfig
    from soku_rl.env.wrappers.learning import LearningConfig, LearningInterface
    from soku_rl.policy.contract import read_training_contract
    from soku_rl.rl import configure_runtime
    from soku_rl.rl.behavior_cloning import ObservationContractEnv
    from soku_rl.rl.feature_split import copy_split_feature_weights, split_feature_config
    from soku_rl.rl.learner import create_learner, parameter_hash
    from soku_rl.rl.ppo import algorithm_type

    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    configure_runtime({'cpu_threads': config['cpu_threads']})
    device = torch.device(config['device'])
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested for feature conversion but unavailable')
    contract_path = Path(config['training_config']).resolve(strict=True)
    contract_bytes = contract_path.read_bytes()
    training = OmegaConf.to_container(OmegaConf.load(io.StringIO(contract_bytes.decode())), resolve=True)
    interface = LearningInterface(EpisodeConfig.from_dict(training['episode']), LearningConfig(**training['wrappers']))
    read_training_contract(contract_path, interface)
    if contract_path.read_bytes() != contract_bytes:
        raise RuntimeError('source contract changed during feature conversion setup')
    converted = split_feature_config(training)
    checkpoint_bytes = Path(config['checkpoint']).resolve(strict=True).read_bytes()
    source = algorithm_type(training['rl']['policy_type']).load(io.BytesIO(checkpoint_bytes), device=device)
    destination, _ = create_learner(ObservationContractEnv(interface), interface,
        converted['rl'], {'kind': 'fresh'}, str(device), config['seed'])
    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=False)
    # The contract remains usable by the shared policy loader and every MARL
    # scheduler. This artifact has fresh optimizer/counters, not continuation.
    converted.update(training_method='split_ppo_features_weights_only', feature_split=config,
        seed=config['seed'], device=str(device), output=str(output))
    (output / 'config.yaml').write_text(OmegaConf.to_yaml(OmegaConf.create(converted)), encoding='utf-8')
    root = Path(__file__).resolve().parents[1]
    identity = {'source_checkpoint_sha256': hashlib.sha256(checkpoint_bytes).hexdigest(),
        'source_contract_sha256': hashlib.sha256(contract_bytes).hexdigest(),
        'source_hashes': {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in (root / 'src/soku_rl', root / 'tools') for p in sorted(folder.rglob('*.py'))},
        'packages': {name: version(name) for name in ('torch', 'stable-baselines3', 'sb3-contrib')},
        'device': str(device), 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES', '')}
    (output / 'identity.json').write_text(json.dumps(identity, indent=2))
    report = {'success': False, 'algorithm': training['algorithm']['name'],
        'method': 'split_ppo_features_weights_only', 'source_parameter_hash': parameter_hash(source.policy)}
    started = time.perf_counter()
    try:
        report.update(copy_split_feature_weights(source, destination))
        destination.save(output / 'final.zip')
        report.update(success=True, parameter_hash=parameter_hash(destination.policy),
            checkpoint_sha256=hashlib.sha256((output / 'final.zip').read_bytes()).hexdigest())
        print(json.dumps(report, indent=2), flush=True)
    except BaseException as error:
        report['error'] = repr(error)
        raise
    finally:
        report['seconds'] = time.perf_counter() - started
        (output / 'result.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
