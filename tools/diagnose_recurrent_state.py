"""Run the normal BR entry with an observational recurrent-memory diagnostic."""
import runpy

import hydra
import soku_rl.marl.br as br
from soku_rl.rl.recurrent_state_diagnostics import attach_recurrent_state_diagnostic


original_create = br.create_learner


def observed_create(env, interface, config, source, device, seed):
    model, metadata = original_create(env, interface, config, source, device, seed)
    attach_recurrent_state_diagnostic(model, interface.episode.max_frames)
    return model, metadata


@hydra.main(version_base='1.3', config_path='../config', config_name='diagnose_recurrent_state')
def main(cfg):
    if cfg.algorithm.name != 'br' or cfg.rl.policy_type != 'lstm':
        raise ValueError('online memory diagnostic requires recurrent BR training')
    br.create_learner = observed_create
    try:
        entry = runpy.run_path('tools/train.py', run_name='memory_diagnostic_training')
        entry['main'].__wrapped__(cfg)
    finally:
        br.create_learner = original_create


if __name__ == '__main__':
    main()
