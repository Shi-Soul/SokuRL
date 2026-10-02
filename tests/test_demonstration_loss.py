"""Changed-frame emphasis must preserve padding, global scoring and shared PPO artifacts."""
import numpy as np
import pytest
import torch

from soku_rl.rl.demonstration_loss import weighted_action_loss, weighted_validation_nll, validate_change_weight
from soku_rl.rl.behavior_cloning import fit_demonstrations, load_demonstrations
from test_behavior_cloning import dataset
from test_shared_ppo import fixture_config


@pytest.mark.parametrize('value', [0., .9, True, float('nan'), float('inf'), '4'])
def test_invalid_change_weight_fails(value):
    with pytest.raises(ValueError, match='action_change_weight'):
        validate_change_weight(value)


def test_actor_gradient_weights_ignore_padding_and_preserve_uniform_case():
    nll = torch.tensor([2., 3., 5., float('nan')], requires_grad=True)
    changed = torch.tensor([False, True, False, True])
    valid = torch.tensor([True, True, True, False])
    weighted = weighted_action_loss(nll, changed, valid, 4.)
    assert weighted.item() == pytest.approx(19 / 6)
    weighted.backward()
    torch.testing.assert_close(nll.grad, torch.tensor([1 / 6, 4 / 6, 1 / 6, 0.]))
    ordinary = weighted_action_loss(nll, changed, valid, 1.)
    assert torch.equal(ordinary, nll[:3].mean())


def test_validation_uses_global_frame_weights_and_handles_no_changes():
    # Five frames with NLLs 1, 2, 3, 4, 5; only the last two require a change.
    metrics = {'nll': 3., 'changed_samples': 2, 'changed_nll': 4.5}
    assert weighted_validation_nll(metrics, 5, 4.) == pytest.approx(42 / 11)
    assert weighted_validation_nll(metrics, 5, 1.) == 3.
    assert weighted_validation_nll({'nll': 3., 'changed_samples': 0}, 5, 4.) == 3.


@pytest.mark.parametrize('kind', ['mlp', 'lstm'])
def test_weighted_fit_selects_by_declared_objective_and_keeps_validation_labels(dataset, kind):
    directory, interface, _ = dataset
    torch.set_num_threads(1)
    samples, _, _, _ = load_demonstrations(directory, interface)
    # Add actual action transitions while retaining complete-episode boundaries.
    samples = {split: [(packed, 4 if index % 3 == 1 else action, target,
                       -1 if flag == -1 else 1) for index, (packed, action, target, flag) in enumerate(rows)]
               for split, rows in samples.items()}
    original_labels = [row[1:] for row in samples['validation']]
    settings = {'epochs': 3, 'batch_size': 4, 'value_coef': .5, 'action_change_weight': 4.,
                'initial_policy': {'kind': 'fresh'}}
    if kind == 'lstm':
        settings['sequence_length'] = 2
    output = directory / 'weighted-fit'
    output.mkdir()
    result = fit_demonstrations(interface, fixture_config(kind), samples, settings, 'cpu', 7, output)
    assert result['ppo_steps'] == 0 and result['supervised_updates'] > 0
    assert result['initial_policy_hash'] != result['final_policy_hash']
    assert result['selection'] == {'metric': 'weighted_nll', 'action_change_weight': 4.}
    scores = [row['validation']['weighted_nll'] for row in result['history']]
    assert result['best_epoch'] == int(np.argmin(scores))
    assert [row[1:] for row in samples['validation']] == original_labels
    for row in result['history']:
        m = row['validation']
        assert m['weighted_nll'] == pytest.approx(weighted_validation_nll(m, len(original_labels), 4.))
