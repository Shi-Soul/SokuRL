"""Teacher diagnostics use actual completed updates and measured cost increments."""
import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'tools'))
from teacher_curves import teacher_series


def rollouts():
    rows = []
    for index in (1, 2):
        rows.append({'steps': 8192 + index * 2048, 'rollout_seconds': 20., 'update_seconds': 5.,
            'online_teacher': {'ppo_steps': 8192 + index * 2048, 'nll_before': 2., 'nll_after': 1.9,
                'accuracy_before': .4, 'accuracy_after': .5, 'behavior_agreement': .3,
                'query_seconds': 10. + index, 'collection_seconds': 20. + index * 2,
                'seconds': 3., 'current_burn_in_frames': 4000,
                'frames': 500, 'rollout_frames': 2048, 'collected_frames': 8192 + index * 2048,
                'updates': 4 + index}})
    return rows


def test_resumed_cumulative_cost_uses_differences_and_does_not_fabricate_first_baseline():
    rows = rollouts()
    rows.append({'steps': 14336, 'rollout_seconds': 20.})
    original = copy.deepcopy(rows)
    points = teacher_series(rows)
    assert rows == original
    assert [p['steps'] for p in points] == [10240, 12288]
    assert points[0]['measured_extra_fraction'] is None
    assert points[0]['collection_delta_seconds'] is None
    assert points[1]['collection_delta_seconds'] == 2.
    assert points[1]['measured_extra_fraction'] == (2 + 3) / (20 + 5)
    assert points[1]['frames'] == 500


@pytest.mark.parametrize('key,value', [('ppo_steps', 1), ('nll_before', float('nan')),
    ('accuracy_after', 50), ('collected_frames', 1), ('updates', 5), ('query_seconds', 1)])
def test_invalid_alignment_or_metrics_are_rejected(key, value):
    rows = rollouts()
    rows[1]['online_teacher'][key] = value
    with pytest.raises(ValueError):
        teacher_series(rows)


def test_no_teacher_runs_do_not_become_zero_loss_or_zero_overhead():
    assert teacher_series([{'steps': 2048, 'rollout_seconds': 20, 'update_seconds': 1}]) == []


def mixed_rollouts():
    rows = rollouts()
    for row in rows:
        teacher = row['online_teacher']
        teacher.update(online_frames=120, replay_frames=380, online_burn_in_frames=1000, replay_burn_in_frames=3000)
        for key, online in (('nll_before', 1.), ('nll_after', .9), ('accuracy_before', .2), ('accuracy_after', .3)):
            teacher[f'online_{key}'] = online
            teacher[f'replay_{key}'] = (teacher[key] * 500 - online * 120) / 380
    return rows


def test_mixed_sources_preserve_unequal_frame_weighting():
    points = teacher_series(mixed_rollouts())
    assert points[0]['online_frames'] == 120 and points[0]['replay_frames'] == 380
    assert points[0]['online_nll_before'] == 1 and points[0]['nll_before'] == 2
    assert points[0]['replay_nll_before'] > points[0]['nll_before']


@pytest.mark.parametrize('key,value', [('online_frames', True), ('replay_frames', 0),
    ('replay_frames', 381), ('replay_burn_in_frames', 3001),
    ('online_accuracy_after', float('nan')), ('replay_accuracy_before', 2.), ('online_nll_after', .4)])
def test_invalid_source_counts_or_weighting_are_rejected(key, value):
    rows = mixed_rollouts()
    rows[0]['online_teacher'][key] = value
    with pytest.raises(ValueError, match='mixed teacher'):
        teacher_series(rows)


def test_partial_source_metrics_do_not_become_fabricated_zeroes():
    rows = mixed_rollouts()
    del rows[0]['online_teacher']['replay_frames']
    with pytest.raises(ValueError, match='complete source partitions'):
        teacher_series(rows)
