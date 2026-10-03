"""Plot completed online-teacher updates without inventing missing cumulative baselines."""
import json
import math

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np


def validate_teacher_sources(teacher):
    if not any(key.startswith(('online_', 'replay_')) for key in teacher):
        return
    metrics = ('nll_before', 'nll_after', 'accuracy_before', 'accuracy_after')
    required = {f'{source}_{key}' for source in ('online', 'replay')
                for key in (*metrics, 'frames', 'burn_in_frames')}
    if not required <= teacher.keys():
        raise ValueError('mixed teacher metrics require both complete source partitions')
    for source in ('online', 'replay'):
        for key, minimum in (('frames', 1), ('burn_in_frames', 0)):
            value = teacher[f'{source}_{key}']
            if type(value) is not int or value < minimum:
                raise ValueError('mixed teacher counts must be valid integers')
        for key in metrics:
            value = teacher[f'{source}_{key}']
            if not math.isfinite(value) or value < 0 or key.startswith('accuracy') and value > 1:
                raise ValueError('invalid mixed teacher fit metric')
    if (teacher['online_frames'] + teacher['replay_frames'] != teacher['frames']
            or teacher['online_burn_in_frames'] + teacher['replay_burn_in_frames'] != teacher['current_burn_in_frames']):
        raise ValueError('mixed teacher counts do not partition the recorded total')
    for key in metrics:
        weighted = sum(teacher[f'{source}_{key}'] * teacher[f'{source}_frames']
                       for source in ('online', 'replay')) / teacher['frames']
        if not math.isclose(weighted, teacher[key], rel_tol=1e-7, abs_tol=1e-7):
            raise ValueError('mixed teacher fit does not match frame-weighted total')


def teacher_series(rollouts):
    points, previous = [], None
    for rollout in rollouts:
        if 'update_seconds' not in rollout:
            continue
        if 'online_teacher' not in rollout:
            previous = None
            continue
        teacher = rollout['online_teacher']
        if teacher['ppo_steps'] != rollout['steps']:
            raise ValueError('teacher update and completed rollout steps differ')
        for key in ('nll_before', 'nll_after', 'accuracy_before', 'accuracy_after', 'behavior_agreement',
                    'query_seconds', 'collection_seconds', 'seconds', 'current_burn_in_frames'):
            if not math.isfinite(teacher[key]) or teacher[key] < 0:
                raise ValueError(f'invalid teacher metric: {key}')
        if any(teacher[key] > 1 for key in ('accuracy_before', 'accuracy_after', 'behavior_agreement')):
            raise ValueError('teacher accuracy and agreement must be fractions')
        if teacher['collection_seconds'] < teacher['query_seconds']:
            raise ValueError('teacher collection time must include querying')
        for key in ('frames', 'rollout_frames', 'collected_frames', 'updates'):
            if type(teacher[key]) is not int or teacher[key] <= 0:
                raise ValueError(f'invalid teacher count: {key}')
        validate_teacher_sources(teacher)
        cycle = rollout['rollout_seconds'] + rollout['update_seconds']
        if not math.isfinite(cycle) or cycle <= 0:
            raise ValueError('teacher overhead requires a positive measured cycle time')
        point = {'steps': rollout['steps'], **teacher, 'cycle_seconds': cycle,
                 'collection_delta_seconds': None, 'measured_extra_fraction': None}
        if previous is not None:
            if (teacher['updates'] <= previous['updates']
                    or teacher['collected_frames'] - previous['collected_frames'] != teacher['rollout_frames']
                    or teacher['query_seconds'] < previous['query_seconds']
                    or teacher['collection_seconds'] < previous['collection_seconds']):
                raise ValueError('teacher cumulative counters or times are inconsistent')
            delta = teacher['collection_seconds'] - previous['collection_seconds']
            point['collection_delta_seconds'] = delta
            point['measured_extra_fraction'] = (delta + teacher['seconds']) / cycle
        points.append(point)
        previous = teacher
    return points


def plot_teacher(labels, output, palette):
    series = {}
    for label in labels:
        rollouts = json.loads((output / label / 'timing.json').read_text())['rollouts']
        points = teacher_series(rollouts)
        if points:
            series[label] = points
    if not series:
        return
    figure, axes = plt.subplots(3, 2, figsize=(12, 10), layout='constrained')
    titles = ('Teacher-label cross entropy (nats)', 'Teacher-label accuracy on sampled windows',
              'Cumulative agreement with executed actions', 'Supervised window frames per rollout',
              'Actor prefix replay frames per rollout', 'Measured teacher overhead / complete cycle')
    for label, points in series.items():
        color = palette[labels.index(label)]
        x = np.asarray([point['steps'] for point in points]) / 1000
        for axis, metric in ((axes[0, 0], 'nll'), (axes[0, 1], 'accuracy')):
            for moment, style, marker in (('before', '--', 'x'), ('after', '-', 'o')):
                axis.plot(x, [p[f'{metric}_{moment}'] for p in points], color=color, linestyle=style,
                          marker=marker, markersize=3, label=f'{label}: {moment} auxiliary update')
        for axis, metric in zip(axes.flat[2:], ('behavior_agreement', 'frames',
                'current_burn_in_frames', 'measured_extra_fraction'), strict=True):
            axis.plot(x, [p[metric] if p[metric] is not None else np.nan for p in points],
                      color=color, marker='o', markersize=3, label=label)
    for axis, title in zip(axes.flat, titles, strict=True):
        axis.set_title(title, fontsize=10)
        axis.set_xlabel('Environment steps (thousands)')
        axis.set_ylim(bottom=0)
        axis.grid(alpha=.2)
        axis.spines[['top', 'right']].set_visible(False)
        axis.legend(fontsize=8)
    for axis in (axes[0, 1], axes[1, 0]):
        axis.set_ylim(0, 1)
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    axes[2, 1].yaxis.set_major_formatter(PercentFormatter(xmax=1))
    figure.suptitle('Online rule-teacher diagnostics — completed updates only\n'
        'Teacher-label fit on sampled states is not a policy-strength evaluation\n'
        'Window frames include repeated samples; overhead includes collection and auxiliary update\n'
        'First overhead point omitted: no prior cumulative baseline in the snapshot', fontsize=11)
    figure.savefig(output / 'teacher.png', dpi=150, bbox_inches='tight')
    figure.savefig(output / 'teacher.pdf', bbox_inches='tight')
    plt.close(figure)
    (output / 'teacher_series.json').write_text(json.dumps(series, indent=2))
    plot_teacher_sources(series, labels, output, palette)


def plot_teacher_sources(series, labels, output, palette):
    mixed = {label: [point for point in points if 'replay_frames' in point] for label, points in series.items()}
    mixed = {label: points for label, points in mixed.items() if points}
    if not mixed:
        return
    figure, axes = plt.subplots(3, 2, figsize=(12, 10), layout='constrained')
    for label, points in mixed.items():
        color = palette[labels.index(label)]
        x = np.asarray([point['steps'] for point in points]) / 1000
        for column, source in enumerate(('online', 'replay')):
            for row, metric in enumerate(('nll', 'accuracy')):
                for moment, style, marker in (('before', '--', 'x'), ('after', '-', 'o')):
                    axes[row, column].plot(x, [point[f'{source}_{metric}_{moment}'] for point in points],
                        color=color, linestyle=style, marker=marker, markersize=3,
                        label=f'{label}: {moment} update')
            style = '-' if source == 'online' else '--'
            for panel, metric in enumerate(('frames', 'burn_in_frames')):
                axes[2, panel].plot(x, [point[f'{source}_{metric}'] for point in points],
                    color=color, linestyle=style, marker='o', markersize=3, label=f'{label}: {source}')
    titles = ('Online-state cross entropy (nats)', 'Training-demonstration cross entropy (nats)',
              'Online-state label accuracy', 'Training-demonstration label accuracy',
              'Supervised window frames by source', 'Prefix replay frames by source')
    for axis, title in zip(axes.flat, titles, strict=True):
        axis.set_title(title, fontsize=10)
        axis.set_xlabel('Environment steps (thousands)')
        axis.set_ylim(bottom=0)
        axis.grid(alpha=.2)
        axis.spines[['top', 'right']].set_visible(False)
        axis.legend(fontsize=8)
    for axis in axes[1]:
        axis.set_ylim(0, 1)
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    figure.suptitle('Teacher supervision by source — completed updates only\n'
        'Demonstrations use training games only; neither source is a strength evaluation\n'
        'Frames include repeated samples; prefix counts include before/after recomputation', fontsize=11)
    figure.savefig(output / 'teacher_sources.png', dpi=150, bbox_inches='tight')
    figure.savefig(output / 'teacher_sources.pdf', bbox_inches='tight')
    plt.close(figure)
