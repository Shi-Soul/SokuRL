import pytest
import torch
import numpy as np
from gymnasium import spaces
from types import SimpleNamespace
from stable_baselines3.common.distributions import CategoricalDistribution

from soku_rl.rl.command_diagnostics import command_group_totals, summarize_command_groups
from soku_rl.rl.behavior_cloning import ObservationContractEnv, score_samples
from soku_rl.rl.ppo import algorithm_type
from soku_rl.rl.recurrent_cloning import sequence_epoch
from soku_rl.rl.storage import PackedObservation


def test_button_groups_count_directional_commands_and_exclude_padding():
    logits = torch.full((4, 576), -1000.)
    # Actual labels: attack, card-change, spell, then a padded attack frame.
    labels = torch.tensor([64 + 1, 256 + 16, 512 + 32, 1])
    logits[0, 65] = 0.
    logits[1, 256] = 0.
    logits[2, 544] = 0.
    logits[3, 1] = 0.
    distribution = CategoricalDistribution(576).proba_distribution(logits)
    totals = command_group_totals(distribution, labels, torch.tensor([True, True, True, False]))
    score = summarize_command_groups(totals)
    assert totals['frames'] == 3
    assert score['attack_label_frames'] == 1
    assert score['attack_label_rate'] == pytest.approx(1/3)
    assert score['attack_exact_accuracy'] == score['attack_mode_recall'] == 1
    assert score['attack_expected_rate'] == pytest.approx(1/3)
    assert score['change_card_exact_accuracy'] == score['change_card_mode_recall'] == 0
    assert score['change_card_label_nll'] == pytest.approx(1000.)  # finite even when probability underflows
    assert score['spellcard_exact_accuracy'] == 1


def test_uniform_distribution_and_empty_positive_groups_have_explicit_denominators():
    distribution = CategoricalDistribution(576).proba_distribution(torch.zeros(2, 576))
    labels = torch.tensor([256, 256])
    totals = command_group_totals(distribution, labels, torch.ones(2, dtype=torch.bool))
    scores = summarize_command_groups(totals)
    assert scores['attack_expected_rate'] == pytest.approx(7/8)
    assert scores['spellcard_expected_rate'] == pytest.approx(.5)
    assert scores['change_card_expected_rate'] == pytest.approx(.5)
    assert scores['attack_label_frames'] == 0 and 'attack_exact_accuracy' not in scores
    with pytest.raises(ValueError):
        command_group_totals(distribution, labels, torch.zeros(2, dtype=torch.bool))
    with pytest.raises(ValueError):
        summarize_command_groups({})


@pytest.mark.parametrize('kind', ['mlp', 'lstm'])
def test_real_policy_validation_counts_command_groups_without_padding_or_updates(kind):
    torch.set_num_threads(1)
    interface = SimpleNamespace(observation_space=spaces.Box(-1., 1., (2,), np.float32),
        action_space=spaces.Discrete(576))
    kwargs = {'net_arch': [8]}
    if kind == 'lstm':
        kwargs['lstm_hidden_size'] = 8
    model = algorithm_type(kind)('MlpLstmPolicy' if kind == 'lstm' else 'MlpPolicy',
        ObservationContractEnv(interface), n_steps=4, batch_size=4, device='cpu', seed=7, policy_kwargs=kwargs)
    packed = PackedObservation.pack(np.zeros(2, np.float32))
    episodes = [[(packed, action, 0., -1 if index == 0 else 1) for index, action in enumerate(actions)]
        for actions in ([65, 272, 544], [256])]
    if kind == 'lstm':
        scores, _, updates = sequence_epoch(model, episodes, [0, 1], 4, 2, 0., False, 1.)
        assert updates == 0
    else:
        scores = score_samples(model, sum(episodes, []), 4)
    for name in ('attack', 'change_card', 'spellcard'):
        assert scores[f'{name}_label_frames'] == 1
        assert scores[f'{name}_label_rate'] == .25
        assert 0 <= scores[f'{name}_expected_rate'] <= 1
    assert not model.policy.optimizer.state and model.num_timesteps == 0
