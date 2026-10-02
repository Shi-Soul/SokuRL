import copy
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest

from soku_rl.policy.population import UniformPolicy
from soku_rl.rl.curriculum import AdaptiveActionNoise, AdaptiveEpisodeMixture, create_curriculum
from soku_rl.rl.episode_metrics import grouped_episode_metrics
from test_adaptive_curriculum import settings
from test_episode_metrics import episode


class ScriptPolicy:
    name = 'a'
    fingerprint = 'script-with-private-counter'

    def __init__(self):
        self.spawned = []

    def spawn(self, seed):
        self.spawned.append(seed)
        return ScriptActor(seed)


class ScriptActor:
    def __init__(self, seed):
        self.counter = seed % 576

    def act(self, observation):
        self.counter = (self.counter + 1) % 576
        return self.counter


def schedule(probability):
    opponent = ScriptPolicy()
    config = settings() | {'kind': 'adaptive_episode_mixture', 'initial_random_probability': probability,
        'min_random_probability': 0., 'max_random_probability': 1., 'warmup_episodes': 1, 'update_every': 1}
    return create_curriculum(config, [opponent], [1.], 576), opponent


@pytest.mark.parametrize('probability,selection', [(0., 'original'), (1., 'uniform')])
def test_endpoints_execute_the_intact_original_or_uniform_policy(probability, selection):
    controller, opponent = schedule(probability)
    actor, context = controller.spawn(opponent, 117)
    reference = ScriptActor(117) if selection == 'original' else UniformPolicy('reference', 576).spawn(117)
    assert context['curriculum']['selected_policy'] == selection
    assert context['curriculum']['random_probability'] == probability
    assert opponent.spawned == ([117] if selection == 'original' else [])
    assert [actor.act({}) for _ in range(1024)] == [reference.act({}) for _ in range(1024)]


def test_seeded_selection_is_order_independent_and_each_game_keeps_its_actor():
    controller, opponent = schedule(.5)
    selections = []
    for seed in range(40):
        actor, context = controller.spawn(opponent, seed)
        reverse, reverse_opponent = schedule(.5)
        for unrelated in range(50, 55):
            reverse.spawn(reverse_opponent, unrelated)
        twin, twin_context = reverse.spawn(reverse_opponent, seed)
        assert context == twin_context
        assert [actor.act({}) for _ in range(20)] == [twin.act({}) for _ in range(20)]
        selected = context['curriculum']['selected_policy']
        selections.append(selected)
        # A feedback update changes the future mixture, not this actor or its memory.
        controller.observe(context | {'opponent': 'a', 'player': 0}, {'outcome': 'p1_win'})
        assert [actor.act({}) for _ in range(20)] == [twin.act({}) for _ in range(20)]
        assert context['curriculum']['random_probability'] == .5
        controller, opponent = schedule(.5)
    assert set(selections) == {'original', 'uniform'}


def test_mixture_fingerprint_and_feedback_report_both_probability_and_actual_policy():
    first, opponent = schedule(.5)
    actor, context = first.spawn(opponent, 7)
    before = copy.deepcopy(context)
    event = first.observe(context | {'opponent': 'a', 'player': 1}, {'outcome': 'p2_win'})
    assert context == before
    assert event['selected_policy'] == context['curriculum']['selected_policy']
    assert event['won'] and event['reason'] == 'decrease_uniform'
    assert event['episode_random_probability'] == .5
    assert event['next_random_probability'] == pytest.approx(.45)
    _, next_context = first.spawn(opponent, 7)
    assert next_context['curriculum']['training_opponent_fingerprint'] != context['curriculum']['training_opponent_fingerprint']
    expected = (opponent.fingerprint if event['selected_policy'] == 'original' else 'uniform-576-v1')
    assert context['curriculum']['selected_policy_fingerprint'] == expected


def test_sidecar_restores_selection_and_rejects_action_noise_checkpoints(tmp_path):
    first, opponent = schedule(.5)
    for seed in range(15):
        _, context = first.spawn(opponent, seed)
        first.observe(context | {'opponent': 'a', 'player': seed % 2}, {'outcome': 'p1_win'})
    checkpoint = tmp_path / 'policy.zip'
    checkpoint.write_bytes(b'checkpoint-identity')
    first.save(checkpoint, 123)
    second, other = schedule(.5)
    second.restore({'kind': 'checkpoint', 'path': str(checkpoint)})
    assert second.snapshot() == first.snapshot()
    assert second.snapshot()['kind'] == 'adaptive_episode_mixture'
    for seed in range(20, 30):
        a, ca = first.spawn(opponent, seed)
        b, cb = second.spawn(other, seed)
        assert ca == cb
        assert [a.act({}) for _ in range(10)] == [b.act({}) for _ in range(10)]
    old = AdaptiveActionNoise(first.config | {'kind': 'adaptive_action_noise'}, [opponent], [1.], 576)
    with pytest.raises(ValueError, match='kind'):
        old.restore({'kind': 'checkpoint', 'path': str(checkpoint)})


def test_hydra_inherits_feedback_parameters_without_wrapping_evaluation_opponents():
    with initialize_config_dir(config_dir=str(Path(__file__).parents[1] / 'config'), version_base='1.3'):
        base = ['algorithm=br', 'rules=god', '+br_opponents=god_all']
        old = OmegaConf.to_container(compose(config_name='train', overrides=base + ['+curriculum=adaptive_noise']).algorithm, resolve=True)
        new = OmegaConf.to_container(compose(config_name='train', overrides=base + ['+curriculum=adaptive_episode_mixture']).algorithm, resolve=True)
    assert new['curriculum']['kind'] == AdaptiveEpisodeMixture.kind
    new['curriculum']['kind'] = old['curriculum']['kind']
    assert old == new
    assert len(new['opponents']) == 27
    assert all(row['policy']['kind'] == 'rule' for row in new['opponents'])


def test_combat_and_wins_are_separated_by_actual_curriculum_policy():
    uniform = episode(0, 'p1_win', 'god', 0, 100)
    original = episode(1, 'p1_win', 'god', 0, 10000)
    legacy = episode(0, 'time_limit', 'god', 0, 5000)
    uniform['training_context']['curriculum'] = {'selected_policy': 'uniform'}
    original['training_context']['curriculum'] = {'selected_policy': 'original'}
    groups = grouped_episode_metrics([uniform, original, legacy])['groups']
    kinds = {row['key']: row for row in groups['curriculum_policy']}
    assert kinds['uniform']['win_rate'] == 1 and kinds['original']['win_rate'] == 0
    assert kinds['original']['combat']['means']['own_hp_loss'] == 10000
    assert sum(row['episodes'] for row in kinds.values()) == 2  # legacy is not labeled uniform
    assert len(groups['curriculum_matchup']) == len(groups['curriculum_opponent']) == 2
    assert 'curriculum_policy' not in grouped_episode_metrics([legacy])['groups']
    original['training_context']['curriculum']['selected_policy'] = 'invalid'
    with pytest.raises(ValueError, match='unknown episode'):
        grouped_episode_metrics([original])
