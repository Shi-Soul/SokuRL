"""A BR benchmark must swap characters with the model and count all outcomes."""
import json

import pytest

from soku_rl.evaluation.br import benchmark_br, matchup_plan
from soku_rl.policy.population import SeatPolicies, UniformPolicy
from test_matchup_response import SeatGame
from soku_rl.policy.matchups import select_opponents


class EvaluationGame(SeatGame):
    def __init__(self, outcome):
        self.outcome = outcome
        self.resets = []

    def reset_matchups(self, seeds, matches):
        self.resets.append(dict(matches))
        return super().reset_matchups(seeds, matches)

    def step(self, actions):
        observations, rewards, terms, truncs, infos = super().step(actions)
        for pair in infos.values():
            for info in pair.values():
                info.update(outcome=self.outcome, frame=1)
        return observations, rewards, terms, truncs, infos


def benchmark_inputs():
    strategies = {name: SeatPolicies(name, (UniformPolicy(name, 4), UniformPolicy(name, 4)))
                  for name in ("learned", "reimu", "remilia")}
    learner = {"character": 1, "palette": 0, "deck": 0}
    setups = {name: {"character": c, "palette": 0, "deck": 0}
              for name, c in (("reimu", 0), ("remilia", 6))}
    config = {"world_seeds": [101, 103], "policy_seed": 19, "alpha": .05}
    return strategies, learner, setups, config


def test_pairing_keeps_logical_policy_seeds_and_hashes_character_setup():
    strategies, learner, setups, config = benchmark_inputs()
    plan = matchup_plan(strategies, "learned", learner, setups, config, "game")
    assert len(plan) == 8
    for trial in plan:
        mate, = [t for t in plan if t.block_id == trial.block_id and t.trial_id != trial.trial_id]
        assert mate.policy_seeds[::-1] == trial.policy_seeds
        assert mate.world_seed == trial.world_seed
        assert trial.match[f"player_{trial.learner_seat}"] == learner
        assert trial.match[f"player_{1-trial.learner_seat}"] == setups[trial.opponent]
    changed = matchup_plan(strategies, "learned", learner | {"character": 2}, setups, config, "game")
    assert not {t.trial_id for t in plan} & {t.trial_id for t in changed}


def test_explicit_screening_panel_keeps_declared_names_and_order():
    population = [{"name": "reimu"}, {"name": "remilia"}, {"name": "suwako"}]
    assert select_opponents(population, ["suwako", "reimu"]) == [population[2], population[0]]
    assert select_opponents(population, "all") == population
    for names in ([], ["reimu", "reimu"], "reimu", ["unknown"]):
        with pytest.raises(ValueError):
            select_opponents(population, names)


def test_completed_game_survives_failure_of_another_slot(tmp_path):
    class FailingGame(EvaluationGame):
        def step(self, actions):
            if 0 not in actions:
                saved = json.loads((tmp_path / "progress.json").read_text())
                assert len(saved["games"]) == 1
                assert saved["summary"]["missing_games"] == 7
                raise RuntimeError("remaining slot failed")
            observations, rewards, terms, truncs, infos = super().step(actions)
            for slot, pair in terms.items():
                if slot != 0:
                    for agent in pair:
                        pair[agent] = False
            return observations, rewards, terms, truncs, infos

    strategies, learner, setups, config = benchmark_inputs()
    with pytest.raises(RuntimeError, match="remaining slot failed"):
        benchmark_br(FailingGame("p1_win"), strategies, "learned", learner,
                     setups, config, "game", tmp_path)
    record, = json.loads((tmp_path / "progress.json").read_text())["games"]
    assert (tmp_path / record["replay"]).is_file()
    failures = json.loads((tmp_path / "failure.json").read_text())["active_trials"]
    assert len(failures) == 7
    assert record["trial_id"] not in {trial["trial_id"] for trial in failures}


@pytest.mark.parametrize("outcome", ["p1_win", "p2_win", "double_ko", "time_limit"])
def test_benchmark_counts_seats_and_censoring_without_confusing_timeouts(tmp_path, outcome):
    strategies, learner, setups, config = benchmark_inputs()
    game = EvaluationGame(outcome)
    report = benchmark_br(game, strategies, "learned", learner, setups, config, "game", tmp_path)
    assert report["summary"]["missing_games"] == 0
    assert len(report["games"]) == 8
    assert len(report["by_opponent_and_seat"]) == 4
    for row in report["by_opponent_and_seat"]:
        assert row["games"] == 2
        if outcome in ("double_ko", "time_limit"):
            assert row["counts"][outcome] == 2
            assert row["win_rate"] == 0
        else:
            won = outcome == ("p1_win" if row["learner_seat"] == 0 else "p2_win")
            assert row["win_rate"] == float(won)
            assert row["counts"]["win" if won else "loss"] == 2
    saved = json.loads((tmp_path / "plan.json").read_text())
    for slot, trial in enumerate(saved):
        assert game.resets[0][slot].player_0.character == trial["match"]["player_0"]["character"]
        assert (tmp_path / f'{trial["trial_id"]}.npz').is_file()
