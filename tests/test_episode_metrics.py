import pytest

from soku_rl.rl.episode_metrics import grouped_episode_metrics, summarize_episodes


def episode(player, outcome, opponent, character, loss):
    return {"outcome": outcome, "frame": 120, "episode": {"l": 60},
        "training_context": {"player": player, "opponent": opponent,
            "match": {f"player_{1-player}": {"character": character}}},
        "combat_metrics": {"available": True, "own_hp_loss": loss,
            "opponent_hp_loss": 10000 - loss, "own_hp_loss_frames": 2,
            "opponent_hp_loss_frames": 3, "own_final_hp": 10000 - loss,
            "opponent_final_hp": loss, "own_spell_action_entries": 1,
            "opponent_spell_action_entries": 2}}


def test_relative_wins_and_combat_are_stratified_by_seat_and_opponent():
    records = [episode(0, "p1_win", "rush", 0, 100),
               episode(1, "p2_win", "rush", 0, 300),
               episode(0, "p2_win", "god", 6, 10000),
               episode(1, "time_limit", "god", 6, 5000)]
    result = grouped_episode_metrics(records)
    assert result["overall"]["counts"] == dict(win=2, loss=1, double_ko=0, time_limit=1)
    assert result["overall"]["win_rate"] == .5
    assert result["overall"]["mean_frames"] == 120
    assert result["overall"]["mean_decisions"] == 60
    seats = {row["key"]: row for row in result["groups"]["learner_seat"]}
    assert seats[0]["combat"]["means"]["own_hp_loss"] == 5050
    assert seats[1]["combat"]["means"]["own_hp_loss"] == 2650
    opponents = {row["key"]: row for row in result["groups"]["opponent"]}
    assert opponents["rush"]["win_rate"] == 1
    assert opponents["god"]["win_rate"] == 0
    assert len(result["groups"]["opponent_character"]) == 2
    assert len(result["groups"]["matchup"]) == 4


def test_missing_combat_and_legacy_actions_do_not_become_zero():
    measured = episode(0, "p1_win", "rush", 0, 100)
    legacy = episode(1, "p1_win", "rush", 0, 300)
    del legacy["combat_metrics"]["own_spell_action_entries"]
    del legacy["combat_metrics"]["opponent_spell_action_entries"]
    missing = episode(0, "double_ko", "random", 0, 500)
    del missing["combat_metrics"]
    del missing["training_context"]["match"]
    result = grouped_episode_metrics([measured, legacy, missing])
    combat = result["overall"]["combat"]
    assert combat["episodes"] == 3
    assert combat["measured_episodes"] == 2
    assert combat["action_measured_episodes"] == 1
    assert combat["means"]["own_hp_loss"] == 200
    assert combat["action_means"]["own_spell_action_entries"] == 1
    assert result["groups"]["opponent_character"][0]["episodes"] == 2


def test_empty_rollout_has_counts_but_no_invented_means():
    result = summarize_episodes([])
    assert result["episodes"] == 0
    assert "win_rate" not in result
    assert "means" not in result["combat"]


def test_unresolved_random_seat_is_rejected():
    record = episode(0, "p1_win", "rush", 0, 100)
    record["training_context"]["player"] = "random"
    with pytest.raises(ValueError, match="actual learner seat"):
        summarize_episodes([record])


def test_empty_rollout_clears_stale_combat_scalars_before_delayed_dump(tmp_path):
    import csv
    import time
    from types import SimpleNamespace
    from stable_baselines3.common.logger import configure
    from soku_rl.rl.training import EpisodeRecords

    logger = configure(str(tmp_path / "scalars"), ["csv"])
    from soku_rl.rl.curriculum import FixedOpponentSchedule
    callback = EpisodeRecords(tmp_path, 100, FixedOpponentSchedule())
    callback.model = SimpleNamespace(logger=logger)
    callback.num_timesteps = 8
    callback.rollout_started = time.perf_counter()
    callback.records = [episode(0, "p1_win", "rush", 0, 100)]
    callback._on_rollout_end()
    assert logger.name_to_value["combat/own_hp_loss"] == 100
    assert logger.name_to_value["combat/win_rate"] == 1
    logger.record("train/loss", .123)
    callback.rollout_record_start = 1
    callback.rollout_started = time.perf_counter()
    callback._on_rollout_end()
    assert logger.name_to_value["combat/episodes"] == 0
    assert logger.name_to_value["combat/own_hp_loss"] is None
    assert logger.name_to_value["combat/win_rate"] is None
    assert logger.name_to_value["train/loss"] == .123
    logger.dump(16)
    logger.close()
    row = list(csv.DictReader((tmp_path / "scalars/progress.csv").open()))[0]
    assert row["combat/episodes"] == "0" and row["combat/own_hp_loss"] == ""
