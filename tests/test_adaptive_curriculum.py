"""Feedback direction, episode freezing, and checkpoint-complete BR curricula."""
import copy
import json
from pathlib import Path

from hydra import compose, initialize_config_dir
import numpy as np
from omegaconf import OmegaConf
import pytest

from soku_rl.marl.br import train_br
from soku_rl.policy.population import UniformPolicy
from soku_rl.rl.curriculum import AdaptiveActionNoise, FixedOpponentSchedule
from soku_rl.rl.matchup_env import MatchupMixtureVecEnv
from test_matchup_response import SeatGame
from test_shared_ppo import fixture_config, fixture_env, save_contract, torch


def settings():
    return {"kind": "adaptive_action_noise", "initial_random_probability": .5,
        "min_random_probability": .1, "max_random_probability": .9,
        "target_win_rate": .5, "deadband": .05, "ema_half_life": 50.,
        "warmup_episodes": 20, "update_every": 10, "gain": .2, "max_change": .05}


def controller(config):
    return AdaptiveActionNoise(config, [UniformPolicy("a", 4), UniformPolicy("b", 4)], [.5, .5], 4)


def observe(schedule, name, player, outcome):
    _, context = schedule.spawn(UniformPolicy(name, 4), 13)
    return schedule.observe(context | {"opponent": name, "player": player}, {"outcome": outcome})


def test_ema_seat_routing_warmup_and_per_opponent_isolation():
    schedule = controller(settings())
    outcomes = []
    for i in range(20):
        player = i % 2
        won = i % 3 == 0
        outcomes.append(won)
        event = observe(schedule, "a", player, f"p{player + 1 if won else 2 - player}_win")
        assert event["won"] == won
        if i < 19:
            assert event["reason"] == "warmup"
            assert event["next_random_probability"] == .5
    decay = 2 ** (-1 / 50)
    expected = sum(w * decay ** (19 - i) for i, w in enumerate(outcomes)) / sum(decay ** i for i in range(20))
    assert event["ema_win_rate"] == pytest.approx(expected)
    assert .5 < event["next_random_probability"] <= .55
    assert schedule.states["b"]["episodes"] == 0
    assert "ema_win_rate" not in schedule.snapshot()["states"]["b"]
    assert "curriculum/opponent_1/ema_win_rate" not in schedule.scalar_metrics()


@pytest.mark.parametrize("outcome,direction,bound", [
    ("p1_win", "decrease_uniform", .1), ("p2_win", "increase_uniform", .9),
    ("time_limit", "increase_uniform", .9), ("double_ko", "increase_uniform", .9)])
def test_bounded_gradual_adjustments_for_wins_and_nonwins(outcome, direction, bound):
    schedule = controller(settings())
    for n in range(1, 201):
        event = observe(schedule, "a", 0, outcome)
        assert abs(event["next_random_probability"] - event["previous_random_probability"]) <= .050000001
        if n == 20:
            assert event["reason"] == direction
        if n == 21:
            assert event["reason"] == "interval"
    assert event["next_random_probability"] == bound
    assert event["reason"] == "clamped"


def test_deadband_and_recovery_reverse_the_direction_without_resetting_history():
    schedule = controller(settings())
    for n in range(20):
        event = observe(schedule, "a", 0, "p1_win" if n % 2 else "p2_win")
    assert event["reason"] == "deadband"
    assert event["next_random_probability"] == .5
    for _ in range(80):
        observe(schedule, "a", 0, "p1_win")
    for _ in range(200):
        event = observe(schedule, "a", 1, "p1_win")
    assert event["ema_win_rate"] < .1
    assert event["next_random_probability"] > .5
    assert event["episodes"] == 300


@pytest.mark.parametrize("key,value", [("initial_random_probability", float("nan")),
    ("initial_random_probability", 1.1), ("ema_half_life", 0), ("warmup_episodes", True),
    ("update_every", 0), ("deadband", .5), ("gain", -1), ("max_change", 0)])
def test_invalid_controller_settings_fail_early(key, value):
    with pytest.raises(ValueError):
        controller(settings() | {key: value})


class OutcomeGame(SeatGame):
    def step(self, actions):
        result = super().step(actions)
        for pair in result[-1].values():
            for info in pair.values():
                info["outcome"] = "p1_win"
        return result


def test_parallel_completions_update_future_games_and_freeze_current_actors():
    game = OutcomeGame()
    opponents = [UniformPolicy("a", 4), UniformPolicy("b", 4)]
    setup = {"character": 1, "palette": 0, "deck": 0}
    view = MatchupMixtureVecEnv(game, "random", opponents, [.5, .5], 123, setup, [setup, setup])
    schedule = controller(settings() | {"warmup_episodes": 1, "update_every": 1})
    view.curriculum = schedule
    try:
        view.reset()
        original = dict(view.actors)
        contexts = copy.deepcopy(view.episode_context)
        view.step_async(np.zeros(8, dtype=int))
        _, _, _, infos = view.step_wait()
        assert {row["player"] for row in contexts.values()} == {0, 1}
        assert sum(row["episodes"] for row in schedule.states.values()) == 8
        for slot, info in enumerate(infos):
            name = contexts[slot]["opponent"]
            assert info["curriculum_event"]["won"] == (contexts[slot]["player"] == 0)
            assert info["training_context"]["curriculum"]["random_probability"] == .5
            assert original[slot].random_probability == .5
            next_name = view.episode_context[slot]["opponent"]
            assert view.actors[slot].random_probability == schedule.states[next_name]["random_probability"]
            assert info["training_context"]["opponent_fingerprint"] == next(p for p in opponents if p.name == name).fingerprint
    finally:
        view.close()


def test_sidecar_restores_future_feedback_exactly_and_rejects_mismatch(tmp_path):
    first = controller(settings())
    for n in range(67):
        observe(first, "a" if n % 2 else "b", n % 2, "p1_win")
    checkpoint = tmp_path / "model.zip"
    checkpoint.write_bytes(b"test-model")
    first.save(checkpoint, 1234)
    source = {"kind": "checkpoint", "path": str(checkpoint)}
    second = controller(settings())
    second.restore(source)
    assert second.snapshot() == first.snapshot()
    for n in range(103):
        args = ("a" if n % 3 else "b", n % 2, "p2_win" if n % 4 else "time_limit")
        assert observe(first, *args) == observe(second, *args)
    assert second.snapshot() == first.snapshot()
    with pytest.raises(ValueError, match="configuration"):
        FixedOpponentSchedule().restore(source)
    with pytest.raises(ValueError, match="config"):
        controller(settings() | {"gain": .3}).restore(source)
    checkpoint.write_bytes(b"different-model")
    with pytest.raises(ValueError, match="hash"):
        controller(settings()).restore(source)
    checkpoint.with_suffix(".curriculum.json").unlink()
    with pytest.raises(FileNotFoundError):
        controller(settings()).restore(source)


def test_active_episode_keeps_its_probability_when_another_episode_finishes(monkeypatch):
    game = OutcomeGame()
    step = game.step

    def partial(actions):
        result = step(actions)
        for slot, pair in result[2].items():
            if slot:
                pair.update(player_0=False, player_1=False)
        return result

    monkeypatch.setattr(game, "step", partial)
    opponent = UniformPolicy("a", 4)
    setup = {"character": 1, "palette": 0, "deck": 0}
    view = MatchupMixtureVecEnv(game, 0, [opponent], [1.], 123, setup, [setup])
    schedule = AdaptiveActionNoise(settings() | {"warmup_episodes": 1, "update_every": 1},
        [opponent], [1.], 4)
    view.curriculum = schedule
    try:
        view.reset()
        previous = view.actors[1]
        view.step_async(np.zeros(8, dtype=int))
        view.step_wait()
        assert schedule.states["a"]["random_probability"] == .45
        assert view.actors[0].random_probability == .45
        assert view.actors[1] is previous
        assert view.actors[1].random_probability == .5
        assert view.episode_context[1]["curriculum"]["random_probability"] == .5
    finally:
        view.close()


@pytest.mark.parametrize("checkpoint", ["final.zip", "checkpoints/updated_8_steps.zip", "checkpoints/ppo_8_steps.zip"])
def test_actual_ppo_training_records_and_resumes_curriculum(tmp_path, checkpoint):
    torch.set_num_threads(1)
    env = fixture_env()
    config = fixture_config("mlp") | {"name": "br", "player": 1,
        "matchups": {"mode": "fixed"}, "timesteps": 16, "checkpoint_every": 8,
        "initial_policy": {"kind": "fresh"}, "curriculum": settings() | {
            "warmup_episodes": 1, "update_every": 1},
        "opponents": [{"name": "random", "probability": 1., "policy": {"kind": "uniform"}}]}
    first, second, third = [tmp_path / name for name in ("first", "second", "weights")]
    for path in (first, second, third):
        path.mkdir()
    try:
        result = train_br(env, config, "cpu", 13, first)
        progress = json.loads((first / "progress.json").read_text())
        assert progress["curriculum"] == result["curriculum"]
        assert all("curriculum_event" in row for row in progress["episodes"])
        assert "curriculum/opponent_0/ema_win_rate" in (first / "scalars/progress.csv").read_text()
        contract = save_contract(first, env, config)
        path = first / checkpoint
        state = json.loads(path.with_suffix(".curriculum.json").read_text())
        config["initial_policy"] = {"kind": "checkpoint", "path": str(path), "training_config": contract}
        resumed = train_br(env, config, "cpu", 14, second)
        new_records = json.loads((second / "progress.json").read_text())["episodes"]
        assert resumed["start_steps"] == state["steps"]
        assert resumed["curriculum"]["states"]["random"]["episodes"] == state["states"]["random"]["episodes"] + len(new_records)
        assert new_records[0]["training_context"]["curriculum"]["random_probability"] == state["states"]["random"]["random_probability"]
        config["initial_policy"]["kind"] = "weights"
        reset = train_br(env, config, "cpu", 14, third)
        assert reset["start_steps"] == 0
        assert reset["curriculum"]["states"]["random"]["episodes"] == len(progress["episodes"])
    finally:
        env.close()


def test_hydra_curriculum_leaves_full_god_population_for_independent_evaluation():
    with initialize_config_dir(config_dir=str(Path(__file__).parents[1] / "config"), version_base="1.3"):
        cfg = compose(config_name="train", overrides=["algorithm=br", "rules=god",
            "wrappers=superhuman_learning", "track=superhuman_combat",
            "+br_opponents=god_all", "+curriculum=adaptive_noise"])
        algorithm = OmegaConf.to_container(cfg.algorithm, resolve=True)
    assert algorithm["curriculum"]["ema_half_life"] == 50
    assert len(algorithm["opponents"]) == 27
    assert all(row["policy"]["kind"] == "rule" for row in algorithm["opponents"])
