"""Cross-seat responses must route actions, rewards and terminal state together."""
from dataclasses import asdict, replace
import json

from gymnasium import spaces
import numpy as np
from omegaconf import OmegaConf
import pytest

from soku_rl.env.match import MatchConfig, PlayerSetup
from soku_rl.env.wrappers.learning import LearningInterface
from soku_rl.policy.contract import read_training_contract
from soku_rl.policy.population import UniformPolicy
from soku_rl.rl.matchup_env import MatchupMixtureVecEnv
from test_shared_ppo import fixture_env, fixture_config, save_contract, torch
from soku_rl.marl.br import train_br
from hydra import compose, initialize_config_dir
from pathlib import Path


class SeatGame:
    num_envs = 8
    single_observation_space = spaces.Box(-100, 100, (1,), np.float32)
    single_action_space = spaces.Discrete(4)

    def reset_matchups(self, seeds, matches):
        self.matches = matches
        return ({s: {f"player_{p}": np.array([10 * s + p], np.float32) for p in (0, 1)} for s in seeds},
                {s: {f"player_{p}": {} for p in (0, 1)} for s in seeds})

    def step(self, actions):
        self.actions = actions
        observations = {s: {f"player_{p}": np.array([-10 * s - p - 1], np.float32)
                            for p in (0, 1)} for s in actions}
        rewards = {s: {"player_0": 1., "player_1": -1.} for s in actions}
        terms = {s: dict.fromkeys(("player_0", "player_1"), True) for s in actions}
        truncs = {s: dict.fromkeys(("player_0", "player_1"), False) for s in actions}
        infos = {s: {f"player_{p}": {"episode": 1, "base_reward": rewards[s][f"player_{p}"]}
                     for p in (0, 1)} for s in actions}
        return observations, rewards, terms, truncs, infos


def test_sampled_seats_keep_learner_character_and_route_terminal_records():
    game = SeatGame()
    learner = {"character": 1, "palette": 0, "deck": 0}
    setups = [{"character": c, "palette": 0, "deck": 0} for c in (0, 6)]
    opponents = [UniformPolicy(str(c), 4) for c in (0, 6)]
    view = MatchupMixtureVecEnv(game, "random", opponents, [.5, .5], 123, learner, setups)
    try:
        observations = view.reset()
        seen = set()
        for _ in range(4):
            players = dict(view.players)
            for s, p in players.items():
                match = asdict(game.matches[s])
                assert match[f"player_{p}"] == learner
                assert match[f"player_{1-p}"] == setups[view.opponent_indices[s]]
                assert observations[s, 0] == 10 * s + p
                seen.add((p, view.opponent_indices[s]))
            view.step_async(np.full(8, 3))
            observations, rewards, dones, infos = view.step_wait()
            assert dones.all()
            for s, p in players.items():
                assert game.actions[s][f"player_{p}"] == 3
                assert rewards[s] == (1 if p == 0 else -1)
                assert infos[s]["terminal_observation"][0] == -10 * s - p - 1
                assert infos[s]["training_context"]["player"] == p
                assert infos[s]["training_context"]["base_return"] == rewards[s]
                assert infos[s]["training_context"]["match"][f"player_{p}"] == learner
        assert seen == {(0, 0), (0, 1), (1, 0), (1, 1)}
    finally:
        view.close()


def test_sampled_br_updates_one_model_and_retains_matchup_context(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    env = fixture_env()
    backend = env.env.backend
    monkeypatch.setattr(backend, "reset_matchups", lambda seeds, matches: backend.reset_slots(seeds), raising=False)
    config = fixture_config("mlp") | {"name": "br", "player": "random", "timesteps": 16,
        "checkpoint_every": 8, "initial_policy": {"kind": "fresh"},
        "matchups": {"mode": "sampled", "learner": {"character": 1, "palette": 0, "deck": 0}},
        "opponents": [{"name": "random", "probability": 1., "policy": {"kind": "uniform"},
                       "setup": {"character": 6, "palette": 0, "deck": 0}}]}
    try:
        report = train_br(env, config, "cpu", 19, tmp_path)
        assert report["additional_steps"] == 16
        assert (tmp_path / "final.zip").is_file()
        records = json.loads((tmp_path / "progress.json").read_text())["episodes"]
        assert {r["training_context"]["player"] for r in records} == {0, 1}
        assert all(r["training_context"]["match"][f'player_{r["training_context"]["player"]}']["character"] == 1
                   for r in records)
        path = save_contract(tmp_path, env, config)
        swapped = MatchConfig(PlayerSetup(19, 0, 0), PlayerSetup(1, 0, 0))
        target = LearningInterface(replace(env.interface.episode, match=swapped), env.interface.config)
        assert read_training_contract(path, target)["algorithm"]["name"] == "br"
        wrong_timing = LearningInterface(replace(target.episode, latency_frames=5), target.config)
        with pytest.raises(ValueError, match="episode configurations differ"):
            read_training_contract(path, wrong_timing)
        wrong_character = LearningInterface(replace(target.episode,
            match=MatchConfig(PlayerSetup(19, 0, 0), PlayerSetup(2, 0, 0))), target.config)
        with pytest.raises(ValueError, match="trained learner"):
            read_training_contract(path, wrong_character)
    finally:
        env.close()


def test_full_god_roster_pairs_every_original_script_with_its_character():
    with initialize_config_dir(config_dir=str(Path(__file__).parents[1] / "config"), version_base="1.3"):
        cfg = compose(config_name="train", overrides=["algorithm=br", "rules=god",
            "wrappers=superhuman_learning", "track=superhuman", "+br_opponents=god_all"])
        opponents = OmegaConf.to_container(cfg.algorithm.opponents, resolve=True)
    assert len(opponents) == len({o["name"] for o in opponents}) == 27
    assert sum(o["probability"] for o in opponents) == pytest.approx(1.)
    assert {o["setup"]["character"] for o in opponents} == set(range(20))
    assert all(int(o["policy"]["rules"]["god"]["script"][:2]) == o["setup"]["character"] for o in opponents)


def test_mismatched_god_character_is_rejected_before_policy_load(tmp_path):
    env = fixture_env()
    try:
        with pytest.raises(ValueError, match="script and opponent character"):
            train_br(env, {"matchups": {"mode": "sampled", "learner": asdict(PlayerSetup(1, 0, 0))},
                "opponents": [{"name": "wrong", "probability": 1., "setup": asdict(PlayerSetup(6, 0, 0)),
                    "policy": {"kind": "rule", "name": "god", "rules": {"god": {"script": "00_reimu_main.ai"}}}}]},
                "cpu", 1, tmp_path)
    finally:
        env.close()
