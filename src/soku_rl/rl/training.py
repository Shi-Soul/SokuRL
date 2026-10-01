"""Train one SB3 PPO policy per seat against a fixed rule population."""
import json

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from soku_rl.policy.loader import load_policy


from soku_rl.rl.ppo import initialize_ppo, parameter_hash


class EpisodeRecords(BaseCallback):
    def __init__(self, directory):
        super().__init__()
        self.directory = directory
        self.records = []

    def _on_step(self):
        for done, info in zip(self.locals["dones"], self.locals["infos"], strict=True):
            if done:
                self.records.append({key: info[key] for key in (
                    "episode", "frame", "outcome", "decision_frames", "latency_frames", "training_context")})
        return True

    def _on_rollout_end(self):
        (self.directory / "progress.json").write_text(json.dumps({
            "steps": self.num_timesteps, "episodes": self.records}, indent=2), encoding="utf-8")


def train_ppo(env, config, device, seed, directory):
    if config["timeout_payoff"] != "zero_at_horizon":
        raise ValueError("PPO requires the declared finite-horizon payoff")
    if config["players"] != [0, 1]:
        raise ValueError("train both seat policies for seat-swapped evaluation")
    if config["timesteps_per_player"] < 1 or config["checkpoint_every"] < env.num_envs:
        raise ValueError("invalid PPO training or checkpoint interval")
    if not config["opponents"] or len(set(config["opponents"])) != len(config["opponents"]):
        raise ValueError("fixed opponent roster must be nonempty and distinct")
    opponents = [load_policy(name, {"kind": "rule", "name": name, "rules": config["rules"]},
                             env.interface, device) for name in config["opponents"]]
    weights = np.full(len(opponents), 1 / len(opponents))
    results = {}
    for player in config["players"]:
        destination = directory / f"player_{player}"
        destination.mkdir()
        from soku_rl.marl.br import train_response
        response = config | {"player": player, "timesteps": config["timesteps_per_player"],
            "initial_policy": config["initial_policies"][f"player_{player}"]}
        results[f"player_{player}"] = train_response(
            env, response, opponents, weights, device, seed + player, destination)
    return results
