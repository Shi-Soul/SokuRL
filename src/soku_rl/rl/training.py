"""Train one SB3 PPO policy per seat against a fixed rule population."""
import json
import time

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from soku_rl.policy.loader import load_policy


from soku_rl.rl.ppo import initialize_ppo, parameter_hash


class EpisodeRecords(BaseCallback):
    def __init__(self, directory, checkpoint_every):
        super().__init__()
        self.directory = directory
        self.records = []
        self.checkpoint_every = checkpoint_every
        self.timings = []
        self.pending_update = False
        self.last_saved_steps = 0

    def _on_rollout_start(self):
        if self.pending_update:
            self._finish_update()
        self.rollout_started = time.perf_counter()

    def _on_step(self):
        for done, info in zip(self.locals["dones"], self.locals["infos"], strict=True):
            if done:
                self.records.append({key: info[key] for key in (
                    "episode", "frame", "outcome", "decision_frames", "latency_frames", "training_context")})
        return True

    def _on_rollout_end(self):
        self.rollout_finished = time.perf_counter()
        self.timings.append({"steps": self.num_timesteps,
            "rollout_seconds": self.rollout_finished - self.rollout_started})
        self.pending_update = True
        (self.directory / "progress.json").write_text(json.dumps({
            "steps": self.num_timesteps, "episodes": self.records}, indent=2), encoding="utf-8")
        self._write_timings("updating")

    def _finish_update(self):
        self.timings[-1].update(update_seconds=time.perf_counter() - self.rollout_finished,
                               ppo_n_updates=self.model._n_updates)
        steps = self.model.num_timesteps
        if not self.last_saved_steps or steps - self.last_saved_steps >= self.checkpoint_every:
            checkpoints = self.directory / "checkpoints"
            checkpoints.mkdir(exist_ok=True)
            path = checkpoints / f"updated_{steps}_steps.zip"
            self.model.save(path)
            self.last_saved_steps = steps
            self.timings[-1]["updated_checkpoint"] = str(path)
        self.pending_update = False
        self._write_timings("sampling")

    def _write_timings(self, phase):
        (self.directory / "timing.json").write_text(json.dumps({
            "phase": phase, "rollouts": self.timings}, indent=2), encoding="utf-8")

    def _on_training_end(self):
        if self.pending_update:
            self._finish_update()
        self._write_timings("finished")


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
            "initial_policy": config["initial_policies"][f"player_{player}"], "matchups": {"mode": "fixed"}}
        results[f"player_{player}"] = train_response(
            env, response, opponents, weights, device, seed + player, destination)
    return results
