"""Train one SB3 PPO policy per seat against a fixed rule population."""
import hashlib
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.logger import configure

from soku_rl.policy.rules.observed_rules import RulePolicy
from soku_rl.policy.rules.strategies import rule_implementation
from soku_rl.env.wrappers.learning import LearningRulePolicy
from soku_rl.rl.opponent_env import OpponentMixtureVecEnv
from soku_rl.policy.contract import read_training_contract


from soku_rl.rl.ppo import initialize_ppo, parameter_hash, create_ppo


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
    episode = env.episodes[0].config
    source = rule_implementation()
    opponents = [LearningRulePolicy(RulePolicy(name, config["rules"], episode, source), env.interface)
                 for name in config["opponents"]]
    weights = np.full(len(opponents), 1 / len(opponents))
    results = {}
    for player in config["players"]:
        destination = directory / f"player_{player}"
        destination.mkdir()
        view = OpponentMixtureVecEnv(env, player, opponents, weights, seed + player)
        try:
            model, source = create_ppo(view, env.interface, config,
                config["initial_policies"][f"player_{player}"], device, seed + player)
            model.set_logger(configure(str(destination / "scalars"), ["csv", "stdout"]))
            initial = parameter_hash(model.policy)
            start_steps = model.num_timesteps
            callbacks = CallbackList([
                EpisodeRecords(destination),
                CheckpointCallback(save_freq=config["checkpoint_every"] // env.num_envs,
                                   save_path=str(destination / "checkpoints"), name_prefix="ppo"),
            ])
            model.learn(total_timesteps=config["timesteps_per_player"], callback=callbacks,
                        reset_num_timesteps=False)
            final = parameter_hash(model.policy)
            if initial == final:
                raise RuntimeError("PPO completed without a policy update")
            path = destination / "final.zip"
            model.save(path)
            results[f"player_{player}"] = {"steps": model.num_timesteps, "start_steps": start_steps,
                "additional_steps": model.num_timesteps - start_steps, "initial_policy": source,
                "initial_policy_hash": initial, "final_policy_hash": final,
                "checkpoint": str(path), "opponents": {p.name: p.fingerprint for p in opponents}}
        finally:
            view.close()
    return results
