"""Train an approximate best response to an explicit frozen strategy mixture."""
from dataclasses import dataclass
import numpy as np
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.logger import configure

from soku_rl.policy.loader import load_policy
from soku_rl.rl.opponent_env import OpponentMixtureVecEnv
from soku_rl.rl.matchup_env import MatchupMixtureVecEnv
from soku_rl.rl.ppo import create_ppo, parameter_hash
from soku_rl.rl.training import EpisodeRecords, ResponseCheckpointCallback
from soku_rl.rl.curriculum import create_curriculum
from soku_rl.policy.matchups import opponent_interface


@dataclass(frozen=True)
class OpponentEntry:
    name: str
    policy: object

    @property
    def fingerprint(self):
        return self.policy.fingerprint

    def spawn(self, seed):
        return self.policy.spawn(seed)


def train_response(env, config, opponents, probabilities, device, seed, directory):
    """Shared response lifecycle; callers supply policies, never a second PPO."""
    if (type(config["timesteps"]) is not int or config["timesteps"] < 1
            or type(config["checkpoint_every"]) is not int
            or config["checkpoint_every"] < env.num_envs):
        raise ValueError("invalid BR training or checkpoint interval")
    matchups = config["matchups"]
    if matchups == {"mode": "fixed"}:
        if config["player"] not in (0, 1):
            raise ValueError("random seats require explicit learner and opponent setups")
        view = OpponentMixtureVecEnv(env, config["player"], opponents, probabilities, seed)
    elif matchups["mode"] == "sampled":
        view = MatchupMixtureVecEnv(env, config["player"], opponents, probabilities, seed,
            matchups["learner"], [entry["setup"] for entry in config["opponents"]])
    else:
        raise ValueError("BR matchups must be fixed or sampled")
    try:
        curriculum = create_curriculum(config.get("curriculum", {"kind": "fixed"}),
            opponents, probabilities, int(env.single_action_space.n))
        curriculum.restore(config["initial_policy"])
        view.curriculum = curriculum
        model, source = create_ppo(view, env.interface, config,
            config["initial_policy"], device, seed)
        model.set_logger(configure(str(directory / "scalars"), ["csv", "stdout"]))
        initial = parameter_hash(model.policy)
        start_steps = model.num_timesteps
        callbacks = CallbackList([
            EpisodeRecords(directory, config["checkpoint_every"], curriculum),
            ResponseCheckpointCallback(directory, config["checkpoint_every"] // env.num_envs, curriculum),
        ])
        model.learn(total_timesteps=config["timesteps"], callback=callbacks,
                    reset_num_timesteps=False)
        final = parameter_hash(model.policy)
        if initial == final:
            raise RuntimeError("BR completed without a policy update")
        path = directory / "final.zip"
        model.save(path)
        curriculum.save(path, model.num_timesteps)
        return {"steps": model.num_timesteps, "start_steps": start_steps,
            "additional_steps": model.num_timesteps - start_steps, "initial_policy": source,
            "initial_policy_hash": initial, "final_policy_hash": final,
            "checkpoint": str(path), "player": config["player"],
            "opponents": {p.name: p.fingerprint for p in opponents},
            "opponent_probabilities": view.probabilities.tolist(), "curriculum": curriculum.snapshot()}
    finally:
        view.close()


def train_br(env, config, device, seed, directory):
    population = config["opponents"]
    if not population or len({entry["name"] for entry in population}) != len(population):
        raise ValueError("BR opponents must be nonempty and have distinct names")
    probabilities = np.asarray([entry["probability"] for entry in population], dtype=float)
    if (not np.isfinite(probabilities).all() or (probabilities < 0).any()
            or not np.isclose(probabilities.sum(), 1)):
        raise ValueError("BR opponent probabilities must form a distribution")
    opponents = []
    for entry in population:
        interface = env.interface
        if config["matchups"]["mode"] == "sampled":
            interface = opponent_interface(interface, config["matchups"]["learner"], entry)
        opponents.append(OpponentEntry(entry["name"], load_policy(entry["name"], entry["policy"], interface, device)))
    return train_response(env, config, opponents, probabilities, device, seed, directory)
