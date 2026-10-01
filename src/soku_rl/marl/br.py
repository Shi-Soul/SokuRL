"""Train an approximate best response to an explicit frozen strategy mixture."""
import numpy as np
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.logger import configure

from soku_rl.policy.loader import load_policy
from soku_rl.rl.opponent_env import OpponentMixtureVecEnv
from soku_rl.rl.ppo import create_ppo, parameter_hash
from soku_rl.rl.training import EpisodeRecords


def train_response(env, config, opponents, probabilities, device, seed, directory):
    """Shared response lifecycle; callers supply policies, never a second PPO."""
    if (type(config["timesteps"]) is not int or config["timesteps"] < 1
            or type(config["checkpoint_every"]) is not int
            or config["checkpoint_every"] < env.num_envs):
        raise ValueError("invalid BR training or checkpoint interval")
    view = OpponentMixtureVecEnv(env, config["player"], opponents, probabilities, seed)
    try:
        model, source = create_ppo(view, env.interface, config,
            config["initial_policy"], device, seed)
        model.set_logger(configure(str(directory / "scalars"), ["csv", "stdout"]))
        initial = parameter_hash(model.policy)
        start_steps = model.num_timesteps
        callbacks = CallbackList([
            EpisodeRecords(directory),
            CheckpointCallback(save_freq=config["checkpoint_every"] // env.num_envs,
                save_path=str(directory / "checkpoints"), name_prefix="ppo"),
        ])
        model.learn(total_timesteps=config["timesteps"], callback=callbacks,
                    reset_num_timesteps=False)
        final = parameter_hash(model.policy)
        if initial == final:
            raise RuntimeError("BR completed without a policy update")
        path = directory / "final.zip"
        model.save(path)
        return {"steps": model.num_timesteps, "start_steps": start_steps,
            "additional_steps": model.num_timesteps - start_steps, "initial_policy": source,
            "initial_policy_hash": initial, "final_policy_hash": final,
            "checkpoint": str(path), "player": config["player"],
            "opponents": {p.name: p.fingerprint for p in opponents},
            "opponent_probabilities": view.probabilities.tolist()}
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
    opponents = [load_policy(entry["name"], entry["policy"], env.interface, device)
                 for entry in population]
    return train_response(env, config, opponents, probabilities, device, seed, directory)
