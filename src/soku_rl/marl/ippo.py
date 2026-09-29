"""Independent PPO self-play: shared game collection and separate learner parameters."""
import json

from stable_baselines3.common.logger import configure

from soku_rl.policy.population import UniformPolicy
from soku_rl.rl.joint import JointRollouts
from soku_rl.rl.opponent_env import OpponentMixtureVecEnv
from soku_rl.rl.ppo import create_ppo, parameter_hash, snapshot


def train_ippo(env, config, device, seed, directory):
    if config["timesteps_per_player"] < 1 or config["checkpoint_every"] < 1:
        raise ValueError("IPPO training and checkpoint budgets must be positive")
    models, views, initial = [], [], []
    try:
        for player in (0, 1):
            destination = directory / f"player_{player}"
            destination.mkdir()
            view = OpponentMixtureVecEnv(env, player,
                [UniformPolicy("unused-space-view", env.single_action_space.n)], [1.], seed + player)
            views.append(view)
            model, _ = create_ppo(view, env.interface, config,
                config["initial_policies"][f"player_{player}"], device, seed + player)
            model.set_logger(configure(str(destination / "scalars"), ["csv"]))
            models.append(model)
            initial.append(parameter_hash(model.policy))
        starts = [model.num_timesteps for model in models]
        targets = [value + config["timesteps_per_player"] for value in starts]
        collector = JointRollouts(env, models, seed)
        updates = 0
        while models[0].num_timesteps < targets[0]:
            collector.update(targets)
            updates += 1
            if updates % config["checkpoint_every"] == 0:
                for player, model in enumerate(models):
                    model.save(directory / f"player_{player}" / f"step-{model.num_timesteps}.zip")
        result = {"format": "sokurl-shared-ppo-v1", "updates": updates, "games": collector.records}
        for player, model in enumerate(models):
            if parameter_hash(model.policy) == initial[player]:
                raise RuntimeError("IPPO completed without updating both policies")
            policy = snapshot(f"player_{player}", model, directory / f"player_{player}" / "final.zip")
            result[f"player_{player}"] = {"steps": model.num_timesteps,
                "additional_steps": model.num_timesteps - starts[player], "fingerprint": policy.fingerprint}
        (directory / "progress.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    finally:
        for view in views:
            view.close()
