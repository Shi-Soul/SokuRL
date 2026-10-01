"""Measure shared PPO update overhead on synthetic observations, not strength."""
import gc
import hashlib
import json
from pathlib import Path
import time
import zlib

import hydra
from omegaconf import OmegaConf


@hydra.main(version_base="1.3", config_path="../config", config_name="profile_ppo")
def main(cfg):
    import gymnasium as gym
    import numpy as np
    import torch
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv
    from soku_rl.env import EpisodeConfig
    from soku_rl.env.wrappers.learning import LearningConfig, LearningInterface
    from soku_rl.rl import ppo_settings
    from soku_rl.rl.ppo import create_ppo
    from soku_rl.rl.storage import PackedObservation
    from soku_rl.env.observation.memory_schema import (
        FIGHTER_NAMES, FIGHTER_WIDTH, MAX_OBJECTS, OBJECT_WIDTH, PLAYER_WIDTH, WORLD_NAMES)

    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("this optimizer diagnostic requires the configured CUDA device")
    threads = config["profile"]["cpu_threads"]
    if not threads or any(type(n) is not int or n < 1 for n in threads):
        raise ValueError("positive CPU thread counts are required")
    updates = config["profile"]["updates"]
    if type(updates) is not int or updates < 2:
        raise ValueError("at least two updates are required")
    settings = ppo_settings(config)
    interface = LearningInterface(EpisodeConfig.from_dict(config["episode"]), LearningConfig(**config["wrappers"]))
    active = config["profile"]["active_objects"]
    if (interface.episode.observation_mode != "privileged_state" or interface.episode.history_frames != 1
            or type(active) is not int or not 0 <= active <= MAX_OBJECTS):
        raise ValueError("profile requires one privileged frame and a valid active object count")
    template = np.zeros(interface.observation_space.shape, np.float32)
    for seat in (0, 1):
        offset = (len(WORLD_NAMES) + seat * PLAYER_WIDTH) * 2
        template[offset:offset + FIGHTER_WIDTH * 2] = .001
        count = offset + FIGHTER_NAMES.index("obj_n") * 2
        template[count:count + 2] = [0, active / 65536.]
        start = offset + FIGHTER_WIDTH * 2
        template[start:start + active * OBJECT_WIDTH * 2] = .001
    if config["profile"]["codec"] == "legacy_zlib":
        def pack_legacy(cls, observation):
            values = np.asarray(observation)
            return cls(values.shape, values.dtype.str, zlib.compress(values.tobytes(), 1))
        PackedObservation.pack = classmethod(pack_legacy)
    elif config["profile"]["codec"] != "installed":
        raise ValueError("profile.codec must be installed or legacy_zlib")
    output = Path(config["output"]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    root = Path(__file__).resolve().parents[1]
    sources = sorted((root / "src/soku_rl/rl").glob("*.py")) + [Path(__file__)]
    report = {"success": False, "synthetic_only": True, "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(device), "results": [],
        "source_hashes": {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}

    class SyntheticEnv(gym.Env):
        observation_space = interface.observation_space
        action_space = interface.action_space

        def reset(self, **kwargs):
            super().reset(**kwargs)
            self.frame = 0
            return template.copy(), {}

        def step(self, action):
            self.frame += 1
            return (template.copy(),
                    float(self.frame % 3 - 1), self.frame == 64, False, {})

    class Timings(BaseCallback):
        def __init__(self):
            super().__init__()
            self.samples = []
            self.pending_update = False

        def _on_rollout_start(self):
            torch.cuda.synchronize(device)
            now = time.perf_counter()
            if self.pending_update:
                self.samples[-1]["update_seconds"] = now - self.finished
            self.started = now

        def _on_step(self):
            return True

        def _on_rollout_end(self):
            torch.cuda.synchronize(device)
            self.finished = time.perf_counter()
            self.samples.append({"rollout_seconds": self.finished - self.started})
            self.pending_update = True

        def _on_training_end(self):
            torch.cuda.synchronize(device)
            self.samples[-1]["update_seconds"] = time.perf_counter() - self.finished

    try:
        for count in threads:
            torch.set_num_threads(count)
            env = DummyVecEnv([SyntheticEnv for _ in range(config["num_envs"])])
            try:
                model, _ = create_ppo(env, interface, settings, {"kind": "fresh"}, device, config["seed"])
                timer = Timings()
                torch.cuda.reset_peak_memory_stats(device)
                model.learn(total_timesteps=updates * config["num_envs"] * settings["ppo"]["n_steps"], callback=timer)
                report["results"].append({"cpu_threads": count, "samples": timer.samples,
                    "peak_cuda_bytes": torch.cuda.max_memory_allocated(device)})
                (output / "progress.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
                del model
            finally:
                env.close()
            gc.collect()
            torch.cuda.empty_cache()
        report["success"] = True
    finally:
        (output / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
