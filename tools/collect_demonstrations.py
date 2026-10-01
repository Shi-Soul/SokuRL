"""Collect independent rule demonstrations under the exact shared BR interface."""
from contextlib import closing
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
import time

import hydra
from omegaconf import OmegaConf


@hydra.main(version_base="1.3", config_path="../config", config_name="collect_demonstrations")
def main(cfg):
    from soku_rl.env import EpisodeConfig, TwoPlayerVectorEnv
    from soku_rl.env.worker_pipe import WorkerBackend
    from soku_rl.env.wrappers.learning import LearningConfig, LearningInterface, LearningVectorEnv
    from soku_rl.policy.loader import load_policy
    from soku_rl.policy.matchups import opponent_interface
    from soku_rl.rl import configure_runtime, ppo_settings, validate_payoff
    from soku_rl.rl.demonstrations import collect_demonstrations, demonstration_plan

    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    ppo_settings(config)
    configure_runtime(config["rl"])
    if (config["algorithm"]["name"] != "br" or config["algorithm"]["matchups"]["mode"] != "sampled"
            or config["algorithm"]["ppo"]["gamma"] != 1. or config["teacher"]["kind"] != "rule"
            or "curriculum" in config["algorithm"]):
        raise ValueError("rule demonstrations require sampled BR, gamma=1 and explicit fixed opponents")
    interface = LearningInterface(EpisodeConfig.from_dict(config["episode"]), LearningConfig(**config["wrappers"]))
    validate_payoff(interface, config["algorithm"])
    learner = config["algorithm"]["matchups"]["learner"]
    population = config["algorithm"]["opponents"]
    excluded = {seed for block in config["excluded"].values() for seed in block["world_seeds"]}
    plan = demonstration_plan(config["collection"], population, config["seed"], excluded)
    teacher = load_policy("teacher", config["teacher"], interface, config["device"])
    opponents = [load_policy(entry["name"], entry["policy"],
        opponent_interface(interface, learner, entry), config["device"]) for entry in population]
    destination = Path(config["output"]).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    root = Path(__file__).resolve().parents[1]
    identity = {"source_hashes": {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in (root / "src/soku_rl", root / "tools") for p in sorted(folder.rglob("*.py"))},
        "sampling": "rules only; no neural training or CUDA allocation",
        "packages": {name: version(name) for name in ("torch", "numpy", "gymnasium", "lupa")}}
    (destination / "identity.json").write_text(json.dumps(identity, indent=2), encoding="utf-8")
    report = {"success": False, "method": "rule_demonstrations"}
    started = time.perf_counter()
    try:
        with closing(WorkerBackend(log_path=destination / "worker.log", **config["runtime"])) as backend:
            backend.configure_observation(interface.episode.backend_observation())
            identity["runtime"] = backend.identity
            (destination / "identity.json").write_text(json.dumps(identity, indent=2), encoding="utf-8")
            env = LearningVectorEnv(TwoPlayerVectorEnv(backend, config["num_envs"], interface.episode), interface.config)
            report["result"] = collect_demonstrations(env, plan, learner, population, teacher,
                opponents, destination / "episodes")
        report["success"] = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        report["seconds"] = time.perf_counter() - started
        (destination / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
