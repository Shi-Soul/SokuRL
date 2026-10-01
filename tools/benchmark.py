"""Evaluate a saved two-seat policy against the fixed rule roster."""
from contextlib import closing
import json
from pathlib import Path

import hydra
from omegaconf import OmegaConf


@hydra.main(version_base="1.3", config_path="../config", config_name="benchmark")
def main(cfg):
    run(cfg)


def run(cfg):
    import torch
    from soku_rl.policy.population import SeatPolicies
    from soku_rl.policy.loader import load_policy
    from soku_rl.env import EpisodeConfig, TwoPlayerVectorEnv
    from soku_rl.env.wrappers.learning import LearningConfig, LearningInterface, LearningVectorEnv
    from soku_rl.evaluation.benchmark import benchmark
    from soku_rl.env.worker_pipe import WorkerBackend
    from soku_rl.rl import configure_runtime

    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    configure_runtime(config["rl"])
    episode = EpisodeConfig.from_dict(config["episode"])
    learning = LearningConfig(**config["wrappers"])
    interface = LearningInterface(episode, learning)
    if config["track"] == "human" and episode.observation_mode != "state":
        raise ValueError("this rule benchmark requires the human state observation")
    if config["track"] == "superhuman" and episode.observation_mode != "privileged_state":
        raise ValueError("the superhuman benchmark requires complete privileged state")
    opponents = config["benchmark"]["opponents"]
    if len(opponents) != len(set(opponents)) or "idle" in opponents:
        raise ValueError("the benchmark roster must contain distinct non-idle rules")
    candidate = config["candidate"]
    if candidate["name"] in opponents:
        raise ValueError("candidate name collides with an opponent")
    device = torch.device(config["device"])
    roles = tuple(load_policy(candidate["name"], (candidate[a] | {"rules": config["rules"]} if candidate[a]["kind"] == "rule" else candidate[a]), interface, device)
                  for a in ("player_0", "player_1"))
    strategies = {candidate["name"]: SeatPolicies(candidate["name"], roles)}
    for name in opponents:
        rule = load_policy(name, {"kind": "rule", "name": name, "rules": config["rules"]}, interface, device)
        strategies[name] = SeatPolicies(name, (rule, rule))
    directory = Path(config["output"]).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    report = {"success": False}
    try:
        with closing(WorkerBackend(log_path=directory / "worker.log", **config["runtime"])) as backend:
            backend.configure_observation(episode.backend_observation())
            report["runtime"] = backend.identity
            (directory / "runtime.json").write_text(json.dumps(backend.identity, indent=2), encoding="utf-8")
            env = LearningVectorEnv(TwoPlayerVectorEnv(backend, config["num_envs"], episode), learning)
            report["result"] = benchmark(env, strategies, candidate["name"], config["benchmark"],
                backend.identity["fingerprints"]["game_id"], directory)
        report["success"] = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        (directory / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
