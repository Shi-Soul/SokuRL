"""Evaluate a saved single BR with paired seats and its fixed learner character."""
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import time

import hydra
from omegaconf import OmegaConf


@hydra.main(version_base="1.3", config_path="../config", config_name="benchmark_br")
def main(cfg):
    import torch
    from soku_rl.env import EpisodeConfig, TwoPlayerVectorEnv
    from soku_rl.env.worker_pipe import WorkerBackend
    from soku_rl.env.wrappers.learning import LearningConfig, LearningInterface, LearningVectorEnv
    from soku_rl.evaluation.br import benchmark_br
    from soku_rl.policy.loader import load_policy
    from soku_rl.policy.matchups import opponent_interface, select_opponents
    from soku_rl.policy.population import SeatPolicies
    from soku_rl.rl import configure_runtime

    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    configure_runtime(config["rl"])
    source = Path(config["training_directory"]).resolve(strict=True)
    training_path = source / "config.yaml"
    training = OmegaConf.to_container(OmegaConf.load(training_path), resolve=True, throw_on_missing=True)
    algorithm = training["algorithm"]
    if algorithm["name"] != "br" or algorithm["matchups"]["mode"] != "sampled":
        raise ValueError("cross-seat evaluation requires a sampled-matchup BR artifact")
    if config["require_complete"]:
        result = json.loads((source / "result.json").read_text(encoding="utf-8"))
        if result["success"] is not True or result["algorithm"] != "br":
            raise ValueError("completed BR evaluation requires successful training")
    if config["opponent_source"] == "training":
        population = algorithm["opponents"]
    elif config["opponent_source"] == "config":
        population = config["algorithm"]["opponents"]
    else:
        raise ValueError("opponent_source must be training or config")
    population = select_opponents(population, config["opponent_names"])
    model_path = (source / config["checkpoint"]).resolve(strict=True)
    if not model_path.is_relative_to(source):
        raise ValueError("checkpoint must belong to training_directory")
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; no CPU fallback")
    episode = EpisodeConfig.from_dict(training["episode"])
    wrappers = LearningConfig(**training["wrappers"])
    interface = LearningInterface(episode, wrappers)
    learner = algorithm["matchups"]["learner"]
    candidate = "learned-br"
    if candidate in {p["name"] for p in population}:
        raise ValueError("candidate name collides with an opponent")
    kind = {"mlp": "sb3", "lstm": "sb3_recurrent"}[algorithm["policy_type"]]
    output = Path(config["output"]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    # Persist the actual source contract and opponent setups, not train defaults.
    config.update(source_training=training, evaluation_opponents=population,
        checkpoint_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
        training_config_sha256=hashlib.sha256(training_path.read_bytes()).hexdigest())
    (output / "config.yaml").write_text(OmegaConf.to_yaml(OmegaConf.create(config)), encoding="utf-8")
    report = {"success": False, "phase": "loading_policies",
        "device": str(device), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"}
    started = time.perf_counter()
    try:
        policy = load_policy(candidate, {"kind": kind, "path": str(model_path),
            "training_config": str(training_path)}, interface, device)
        strategies = {candidate: SeatPolicies(candidate, (policy, policy))}
        setups = {}
        for entry in population:
            matched = opponent_interface(interface, learner, entry)
            opponent = load_policy(entry["name"], entry["policy"], matched, device)
            strategies[entry["name"]] = SeatPolicies(entry["name"], (opponent, opponent))
            setups[entry["name"]] = entry["setup"]
        report["phase"] = "running_games"
        with closing(WorkerBackend(log_path=output / "worker.log", **config["runtime"])) as backend:
            backend.configure_observation(episode.backend_observation())
            report["runtime"] = backend.identity
            env = LearningVectorEnv(TwoPlayerVectorEnv(backend, config["num_envs"], episode), wrappers)
            report["result"] = benchmark_br(env, strategies, candidate, learner, setups,
                config["benchmark"], backend.identity["fingerprints"]["game_id"], output)
        report["success"] = True
        report["phase"] = "finished"
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        report["seconds"] = time.perf_counter() - started
        (output / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
