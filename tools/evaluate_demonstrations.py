"""Compare frozen shared-PPO models on separate, hashed demonstration validation sets."""
import hashlib
from importlib.metadata import version
import io
import json
import os
from pathlib import Path
import re
import time

import hydra
from omegaconf import OmegaConf


@hydra.main(version_base="1.3", config_path="../config", config_name="evaluate_demonstrations")
def main(cfg):
    import torch
    from soku_rl.env import EpisodeConfig
    from soku_rl.env.wrappers.learning import LearningConfig, LearningInterface
    from soku_rl.policy.contract import read_training_contract
    from soku_rl.rl.behavior_cloning import load_demonstrations
    from soku_rl.rl.demonstration_evaluation import score_validation
    from soku_rl.rl.ppo import algorithm_type, parameter_hash

    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    if type(config["cpu_threads"]) is not int or config["cpu_threads"] < 1:
        raise ValueError("cpu_threads must be positive")
    torch.set_num_threads(config["cpu_threads"])
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for validation but unavailable")
    for key in ("models", "datasets"):
        if not isinstance(config[key], dict) or not config[key] or any(
                not isinstance(label, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", label) for label in config[key]):
            raise ValueError(f"{key} requires nonempty, uniquely named entries")
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    if (output / "config.yaml").exists():
        raise ValueError("use a fresh demonstration evaluation directory")
    (output / "config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))
    root = Path(__file__).resolve().parents[1]
    identity = {"packages": {name: version(name) for name in ("torch", "numpy", "gymnasium",
        "stable-baselines3", "sb3-contrib")}, "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "source_hashes": {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for folder in (root / "src/soku_rl", root / "tools") for path in sorted(folder.rglob("*.py"))}}
    (output / "identity.json").write_text(json.dumps(identity, indent=2))
    report = {"success": False, "measurement": "teacher-label fit on fixed validation games, not game strength",
        "models": {}}
    started = time.perf_counter()
    try:
        for label, spec in config["models"].items():
            if not isinstance(spec, dict) or set(spec) != {"checkpoint", "training_config"}:
                raise ValueError("each model requires checkpoint and training_config")
            training_path = Path(spec["training_config"]).resolve(strict=True)
            contract_bytes = training_path.read_bytes()
            training = OmegaConf.to_container(OmegaConf.load(io.StringIO(contract_bytes.decode())), resolve=True)
            interface = LearningInterface(EpisodeConfig.from_dict(training["episode"]),
                LearningConfig(**training["wrappers"]))
            read_training_contract(training_path, interface)
            if training_path.read_bytes() != contract_bytes:
                raise RuntimeError("training contract changed during validation setup")
            path = Path(spec["checkpoint"]).resolve(strict=True)
            # Load the exact bytes hashed here, so an active writer cannot change
            # which checkpoint this result describes between hashing and load.
            checkpoint_bytes = path.read_bytes()
            from soku_rl.rl.learner import learner_kind
            from soku_rl.rl.dqn import DoubleDQN
            kind = learner_kind(training["rl"])
            loader = DoubleDQN if kind == "dqn" else algorithm_type(training["rl"]["policy_type"])
            model = loader.load(io.BytesIO(checkpoint_bytes), device=device)
            if model.observation_space != interface.observation_space or model.action_space != interface.action_space:
                raise ValueError("checkpoint spaces disagree with its training contract")
            initial_hash, initial_steps = parameter_hash(model.policy), model.num_timesteps
            entry = {"checkpoint": str(path), "checkpoint_sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
                "training_config": str(training_path), "training_config_sha256": hashlib.sha256(contract_bytes).hexdigest(),
                "policy_parameter_hash": initial_hash, "learner": kind, "learner_steps": initial_steps,
                **({"ppo_steps": initial_steps} if kind == "ppo" else {}), "datasets": {}}
            del checkpoint_bytes
            for dataset_label, source in config["datasets"].items():
                dataset_path = Path(source).resolve(strict=True)
                dataset_config = (dataset_path / "config.yaml").read_bytes()
                samples, manifest, _, digest = load_demonstrations(source, interface)
                if (dataset_path / "config.yaml").read_bytes() != dataset_config:
                    raise RuntimeError("dataset contract changed during validation setup")
                validation = [row for row in manifest["episodes"] if row["split"] == "validation"]
                entry["datasets"][dataset_label] = {"path": str(dataset_path), "manifest_sha256": digest,
                    "config_sha256": hashlib.sha256(dataset_config).hexdigest(),
                    "validation_episodes": [{key: row[key] for key in
                        ("id", "world_seed", "learner_seat", "steps", "sha256")} for row in validation],
                    **score_validation(model, samples, manifest, config["batch_size"], config["sequence_length"])}
                del samples
            if parameter_hash(model.policy) != initial_hash or model.num_timesteps != initial_steps:
                raise RuntimeError("read-only validation changed model parameters or training steps")
            report["models"][label] = entry
            print(label, {name: values["groups"]["overall"] for name, values in entry["datasets"].items()}, flush=True)
            del model
        report["success"] = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        report["seconds"] = time.perf_counter() - started
        (output / "result.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
