"""Measure complete CPU policy decisions, including memory and action sampling."""
import json
from pathlib import Path
import platform
import sys
import time

import hydra
import numpy as np
from omegaconf import OmegaConf

from soku_rl.env import EpisodeConfig
from soku_rl.learning_wrappers import LearningConfig, LearningInterface
from soku_rl.onnx_policy import OnnxPolicy


@hydra.main(version_base="1.3", config_path="../config", config_name="benchmark_inference")
def main(cfg):
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    if config["device"] != "cpu" or config["candidate"]["policy"]["kind"] != "onnx_recurrent":
        raise ValueError("this deployment benchmark requires the CPU ONNX actor")
    if type(config["decisions"]) is not int or config["decisions"] < 2048 or config["maximum_p99_ms"] <= 0:
        raise ValueError("measure at least 2048 decisions with an explicit positive latency bound")
    interface = LearningInterface(EpisodeConfig(**config["episode"]), LearningConfig(**config["wrappers"]))
    started = time.perf_counter()
    policy = OnnxPolicy(config["candidate"]["name"], config["candidate"]["policy"]["path"], interface)
    load_seconds = time.perf_counter()-started
    rng = np.random.default_rng(config["seed"])
    observations = rng.uniform(-1, 1, size=(128, *policy.shape)).astype(np.float32)
    actor = policy.spawn(config["seed"])
    for observation in observations:
        actor.act(observation)
    samples = []
    for index in range(config["decisions"]):
        if index % 512 == 0:
            actor = policy.spawn(config["seed"]+index)
        started = time.perf_counter_ns()
        actor.act(observations[index % len(observations)])
        samples.append((time.perf_counter_ns()-started)/1e6)
    latency = {name: float(np.percentile(samples, q)) for name, q in (("p50", 50), ("p95", 95), ("p99", 99), ("maximum", 100))}
    report = {"success": latency["p99"] <= config["maximum_p99_ms"],
        "model_sha256": policy.fingerprint, "platform": platform.platform(),
        "providers": policy.session.get_providers(), "torch_imported": "torch" in sys.modules,
        "decisions": config["decisions"], "load_seconds": load_seconds, "latency_ms": latency,
        "maximum_p99_ms": config["maximum_p99_ms"], "intra_op_threads": 1, "inter_op_threads": 1}
    directory = Path(config["output"]).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    (directory / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not report["success"]:
        raise RuntimeError("CPU policy did not meet the configured inference latency bound")


if __name__ == "__main__":
    main()
