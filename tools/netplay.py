"""Run a local policy while its worker owns one real network game."""
from contextlib import ExitStack, closing
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import time

import hydra
import numpy as np
from omegaconf import OmegaConf

from soku_rl.env import EpisodeConfig
from soku_rl.learning_wrappers import LearningConfig, LearningInterface
from soku_rl.network_session import run_session
from soku_rl.worker_pipe import WorkerConnection


@hydra.main(version_base="1.3", config_path="../config", config_name="netplay")
def main(cfg):
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    episode = EpisodeConfig(**config["episode"])
    if (episode.observation_mode, episode.decision_frames, episode.latency_frames) != ("state", 3, 5):
        raise ValueError("the network bridge requires public state, decision_frames=3 and latency_frames=5")
    if config["network"]["role"] not in ("host", "join"):
        raise ValueError("network role must be host or join")
    if config["human"]["enabled"] and config["network"]["role"] != "host":
        raise ValueError("local human play requires the AI to host")
    seat = ("host", "join").index(config["network"]["role"])
    interface = LearningInterface(episode, LearningConfig(**config["wrappers"]))
    if (type(config["session"]["matches"]) is not int or config["session"]["matches"] < 1
            or config["session"]["timeout"] <= 0):
        raise ValueError("positive match count and session timeout are required")
    spec = config["candidate"]["policy"]
    if spec["kind"] == "onnx_recurrent":
        from soku_rl.onnx_policy import OnnxPolicy
        if config["device"] != "cpu":
            raise ValueError("the deployment model requires device=cpu")
        policy = OnnxPolicy(config["candidate"]["name"], spec["path"], interface)
    else:
        import torch
        from soku_rl.checkpoint_policy import load_policy
        torch.set_num_threads(1)
        policy = load_policy(config["candidate"]["name"], spec, interface, torch.device(config["device"]))
    # Initialize model kernels before the original engine starts its frame clock.
    warm = policy.spawn(config["seed"])
    for _ in range(8):
        warm.act(np.zeros(interface.observation_space.shape, np.float32))
    del warm
    directory = Path(config["output"]).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    report = {"success": False, "policy_fingerprint": policy.fingerprint,
              "policy_kind": spec["kind"], "device": config["device"],
              "seat": seat, "wins_required": 2}
    started = time.monotonic()
    try:
        with ExitStack() as stack:
            connection = stack.enter_context(closing(WorkerConnection(log_path=directory / "worker.log", **config["runtime"])))
            if connection.identity["kind"] != "network":
                raise ValueError("netplay requires the dedicated network worker")
            (directory / "runtime.json").write_text(json.dumps(connection.identity, indent=2), encoding="utf-8")
            report["game"] = connection.request("start", {"network": config["network"],
                                                        "visibility": asdict(episode.visibility)})
            (directory / "game.json").write_text(json.dumps(report["game"], indent=2), encoding="utf-8")
            if config["human"]["enabled"]:
                connection.request("wait_host", {})
                human_runtime = config["runtime"] | {"mute_audio": config["human"]["mute_audio"]}
                human = stack.enter_context(closing(WorkerConnection(log_path=directory / "human-worker.log", **human_runtime)))
                human_settings = config["network"] | {"role": "join", "address": "127.0.0.1",
                                                     "automate_menu": config["human"]["automate_menu"]}
                report["human_game"] = human.request("start", {"network": human_settings,
                                                               "visibility": asdict(episode.visibility)})
                (directory / "human-game.json").write_text(json.dumps(report["human_game"], indent=2), encoding="utf-8")
                connection.request("watch_local_peer", {"pid": report["human_game"]["pid"]})
                print(f"人类玩家窗口已启动。请在后打开的窗口中选人并操作；推理设备：{config['device']}。", flush=True)
            with gzip.open(directory / "events.jsonl.gz", "wt", encoding="utf-8") as events:
                def record(value):
                    events.write(json.dumps({"seconds": time.monotonic()-started, **value})+"\n")
                    if value["kind"] != "frame" or value["events"]:
                        events.flush()
                    if value["kind"] == "frame" and value["events"]:
                        for event in value["events"]:
                            if event["kind"] == "round_started":
                                print(f"第 {event['match']} 场，第 {event['round']+1} 局开始。", flush=True)
                            elif event["kind"] == "match_finished":
                                print(f"本场结束，比分 {event['scores'][0]}:{event['scores'][1]}。", flush=True)
                report["result"] = run_session(connection, policy, interface, seat, config["seed"],
                    config["session"]["matches"], config["session"]["timeout"], record)
        report["success"] = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        report["seconds"] = time.monotonic()-started
        (directory / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
