"""Run a local policy while its worker owns one real network game."""
from contextlib import ExitStack
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
from soku_rl.play_policy import load_play_policy
from network_launch import start_games


@hydra.main(version_base="1.3", config_path="../config", config_name="netplay")
def main(cfg):
    if cfg.human.enabled:
        if type(cfg.human.seat) is not int or cfg.human.seat not in (1, 2):
            raise ValueError("human.seat must be 1 or 2")
        cfg.network.role = "join" if cfg.human.seat == 1 else "host"
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    episode = EpisodeConfig.from_dict(config["episode"])
    if (episode.observation_mode, episode.decision_frames, episode.latency_frames) != ("state", 3, 5):
        raise ValueError("the network bridge requires public state, decision_frames=3 and latency_frames=5")
    if config["network"]["role"] not in ("host", "join"):
        raise ValueError("network role must be host or join")
    seat = ("host", "join").index(config["network"]["role"])
    interface = LearningInterface(episode, LearningConfig(**config["wrappers"]))
    if (type(config["session"]["matches"]) is not int or config["session"]["matches"] < 1
            or config["session"]["timeout"] <= 0):
        raise ValueError("positive match count and session timeout are required")
    spec = config["candidate"]["policy"]
    policy = load_play_policy(config["candidate"], interface, config["rules"], config["device"], seat)
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
            connection, games = start_games(stack, config, asdict(episode.visibility), directory, WorkerConnection)
            report.update(games)
            (directory / "runtime.json").write_text(json.dumps(connection.identity, indent=2), encoding="utf-8")
            for key, game in games.items():
                (directory / f"{key.replace('_', '-')}.json").write_text(json.dumps(game, indent=2), encoding="utf-8")
            if config["human"]["enabled"]:
                character = "魔理沙" if seat == 0 else "灵梦"
                print(f"玩家：{config['human']['seat']}P，窗口 SokuRL - Player；"
                      f"AI：{seat+1}P {character}，策略 {policy.name}；推理设备：{config['device']}。", flush=True)
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
        report["success"] = report["result"]["termination"] == "matches_completed"
        if report["result"]["termination"] == "game_closed":
            print("游戏已关闭，本次启动的对战进程已退出。", flush=True)
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        report["seconds"] = time.monotonic()-started
        (directory / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
