"""Load common policies and play full original matches in one local game."""
from contextlib import closing
import gzip
import json
from pathlib import Path

import hydra
from omegaconf import OmegaConf

from soku_rl.env import EpisodeConfig
from soku_rl.env.worker_pipe import WorkerBackend
from soku_rl.env.wrappers.learning import LearningConfig, LearningInterface
from soku_rl.play.loader import load_play_policy
from soku_rl.play.local_session import run_session


@hydra.main(version_base="1.3", config_path="../config", config_name="local_match")
def main(cfg):
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    if set(config["players"]) != {"player_0", "player_1"}:
        raise ValueError("players must specify player_0 and player_1")
    episode = EpisodeConfig.from_dict(config["episode"])
    interface = LearningInterface(episode, LearningConfig(**config["wrappers"]))
    policies = {}
    for seat in (0, 1):
        candidate = config["players"][f"player_{seat}"]
        if candidate == "human":
            continue
        if not isinstance(candidate, dict):
            raise ValueError("each player must be human or a named policy configuration")
        policies[seat] = load_play_policy(candidate, interface, config["rules"], config["device"], seat)
    if not policies:
        raise ValueError("at least one player must be a policy")
    directory = Path(config["output"]).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    report = {"success": False, "policies": {seat: policy.fingerprint for seat, policy in policies.items()}}
    labels = [f"{seat + 1}P：策略 {policies[seat].name}" if seat in policies else f"{seat + 1}P：玩家"
              for seat in (0, 1)]
    print("；".join(labels) + "。", flush=True)
    try:
        with closing(WorkerBackend(log_path=directory / "worker.log", **config["runtime"])) as connection:
            if connection.identity["kind"] != "local_match":
                raise ValueError("local play requires tools/local_worker.py")
            (directory / "runtime.json").write_text(json.dumps(connection.identity, indent=2), encoding="utf-8")
            with gzip.open(directory / "events.jsonl.gz", "wt", encoding="utf-8") as events:
                def record(value):
                    events.write(json.dumps(value) + "\n")
                    if value["kind"] == "frame" and value["events"]:
                        events.flush()
                        for event in value["events"]:
                            if event["kind"] == "round_started":
                                print(f"第 {event['round'] + 1} 局开始，比分 {event['scores'][0]}:{event['scores'][1]}。", flush=True)
                            elif event["kind"] == "match_finished":
                                print(f"本场结束，比分 {event['scores'][0]}:{event['scores'][1]}。", flush=True)
                report["result"] = run_session(connection, policies, interface, config["seed"],
                    config["session"]["matches"], config["session"]["timeout"], record)
            report["success"] = report["result"]["termination"] == "matches_completed"
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        (directory / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
