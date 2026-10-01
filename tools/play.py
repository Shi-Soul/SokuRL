"""Use one policy and one session interface for local, host and join play."""
from contextlib import closing
from dataclasses import asdict
import gzip
import json
from pathlib import Path

import hydra
from omegaconf import OmegaConf

from soku_rl.env import EpisodeConfig
from soku_rl.env.worker_pipe import WorkerConnection
from soku_rl.env.wrappers.learning import LearningConfig, LearningInterface
from soku_rl.play.loader import load_play_policy
from soku_rl.play.opponents import opponent_catalog
from soku_rl.play.realtime_session import run_session
from soku_rl.play.settings import client_plan
from soku_rl.policy.god.package import ScriptPackage


@hydra.main(version_base="1.3", config_path="../config", config_name="play")
def main(cfg):
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    package = ScriptPackage(config["rules"]["god"]["package"], config["rules"]["god"]["api_source"])
    catalog = opponent_catalog(config["rules"], package, config["checkpoints"])
    if config["operation"] == "list":
        for name, opponent in catalog.items():
            tracks = "/".join("拟人" if track == "human" else "超人" for track in opponent.tracks)
            characters = ",".join(map(str, opponent.characters))
            print(f"{name} | {opponent.label} | {tracks} | AI 角色编号 {characters}")
        return
    if config["operation"] != "play":
        raise ValueError("operation must be list or play")
    if config["opponent"] not in catalog:
        raise ValueError("unknown opponent; use operation=list to list supported opponents")
    plan = client_plan(config["play"])
    ai = next(client for client in plan if client["realtime"])
    candidate, rules = catalog[config["opponent"]].configuration(config["track"], ai["character"], config["rules"])
    interface = LearningInterface(EpisodeConfig.from_dict(config["episode"]), LearningConfig(**config["wrappers"]))
    policy = load_play_policy(candidate, interface, rules, config["device"], ai["seat"])
    output = Path(config["output"]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(cfg, output / "config.yaml", resolve=True)
    report = {"success": False, "opponent": config["opponent"], "policy_fingerprint": policy.fingerprint,
              "track": config["track"], "clients": plan}
    try:
        with closing(WorkerConnection(log_path=output / "worker.log", **config["runtime"])) as connection:
            if connection.identity["kind"] != "realtime_play":
                raise ValueError("play requires tools/play_worker.py")
            games = connection.request("start", {"settings": config["play"], "episode": asdict(interface.episode)})
            (output / "runtime.json").write_text(json.dumps({**connection.identity, **games}, indent=2), encoding="utf-8")
            connection_name = {"local": "本机双引擎", "host": "建房", "join": "加入房间"}[config["play"]["connection"]]
            print(f"{connection_name}；AI：{ai['seat']+1}P，角色编号 {ai['character']}，对手 {config['opponent']}。", flush=True)
            with gzip.open(output / "events.jsonl.gz", "wt", encoding="utf-8") as stream:
                def record(event):
                    stream.write(json.dumps(event, ensure_ascii=False) + "\n")
                    if event["kind"] == "frame":
                        for transition in event["events"]:
                            if transition["kind"] == "match_finished":
                                print(f"本场结束，比分 {transition['scores'][0]}:{transition['scores'][1]}。", flush=True)
                report["result"] = run_session(connection, policy, interface, ai["seat"], config["seed"],
                    config["session"]["matches"], config["session"]["timeout"], record)
            report["success"] = report["result"]["termination"] in {"matches_completed", "game_closed", "disconnected"}
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        (output / "result.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
