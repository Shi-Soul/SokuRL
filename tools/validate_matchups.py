"""Verify real per-slot character changes, both seats and untouched peers."""
from contextlib import closing
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import hydra
from omegaconf import OmegaConf

from soku_rl.env import EpisodeConfig, TwoPlayerVectorEnv
from soku_rl.env.encoding import AGENTS
from soku_rl.env.match import MatchConfig, PlayerSetup
from soku_rl.env.observation.privileged import decode_privileged
from soku_rl.env.worker_pipe import WorkerBackend
from soku_rl.env.wrappers.learning import LearningConfig, LearningVectorEnv
from validate_env import observation_hash


@hydra.main(version_base="1.3", config_path="../config", config_name="validate_matchups")
def main(cfg):
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    episode = EpisodeConfig.from_dict(config["episode"])
    if episode.observation_mode != "privileged_state" or episode.history_frames != 1:
        raise ValueError("matchup validation requires one privileged observation frame")
    stages = [{int(slot): MatchConfig(*(PlayerSetup(c, 0, 0) for c in pair))
               for slot, pair in stage.items()} for stage in config["validation"]["stages"]]
    if set(stages[0]) != set(range(config["num_envs"])):
        raise ValueError("the first stage must initialize every slot")
    frames = config["validation"]["frames_per_stage"]
    if type(frames) is not int or frames < 1:
        raise ValueError("frames_per_stage must be positive")
    output = Path(config["output"]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    root = Path(__file__).resolve().parents[1]
    sources = [*sorted((root / "src/soku_rl").rglob("*.py")), *sorted((root / "tools").rglob("*.py"))]
    report = {"success": False, "stages": [], "source_hashes": {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    started = time.perf_counter()
    try:
        with closing(WorkerBackend(log_path=output / "worker.log", **config["runtime"])) as backend:
            backend.configure_observation(episode.backend_observation())
            report["runtime"] = backend.identity
            env = LearningVectorEnv(TwoPlayerVectorEnv(backend, config["num_envs"], episode),
                                    LearningConfig(**config["wrappers"]))
            observations, matches = {}, {}
            idle = env.interface.action(256)
            for index, selected in enumerate(stages):
                peers = set(observations) - set(selected)
                hashes = {s: observation_hash(observations[s][AGENTS[0]]) for s in peers}
                # The next step must advance each untouched peer by exactly one
                # frame, including the raw frame check in Episode.step.
                seeds = {s: config["seed"] + index * env.num_envs + s for s in selected}
                fresh, infos = env.reset_matchups(seeds, selected)
                observations.update(fresh)
                matches.update(selected)
                actual = {}
                for slot, pair in fresh.items():
                    expected = (selected[slot].player_0.character, selected[slot].player_1.character)
                    for p, agent in enumerate(AGENTS):
                        state = decode_privileged(env.interface.base_observation(pair[agent]))
                        characters = tuple(int(f["char"]) for f in state.players)
                        if characters != (expected[p], expected[1-p]) or infos[slot][agent]["frame"] != 0:
                            raise RuntimeError(f"incorrect reset characters/frame: slot={slot}, seat={p}, chars={characters}")
                    actual[slot] = list(expected)
                for _ in range(frames):
                    previous = {s: env.episodes[s].frame for s in observations}
                    observations, _, _, _, infos = env.step({s: dict.fromkeys(AGENTS, idle) for s in observations})
                    for s in observations:
                        if infos[s][AGENTS[0]]["frame"] != previous[s] + 1:
                            raise RuntimeError("a paused peer skipped frames during matchup reset")
                report["stages"].append({"index": index, "seeds": seeds, "characters": actual,
                    "matches": {s: asdict(m) for s, m in selected.items()}, "untouched_slots": sorted(peers),
                    "peer_observations_before_reset": hashes,
                    "frames": {s: env.episodes[s].frame for s in observations}})
                (output / "progress.json").write_text(json.dumps(report["stages"], indent=2), encoding="utf-8")
        report["success"] = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        report["seconds"] = time.perf_counter() - started
        (output / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
