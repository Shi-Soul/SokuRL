"""Compare injected episodes with unmodified official replay playback."""
import ctypes
import hashlib
import json
from pathlib import Path
import sys

import hydra
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from bridge_shared import calculate_state_hash
from frame_validation import copy_state, state_diff
from game_batch import SokuGameBatch
from replay_validation import launch_replay_checkpoint
import sokurl
from soku_rl.env import EpisodeConfig
from soku_rl.env.encoding import Decision
from soku_rl.replay import Replay


def compare(expected, actual):
    normalized = copy_state(actual)
    normalized.battleSubMode = expected.battleSubMode
    normalized.stateHash = calculate_state_hash(normalized)
    if bytes(normalized) != bytes(expected):
        raise AssertionError({"frame": expected.frameId, "differences": state_diff(expected, normalized)[:20]})


@hydra.main(version_base="1.3", config_path="../../config", config_name="replay_parity")
def main(config):
    output = Path(config.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    episode = EpisodeConfig.from_dict(OmegaConf.to_container(config.episode))
    sokurl.GAME_DIR = Path(config.validation.game_directory).resolve(strict=True)
    sokurl.GAME_EXE = sokurl.GAME_DIR / "th123.exe"
    sokurl.SKIPINTRO_INI = sokurl.GAME_DIR / "modules/SkipIntro/SkipIntro.ini"
    frames = int(config.validation.frames)
    schedule = tuple(tuple(int(v) for v in keys) for keys in config.validation.inputs)
    if not 0 < frames < episode.max_frames or not schedule:
        raise ValueError("diagnostic frames must be positive and below the episode limit")
    report = {"success": False, "episodes": [], "bridge_sha256": hashlib.sha256(
        (sokurl.GAME_DIR / "modules/SokuRLBridge/SokuRLBridge.dll").read_bytes()).hexdigest()}
    OmegaConf.save(config, output / "config.yaml")
    game = SokuGameBatch(float(config.runtime.launch_timeout))
    game.configure_observation(episode.backend_observation())
    game.enable_recording()
    traces = []
    try:
        for seed in config.validation.seeds:
            current = game.reset_slots({0: int(seed)})[0]
            trace = [copy_state(game.clients[0].snapshot().latest)]
            while len(trace) <= frames and not current.ended:
                index = (len(trace) - 1) // int(config.validation.hold_frames)
                keys = schedule[index % len(schedule)]
                actions = (Decision(keys, "replay_diagnosis"), Decision((0,) * 8, "replay_diagnosis"))
                current = game.step({0: actions})[0]
                trace.append(copy_state(game.clients[0].snapshot().latest))
            traces.append(trace)
        game.close()  # Includes interrupted episodes, not only knockouts.
        records = game.take_replays()
        if len(records) != len(traces):
            raise AssertionError("a reset or close lost an episode replay")
        for index, (record, trace) in enumerate(zip(records, traces, strict=True)):
            path = output / f"episode-{index}.rep"
            path.write_bytes(record["data"])
            replay = Replay.decode(record["data"])
            if replay.encode() != record["data"] or replay.matches[0].seed != record["seed"]:
                raise AssertionError("saved replay metadata differs from the source episode")
            with (output / f"episode-{index}.frames").open("wb") as stream:
                for state in trace:
                    stream.write(bytes(state))
            instance = launch_replay_checkpoint(path, float(config.runtime.launch_timeout), True)
            try:
                compare(trace[0], instance.state)
                for expected in trace[1:]:
                    compare(expected, instance.step_native(10.0))
            finally:
                instance.close()
            report["episodes"].append({key: value for key, value in record.items() if key != "data"} |
                                      {"compared_frames": len(trace), "replay": path.name})
        report["success"] = True
    except Exception as error:
        report["error"] = repr(error)
        raise
    finally:
        game.close()
        (output / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
