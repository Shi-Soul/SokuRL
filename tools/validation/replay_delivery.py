"""Check replay delivery through the real child-process environment transport."""
import hashlib
import json
from pathlib import Path

import hydra
from omegaconf import OmegaConf

from soku_rl.env import EpisodeConfig
from soku_rl.env.encoding import Decision
from soku_rl.env.worker_pipe import WorkerBackend
from soku_rl.replay import Replay


@hydra.main(version_base="1.3", config_path="../../config", config_name="replay_delivery")
def main(config):
    output = Path(config.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    episode = EpisodeConfig.from_dict(OmegaConf.to_container(config.episode, resolve=True))
    runtime = OmegaConf.to_container(config.runtime, resolve=True, throw_on_missing=True)
    seeds = list(config.validation.seeds)
    if len(seeds) != 3 or episode.max_frames < 3:
        raise ValueError("delivery validation requires three seeds and at least three frames")
    worker = WorkerBackend(log_path=output / "worker.log", **runtime)
    expected = []
    report = {"success": False, "runtime": worker.identity}
    OmegaConf.save(config, output / "config.yaml")
    try:
        worker.configure_observation(episode.backend_observation())
        for seed, frames, reason in zip(seeds,
                (episode.max_frames, episode.max_frames // 2, episode.max_frames // 3),
                ("time_limit", "reset", "close"), strict=True):
            worker.reset_slots({0: seed})
            for frame in range(frames):
                inputs = (1, 0, int(frame % 30 < 15), 0, 0, 0, 0, 0)
                state = worker.step({0: (Decision(inputs, "recording_validation"),
                                         Decision((0,) * 8, "recording_validation"))})[0]
                if state.ended:
                    raise RuntimeError("diagnostic input schedule unexpectedly ended the battle")
            expected.append((seed, frames, reason))
        try:
            worker.step({0: (Decision((2,) + (0,) * 7, "invalid_input"),
                             Decision((0,) * 8, "invalid_input"))})
        except RuntimeError as error:
            if "invalid logical input" not in str(error):
                raise
        else:
            raise AssertionError("the worker accepted invalid input")
        worker.close()
        files = sorted(worker.replay_directory.glob("*.json"))
        if len(files) != len(expected):
            raise AssertionError(f"saved {len(files)} episodes, expected {len(expected)}")
        records = []
        for path, fields in zip(files, expected, strict=True):
            metadata = json.loads(path.read_text(encoding="utf-8"))
            if tuple(metadata[name] for name in ("seed", "frames", "reason")) != fields:
                raise AssertionError(f"incorrect saved episode metadata: {metadata}")
            data = (path.parent / metadata["replay"]).read_bytes()
            replay = Replay.decode(data)
            if replay.matches[0].seed != fields[0] or hashlib.sha256(data).hexdigest() != metadata["sha256"]:
                raise AssertionError("saved replay differs from its episode metadata")
            records.append(metadata)
        report.update(success=True, episodes=records)
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        worker.close()
        (output / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"success": True, "episodes": records}, indent=2))


if __name__ == "__main__":
    main()
