"""Measure real game pacing with active and deliberately stalled AI transport."""
import json
import os
from pathlib import Path
import sys
import time
import traceback

import hydra
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from game_runtime.startup import configure_game
from game_runtime.privileged import PrivilegedReader
from play_runtime.channels import RealtimeHistory, RealtimeInput
from soku_rl.env.match import MatchConfig, PlayerSetup
import sokurl


def status(client):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        value = client.status()
        if value is not None:
            return value
        time.sleep(.001)
    raise TimeoutError("could not read realtime input status")


def case(cfg, seat, character):
    os.environ["SOKURL_REALTIME_SEAT"] = str(seat)
    selections = [PlayerSetup(character, 0, 0), PlayerSetup(character, 1, 1)]
    processes = sokurl._launch_vs_group_from_title(1, cfg.runtime.launch_timeout,
        headless=not cfg.validation.render, unlimited=False, seeds=(cfg.seed,), pause_at_start=False,
        capture_images=False, capture_state=False, match=MatchConfig(*selections))
    process = processes[0]
    history = channel = None
    result = {"seat": seat, "character": character, "pid": process.pid,
              "render": cfg.validation.render, "success": False}
    try:
        history, channel = RealtimeHistory(process.pid), RealtimeInput(process.pid, seat)
        cursor, captures, decoded = 0, [], 0
        reader = PrivilegedReader(None)
        deadline = time.monotonic() + cfg.runtime.launch_timeout
        while not decoded or time.monotonic() < deadline:
            cursor, frames = history.read_after(cursor)
            if not decoded and frames:
                deadline = time.monotonic() + cfg.validation.observation_seconds
            for frame in frames:
                reader.memory = frame.memory
                observations = reader.observe_snapshot(frame.raw)
                if any(p["char"] != character for p in observations[0].players):
                    raise AssertionError("wrong character captured")
                captures.append(frame.capture_seconds)
                decoded += 1
            if not decoded and time.monotonic() >= deadline:
                raise TimeoutError(f"game produced no battle frames; input status: {status(channel)}")
            if process.poll() is not None:
                raise RuntimeError("game exited during observation validation")
            time.sleep(.001)
        before = status(channel)
        stalled_start = time.monotonic()
        # This is the defect reproduction: no policy inference, state reads or
        # action writes for the whole interval. Only the game keeps executing.
        time.sleep(cfg.validation.stalled_seconds)
        after = status(channel)
        elapsed = time.monotonic() - stalled_start
        stalled_fps = (after["frame"]-before["frame"])/elapsed
        result.update(decoded_frames=decoded, stalled_seconds=elapsed,
            stalled_frames=after["frame"]-before["frame"], stalled_fps=stalled_fps,
            capture_mean_seconds=sum(captures)/len(captures), capture_max_seconds=max(captures),
            before=before, after=after)
        if not 55 <= stalled_fps <= 65:
            raise AssertionError(f"game failed real-time pacing while AI stalled: {stalled_fps}")
        try:
            history.read_after(cursor)
        except BufferError:
            result["history_overrun_reported"] = True
        else:
            raise AssertionError("stalled observation consumer silently lost history")
        result["success"] = True
    except Exception as error:
        result["error"] = repr(error)
        result["traceback"] = traceback.format_exc()
    finally:
        if history is not None:
            history.close()
        if channel is not None:
            channel.close()
        sokurl.shutdown(5., process.pid)
    return result


@hydra.main(version_base="1.3", config_path="../../config", config_name="realtime_validation")
def main(cfg):
    output = Path(cfg.output)
    output.mkdir(parents=True, exist_ok=False)
    configure_game(cfg.runtime.game_directory)
    OmegaConf.save(cfg, output / "config.yaml")
    results = []
    for character in cfg.validation.characters:
        for seat in cfg.validation.seats:
            result = case(cfg, seat, character)
            results.append(result)
            (output / "result.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(json.dumps(result), flush=True)
            if not result["success"]:
                raise RuntimeError("realtime validation failed; inspect saved report")


if __name__ == "__main__":
    main()
