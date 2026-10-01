"""Check full snapshots and expiring local input across two real network engines."""
from contextlib import ExitStack, closing
import json
import os
from pathlib import Path
import time
import traceback

import hydra
from omegaconf import OmegaConf

from game_runtime.privileged import PrivilegedReader
from game_runtime.startup import configure_game
from network_runtime.game import NetworkGame
from play_runtime.channels import RealtimeHistory, RealtimeInput


def compare(raw):
    return tuple((p.characterId, p.hp, tuple(getattr(p.input, key) for key, _ in p.input._fields_))
                 for p in (raw.p1, raw.p2))


def status(channel):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        value = channel.status()
        if value is not None:
            return value
        time.sleep(.001)
    raise TimeoutError("no stable realtime input status")


def validate(cfg, report):
    visibility = OmegaConf.to_container(cfg.episode.visibility, resolve=True)
    with ExitStack() as stack:
        games, histories, channels, readers = [], [], [], []
        for seat, role in enumerate(("host", "join")):
            os.environ["SOKURL_REALTIME_SEAT"] = str(seat)
            os.environ["SOKURL_VS_PAUSE_AT_START"] = "0"
            game = stack.enter_context(closing(NetworkGame({"role": role, "address": "127.0.0.1",
                "port": cfg.validation.port, "automate_menu": False}, visibility,
                cfg.runtime.launch_timeout, cfg.validation.render)))
            games.append(game)
            histories.append(stack.enter_context(closing(RealtimeHistory(game.process.pid))))
            channels.append(stack.enter_context(closing(RealtimeInput(game.process.pid, seat))))
            readers.append(PrivilegedReader(None))
            if seat == 0:
                game.wait_host()
        report["pids"] = [game.process.pid for game in games]
        cursors, pulses, submitted = [0, 0], [0., 0.], [False, False]
        frames, applied = [{}, {}], [False, False]
        deadline = time.monotonic() + cfg.validation.timeout
        while time.monotonic() < deadline:
            for seat, game in enumerate(games):
                if game.process.poll() is not None:
                    raise RuntimeError(f"network game {seat} exited")
                latest = game.clients["state"].read(2.)
                now = time.monotonic()
                if latest.scene in (8, 9) and now >= pulses[seat]:
                    menu = game.clients["menu"]
                    if menu.block.commandSeq == menu.block.ackSeq:
                        menu.menu_choose_character(cfg.validation.characters[seat])
                        pulses[seat] = now + .1
                cursors[seat], batch = histories[seat].read_after(cursors[seat])
                for frame in batch:
                    readers[seat].memory = frame.memory
                    observations = readers[seat].observe_snapshot(frame.raw)
                    if [p["char"] for p in observations[0].players] != list(cfg.validation.characters):
                        raise AssertionError("network selected the wrong characters")
                    frames[seat][frame.match.frame] = compare(frame.raw)
                if batch and not submitted[seat]:
                    keys = (1 if seat == 0 else -1, 0, 1, 0, 0, 0, 0, 0)
                    submitted[seat] = channels[seat].submit(batch[-1].match, keys, 5, 12)
                current = status(channels[seat])
                if current["acknowledged"]:
                    if current["result"] != "accepted":
                        raise AssertionError(f"network rejected input: {current}")
                    applied[seat] |= current["applied"] == 1
            if all(len(values) >= cfg.validation.frames for values in frames):
                break
            time.sleep(.001)
        else:
            raise TimeoutError("two network engines did not produce enough complete snapshots")
        common = sorted(set(frames[0]) & set(frames[1]))
        if len(common) < cfg.validation.frames or not all(applied):
            raise AssertionError(f"insufficient common frames or missing inputs: {len(common)}, {applied}")
        if any(frames[0][frame] != frames[1][frame] for frame in common):
            raise AssertionError("peers disagree on characters, HP or engine inputs")
        before = [status(channel) for channel in channels]
        started = time.monotonic()
        time.sleep(cfg.validation.stalled_seconds)
        elapsed = time.monotonic() - started
        after = [status(channel) for channel in channels]
        report.update(common_frames=len(common), applied=applied, before=before, after=after,
            stalled_seconds=elapsed, stalled_fps=[(b["frame"]-a["frame"])/elapsed
                                                  for a, b in zip(before, after)])
        if any(b["frame"] <= a["frame"] or any(b["held"]) for a, b in zip(before, after)):
            raise AssertionError("game stopped while Python stalled or input failed to expire")
        if not cfg.validation.render and any(not 55 <= fps <= 65 for fps in report["stalled_fps"]):
            raise AssertionError("headless network game failed real-time pacing")
        for history, cursor in zip(histories, cursors):
            try:
                history.read_after(cursor)
            except BufferError:
                continue
            raise AssertionError("lost history was not reported")


@hydra.main(version_base="1.3", config_path="../../config", config_name="realtime_network_validation")
def main(cfg):
    output = Path(cfg.output)
    output.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(cfg, output / "config.yaml")
    configure_game(cfg.runtime.game_directory)
    report = {"success": False, "characters": list(cfg.validation.characters), "render": cfg.validation.render}
    try:
        validate(cfg, report)
        report["success"] = True
    except Exception:
        report["error"] = traceback.format_exc()
        raise
    finally:
        (output / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
