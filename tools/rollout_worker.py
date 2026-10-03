"""Serve trusted parent requests using the existing Windows game controller."""
import sys
import traceback
import os
from pathlib import Path
import shutil

from soku_rl.env.worker_pipe import PROTOCOL, receive, send
from game_runtime.batch import RESET_METHODS, SokuGameBatch
from game_runtime.identity import fingerprints
from game_runtime.startup import configure_game
from game_runtime.offline_snapshot import offline_snapshot_requested


def main():
    requests, replies = sys.stdin.buffer, sys.stdout.buffer
    # Human logs must not corrupt the framed binary replies.
    sys.stdout = sys.stderr
    operation, config = receive(requests)
    if operation != "initialize" or config["protocol"] != PROTOCOL:
        raise ValueError("unsupported rollout worker protocol")
    game = config["game_directory"]
    if "SOKURL_ISOLATED_GAME_ROOT" in os.environ:
        # Win32 named mutexes do not coordinate independent Wine servers.
        # Private writable profiles/INI files keep concurrent workers separate.
        game = os.environ["SOKURL_ISOLATED_GAME_ROOT"]
        shutil.copytree(Path(config["game_directory"]), Path(game))
    configure_game(game)
    backend = SokuGameBatch(config["launch_timeout"])
    backend.enable_recording()
    send(replies, {"ok": True, "value": {"protocol": PROTOCOL, "reset_methods": RESET_METHODS,
                                         "privileged_transport": "native_snapshot" if offline_snapshot_requested() else "process_memory",
                                         "game_directory": str(Path(game).resolve()),
                                         "game_source": config["game_directory"],
                                         "fingerprints": fingerprints()}})
    try:
        while True:
            try:
                operation, payload = receive(requests)
            except EOFError:
                break
            try:
                if operation == "configure_observation":
                    backend.configure_observation(payload)
                    value = {"observation_mode": payload}
                elif operation == "reset":
                    value = backend.reset_slots(payload)
                elif operation == "reset_matchups":
                    value = backend.reset_matchups(payload["seeds"], payload["matches"])
                elif operation == "step":
                    value = backend.step(payload)
                elif operation == "close":
                    backend.close()
                    send(replies, {"ok": True, "value": {"value": {}, "replays": backend.take_replays()}})
                    break
                else:
                    raise ValueError(f"unsupported worker operation: {operation}")
                send(replies, {"ok": True, "value": {"value": value, "replays": backend.take_replays()}})
            except Exception:
                error = traceback.format_exc()
                try:
                    backend.close()
                except Exception:
                    error += "\nEpisode cleanup also failed:\n" + traceback.format_exc()
                send(replies, {"ok": False, "error": error, "replays": backend.take_replays()})
                raise
    finally:
        backend.close()


if __name__ == "__main__":
    main()
