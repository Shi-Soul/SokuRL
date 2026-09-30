"""Serve trusted parent requests using the existing Windows game controller."""
import sys
import traceback

from soku_rl.env.worker_pipe import PROTOCOL, receive, send
from game_batch import RESET_METHODS, SokuGameBatch
from runtime_identity import fingerprints
from game_runtime.startup import configure_game


def main():
    requests, replies = sys.stdin.buffer, sys.stdout.buffer
    # Human logs must not corrupt the framed binary replies.
    sys.stdout = sys.stderr
    operation, config = receive(requests)
    if operation != "initialize" or config["protocol"] != PROTOCOL:
        raise ValueError("unsupported rollout worker protocol")
    configure_game(config["game_directory"])
    backend = SokuGameBatch(config["launch_timeout"])
    backend.enable_recording()
    send(replies, {"ok": True, "value": {"protocol": PROTOCOL, "reset_methods": RESET_METHODS,
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
