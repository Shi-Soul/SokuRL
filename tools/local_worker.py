"""Serve local match frames through the shared worker and replay delivery protocol."""
import sys
import traceback

from game_runtime.local_match import LocalMatch
from game_runtime.startup import configure_game
from game_runtime.identity import fingerprints
from soku_rl.env import EpisodeConfig
from soku_rl.env.worker_pipe import PROTOCOL, receive, send


def main():
    requests, replies = sys.stdin.buffer, sys.stdout.buffer
    sys.stdout = sys.stderr
    operation, config = receive(requests)
    if operation != "initialize" or config["protocol"] != PROTOCOL:
        raise ValueError("unsupported local match worker protocol")
    configure_game(config["game_directory"])
    send(replies, {"ok": True, "value": {"protocol": PROTOCOL, "kind": "local_match",
                                         "fingerprints": fingerprints()}})
    game, recording = None, False
    completed = []

    def finish(reason):
        nonlocal recording
        if recording:
            replay = game.replay()
            completed.append({"slot": 0, "seed": game.seed,
                "frames": game.client.snapshot().game_frame, "reason": reason,
                "scope": "match", "data": replay.encode()})
            recording = False

    try:
        while True:
            try:
                operation, payload = receive(requests)
            except EOFError:
                break
            try:
                if operation == "start" and game is None:
                    episode = EpisodeConfig.from_dict(payload["episode"])
                    game = LocalMatch(episode, payload["seed"], config["launch_timeout"])
                    recording = True
                    value = {"closed": False, "frame": game.read()}
                elif operation == "step" and game is not None:
                    try:
                        game.step(payload)
                        value = {"closed": False, "frame": game.read()}
                    except EOFError:
                        recording = False  # The original process no longer holds its replay queue.
                        game.close()
                        game = None
                        value = {"closed": True, "replay_saved": False}
                elif operation == "finish" and game is not None:
                    finish("match_finished")
                    value = {}
                elif operation == "reset" and game is not None:
                    finish("reset")
                    game.reset(payload["seed"])
                    recording = True
                    value = {"closed": False, "frame": game.read()}
                elif operation == "close":
                    finish("close")
                    if game is not None:
                        game.close()
                        game = None
                    send(replies, {"ok": True, "value": {"value": {}, "replays": completed}})
                    break
                else:
                    raise ValueError(f"unsupported local match operation: {operation}")
                send(replies, {"ok": True, "value": {"value": value, "replays": completed}})
                completed.clear()
            except Exception:
                error = traceback.format_exc()
                try:
                    finish("error")
                except Exception:
                    error += "\nReplay recovery failed:\n" + traceback.format_exc()
                if game is not None:
                    try:
                        game.close()
                    except Exception:
                        error += "\nGame cleanup failed:\n" + traceback.format_exc()
                    game = None
                send(replies, {"ok": False, "error": error, "replays": completed})
                raise
    finally:
        if game is not None:
            game.close()


if __name__ == "__main__":
    main()
