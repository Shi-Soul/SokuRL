"""Serve a trusted local parent without exposing offline game control."""
import sys
import traceback

from soku_rl.env.worker_pipe import PROTOCOL, receive, send
from network_game import NetworkGame
from runtime_identity import fingerprints
from game_runtime.startup import configure_game


def main():
    requests, replies = sys.stdin.buffer, sys.stdout.buffer
    sys.stdout = sys.stderr
    game = None
    operation, initialization = receive(requests)
    if operation != "initialize" or initialization["protocol"] != PROTOCOL:
        raise ValueError("unsupported network worker protocol")
    configure_game(initialization["game_directory"])
    send(replies, {"ok": True, "value": {"protocol": PROTOCOL, "kind": "network",
                                         "fingerprints": fingerprints()}})
    try:
        while True:
            try:
                operation, payload = receive(requests)
            except EOFError:
                break
            try:
                if operation == "start" and game is None:
                    game = NetworkGame(payload["network"], payload["visibility"], initialization["launch_timeout"])
                    value = {"pid": game.process.pid}
                elif operation == "poll" and game is not None:
                    value = game.poll()
                elif operation == "wait_host" and game is not None:
                    value = game.wait_host()
                elif operation == "watch_local_peer" and game is not None:
                    value = game.watch_local_peer(**payload)
                elif operation == "set_caption" and game is not None:
                    value = game.set_caption(**payload)
                elif operation == "submit" and game is not None:
                    value = game.submit(**payload)
                elif operation == "close":
                    if game is not None:
                        game.close()
                        game = None
                    send(replies, {"ok": True, "value": {}})
                    break
                else:
                    raise ValueError(f"unsupported network worker operation: {operation}")
                send(replies, {"ok": True, "value": value})
            except Exception:
                send(replies, {"ok": False, "error": traceback.format_exc()})
                raise
    finally:
        if game is not None:
            game.close()


if __name__ == "__main__":
    main()
