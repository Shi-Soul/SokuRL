"""Serve one unified play session while the original engines run independently."""
import sys
import traceback

from game_runtime.identity import fingerprints
from game_runtime.startup import configure_game
from play_runtime.session import RealtimeSession
from soku_rl.env import EpisodeConfig
from soku_rl.env.worker_pipe import PROTOCOL, receive, send


def main():
    requests, replies = sys.stdin.buffer, sys.stdout.buffer
    sys.stdout = sys.stderr
    operation, initialization = receive(requests)
    if operation != "initialize" or initialization["protocol"] != PROTOCOL:
        raise ValueError("unsupported play worker initialization")
    configure_game(initialization["game_directory"])
    send(replies, {"ok": True, "value": {"protocol": PROTOCOL, "kind": "realtime_play", "fingerprints": fingerprints()}})
    sessions = []
    try:
        while True:
            try:
                operation, payload = receive(requests)
            except EOFError:
                break
            try:
                if operation == "start" and not sessions:
                    session = RealtimeSession(payload["settings"], EpisodeConfig.from_dict(payload["episode"]),
                                              initialization["launch_timeout"])
                    sessions.append(session)
                    value = session.identity()
                elif operation == "poll" and sessions:
                    value = sessions[0].poll()
                elif operation == "submit" and sessions:
                    value = sessions[0].submit(**payload)
                elif operation == "close":
                    for session in sessions:
                        session.close()
                    sessions.clear()
                    send(replies, {"ok": True, "value": {}})
                    break
                else:
                    raise ValueError(f"unsupported play operation: {operation}")
                send(replies, {"ok": True, "value": value})
            except Exception:
                send(replies, {"ok": False, "error": traceback.format_exc()})
                raise
    finally:
        for session in sessions:
            session.close()


if __name__ == "__main__":
    main()
