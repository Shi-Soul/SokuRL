"""Start only workspace-owned virtual display and audio services."""
import json
import os
import pwd
from pathlib import Path
import secrets
import socket
import struct
import subprocess
import time

from linux_runtime.environment import contained, native_environment, restrict_writes


def start(config):
    if os.geteuid() != 0:
        raise PermissionError("Xvfb -nolock requires root; invoke this operation explicitly with sudo")
    state = Path(config["state"]).resolve()
    root = Path(config["root"]).resolve()
    number = config["display_number"]
    if type(number) is not int or number < 1 or config["display"] != f"127.0.0.1:{number}":
        raise ValueError("display and display_number must name the same local TCP display")
    with socket.socket() as connection:
        if connection.connect_ex(("127.0.0.1", 6000 + number)) == 0:
            raise RuntimeError("display is already listening; use operation=check, do not restart it")
    pulse = contained(root, config["pulse_socket"])
    if pulse.exists():
        raise FileExistsError(f"inspect the recorded audio service before replacing its socket: {pulse}")
    env = native_environment(config)
    for folder in (state / "display", state / "display/xkb-cache", state / "config/pulse", pulse.parent):
        folder.mkdir(parents=True, exist_ok=True)
    auth = contained(root, config["xauthority"])
    auth.parent.mkdir(parents=True, exist_ok=True)
    fields = (b"", str(number).encode(), b"MIT-MAGIC-COOKIE-1", secrets.token_bytes(16))
    auth.write_bytes(struct.pack(">H", 65535) + b"".join(struct.pack(">H", len(v)) + v for v in fields))
    auth.chmod(0o600)
    cookie = state / "config/pulse/cookie"
    if not cookie.exists():
        cookie.write_bytes(secrets.token_bytes(256))
        cookie.chmod(0o600)
    account = pwd.getpwuid(config["uid"])
    for path in (auth, cookie, pulse.parent, state / "audio"):
        if path.stat().st_uid != account.pw_uid:
            path.chown(account.pw_uid, account.pw_gid)
    env.update(PULSE_CONFIG_PATH=str(cookie.parent), PULSE_STATE_PATH=str(state / "audio/state"),
        PULSE_RUNTIME_PATH=str(pulse.parent))
    commands = {
        "audio": [config["pulseaudio"], "-n", "--daemonize=no", "--use-pid-file=no", "--exit-idle-time=-1",
            "--log-target=stderr", "--enable-memfd=true", "-L", "module-null-sink sink_name=sokurl rate=44100 channels=2",
            "-L", f"module-native-protocol-unix socket={pulse} auth-anonymous=1"],
        "display": [config["xvfb"], f":{number}", "-screen", "0", "1280x720x24", "-nolock", "-nolisten", "unix",
            "-listen", "tcp", "-auth", str(auth), "-fbdir", str(state / "display"), "-noreset"],
    }
    processes = []
    records = {}
    try:
        for name, command in commands.items():
            child_env = env.copy()
            if name == "display":
                child_env.update(LD_PRELOAD=str(state / "lib/XkbDirectory.so"), SOKURL_XKB_CACHE=str(state / "display/xkb-cache"))
            uid = 0 if name == "display" else config["uid"]
            with (state / f"{name}.log").open("wb") as output:
                process = subprocess.Popen(command, env=child_env, cwd=state, stdin=subprocess.DEVNULL,
                    stdout=output, stderr=subprocess.STDOUT, user=uid, group=0 if uid == 0 else account.pw_gid, extra_groups=(),
                    start_new_session=True, preexec_fn=lambda: restrict_writes(root))
            processes.append(process)
            records[name] = {"pid": process.pid, "command": command}
        time.sleep(1)
        if any(p.poll() is not None for p in processes):
            raise RuntimeError(f"runtime service exited; inspect {state}/audio.log and display.log")
        (state / "services.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(json.dumps(records, indent=2))
    except BaseException:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=10)
        raise
