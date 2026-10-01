"""Own a Wine server in the same Landlock domain as one rollout worker."""
import json
from pathlib import Path
import shutil
import subprocess
from uuid import uuid4

from linux_runtime.environment import contained, windows, wine_environment


def run_worker(config, arguments):
    root = Path(config["root"]).resolve(strict=True)
    session = contained(root, Path(config["state"]) / "workers" / uuid4().hex)
    session.mkdir(parents=True, exist_ok=False)
    prefix = session / "prefix"
    source = contained(root, config["prefix_source"])
    shutil.copytree(source, prefix, symlinks=True)
    server = session / "server"
    server.mkdir()
    env = wine_environment(config | {"prefix": str(prefix)})
    env["SOKURL_WINE_SERVER_ROOT"] = str(server)
    game = session / "game"
    env["SOKURL_ISOLATED_GAME_ROOT"] = windows(game)
    command = [config["wine"], config["windows_python"], "-B", *arguments]
    metadata = {"prefix": str(prefix), "server_root": str(server), "game": str(game), "command": command}
    path = session / "session.json"
    process = subprocess.Popen(command, env=env)
    metadata["pid"] = process.pid
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    try:
        result = process.wait()
        metadata["exit_code"] = result
        return result
    finally:
        # The unique prefix AND redirected server root identify this worker's
        # private server. Never issue these commands with the shared profile.
        wineserver = str(Path(config["wine"]).resolve().with_name("wineserver"))
        stopped = subprocess.run([wineserver, "-k"], env=env, timeout=30, check=False)
        waited = subprocess.run([wineserver, "-w"], env=env, timeout=30, check=False)
        metadata["server_stop_exit"] = stopped.returncode
        metadata["server_wait_exit"] = waited.returncode
        if stopped.returncode == waited.returncode == 0:
            shutil.rmtree(prefix)
            if game.exists():
                shutil.rmtree(game)
        path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
