"""Launch an unchanged Python entry point with the Linux workspace environment."""
from pathlib import Path
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from linux_runtime.environment import REPO, native_environment, restrict_writes, settings, use_workspace_user, wine_environment


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in {"native", "wine"}:
        raise ValueError("use scripts/linux.sh or scripts/wine-python.sh with Python arguments")
    config = settings()
    use_workspace_user(config)
    mode, arguments = sys.argv[1], sys.argv[2:]
    if mode == "native":
        env = native_environment(config)
        command = [sys.executable, "-B", *arguments]
    else:
        env = wine_environment(config)
        command = [config["wine"], config["windows_python"], "-B", *arguments]
    os.chdir(REPO)
    restrict_writes(config["root"])
    os.execvpe(command[0], command, env)


if __name__ == "__main__":
    main()
