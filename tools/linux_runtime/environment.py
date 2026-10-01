"""Resolve the shared Hydra machine profile and contain runtime storage."""
import ctypes
import os
from pathlib import Path
import sys

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


REPO = Path(__file__).resolve().parents[2]


def settings():
    with initialize_config_dir(version_base="1.3", config_dir=str(REPO / "config")):
        config = compose(config_name="linux")
    return OmegaConf.to_container(config.linux, resolve=True, throw_on_missing=True)


def contained(root, value):
    path = Path(value).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"path is outside the Linux workspace: {path}")
    return path


def windows(path):
    return "Z:" + str(Path(path).resolve()).replace("/", "\\")


def native_environment(config):
    root = Path(config["root"]).resolve(strict=True)
    contained(root, REPO)
    for key in ("state", "wine", "windows_python", "prefix", "toolchain", "game", "xauthority", "pulse_socket", "xvfb"):
        contained(root, config[key])
    for value in config["windows_packages"]:
        contained(root, value)
    state = contained(root, config["state"])
    paths = {name: state / name for name in ("tmp", "cache", "config", "home", "audio", "server", "matplotlib")}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return os.environ | {
        "HOME": str(paths["home"]), "TMPDIR": str(paths["tmp"]),
        "XDG_CACHE_HOME": str(paths["cache"]), "XDG_CONFIG_HOME": str(paths["config"]),
        "XDG_RUNTIME_DIR": str(paths["audio"]), "MPLCONFIGDIR": str(paths["matplotlib"]),
        "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": os.pathsep.join(str(REPO / p) for p in ("src", "tools")),
        "CUDA_VISIBLE_DEVICES": str(config["cuda_devices"]),
        "DISPLAY": config["display"], "XAUTHORITY": str(Path(config["xauthority"]).resolve()),
        "PULSE_SERVER": "unix:" + str(Path(config["pulse_socket"]).resolve()),
        "PULSE_COOKIE": str(state / "config/pulse/cookie"),
        "__GL_SHADER_DISK_CACHE_PATH": str(paths["cache"]),
    }


def wine_environment(config):
    env = native_environment(config)
    state = Path(config["state"]).resolve()
    env.update({
        "WINEPREFIX": str(Path(config["prefix"]).resolve()), "WINEDEBUG": "-all",
        "WINESERVER": str(Path(config["wine"]).resolve().with_name("wineserver")),
        "SOKURL_WINE_SERVER_ROOT": str(state / "server"),
        "LD_PRELOAD": ":".join(str(state / "lib" / name) for name in ("ServerDirectory.so", "MappingMemory.so")),
        "WINEDLLOVERRIDES": "d3d9=n,b;mscoree,mshtml=d", "LIBGL_ALWAYS_SOFTWARE": "1", "LP_NUM_THREADS": "1",
        "TEMP": windows(state / "tmp"), "TMP": windows(state / "tmp"),
        "PYTHONPATH": ";".join(windows(p) for p in [REPO / "src", REPO / "tools", *config["windows_packages"]]),
    })
    return env


def restrict_writes(root):
    """Allow filesystem mutation only in root; GPU device access is not storage."""
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    version = libc.syscall(444, 0, 0, 1)
    if version < 1:
        raise RuntimeError("Linux Landlock ABI 1 or newer is required")
    rights = (1 << 1) | sum(1 << bit for bit in range(4, 13))
    if version >= 2:
        rights |= 1 << 13
    if version >= 3:
        rights |= 1 << 14
    attribute = ctypes.c_uint64(rights)
    descriptor = libc.syscall(444, ctypes.byref(attribute), ctypes.sizeof(attribute), 0)
    if descriptor < 0:
        raise OSError(ctypes.get_errno(), "create Landlock ruleset")

    class PathRule(ctypes.Structure):
        _pack_ = 1
        _fields_ = [("allowed", ctypes.c_uint64), ("parent", ctypes.c_int32)]

    devices = [Path("/dev/null"), *Path("/dev").glob("nvidia*")]
    try:
        for path, access in [(Path(root).resolve(), rights), *[(p, 1 << 1) for p in devices if p.is_file() or p.is_char_device()]]:
            parent = os.open(path, os.O_PATH | os.O_CLOEXEC)
            try:
                rule = PathRule(access, parent)
                if libc.syscall(445, descriptor, 1, ctypes.byref(rule), 0) != 0:
                    raise OSError(ctypes.get_errno(), f"allow workspace path {path}")
            finally:
                os.close(parent)
        if libc.prctl(38, 1, 0, 0, 0) != 0 or libc.syscall(446, descriptor, 0) != 0:
            raise OSError(ctypes.get_errno(), "enforce workspace storage restriction")
    finally:
        os.close(descriptor)


def use_workspace_user(config):
    uid = config["uid"]
    if type(uid) is not int or uid < 1:
        raise ValueError("linux.uid must be a non-root user ID")
    if os.geteuid() != uid:
        command = ["sudo", "-n", "-u", f"#{uid}", sys.executable, "-B", *sys.argv]
        os.execvp(command[0], command)
