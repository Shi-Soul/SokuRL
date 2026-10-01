"""Build and deploy x86 MSVC runtime modules entirely through Wine on Linux."""
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess

from linux_runtime.environment import REPO, contained, native_environment, windows, wine_environment


MODULES = {"d3d9.dll": "d3d9.dll", **{name + ".dll": f"modules/{name}/{name}.dll"
    for name in ("WindowResizer", "MemoryPatch", "SkipIntro", "ReplayDnD", "SokuRLBridge")}}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(config):
    root, state = Path(config["root"]).resolve(), Path(config["state"]).resolve()
    env = native_environment(config)
    library = state / "lib"
    library.mkdir(exist_ok=True)
    for name in ("ServerDirectory", "MappingMemory", "XkbDirectory"):
        subprocess.run(["gcc", "-shared", "-fPIC", "-O2", "-Wall", "-Wextra",
            str(REPO / "native/WineWorkspace" / (name + ".c")), "-ldl", "-o", str(library / (name + ".so"))],
            check=True, env=env)
    for source_key, target_key in (("prefix_source", "prefix"), ("game_source", "game")):
        source = contained(root, config[source_key])
        target = contained(root, config[target_key])
        if source == target:
            raise ValueError(f"{source_key} and {target_key} must be separate")
        if target.exists():
            continue
        # A complete copy also works when the source and destination mounts differ.
        shutil.copytree(source, target, symlinks=True)
    mods = REPO / "third_party/SokuMods"
    if not mods.exists():
        shutil.copytree(Path(config["toolchain"]) / "SokuMods", mods)
    game = Path(config["game"])
    for name in ("th123.exe", "th123a.dat", "th123b.dat", "th123c.dat"):
        if not (game / name).is_file():
            raise FileNotFoundError(f"incomplete game runtime: {game / name}")
    if hashlib.md5((game / "th123.exe").read_bytes()).hexdigest() != "df35d1fbc7b583317adabe8cd9f53b2e":
        raise RuntimeError("game executable is not the supported 1.10a build")


def compiler_environment(config):
    env = wine_environment(config)
    chain = Path(config["toolchain"]).resolve()
    env.update({
        "WINEPATH": ";".join(windows(chain / p) for p in ("msvc/bin", "sdk/bin", "cmake/bin")),
        "INCLUDE": ";".join(windows(chain / p) for p in ("msvc/include", "sdk/include/ucrt", "sdk/include/shared", "sdk/include/um", "sdk/include/winrt", "sdk/include/cppwinrt")),
        "LIB": ";".join(windows(chain / p) for p in ("msvc/lib", "sdk/lib/um", "sdk/lib/ucrt")),
    })
    return env


def build(config):
    chain = Path(config["toolchain"]).resolve()
    env = compiler_environment(config)
    cmake = [config["wine"], str(chain / "cmake/bin/cmake.exe")]
    outputs = {}
    for module in ("RuntimeModules", "SokuRLBridge"):
        folder = REPO / "build/linux" / module
        subprocess.run([*cmake, "-S", windows(REPO / "native" / module), "-B", windows(folder),
            "-G", "NMake Makefiles", "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_TRY_COMPILE_CONFIGURATION=Release",
            "-DCMAKE_MAKE_PROGRAM=" + windows(chain / "msvc/bin/nmake.exe"),
            "-DCMAKE_C_COMPILER=" + windows(chain / "msvc/bin/cl.exe"),
            "-DCMAKE_CXX_COMPILER=" + windows(chain / "msvc/bin/cl.exe")], check=True, env=env)
        subprocess.run([*cmake, "--build", windows(folder)], check=True, env=env)
        for name in MODULES:
            artifact = folder / name
            if artifact.exists():
                data = artifact.read_bytes()
                offset = struct.unpack_from("<I", data, 0x3c)[0]
                if struct.unpack_from("<H", data, offset + 4)[0] != 0x014c:
                    raise RuntimeError(f"not an x86 DLL: {artifact}")
                outputs[name] = {"path": str(artifact), "sha256": digest(artifact)}
    if outputs.keys() != MODULES.keys():
        raise RuntimeError("not all runtime DLLs were produced")
    report = {"modules": outputs, "compiler_sha256": digest(chain / "msvc/bin/cl.exe"),
        "cmake_sha256": digest(chain / "cmake/bin/cmake.exe")}
    (REPO / "build/linux/artifacts.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def test_native(config):
    chain = Path(config["toolchain"]).resolve()
    subprocess.run([config["wine"], str(chain / "cmake/bin/ctest.exe"), "--test-dir",
        windows(REPO / "build/linux/SokuRLBridge"), "--output-on-failure", "--no-tests=error"],
        check=True, env=compiler_environment(config))


def deploy(config):
    import psutil
    game = Path(config["game"]).resolve()
    for process in psutil.process_iter(["pid", "exe", "cmdline"]):
        try:
            arguments = " ".join(process.info["cmdline"] or [])
            executable = process.info["exe"] or ""
            if "th123.exe" in arguments.lower() + executable.lower() and (
                    str(game) in arguments + executable or windows(game) in arguments):
                raise RuntimeError(f"game directory is in use by PID {process.pid}")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    report = json.loads((REPO / "build/linux/artifacts.json").read_text(encoding="utf-8"))
    for name, target in MODULES.items():
        source = Path(report["modules"][name]["path"])
        if digest(source) != report["modules"][name]["sha256"]:
            raise RuntimeError(f"build artifact changed: {source}")
        destination = game / target
        print(f"Deploy {source} -> {destination}", flush=True)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            backup = Path(config["state"]) / "backups" / (name + "-" + digest(destination))
            backup.parent.mkdir(exist_ok=True)
            if not backup.exists():
                shutil.copy2(destination, backup)
        shutil.copy2(source, destination)
    (Path(config["state"]) / "deployed.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
