"""Manage the Linux developer runtime using the shared Hydra configuration space."""
import json
from pathlib import Path
import subprocess

import hydra
from omegaconf import OmegaConf

from linux_runtime.build import build, deploy, prepare, test_native
from linux_runtime.environment import REPO, native_environment, wine_environment


def check(config):
    import torch
    env = native_environment(config)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; GPU training cannot start")
    for name in ("wine", "windows_python", "xauthority", "pulse_socket"):
        if not Path(config[name]).exists():
            raise FileNotFoundError(config[name])
    subprocess.run([config["wine"], config["windows_python"], "-B", "-c",
        "import sys,numpy,hydra,psutil,lupa.lua51; print('Wine Python:',sys.version); print('Game worker dependencies: OK')"],
        check=True, env=wine_environment(config))
    print(json.dumps({"repository": str(REPO), "CUDA_VISIBLE_DEVICES": env["CUDA_VISIBLE_DEVICES"],
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0), "game": config["game"]}, indent=2))


@hydra.main(version_base="1.3", config_path="../config", config_name="linux")
def main(cfg):
    config = OmegaConf.to_container(cfg.linux, resolve=True, throw_on_missing=True)
    operations = {"prepare": prepare, "build": build, "test": test_native, "deploy": deploy, "check": check}
    if cfg.operation == "services":
        from linux_runtime.services import start
        start(config)
    elif cfg.operation in operations:
        operations[cfg.operation](config)
    else:
        raise ValueError(f"unknown Linux operation: {cfg.operation}")


if __name__ == "__main__":
    main()
