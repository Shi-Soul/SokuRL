"""Launch final BR validation when owned training jobs finish successfully."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

import hydra
from omegaconf import OmegaConf
import psutil


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def live_job(directory, role):
    launch = json.loads((directory / "launch.json").read_text())
    try:
        arguments = psutil.Process(launch[role + "_pid"]).cmdline()
    except psutil.NoSuchProcess:
        return False
    outputs = [Path(arg.removeprefix("output=")).resolve()
               for arg in arguments if arg.startswith("output=")]
    return directory.resolve() in outputs


def finished_training(directory, expected_steps):
    result = json.loads((directory / "result.json").read_text())
    if result["success"] is not True or result["algorithm"] != "br":
        raise RuntimeError(f"training did not succeed: {directory}")
    manifest = json.loads((directory / "final.replay.json").read_text())
    if result["result"]["steps"] != expected_steps or manifest["steps"] != expected_steps:
        raise ValueError(f"final training budget differs: {directory}")
    for name, key in (("final.zip", "model_sha256"), ("final.replay.pkl", "replay_sha256")):
        if digest(directory / name) != manifest[key]:
            raise ValueError(f"final checkpoint integrity mismatch: {directory / name}")
    return manifest


def source_commit(repo):
    git_dir = repo / ".git"
    if git_dir.is_file():
        git_dir = (repo / git_dir.read_text().strip().removeprefix("gitdir: ")).resolve()
    return subprocess.check_output(["git", f"--git-dir={git_dir}", f"--work-tree={repo}",
                                    "rev-parse", "HEAD"], text=True).strip()


@hydra.main(version_base="1.3", config_path="../config", config_name="queue_dqn_validation")
def main(cfg):
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    if not 1 <= config["poll_seconds"] <= 60 or config["max_evaluations"] < 1:
        raise ValueError("invalid queue polling or concurrency budget")
    repo = Path.cwd()
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))
    pending = {job["name"]: job for job in config["jobs"]}
    if len(pending) != len(config["jobs"]):
        raise ValueError("duplicate queue job names")
    launched, processes = {}, {}
    ready = {}
    status = {"phase": "waiting_for_training", "launched": launched}
    try:
        while pending:
            for name, process in processes.items():
                code = process.poll()
                if code is not None and code != 0:
                    raise RuntimeError(f"queued evaluation exited {code}: {name}")
            active = sum(live_job(Path(path), "evaluator") for path in config["existing_evaluations"])
            active += sum(process.poll() is None for process in processes.values())
            free_mib = int(subprocess.check_output(["nvidia-smi", f"--id={config['gpu']}",
                "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).strip())
            available_gib = psutil.virtual_memory().available / 2**30
            for name, job in list(pending.items()):
                training = Path(job["training_directory"])
                if live_job(training, "trainer"):
                    continue
                if name not in ready:
                    ready[name] = finished_training(training, config["expected_steps"])
                if (active >= config["max_evaluations"] or free_mib < config["minimum_free_gpu_mib"]
                        or available_gib < config["minimum_available_ram_gib"]):
                    break
                destination = Path(job["output"])
                if destination.exists():
                    raise FileExistsError(destination)
                command = ["bash", "scripts/linux.sh", "tools/benchmark_br.py",
                    f"training_directory={training}", "checkpoint=final.zip", "require_complete=true",
                    "evaluation=validation", f"num_envs={config['num_envs']}",
                    f"rl.cpu_threads={config['cpu_threads']}", f"linux.cuda_devices={config['gpu']}",
                    f"output={destination}"]
                commit = source_commit(repo)
                with (output / f"{name}.stdout.log").open("x") as stream:
                    process = subprocess.Popen(command, cwd=repo, stdout=stream, stderr=subprocess.STDOUT)
                record = {"evaluator_pid": process.pid, "source_commit": commit,
                    "physical_gpu": config["gpu"], "argv": command,
                    "checkpoint_manifest": ready[name], "output": str(destination),
                    "queue_script_sha256": digest(Path(__file__)),
                    "benchmark_script_sha256": digest(repo / "tools/benchmark_br.py")}
                (output / f"{name}.launch.json").write_text(json.dumps(record, indent=2))
                launched[name], processes[name] = record, process
                del pending[name]
                active += 1
                # Recheck resources after initialization before starting another.
                break
            status.update(pending=list(pending), active_evaluations=active,
                          free_gpu_mib=free_mib, available_ram_gib=available_gib,
                          updated_utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
            (output / "status.json").write_text(json.dumps(status, indent=2))
            if pending:
                time.sleep(config["poll_seconds"])
        status["phase"] = "all_evaluations_launched"
    except BaseException as error:
        status.update(phase="failed", error=repr(error))
        raise
    finally:
        (output / "status.json").write_text(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
