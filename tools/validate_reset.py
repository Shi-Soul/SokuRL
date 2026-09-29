"""Compare repeated in-process episodes against fresh-process trajectories."""
from contextlib import closing
import hashlib
import json
from pathlib import Path
import pickle
import random
import time

import hydra
from omegaconf import OmegaConf

from soku_rl.env import EpisodeConfig
from soku_rl.env.encoding import AGENTS, NUM_ACTIONS
from soku_rl.env.control import ControlConfig, DelayedControls
from soku_rl.pixels import RGBFrame
from soku_rl.worker_pipe import WorkerBackend


def save_images(state, directory):
    if any(not isinstance(value, RGBFrame) for value in state.observations):
        raise TypeError("reset pixel capture requires RGB observations for both players")
    if any(image.frame != state.frame for image in state.observations):
        raise ValueError("image and simulation frame differ")
    directory.mkdir(parents=True, exist_ok=True)
    for player, image in enumerate(state.observations):
        header = f"P6\n{image.width} {image.height}\n255\n".encode("ascii")
        (directory / f"frame-{state.frame}-player-{player}.ppm").write_bytes(header + image.pixels)


def record(backend, seed, actions, capture_frames, capture_directory):
    start = time.perf_counter()
    state = backend.reset_slots({0: seed})[0]
    reset_seconds = time.perf_counter() - start
    initial = dict(state.diagnostics)
    hashes, observations = [], []
    for index in range(len(actions) + 1):
        hashes.append(state.diagnostics["hash"])
        observations.append(hashlib.sha256(pickle.dumps(state.observations, protocol=5)).hexdigest())
        if state.frame in capture_frames:
            save_images(state, capture_directory)
        if state.ended or index == len(actions):
            break
        state = backend.step({0: actions[index]})[0]
    return {"seed": seed, "pid": initial["pid"], "segment": initial["segment"],
            "reset_seconds": reset_seconds, "initial": initial, "hashes": hashes,
            "observations": observations, "frames": state.frame, "outcome": state.outcome.value}


def compare(reference, actual):
    for key in ("hashes", "observations"):
        left, right = reference[key], actual[key]
        for frame, pair in enumerate(zip(left, right)):
            if pair[0] != pair[1]:
                raise RuntimeError(f"seed {actual['seed']}: {key} differ at frame {frame}")
        if len(left) != len(right):
            raise RuntimeError(f"seed {actual['seed']}: different episode lengths")
    if reference["outcome"] != actual["outcome"]:
        raise RuntimeError("reset changed the game outcome")


@hydra.main(version_base="1.3", config_path="../config", config_name="validate_reset")
def main(cfg):
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    episode = EpisodeConfig(**config["episode"])
    validation = config["reset_validation"]
    seeds = validation["seeds"]
    if not seeds or len(set(seeds)) != len(seeds) or validation["cycles"] < 1:
        raise ValueError("distinct seeds and a positive cycle count are required")
    pause = validation["pause_after_episode_seconds"]
    if not 0 <= pause <= 60:
        raise ValueError("episode pause must be between zero and sixty seconds")
    capture_frames = validation["capture_frames"]
    if (any(type(frame) is not int or not 0 <= frame <= episode.max_frames for frame in capture_frames)
            or len(set(capture_frames)) != len(capture_frames)):
        raise ValueError("capture frames must be distinct integer frames within the episode horizon")
    if capture_frames and episode.observation_mode != "image":
        raise ValueError("pixel capture requires episode.observation_mode=image")
    rng = random.Random(validation["action_seed"])
    controls = DelayedControls(ControlConfig(episode.decision_frames, episode.latency_frames))
    controls.reset()
    actions = []
    for frame in range(episode.max_frames):
        if frame % episode.decision_frames == 0:
            controls.submit(frame, {agent: rng.randrange(NUM_ACTIONS) for agent in AGENTS})
        actions.append(controls.inputs(frame))
    output = Path(config["output"]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    report = {"success": False, "episodes": []}

    def backend(name):
        result = WorkerBackend(log_path=output / (name + ".log"), **config["runtime"])
        try:
            result.configure_observation(episode.backend_observation())
        except BaseException:
            result.close()
            raise
        return result

    def save(name, value):
        (output / (name + ".json")).write_text(json.dumps(value, indent=2), encoding="utf-8")
        report["episodes"].append({k: v for k, v in value.items()
                                   if k not in {"hashes", "observations"}} | {"trace": name + ".json"})
        save_report()

    def save_report():
        (output / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    try:
        reference = {}
        for seed in seeds:
            with closing(backend(f"fresh-{seed}")) as worker:
                reference[seed] = record(worker, seed, actions, capture_frames, output / f"fresh-{seed}")
                report["runtime"] = worker.identity
                save(f"fresh-{seed}", reference[seed])
        with closing(backend("persistent")) as worker:
            pid, segment = None, None
            peer_seed = seeds[-1]
            peer = worker.reset_slots({1: peer_seed})[1]
            peer_pid = peer.diagnostics["pid"]
            peer_frames = 0

            def check_peer(state):
                expected = reference[peer_seed]
                digest = hashlib.sha256(pickle.dumps(state.observations, protocol=5)).hexdigest()
                if (state.diagnostics["pid"] != peer_pid or state.diagnostics["segment"] != 0 or
                        state.diagnostics["hash"] != expected["hashes"][peer_frames] or
                        digest != expected["observations"][peer_frames]):
                    raise RuntimeError(f"reset changed the unselected slot at frame {peer_frames}")

            check_peer(peer)
            for cycle in range(validation["cycles"]):
                for seed in seeds:
                    actual = record(worker, seed, actions, capture_frames, output / f"cycle-{cycle}-{seed}")
                    save(f"cycle-{cycle}-{seed}", actual)
                    compare(reference[seed], actual)
                    method = worker.identity["reset_methods"][episode.observation_mode]
                    if method == "native_scene_reload":
                        if pid is not None and (actual["pid"] != pid or actual["segment"] != segment + 1):
                            raise RuntimeError("scene reset changed PID or failed to advance episode segment")
                    elif method == "process_restart":
                        if actual["segment"] != 0:
                            raise RuntimeError("new game process did not start at episode segment zero")
                    else:
                        raise ValueError(f"unsupported reset method: {method}")
                    pid, segment = actual["pid"], actual["segment"]
                    time.sleep(pause)
                    if not peer.ended and peer_frames < len(actions):
                        peer = worker.step({1: actions[peer_frames]})[1]
                        peer_frames += 1
                        check_peer(peer)
            while not peer.ended and peer_frames < len(actions):
                peer = worker.step({1: actions[peer_frames]})[1]
                peer_frames += 1
                check_peer(peer)
            report["unselected_slot"] = {"pid": peer_pid, "frames": peer_frames,
                                         "outcome": peer.outcome.value, "matched": True}
        report["success"] = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        save_report()


if __name__ == "__main__":
    main()
