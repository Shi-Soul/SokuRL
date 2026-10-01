"""Load a play opponent without importing training libraries for rules or ONNX."""
from dataclasses import replace
import json
from pathlib import Path

from omegaconf import OmegaConf

from soku_rl.env import EpisodeConfig
from soku_rl.env.wrappers.learning import LearningInterface
from soku_rl.policy.loader import load_policy


def checkpoint_interface(spec, interface):
    if spec["kind"] == "onnx_recurrent":
        path = Path(spec["path"]).resolve(strict=True)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        training_path = path.parent / manifest["training_config"]
    else:
        training_path = Path(spec["training_config"])
    training = OmegaConf.to_container(OmegaConf.load(training_path), resolve=True)
    match = EpisodeConfig.from_dict(training["episode"]).match
    # Character selection is a live match setting, not a tensor/clock change.
    # The shared checkpoint loader still checks every observation, timing and
    # wrapper field against the artifact's original training contract.
    return LearningInterface(replace(interface.episode, match=match), interface.config)


def load_play_policy(candidate, interface, rules, device, seat):
    spec = candidate["policy"]
    if seat not in (0, 1):
        raise ValueError("policy seat must be 0 or 1")
    if spec["kind"] == "rule":
        spec = spec | {"rules": rules}
    if spec["kind"] in {"sb3", "sb3_recurrent", "nfsp_average", "psro_mixture", "benchmarl_ippo", "onnx_recurrent"}:
        interface = checkpoint_interface(spec, interface)
    if spec["kind"] in {"sb3", "sb3_recurrent", "nfsp_average", "psro_mixture", "benchmarl_ippo"}:
        import torch
        torch.set_num_threads(1)
    return load_policy(candidate["name"], spec, interface, device)
