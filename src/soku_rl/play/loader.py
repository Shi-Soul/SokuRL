"""Load a play opponent without importing training libraries for rules or ONNX."""
from soku_rl.policy.loader import load_policy


def load_play_policy(candidate, interface, rules, device, seat):
    spec = candidate["policy"]
    if seat not in (0, 1):
        raise ValueError("policy seat must be 0 or 1")
    if spec["kind"] == "rule":
        spec = spec | {"rules": rules}
    if spec["kind"] in {"nfsp_average", "psro_mixture", "benchmarl_ippo"}:
        if spec["player"] != f"player_{seat}":
            raise ValueError("checkpoint seat differs from the AI seat")
    if spec["kind"] in {"sb3", "sb3_recurrent", "nfsp_average", "psro_mixture", "benchmarl_ippo"}:
        import torch
        torch.set_num_threads(1)
    return load_policy(candidate["name"], spec, interface, device)
