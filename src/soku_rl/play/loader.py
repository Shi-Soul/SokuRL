"""Load a play opponent without importing training libraries for rules or ONNX."""
from soku_rl.policy.rules.observed_rules import LearningRulePolicy
from soku_rl.policy.rules.observed_rules import RulePolicy
from soku_rl.policy.rules.strategies import rule_implementation


def load_play_policy(candidate, interface, rules, device, seat):
    spec = candidate["policy"]
    if seat not in (0, 1):
        raise ValueError("policy seat must be 0 or 1")
    if spec["kind"] == "rule":
        if spec["name"] not in rules["roster"]:
            raise ValueError("play opponent must belong to the configured rule roster")
        return LearningRulePolicy(RulePolicy(spec["name"], rules, interface.episode,
                                             rule_implementation()), interface)
    if spec["kind"] == "onnx_recurrent":
        from soku_rl.policy.onnx import OnnxPolicy
        if device != "cpu":
            raise ValueError("the deployment model requires device=cpu")
        return OnnxPolicy(candidate["name"], spec["path"], interface)
    if spec["kind"] in {"nfsp_average", "psro_mixture", "benchmarl_ippo"}:
        if spec["player"] != f"player_{seat}":
            raise ValueError("checkpoint seat differs from the AI seat")
    import torch
    from soku_rl.policy.checkpoint import load_policy
    torch.set_num_threads(1)
    return load_policy(candidate["name"], spec, interface, torch.device(device))
