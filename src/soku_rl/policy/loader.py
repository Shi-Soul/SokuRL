"""Load every policy type through the same observation and action contract."""
def load_policy(name, spec, interface, device):
    kind = spec["kind"]
    if kind == "rule":
        from soku_rl.policy.rules.observed_rules import RulePolicy, LearningRulePolicy
        from soku_rl.policy.rules.strategies import rule_implementation
        rules = spec["rules"]
        if spec["name"] not in rules["roster"]:
            raise ValueError("rule policy must belong to the configured roster")
        policy = RulePolicy(spec["name"], rules, interface.episode, rule_implementation())
        return LearningRulePolicy(policy, interface)
    if kind == "uniform":
        from soku_rl.policy.population import UniformPolicy
        return UniformPolicy(name, interface.action_space.n)
    if kind == "onnx_recurrent":
        from soku_rl.policy.onnx import OnnxPolicy
        if str(device) != "cpu":
            raise ValueError("the deployment model requires device=cpu")
        return OnnxPolicy(name, spec["path"], interface)
    if kind in {"sb3", "sb3_recurrent", "nfsp_average", "psro_mixture", "benchmarl_ippo"}:
        from soku_rl.policy.checkpoint import load_checkpoint
        return load_checkpoint(name, spec, interface, device)
    raise ValueError(f"unsupported policy kind: {kind}")
