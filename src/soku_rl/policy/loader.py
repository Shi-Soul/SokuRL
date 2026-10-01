"""Load every policy type through the same observation and action contract."""
def load_policy(name, spec, interface, device):
    kind = spec["kind"]
    if kind == "action_noise":
        from soku_rl.policy.action_noise import ActionNoisePolicy
        if set(spec) != {"kind", "policy", "random_probability"}:
            raise ValueError("action_noise requires policy and random_probability")
        return ActionNoisePolicy(name, load_policy(name, spec["policy"], interface, device),
            int(interface.action_space.n), spec["random_probability"])
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
    if kind in {"sb3", "sb3_dqn", "sb3_recurrent", "nfsp_average", "psro_mixture", "benchmarl_ippo"}:
        from soku_rl.policy.checkpoint import load_checkpoint
        return load_checkpoint(name, spec, interface, device)
    raise ValueError(f"unsupported policy kind: {kind}")
