"""Shared reinforcement-learning settings, independent of MARL scheduling."""


def ppo_settings(config):
    shared = config["rl"]
    algorithm = config["algorithm"]
    learner = algorithm["response"] if algorithm["name"] == "psro" else algorithm
    for name in ("policy_type", "timeout_payoff", "ppo"):
        if learner[name] != shared[name]:
            raise ValueError(f"algorithm-specific {name} differs from the shared rl configuration; configure rl.{name}")
    if shared["policy_type"] not in {"mlp", "lstm"}:
        raise ValueError("shared PPO policy_type must be mlp or lstm")
    if shared["timeout_payoff"] != "zero_at_horizon":
        raise ValueError("shared PPO requires zero_at_horizon")
    return shared
