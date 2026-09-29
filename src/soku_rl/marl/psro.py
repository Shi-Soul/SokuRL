"""Use OpenSpiel PSROSolver with live vector rollouts instead of tree traversal."""
import json
import hashlib
from pathlib import Path

from open_spiel.python.algorithms.psro_v2.psro_v2 import PSROSolver

from soku_rl.evaluation.population import PopulationEvaluator
from soku_rl.policy.population import UniformPolicy, PPOPolicy
from soku_rl.marl.response import PPOResponseOracle
from soku_rl.policy.recurrent import RecurrentPPOPolicy
from soku_rl.env.encoding import AGENTS


class PlayerRoles:
    """Metadata required by the sampled meta-solver, not a pyspiel game tree."""
    def num_players(self):
        return 2


def policy_artifact(policy, directory):
    metadata = {"name": policy.name, "fingerprint": policy.fingerprint}
    if isinstance(policy, UniformPolicy):
        return metadata | {"kind": "uniform", "num_actions": int(policy.num_actions)}
    if isinstance(policy, PPOPolicy):
        kind = "sb3_recurrent" if isinstance(policy, RecurrentPPOPolicy) else "sb3"
        return metadata | {"kind": kind, "path": str(policy.path.relative_to(directory))}
    raise TypeError("unsupported PSRO population member")


def initial_policies(config, interface, device, directory):
    """Validate both seat contracts, then retain independent checkpoint copies."""
    from soku_rl.policy.loader import load_policy

    if set(config) != set(AGENTS):
        raise ValueError("PSRO initial population requires both player roles")
    policies = []
    for index, seat in enumerate(AGENTS):
        spec = config[seat]
        if spec == {"kind": "uniform"}:
            policy = UniformPolicy(f"uniform-p{index}", interface.action_space.n)
        elif spec["kind"] in {"sb3", "sb3_recurrent"}:
            policy = load_policy(f"initial-p{index}", spec, interface, device)
        else:
            raise ValueError("initial policies must be uniform, sb3, or sb3_recurrent")
        policies.append(policy)
    saved = []
    for policy in policies:
        if isinstance(policy, PPOPolicy):
            data = policy.path.read_bytes()
            if hashlib.sha256(data).hexdigest() != policy.fingerprint:
                raise ValueError("initial checkpoint changed while loading")
            path = Path(directory) / (policy.name + ".zip")
            with path.open("xb") as stream:
                stream.write(data)
            policy = type(policy)(policy.name, policy.model, path)
        saved.append(policy)
    return saved


class SampledPSROSolver(PSROSolver):
    def __init__(self, evaluator, oracle, simulations, prd_iterations, policies):
        self.evaluator = evaluator
        super().__init__(
            PlayerRoles(), oracle, simulations,
            initial_policies=policies,
            rectifier="", training_strategy_selector="probabilistic",
            meta_strategy_method="prd", sample_from_marginals=True,
            number_policies_selected=1, symmetric_game=False,
            prd_iterations=prd_iterations,
        )

    def sample_episodes(self, policies, num_episodes):
        # Upstream's recursive state traversal needs a game tree and can exceed
        # Python recursion depth at 7200 frames. Live rollouts need neither.
        return self.evaluator.evaluate(policies, num_episodes)


def train_psro(env, config, device, seed, directory):
    if min(config["iterations"], config["simulations_per_entry"],
           config["prd_iterations"], config["response"]["timesteps_per_response"]) < 1:
        raise ValueError("PSRO iteration and sampling budgets must be positive")
    evaluator = PopulationEvaluator(env, seed, config["timeout_payoff"])
    oracle = PPOResponseOracle(env, config["response"], device, seed + 1, directory)
    if (config["response"]["initialization"] == "parent_weights" and
            any(spec["kind"] == "sb3_recurrent" for spec in config["initial_population"].values())):
        raise ValueError("recurrent initial policies require response.initialization=fresh for MLP responses")
    policies = initial_policies(config["initial_population"], env.interface, device, directory)
    solver = SampledPSROSolver(evaluator, oracle, config["simulations_per_entry"], config["prd_iterations"], policies)
    report = {}
    for iteration in range(config["iterations"] + 1):
        if iteration:
            solver.iteration()
        report = {"format": "sokurl-psro-population-v1",
                  "iteration": iteration, "timeout_payoff": config["timeout_payoff"],
                  "meta_game": [v.tolist() for v in solver.get_meta_game()],
                  "meta_strategies": [v.tolist() for v in solver.get_meta_strategies()],
                  "populations": [[policy_artifact(p, directory)
                                   for p in role] for role in solver.get_policies()],
                  "evaluation_games": evaluator.records}
        destination = Path(directory) / "population.json"
        temporary = destination.with_suffix(".pending.json")
        temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
        temporary.replace(destination)
    return report
