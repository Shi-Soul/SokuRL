"""The single PPO implementation and checkpoint lifecycle for every training method."""
import hashlib
from pathlib import Path

from gymnasium import spaces
from hydra.utils import get_class
from stable_baselines3 import PPO

from soku_rl.policy.contract import read_training_contract
from soku_rl.rl import validate_payoff


def algorithm_type(policy_type):
    if policy_type == "mlp":
        return PPO
    if policy_type == "lstm":
        from sb3_contrib import RecurrentPPO
        return RecurrentPPO
    raise ValueError("PPO policy_type must be mlp or lstm")


def create_ppo(env, interface, config, source, device, seed):
    algorithm = algorithm_type(config["policy_type"])
    policy = ("MultiInput" if isinstance(env.observation_space, spaces.Dict) else
              "Cnn" if len(env.observation_space.shape) == 3 else "Mlp")
    policy += "LstmPolicy" if config["policy_type"] == "lstm" else "Policy"
    return initialize_ppo(algorithm, policy, env, interface, config, source, device, seed)


def snapshot(name, model, path):
    from soku_rl.policy.population import PPOPolicy
    from soku_rl.policy.recurrent import RecurrentPPOPolicy
    model.save(path)
    policy = PPOPolicy if type(model) is PPO else RecurrentPPOPolicy
    return policy(name, model, Path(path))


def parameter_hash(policy):
    digest = hashlib.sha256()
    for name, parameter in sorted(policy.named_parameters()):
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def initialize_ppo(algorithm, policy_type, env, interface, config, source, device, seed):
    validate_payoff(interface, config)
    parameters = dict(config["ppo"])
    architecture = dict(parameters["policy_kwargs"])
    if "features_extractor_class" in architecture:
        architecture["features_extractor_class"] = get_class(architecture["features_extractor_class"])
    parameters["policy_kwargs"] = architecture
    if "rollout_buffer_class" in parameters:
        parameters["rollout_buffer_class"] = get_class(parameters["rollout_buffer_class"])
    elif algorithm is PPO and interface.episode.observation_mode == "privileged_state":
        from soku_rl.rl.buffers import PackedRolloutBuffer
        parameters["rollout_buffer_class"] = PackedRolloutBuffer
    if source == {"kind": "fresh"}:
        return algorithm(policy_type, env, seed=seed, device=device, **parameters), source
    if set(source) != {"kind", "path", "training_config"} or source["kind"] not in {"checkpoint", "weights"}:
        raise ValueError("initial policy must be fresh, a training checkpoint, or policy weights")
    training = read_training_contract(source["training_config"], interface)
    previous = training["rl"] if "rl" in training else training["algorithm"]
    if any(previous[key] != config[key] for key in ("policy_type", "timeout_payoff")):
        raise ValueError("PPO initialization requires the same policy type and payoff")
    if source["kind"] == "checkpoint" and previous["ppo"] != config["ppo"]:
        raise ValueError("continued PPO must retain its algorithm and optimizer configuration")
    path = Path(source["path"]).resolve(strict=True)
    if source["kind"] == "weights":
        if previous["ppo"]["policy_kwargs"] != config["ppo"]["policy_kwargs"]:
            raise ValueError("policy weights require the same network architecture")
        initial = algorithm.load(path, device=device)
        source_steps = initial.num_timesteps
        model = algorithm(policy_type, env, seed=seed, device=device, **parameters)
        model.policy.load_state_dict(initial.policy.state_dict(), strict=True)
    else:
        model = algorithm.load(path, env=env, device=device)
        model.set_random_seed(seed)
        source_steps = model.num_timesteps
    return model, source | {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "source_steps": source_steps}

