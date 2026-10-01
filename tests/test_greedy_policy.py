"""Deterministic deployment preserves PPO observations and per-episode LSTM memory."""
import numpy as np
import pytest
import torch

from test_shared_ppo import fixture_config, fixture_env, save_contract
from soku_rl.policy.loader import load_policy
from soku_rl.rl.behavior_cloning import ObservationContractEnv
from soku_rl.rl.ppo import create_ppo, parameter_hash


@pytest.mark.parametrize("policy_type,kind", [("mlp", "sb3"), ("lstm", "sb3_recurrent")])
def test_greedy_matches_sb3_predict_and_keeps_episode_memory_private(tmp_path, policy_type, kind):
    torch.set_num_threads(1)
    env = fixture_env()
    config = fixture_config(policy_type) | {"name": "br"}
    model, _ = create_ppo(ObservationContractEnv(env.interface), env.interface, config,
        {"kind": "fresh"}, "cpu", 17)
    checkpoint = tmp_path / "model.zip"
    model.save(checkpoint)
    contract = save_contract(tmp_path, env, config)
    source = {"kind": kind, "path": str(checkpoint), "training_config": contract}
    policy = load_policy("greedy", {"kind": "greedy", "policy": source}, env.interface, "cpu")
    sampled = load_policy("sampled", source, env.interface, "cpu")
    renamed = load_policy("renamed", {"kind": "greedy", "policy": source}, env.interface, "cpu")
    assert policy.fingerprint != sampled.fingerprint
    assert policy.fingerprint == renamed.fingerprint
    before = parameter_hash(policy.model.policy)
    observations = np.random.default_rng(19).normal(size=(12, *env.single_observation_space.shape)).astype(np.float32)
    first, second = policy.spawn(11), policy.spawn(999)
    state = None
    expected = []
    for index, observation in enumerate(observations):
        action, state = model.predict(observation, state=state,
            episode_start=np.array([index == 0]), deterministic=True)
        expected.append(int(action))
        assert first.act(observation) == expected[-1]
    if policy_type == "lstm":
        assert all(torch.count_nonzero(part) == 0 for part in second.states)
        assert any(torch.count_nonzero(part) > 0 for part in first.states)
        for actual, reference in zip(first.states, state):
            np.testing.assert_allclose(actual.numpy(), reference, atol=1e-6)
    assert [second.act(observation) for observation in observations] == expected
    reset = policy.spawn(11)
    assert [reset.act(observation) for observation in observations] == expected
    assert parameter_hash(policy.model.policy) == before
    env.close()


@pytest.mark.parametrize("spec", [
    {"kind": "greedy"},
    {"kind": "greedy", "policy": {"kind": "uniform"}},
    {"kind": "greedy", "policy": {"kind": "greedy"}},
    {"kind": "greedy", "policy": {"kind": "sb3"}, "temperature": 1},
    {"kind": "greedy", "policy": "sb3"},
])
def test_greedy_rejects_unsupported_or_ambiguous_specifications(spec):
    with pytest.raises(ValueError, match="greedy requires"):
        load_policy("invalid", spec, object(), "cpu")
