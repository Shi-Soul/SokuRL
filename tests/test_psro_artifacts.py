"""Verify saved PSRO mixtures preserve weights, frozen models and episode choice."""
import json

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("open_spiel")
from stable_baselines3 import PPO
import gymnasium as gym

from soku_rl.checkpoint_policy import load_policy
from soku_rl.population import MixturePolicy, PPOPolicy, UniformPolicy
from soku_rl.psro import initial_policies, policy_artifact
from test_policy_artifacts import interface, training_config


def test_saved_population_loads_the_selected_seat_and_exact_ppo(tmp_path):
    contract = interface()
    env = gym.Env()
    env.observation_space, env.action_space = contract.observation_space, contract.action_space
    model = PPO("MlpPolicy", env, device="cpu", n_steps=2, batch_size=2,
                policy_kwargs={"net_arch": [16]})
    checkpoint = tmp_path / "response.zip"
    model.save(checkpoint)
    response = PPOPolicy("response", model, checkpoint)
    uniform = UniformPolicy("uniform", contract.action_space.n)
    entries = [policy_artifact(policy, tmp_path) for policy in (uniform, response)]
    assert entries[1]["path"] == "response.zip"
    saved = {"format": "sokurl-psro-population-v1", "populations": [entries, entries],
             "meta_strategies": [[0., 1.], [1., 0.]]}
    path = tmp_path / "population.json"
    path.write_text(json.dumps(saved))
    spec = {"kind": "psro_mixture", "path": str(path), "player": "player_0",
            "training_config": training_config(tmp_path, contract)}
    first = load_policy("mixture", spec, contract, "cpu")
    second = load_policy("mixture", spec | {"player": "player_1"}, contract, "cpu")
    assert first.fingerprint != second.fingerprint
    assert first.spawn(7).model is first.members[1].model
    assert second.spawn(7).num_actions == contract.action_space.n
    observation = np.full(contract.observation_space.shape, .2, dtype=np.float32)
    rng = np.random.default_rng(7)
    rng.choice(2, p=[0., 1.])
    expected = response.spawn(int(rng.integers(0, 0xFFFFFFFF)))
    actual = first.spawn(7)
    assert [actual.act(observation) for _ in range(12)] == [expected.act(observation) for _ in range(12)]
    saved["populations"][0][1]["fingerprint"] = "modified"
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="fingerprint differs"):
        load_policy("mixture", spec, contract, "cpu")


def test_mixture_selects_once_per_episode_with_independent_reproducible_rng():
    members = [UniformPolicy("one", 1), UniformPolicy("two", 2)]
    policy = MixturePolicy("mixed", members, [.25, .75], "saved-identity")
    actors = [policy.spawn(seed) for seed in range(2000)]
    fraction = sum(actor.num_actions == 2 for actor in actors) / len(actors)
    assert .70 < fraction < .80
    first, second = policy.spawn(9), policy.spawn(9)
    assert first is not second
    assert [first.act(None) for _ in range(32)] == [second.act(None) for _ in range(32)]
    with pytest.raises(ValueError, match="probability distribution"):
        MixturePolicy("bad", members, [1., 1.], "invalid")
    with pytest.raises(ValueError, match="probability distribution"):
        MixturePolicy("bad", members, [-.1, 1.1], "invalid")


@pytest.mark.parametrize("kind", ["sb3", "sb3_recurrent"])
def test_initial_population_remains_loadable_after_source_checkpoint_is_removed(tmp_path, kind):
    contract = interface()
    env = gym.Env()
    env.observation_space, env.action_space = contract.observation_space, contract.action_space
    if kind == "sb3_recurrent":
        Algorithm = pytest.importorskip("sb3_contrib").RecurrentPPO
        model = Algorithm("MlpLstmPolicy", env, device="cpu", n_steps=2, batch_size=2,
                          policy_kwargs={"net_arch": [16], "lstm_hidden_size": 16})
    else:
        model = PPO("MlpPolicy", env, device="cpu", n_steps=2, batch_size=2,
                    policy_kwargs={"net_arch": [16]})
    source = tmp_path / "external.zip"
    model.save(source)
    spec = {"kind": kind, "path": str(source), "training_config": training_config(tmp_path, contract)}
    output = tmp_path / "population"
    output.mkdir()
    policies = initial_policies({seat: spec for seat in ("player_0", "player_1")},
                                contract, "cpu", output)
    assert policies[0].model is not policies[1].model
    assert policies[0].path.read_bytes() == policies[1].path.read_bytes() == source.read_bytes()
    entries = [[policy_artifact(policy, output)] for policy in policies]
    assert all(entry[0]["kind"] == kind for entry in entries)
    source.unlink()
    path = output / "population.json"
    path.write_text(json.dumps({"format": "sokurl-psro-population-v1", "populations": entries,
                               "meta_strategies": [[1.], [1.]]}))
    loaded = load_policy("mixture", {"kind": "psro_mixture", "path": str(path),
        "player": "player_0", "training_config": training_config(output, contract)}, contract, "cpu")
    observation = np.full(contract.observation_space.shape, .2, dtype=np.float32)
    actual = loaded.spawn(17)
    rng = np.random.default_rng(17)
    rng.choice(1, p=[1.])
    expected = policies[0].spawn(int(rng.integers(0, 0xFFFFFFFF)))
    assert [actual.act(observation) for _ in range(12)] == [expected.act(observation) for _ in range(12)]


def test_initial_population_rejects_missing_role_before_writing_artifacts(tmp_path):
    with pytest.raises(ValueError, match="both player roles"):
        initial_policies({"player_0": {"kind": "uniform"}}, interface(), "cpu", tmp_path)
    assert not list(tmp_path.iterdir())
