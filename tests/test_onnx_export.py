"""BC actors retain their observation contract, memory and actions in ONNX play."""
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

from soku_rl.env.wrappers.learning import LearningInterface
from soku_rl.policy.export_actor import export_actor, export_tree
from soku_rl.policy.loader import load_policy
from soku_rl.policy.population import MixturePolicy, PPOPolicy, UniformPolicy
from soku_rl.policy.verification_inputs import verification_observation
from soku_rl.play.deployment import prepare_play_candidate
from soku_rl.play.loader import load_play_policy, play_interface, warm_play_policy
from soku_rl.rl.behavior_cloning import ObservationContractEnv
from soku_rl.rl.learner import create_learner, parameter_hash
from test_shared_ppo import fixture_config, fixture_env
from test_dqn import dqn_config


def source_model(directory, kind, mode):
    env = fixture_env()
    interface = LearningInterface(replace(env.interface.episode, observation_mode=mode), env.interface.config)
    env.close()
    config = dqn_config() if kind == "dqn" else fixture_config(kind)
    if mode == "privileged_state":
        config["dqn" if kind == "dqn" else "ppo"]["policy_kwargs"].update(
            features_extractor_class="soku_rl.rl.address_invariant_features.AddressInvariantCombatFeatures",
            features_extractor_kwargs={"history_frames": 1, "object_features": 2,
                "player_features": 8, "features_dim": 8})
    model, _ = create_learner(ObservationContractEnv(interface), interface, config, {"kind": "fresh"}, "cpu", 5)
    path = directory / "best.zip"
    model.save(path)
    training = {"episode": asdict(interface.episode), "wrappers": asdict(interface.config),
        "training_method": "behavior_cloning", "algorithm": config | {"name": "br"}}
    OmegaConf.save(OmegaConf.create(training), directory / "training.yaml")
    return model, interface, {"kind": "checkpoint", "path": str(path), "training_config": str(directory / "training.yaml")}


@pytest.mark.parametrize("kind,mode", [("mlp", "diagnostic_state"), ("lstm", "state"),
                                      ("lstm", "privileged_state"), ("dqn", "privileged_state")])
def test_export_shared_bc_and_q_networks_preserves_complete_inputs_and_private_memory(tmp_path, kind, mode):
    torch.set_num_threads(1)
    model, interface, spec = source_model(tmp_path, kind, mode)
    before = parameter_hash(model.policy)
    directory = tmp_path / "portable"
    manifest = export_actor({"candidate": {"name": "bc", "policy": spec},
        "verification_steps": 512, "seed": 13, "output": str(directory)})
    assert parameter_hash(model.policy) == before
    assert manifest["verification"]["steps"] == 512
    candidate = {"name": "bc", "policy": {"kind": "onnx", "path": str(directory / "policy.json")}}
    selected = play_interface(candidate, {}, {}, "human" if mode == "state" else "superhuman", [])
    rng = np.random.default_rng(6)
    for seat in (0, 1):
        loaded = load_play_policy(candidate, selected, {}, "cpu", seat)
        assert warm_play_policy(loaded, selected, 17) == 1
        first, independent = loaded.spawn(11), loaded.spawn(11)
        observation = verification_observation(interface, rng, 5)
        assert first.act(observation) == independent.act(observation)
        if kind == "lstm":
            np.testing.assert_array_equal(loaded.spawn(11).hidden, np.zeros((1, 1, 8), np.float32))
            unchanged = independent.hidden.copy()
            first.act(observation)
            np.testing.assert_array_equal(independent.hidden, unchanged)
        assert loaded.session.get_providers() == ["CPUExecutionProvider"]
    with (directory / "training.yaml").open("a") as stream:
        stream.write("\n# changed contract bytes\n")
    with pytest.raises(ValueError, match="checksum"):
        load_policy("changed", candidate["policy"], interface, "cpu")


def test_play_preparation_is_cached_and_parent_runtime_never_imports_torch(tmp_path):
    torch.set_num_threads(1)
    _, interface, spec = source_model(tmp_path, "mlp", "diagnostic_state")
    candidate = {"name": "bc", "policy": spec}
    settings = {"cache": str(tmp_path / "cache"), "verification_steps": 512}
    script = '''
import json, sys
from soku_rl.play.deployment import prepare_play_candidate
from soku_rl.play.loader import play_interface, load_play_policy, warm_play_policy
candidate, settings = json.loads(sys.argv[1])
prepared = prepare_play_candidate(candidate, settings)
interface = play_interface(prepared, {}, {}, "superhuman", [])
policy = load_play_policy(prepared, interface, {}, "cpu", 0)
assert warm_play_policy(policy, interface, 11) == 1
assert "torch" not in sys.modules
print(json.dumps(prepared))
'''
    result = subprocess.run([sys.executable, "-B", "-c", script, json.dumps([candidate, settings])],
        check=True, capture_output=True, text=True, env=os.environ.copy())
    prepared = json.loads(result.stdout.splitlines()[-1])
    from unittest.mock import patch
    with patch("soku_rl.play.deployment.subprocess.run", side_effect=AssertionError("cache should not rebuild")):
        assert prepare_play_candidate(candidate, settings) == prepared
    path = Path(prepared["policy"]["path"])
    manifest = json.loads(path.read_text())
    assert manifest["source_sha256"]
    with (path.parent / manifest["model"]).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        load_policy("bc", prepared["policy"], interface, "cpu")


def test_portable_population_keeps_episode_selection_and_checks_members(tmp_path):
    torch.set_num_threads(1)
    model, interface, spec = source_model(tmp_path, "mlp", "diagnostic_state")
    learned = PPOPolicy("average", model, Path(spec["path"]))
    mixed = MixturePolicy("mixed", [UniformPolicy("random", interface.action_space.n), learned], [.4, .6], "original")
    directory = tmp_path / "mixture"
    export_tree(mixed, interface, OmegaConf.to_container(OmegaConf.load(spec["training_config"])), directory, 512, 7)
    portable = load_policy("mixed", {"kind": "onnx", "path": str(directory / "policy.json")}, interface, "cpu")
    for seed in range(12):
        assert mixed._select(seed)[0].name == portable._select(seed)[0].name
    assert warm_play_policy(portable, interface, 13) == 1
    child = directory / "member-1/policy.json"
    child.write_text(child.read_text() + " ")
    with pytest.raises(ValueError, match="member checksum"):
        load_policy("mixed", {"kind": "onnx", "path": str(directory / "policy.json")}, interface, "cpu")


def test_facing_actor_export_keeps_absolute_commands_when_facing_changes(tmp_path):
    from test_facing_policy import configuration
    from soku_rl.env.observation.memory_schema import FIGHTER_NAMES, WORLD_NAMES
    import onnxruntime as ort

    torch.set_num_threads(1)
    interface, config = configuration()
    model, _ = create_learner(ObservationContractEnv(interface), interface, config, {"kind": "fresh"}, "cpu", 5)
    with torch.no_grad():
        model.policy.action_net.weight.zero_()
        model.policy.action_net.bias.fill_(-5.)
        model.policy.action_net.bias[448] = 5.
    path = tmp_path / "best.zip"
    model.save(path)
    contract = tmp_path / "training.yaml"
    OmegaConf.save(OmegaConf.create({"episode": asdict(interface.episode),
        "wrappers": asdict(interface.config), "algorithm": config}), contract)
    output = tmp_path / "portable"
    manifest = export_actor({"candidate": {"name": "facing", "policy": {
        "kind": "sb3_recurrent", "path": str(path), "training_config": str(contract)}},
        "verification_steps": 512, "seed": 13, "output": str(output)})
    assert manifest["verification"]["maximum_absolute_error"] < 2e-6
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(output / "actor.onnx"), sess_options=options,
                                  providers=["CPUExecutionProvider"])
    states = [np.zeros(manifest["state_shape"], np.float32) for _ in range(2)]
    rng = np.random.default_rng(7)
    signs = set()
    position = 2 * (len(WORLD_NAMES) + FIGHTER_NAMES.index("dir"))
    for step in range(8):
        observation = verification_observation(interface, rng, step)
        facing = observation[position] * 4294967296. + observation[position + 1] * 65536.
        signs.add(facing)
        probabilities, *states = session.run(None, {
            "observation": observation[None], "hidden": states[0], "cell": states[1]})
        assert int(probabilities.argmax(-1)[0]) == (448 if facing > 0 else 64)
        assert (probabilities > 0).all()
    assert signs == {-1., 1.}
