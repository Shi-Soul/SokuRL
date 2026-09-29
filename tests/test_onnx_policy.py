"""Portable policy contracts and recurrent memory must survive CPU deployment."""
from dataclasses import asdict
import hashlib
import json

import numpy as np
from omegaconf import OmegaConf
import pytest

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
from onnx import TensorProto, helper, numpy_helper

from test_env_timing import VISIBILITY
from soku_rl.env import EpisodeConfig
from soku_rl.learning_wrappers import LearningConfig, LearningInterface
from soku_rl.onnx_policy import OnnxPolicy


def artifact(directory):
    interface = LearningInterface(EpisodeConfig(7200, 4, 3, 5, "state", VISIBILITY),
                                  LearningConfig("combat", True, 8, 1.))
    shape, memory, actions = (1, *interface.observation_space.shape), (1, 1, 2), int(interface.action_space.n)
    fields = lambda name, size: helper.make_tensor_value_info(name, TensorProto.FLOAT, size)
    graph = helper.make_graph([
        helper.make_node("Add", ["hidden", "one"], ["hidden_out"]),
        helper.make_node("Add", ["cell", "one"], ["cell_out"]),
        helper.make_node("Softmax", ["logits"], ["probabilities"], axis=-1)],
        "recurrent-contract", [fields("observation", shape), fields("hidden", memory), fields("cell", memory)],
        [fields("probabilities", (1, actions)), fields("hidden_out", memory), fields("cell_out", memory)],
        [numpy_helper.from_array(np.ones(memory, np.float32), "one"),
         numpy_helper.from_array(np.zeros((1, actions), np.float32), "logits")])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10)
    path = directory / "actor.onnx"
    onnx.save(model, path)
    OmegaConf.save(OmegaConf.create({"episode": asdict(interface.episode), "wrappers": asdict(interface.config)}),
                   directory / "training.yaml")
    manifest = {"format": "sokurl-recurrent-onnx-v1", "model": path.name,
                "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "observation_shape": list(interface.observation_space.shape), "state_shape": list(memory),
                "num_actions": actions, "training_config": "training.yaml"}
    description = directory / "policy.json"
    description.write_text(json.dumps(manifest), encoding="utf-8")
    return description, interface


def test_cpu_memory_is_private_and_new_rounds_reset_it(tmp_path):
    path, interface = artifact(tmp_path)
    policy = OnnxPolicy("portable", path, interface)
    first, second = policy.spawn(51), policy.spawn(51)
    observation = np.zeros(interface.observation_space.shape, np.float32)
    assert first.act(observation) == second.act(observation)
    first.act(observation)
    np.testing.assert_array_equal(first.hidden, np.full((1, 1, 2), 2, np.float32))
    np.testing.assert_array_equal(second.hidden, np.ones((1, 1, 2), np.float32))
    np.testing.assert_array_equal(policy.spawn(52).hidden, np.zeros((1, 1, 2), np.float32))
    assert policy.session.get_providers() == ["CPUExecutionProvider"]
    with pytest.raises(ValueError, match="observation"):
        first.act(observation[:-1])


def test_cpu_loader_rejects_changed_weights_and_training_contract(tmp_path):
    path, interface = artifact(tmp_path)
    manifest = json.loads(path.read_text())
    manifest["model_sha256"] = "0"*64
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="checksum"):
        OnnxPolicy("portable", path, interface)
    config = OmegaConf.load(tmp_path / "training.yaml")
    config.episode.latency_frames = 12
    OmegaConf.save(config, tmp_path / "training.yaml")
    with pytest.raises(ValueError, match="episode configurations differ"):
        OnnxPolicy("portable", path, interface)
