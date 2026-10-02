"""Export and numerically verify the frozen online Q network used by DQN actors."""
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from omegaconf import OmegaConf
import torch

from soku_rl.env import EpisodeConfig
from soku_rl.env.wrappers.learning import LearningConfig, LearningInterface
from soku_rl.policy.loader import load_policy


def export_dqn(config):
    import onnx
    import onnxruntime as ort

    spec = config["candidate"]["policy"]
    training = OmegaConf.to_container(OmegaConf.load(spec["training_config"]), resolve=True)
    interface = LearningInterface(EpisodeConfig.from_dict(training["episode"]), LearningConfig(**training["wrappers"]))
    if interface.episode.observation_mode not in {"state", "diagnostic_state"}:
        raise ValueError("DQN CPU export requires public or compact numeric state observations")
    steps = config["verification_steps"]
    if type(steps) is not int or steps < 512:
        raise ValueError("verify at least 512 greedy DQN decisions")
    torch.set_num_threads(1)
    loaded = load_policy(config["candidate"]["name"], spec, interface, "cpu")
    network = loaded.model.q_net.eval()
    shape = interface.observation_space.shape
    directory = Path(config["output"]).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "actor.onnx"
    torch.onnx.export(network, (torch.zeros((1, *shape)),), path, dynamo=False,
        opset_version=17, input_names=["observation"], output_names=["q_values"])
    onnx.checker.check_model(str(path), full_check=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    session.disable_fallback()
    rng = np.random.default_rng(config["seed"])
    maximum_error = 0.
    with torch.inference_mode():
        for _ in range(steps):
            observation = rng.uniform(-1, 1, size=(1, *shape)).astype(np.float32)
            expected = network(torch.from_numpy(observation)).numpy()
            actual, = session.run(None, {"observation": observation})
            np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=2e-6)
            if not np.array_equal(actual.argmax(-1), expected.argmax(-1)):
                raise RuntimeError("portable DQN changed a greedy action during verification")
            maximum_error = max(maximum_error, float(np.max(np.abs(actual - expected))))
    shutil.copyfile(spec["training_config"], directory / "training.yaml")
    manifest = {"format": "sokurl-dqn-onnx-v1", "model": path.name,
        "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_sha256": loaded.fingerprint, "training_config": "training.yaml",
        "observation_shape": list(shape), "num_actions": int(interface.action_space.n),
        "inference": "greedy_online_q", "verification": {"steps": steps,
            "maximum_absolute_error": maximum_error, "greedy_action_mismatches": 0},
        "versions": {"torch": torch.__version__, "onnx": onnx.__version__, "onnxruntime": ort.__version__}}
    (directory / "policy.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    resolved = config | {"episode": training["episode"], "wrappers": training["wrappers"]}
    OmegaConf.save(OmegaConf.create(resolved), directory / "export.yaml")
    return manifest
