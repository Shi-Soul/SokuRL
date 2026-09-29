"""Export and verify a recurrent PPO actor for standalone CPU play."""
import hashlib
import json
from pathlib import Path
import shutil

import hydra
import numpy as np
from omegaconf import OmegaConf
import torch
from torch import nn

from soku_rl.policy.loader import load_policy
from soku_rl.env import EpisodeConfig
from soku_rl.env.wrappers.learning import LearningConfig, LearningInterface


class RecurrentActor(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.features = policy.pi_features_extractor
        self.memory = policy.lstm_actor
        self.policy_net = policy.mlp_extractor.policy_net
        self.action_net = policy.action_net

    def forward(self, observation, hidden, cell):
        sequence = self.features(observation).unsqueeze(0)
        features, (hidden, cell) = self.memory(sequence, (hidden, cell))
        logits = self.action_net(self.policy_net(features.squeeze(0)))
        return logits.softmax(-1), hidden, cell


@hydra.main(version_base="1.3", config_path="../config", config_name="export_policy")
def main(cfg):
    import onnx
    import onnxruntime as ort

    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    spec = config["candidate"]["policy"]
    if spec["kind"] != "sb3_recurrent" or config["episode"]["observation_mode"] != "state":
        raise ValueError("CPU deployment export requires a public-state recurrent PPO policy")
    if type(config["verification_steps"]) is not int or config["verification_steps"] < 512:
        raise ValueError("verify at least 512 sequential recurrent decisions")
    torch.set_num_threads(1)
    interface = LearningInterface(EpisodeConfig.from_dict(config["episode"]), LearningConfig(**config["wrappers"]))
    loaded = load_policy(config["candidate"]["name"], spec, interface, torch.device("cpu"))
    policy = loaded.model.policy.eval()
    actor = RecurrentActor(policy).eval()
    shape, state_shape = interface.observation_space.shape, policy.lstm_hidden_state_shape
    inputs = (torch.zeros((1, *shape)), torch.zeros(state_shape), torch.zeros(state_shape))
    directory = Path(config["output"]).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    model_path = directory / "actor.onnx"
    torch.onnx.export(actor, inputs, model_path, dynamo=False, opset_version=17,
        input_names=["observation", "hidden", "cell"],
        output_names=["probabilities", "hidden_out", "cell_out"])
    onnx.checker.check_model(str(model_path), full_check=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(config["seed"])
    errors = np.zeros(3)
    with torch.inference_mode():
        for step in range(config["verification_steps"]):
            if step % 128 == 0:
                states = (torch.zeros(state_shape), torch.zeros(state_shape))
                portable = (np.zeros(state_shape, np.float32), np.zeros(state_shape, np.float32))
            observation = rng.uniform(-1, 1, size=(1, *shape)).astype(np.float32)
            distribution, states = policy.get_distribution(torch.from_numpy(observation), states, torch.zeros(1))
            expected = (distribution.distribution.probs.numpy(), *(value.numpy() for value in states))
            actual = session.run(None, {"observation": observation, "hidden": portable[0], "cell": portable[1]})
            for index, (reference, value) in enumerate(zip(expected, actual, strict=True)):
                np.testing.assert_allclose(value, reference, rtol=1e-4, atol=2e-6)
                errors[index] = max(errors[index], float(np.max(np.abs(value-reference))))
            portable = tuple(actual[1:])
    shutil.copyfile(spec["training_config"], directory / "training.yaml")
    manifest = {"format": "sokurl-recurrent-onnx-v1", "model": "actor.onnx",
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "source_sha256": loaded.fingerprint, "training_config": "training.yaml",
        "observation_shape": [int(size) for size in shape], "state_shape": [int(size) for size in state_shape],
        "num_actions": int(interface.action_space.n),
        "verification": {"steps": config["verification_steps"], "round_resets": config["verification_steps"]//128,
                         "maximum_absolute_errors": errors.tolist()},
        "versions": {"torch": torch.__version__, "onnx": onnx.__version__, "onnxruntime": ort.__version__}}
    (directory / "policy.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (directory / "export.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
