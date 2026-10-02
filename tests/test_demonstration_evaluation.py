import copy
import hashlib
import json
from pathlib import Path
import runpy

from omegaconf import OmegaConf
import pytest
import torch

from test_behavior_cloning import dataset
from test_shared_ppo import fixture_config
from soku_rl.rl.behavior_cloning import ObservationContractEnv, load_demonstrations
from soku_rl.rl.demonstration_evaluation import score_validation
from soku_rl.rl.ppo import create_ppo, parameter_hash


@pytest.mark.parametrize("kind", ["mlp", "lstm"])
def test_validation_preserves_model_and_seat_weighted_scores(dataset, kind):
    directory, interface, _ = dataset
    torch.set_num_threads(1)
    samples, manifest, _, _ = load_demonstrations(directory, interface)
    model, _ = create_ppo(ObservationContractEnv(interface), interface, fixture_config(kind),
        {"kind": "fresh"}, "cpu", 11)
    before = parameter_hash(model.policy)
    report = score_validation(model, samples, manifest, 4, 2)
    overall, first, second = (report["groups"][key] for key in ("overall", "player_0", "player_1"))
    assert overall["frames"] == first["frames"] + second["frames"] == 6
    assert overall["episodes"] == first["episodes"] + second["episodes"] == 2
    for metric in ("nll", "accuracy", "entropy", "value_mse"):
        assert overall["metrics"][metric] == pytest.approx(sum(
            group["metrics"][metric] * group["frames"] for group in (first, second)) / 6, abs=1e-6)
    assert overall["copy_previous_action"] == {"transitions": 4, "accuracy": 1.}
    assert parameter_hash(model.policy) == before and model.num_timesteps == 0
    assert not model.policy.optimizer.state
    # The read-only scorer never sees training rows.
    samples["train"] = [object()]
    assert score_validation(model, samples, manifest, 4, 2) == report
    learner_manifest = dict(manifest, schema=2, control="learner")
    learner_report = score_validation(model, samples, learner_manifest, 4, 2)
    assert all("value_mse" not in group["metrics"] for group in learner_report["groups"].values())


@pytest.mark.parametrize("problem", ["size", "boundary", "seat", "batch"])
def test_bad_validation_partition_is_rejected(dataset, problem):
    directory, interface, _ = dataset
    samples, manifest, _, _ = load_demonstrations(directory, interface)
    model, _ = create_ppo(ObservationContractEnv(interface), interface, fixture_config("mlp"),
        {"kind": "fresh"}, "cpu", 11)
    manifest = copy.deepcopy(manifest)
    first = next(row for row in manifest["episodes"] if row["split"] == "validation")
    if problem == "size":
        first["steps"] += 1
    elif problem == "boundary":
        samples["validation"][0] = (*samples["validation"][0][:3], 0)
    elif problem == "seat":
        for row in manifest["episodes"]:
            row["learner_seat"] = 0
    with pytest.raises(ValueError):
        score_validation(model, samples, manifest, 3 if problem == "batch" else 4, 2)


@pytest.mark.parametrize("kind", ["mlp", "lstm", "dqn", "nfsp_average", "nfsp_response"])
def test_entrypoint_records_the_exact_loaded_checkpoint_and_fixed_validation_games(dataset, kind):
    directory, interface, _ = dataset
    torch.set_num_threads(1)
    from test_dqn import dqn_config
    from test_shared_ppo import fixture_env
    from soku_rl.marl.nfsp import train_nfsp
    from soku_rl.rl.learner import create_learner
    from stable_baselines3 import PPO
    from soku_rl.rl.dqn import DoubleDQN
    config = fixture_config(kind) if kind in {"mlp", "lstm"} else dqn_config()
    if kind.startswith("nfsp_"):
        config.update(name="nfsp", iterations=1, timesteps_per_iteration=8, anticipatory_param=.1,
            average={"capacity": 12, "batch_size": 4, "updates": 2}, resume={"kind": "fresh"})
        output = directory / "nfsp"
        output.mkdir()
        env = fixture_env()
        try:
            train_nfsp(env, config, "cpu", 11, output)
        finally:
            env.close()
        checkpoint = output / ("player_0/final.zip" if kind == "nfsp_average" else "checkpoint-1/response-p0.zip")
        model = (PPO if kind == "nfsp_average" else DoubleDQN).load(checkpoint, device="cpu")
    else:
        model, _ = create_learner(ObservationContractEnv(interface), interface, config,
            {"kind": "fresh"}, "cpu", 11)
        checkpoint = directory / "checkpoint.zip"
        model.save(checkpoint)
    training = OmegaConf.load(directory / "config.yaml")
    training.rl = {key: config[key] for key in ("policy_type", "timeout_payoff", "ppo", "learner", "dqn")
        if key in config}
    if kind.startswith("nfsp_"):
        training.algorithm = config
    path = directory / "policy-config.yaml"
    OmegaConf.save(training, path)
    output = directory / "scoring"
    cfg = OmegaConf.create({"cpu_threads": 1, "device": "cpu", "batch_size": 4, "sequence_length": 2,
        "models": {"candidate": {"checkpoint": str(checkpoint), "training_config": str(path)}},
        "datasets": {"held_out": str(directory)}, "output": str(output)})
    main = runpy.run_path(str(Path(__file__).parents[1] / "tools/evaluate_demonstrations.py"))["main"].__wrapped__
    main(cfg)
    report = json.loads((output / "result.json").read_text())
    assert report["success"]
    candidate = report["models"]["candidate"]
    assert candidate["checkpoint_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert candidate["policy_parameter_hash"] == parameter_hash(model.policy)
    expected_learner = "dqn" if kind in {"dqn", "nfsp_response"} else "ppo"
    expected_policy = "sb3_dqn" if expected_learner == "dqn" else "sb3_recurrent" if kind == "lstm" else "sb3"
    assert candidate["learner"] == expected_learner
    assert candidate["policy_kind"] == expected_policy
    assert candidate["training_learner"] == config.get("learner", "ppo")
    assert candidate["learner_steps"] == model.num_timesteps
    assert candidate["datasets"]["held_out"]["groups"]["overall"]["frames"] == 6
    assert {row["learner_seat"] for row in candidate["datasets"]["held_out"]["validation_episodes"]} == {0, 1}
    with pytest.raises(ValueError, match="fresh"):
        main(cfg)
