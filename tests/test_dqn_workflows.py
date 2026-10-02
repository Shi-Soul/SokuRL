"""DQN artifacts cross offline fitting, BR, CPU export and original-match play."""
from dataclasses import asdict, replace
import json

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

from soku_rl.env.wrappers.learning import LearningInterface, LearningEpisode
from soku_rl.env.observation_history import ObservationHistory
from soku_rl.play.loader import load_play_policy, play_interface, warm_play_policy
from soku_rl.play.live_policy import LivePolicy
from soku_rl.policy.loader import load_policy
from soku_rl.rl.behavior_cloning import ObservationContractEnv, fit_demonstrations, load_demonstrations
from soku_rl.rl.learner import create_learner, parameter_hash
from test_behavior_cloning import dataset
from test_dqn import dqn_config
from test_shared_ppo import fixture_env
from test_env_timing import RecordingBackend


def save_model(directory, interface):
    config = dqn_config() | {"name": "br"}
    model, _ = create_learner(ObservationContractEnv(interface), interface, config, {"kind": "fresh"}, "cpu", 19)
    path = directory / "final.zip"
    model.save(path)
    contract = directory / "config.yaml"
    OmegaConf.save(OmegaConf.create({"episode": asdict(interface.episode), "wrappers": asdict(interface.config),
        "rl": config, "algorithm": config}), contract)
    return {"kind": "sb3_dqn", "path": str(path), "training_config": str(contract)}, model


@pytest.mark.parametrize("seat", [0, 1])
def test_play_loads_saved_contract_allows_live_characters_and_stays_greedy(tmp_path, seat):
    torch.set_num_threads(1)
    env = fixture_env()
    spec, model = save_model(tmp_path, env.interface)
    candidate = {"name": "dqn", "policy": spec}
    interface = play_interface(candidate, {}, {}, "superhuman", [])
    other_match = replace(interface.episode.match, player_0=replace(interface.episode.match.player_0, character=19))
    selected = LearningInterface(replace(interface.episode, match=other_match), interface.config)
    loaded = load_play_policy(candidate, selected, {}, "cpu", seat)
    assert warm_play_policy(loaded, selected, 1) == 1
    loaded.model.exploration_rate = 1.
    wrapped = load_policy("greedy", {"kind": "greedy", "policy": spec}, interface, "cpu")
    assert wrapped.fingerprint == loaded.fingerprint
    for obs in np.random.default_rng(5).normal(size=(16, *interface.observation_space.shape)).astype(np.float32):
        expected = int(model.predict(obs, deterministic=True)[0])
        assert loaded.spawn_play(9).act(obs) == wrapped.spawn(33).act(obs) == expected
    with pytest.raises(ValueError, match="selected track"):
        play_interface(candidate, {}, {}, "human", [])
    with pytest.raises(ValueError, match="remove overrides"):
        play_interface(candidate, {}, {}, "superhuman", ["episode.history_frames=2"])
    env.close()


@pytest.mark.parametrize("seat", [0, 1])
def test_diagnostic_live_round_clock_history_and_commands_match_training(tmp_path, seat):
    torch.set_num_threads(1)
    env = fixture_env()
    interface = LearningInterface(replace(env.interface.episode, history_frames=4), env.interface.config)
    spec, _ = save_model(tmp_path, interface)
    policy = load_policy("dqn", spec, interface, "cpu")
    live = LivePolicy(policy, interface, seat)
    backend = RecordingBackend()
    for origin in (1500, 9000):
        actor = policy.spawn(42)
        history = ObservationHistory(interface.episode, (live.agent,))
        features = LearningEpisode(interface)
        for frame in range(7):
            backend.frames[0] = min(frame, interface.episode.max_frames)
            pair = backend._state(0).observations
            absolute = tuple(replace(obs, frame=origin + frame) for obs in pair)
            if frame == 0:
                history.reset(frame, pair)
                features.reset_agent(live.agent, history.observations()[live.agent])
                live.start_round(origin, absolute, 42)
            else:
                history.append(frame, pair)
                live.observe(origin + frame, absolute)
            obs = features.observation(live.agent, history.observations()[live.agent], min(frame, interface.episode.max_frames))
            np.testing.assert_array_equal(live.history.observations()[live.agent], history.observations()[live.agent])
            command = interface.command(actor.act(obs))
            assert live.act() == command
            features.record_command(live.agent, command)
        live.stop()
    env.close()


@pytest.mark.parametrize("mode", ["state", "diagnostic_state"])
def test_portable_dqn_matches_torch_and_rejects_corrupted_artifact(tmp_path, mode):
    from soku_rl.policy.export_dqn import export_dqn
    torch.set_num_threads(1)
    env = fixture_env()
    interface = LearningInterface(replace(env.interface.episode, observation_mode=mode), env.interface.config)
    spec, model = save_model(tmp_path, interface)
    output = tmp_path / "portable"
    manifest = export_dqn({"candidate": {"name": "dqn", "policy": spec},
        "verification_steps": 512, "seed": 23, "output": str(output)})
    source = {"kind": "onnx_dqn", "path": str(output / "policy.json")}
    candidate = {"name": "portable", "policy": source}
    restored = play_interface(candidate, {}, {}, "human" if mode == "state" else "superhuman", [])
    loaded = load_play_policy(candidate, restored, {}, "cpu", 0)
    assert warm_play_policy(loaded, restored, 5) == 1
    assert manifest["verification"]["greedy_action_mismatches"] == 0
    for obs in np.random.default_rng(11).normal(size=(20, *interface.observation_space.shape)).astype(np.float32):
        assert loaded.spawn_play(123).act(obs) == int(model.predict(obs, deterministic=True)[0])
    with pytest.raises(ValueError, match="observation"):
        loaded.spawn(1).act(np.zeros(2))
    with pytest.raises(ValueError, match="device=cpu"):
        load_policy("portable", source, interface, "cuda:0")
    manifest["model_sha256"] = "bad"
    (output / "policy.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="checksum"):
        load_policy("portable", source, interface, "cpu")
    env.close()


def test_dqn_demonstration_fit_evaluate_transfer_and_br_training(dataset):
    from soku_rl.rl.demonstration_evaluation import score_validation
    from soku_rl.marl.br import train_br
    directory, interface, _ = dataset
    torch.set_num_threads(1)
    samples, manifest, _, _ = load_demonstrations(directory, interface)
    config = dqn_config() | {"name": "br", "player": 0, "matchups": {"mode": "fixed"},
        "timesteps": 16, "checkpoint_every": 8,
        "opponents": [{"name": "random", "probability": 1., "policy": {"kind": "uniform"}}]}
    config["dqn"]["learning_rate"] = .02
    output = directory / "fit-dqn"
    output.mkdir()
    result = fit_demonstrations(interface, config, samples, {"epochs": 20, "batch_size": 4,
        "value_coef": 0., "action_change_weight": 1., "initial_policy": {"kind": "fresh"}}, "cpu", 7, output)
    contract = output / "config.yaml"
    OmegaConf.save(OmegaConf.create({"episode": asdict(interface.episode), "wrappers": asdict(interface.config),
        "rl": config, "algorithm": config}), contract)
    assert result["learner"] == "dqn" and result["learner_steps"] == 0
    assert result["history"][-1]["validation"]["accuracy"] == 1.
    assert result["history"][-1]["validation"]["nll"] < result["history"][0]["validation"]["nll"] * .5
    source = {"kind": "weights", "path": result["final_checkpoint"], "training_config": str(contract)}
    model, _ = create_learner(ObservationContractEnv(interface), interface, config, source, "cpu", 71)
    assert parameter_hash(model.policy) == result["final_policy_hash"]
    assert not model.policy.optimizer.state
    for online, target in zip(model.q_net.parameters(), model.q_net_target.parameters(), strict=True):
        torch.testing.assert_close(online, target, rtol=0, atol=0)
    before = parameter_hash(model.policy)
    scored = score_validation(model, samples, manifest, 4, 1)
    assert scored["groups"]["overall"]["metrics"]["accuracy"] == 1.
    assert parameter_hash(model.policy) == before
    continued = directory / "continued-dqn"
    continued.mkdir()
    env = fixture_env()
    try:
        config["initial_policy"] = source
        trained = train_br(env, config, "cpu", 9, continued)
        assert trained["initial_policy_hash"] == before and trained["steps"] == 16
        assert (continued / "final.replay.pkl").exists()
    finally:
        env.close()
