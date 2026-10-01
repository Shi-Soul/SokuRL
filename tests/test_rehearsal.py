import copy
from pathlib import Path

from hydra import compose, initialize_config_dir
import numpy as np
from omegaconf import OmegaConf
import pytest
import torch
from stable_baselines3.common.logger import configure

from test_behavior_cloning import dataset
from test_demonstration_sets import pair
from test_shared_ppo import fixture_config, fixture_env, save_contract
from test_recurrent_separate_features import TrainableFeatures
from soku_rl.rl import ppo_settings
from soku_rl.rl.behavior_cloning import ObservationContractEnv
from soku_rl.rl.ppo import algorithm_type, create_ppo, parameter_hash, snapshot
from soku_rl.rl.recurrent_cloning import zero_states
from soku_rl.rl.rehearsal import validate_rehearsal, window_distribution
from soku_rl.rl.storage import PackedObservation


def settings(path):
    return {"datasets": [str(path)], "updates_per_rollout": 2, "sequences": 2,
        "sequence_length": 2, "learning_rate": .01, "seed": 23}


@pytest.mark.parametrize("algorithm", ["ppo", "br", "ippo", "nfsp", "psro"])
def test_shared_optional_settings_reach_every_marl_learner(algorithm):
    with initialize_config_dir(config_dir=str(Path(__file__).parents[1] / "config"), version_base="1.3"):
        cfg = OmegaConf.to_container(compose(config_name="train", overrides=[
            f"algorithm={algorithm}", "rl=recurrent_rehearsal"]), resolve=True)
    shared = ppo_settings(cfg)
    learner = cfg["algorithm"]["response"] if algorithm == "psro" else cfg["algorithm"]
    assert learner["rehearsal"] == shared["rehearsal"]
    learner["rehearsal"] = dict(shared["rehearsal"], seed=9)
    with pytest.raises(ValueError, match="rehearsal"):
        ppo_settings(cfg)


@pytest.mark.parametrize("kind", ["mlp", "lstm"])
def test_actual_ppo_update_rehearsal_resume_weights_and_inference(pair, kind):
    from soku_rl.marl.br import OpponentEntry
    from soku_rl.rl.opponent_env import OpponentMixtureVecEnv
    from test_demonstrations import ConstantPolicy
    first, _, interface = pair
    torch.set_num_threads(1)
    env = fixture_env()
    view = OpponentMixtureVecEnv(env, 0, [OpponentEntry("constant", ConstantPolicy(8))], [1.], 17)
    config = fixture_config(kind) | {"name": "br", "rehearsal": settings(first)}
    model, _ = create_ppo(view, interface, config, {"kind": "fresh"}, "cpu", 7)
    assert sum(map(len, model._rehearsal.episodes)) == 6  # held-out six frames excluded
    model.learn(8)
    assert model.num_timesteps == 8 and model._n_updates == 1
    assert model.rehearsal_state["updates"] == 2
    assert model.rehearsal_state["frames"] > 0
    assert model.logger.name_to_value["rehearsal/seconds"] > 0
    path = first / "rehearsal.zip"
    exported = snapshot("test", model, path)
    assert exported.model is model
    loaded_for_inference = algorithm_type(kind).load(path, device="cpu")
    assert parameter_hash(loaded_for_inference.policy) == parameter_hash(model.policy)
    assert not hasattr(loaded_for_inference, "_rehearsal")  # dataset not pickled into zip
    contract_path = save_contract(first / "second", env, config)
    contract = OmegaConf.load(contract_path)
    contract.rl.rehearsal = config["rehearsal"]
    OmegaConf.save(contract, contract_path)
    source = {"kind": "checkpoint", "path": str(path), "training_config": contract_path}
    resumed, _ = create_ppo(view, interface, config, source, "cpu", 19)
    assert resumed.rehearsal_state == model.rehearsal_state
    resumed.set_logger(configure(folder=None, format_strings=[]))
    before = model.rehearsal_state["updates"]
    # Same optimizer moments, replay RNG and fresh prefix computation reproduce
    # the next auxiliary update despite a different environment seed.
    a = model._rehearsal.update(model)
    b = resumed._rehearsal.update(resumed)
    assert {k: v for k, v in a.items() if k != "seconds"} == pytest.approx(
        {k: v for k, v in b.items() if k != "seconds"})
    assert parameter_hash(resumed.policy) == parameter_hash(model.policy)
    assert resumed.rehearsal_state["updates"] == before + 2
    reset, _ = create_ppo(view, interface, config, source | {"kind": "weights"}, "cpu", 19)
    assert reset.rehearsal_state["updates"] == reset.num_timesteps == 0
    assert not reset.policy.optimizer.state
    changed = copy.deepcopy(config)
    changed["rehearsal"]["seed"] += 1
    with pytest.raises(ValueError, match="rehearsal"):
        create_ppo(view, interface, changed, source, "cpu", 19)
    del changed["rehearsal"]
    with pytest.raises(ValueError, match="rehearsal"):
        create_ppo(view, interface, changed, source, "cpu", 19)
    view.close()
    env.close()


def test_recurrent_window_burn_in_matches_online_history_and_has_actor_gradients():
    torch.set_num_threads(1)
    env = fixture_env()
    config = fixture_config("lstm")
    config["ppo"]["policy_kwargs"].update(share_features_extractor=False,
        features_extractor_class="test_recurrent_separate_features.TrainableFeatures")
    model, _ = create_ppo(ObservationContractEnv(env.interface), env.interface, config,
        {"kind": "fresh"}, "cpu", 7)
    rng = np.random.default_rng(13)
    episode = [(PackedObservation.pack(rng.normal(size=env.single_observation_space.shape).astype(np.float32)),
        int(rng.integers(env.single_action_space.n)), 0., -1 if i == 0 else 1) for i in range(263)]
    model.policy.set_training_mode(False)
    with torch.no_grad():
        states, expected = zero_states(model.policy, 1).pi, []
        for row in episode:
            distribution, states = model.policy.get_distribution(torch.tensor(row[0].unpack())[None], states,
                torch.zeros(1))
            expected.append(float(-distribution.log_prob(torch.tensor([row[1]]))))
    nll, _ = window_distribution(model, episode, 259, 4)
    assert nll.detach().numpy() == pytest.approx(expected[259:], abs=1e-6)
    nll.mean().backward()
    assert model.policy.lstm_actor.weight_hh_l0.grad.abs().sum() > 0
    assert any(p.grad is not None for p in model.policy.pi_features_extractor.parameters())
    assert all(p.grad is None for p in model.policy.vf_features_extractor.parameters())
    assert all(p.grad is None for p in model.policy.lstm_critic.parameters())
    env.close()


@pytest.mark.parametrize("kind", ["mlp", "lstm"])
def test_joint_marl_collection_calls_the_same_rehearsal_update(pair, kind):
    from soku_rl.marl.ippo import train_ippo
    first, _, _ = pair
    env = fixture_env()
    config = fixture_config(kind) | {"name": "ippo", "rehearsal": settings(first)}
    output = first / "joint"
    output.mkdir()
    report = train_ippo(env, config, "cpu", 13, output)
    assert report["updates"] == 1
    for player in (0, 1):
        model = algorithm_type(kind).load(output / f"player_{player}/final.zip", device="cpu")
        assert model.rehearsal_state["updates"] == 2
        assert model.num_timesteps == 8
    env.close()


def test_actor_rehearsal_preserves_private_critic_and_restores_learning_rate(pair):
    first, _, interface = pair
    env = fixture_env()
    config = fixture_config("lstm") | {"rehearsal": settings(first)}
    config["ppo"]["policy_kwargs"].update(share_features_extractor=False,
        features_extractor_class="test_recurrent_separate_features.TrainableFeatures")
    model, _ = create_ppo(ObservationContractEnv(interface), interface, config,
        {"kind": "fresh"}, "cpu", 13)
    model.set_logger(configure(folder=None, format_strings=[]))
    # Existing critic optimizer momentum must not cause a private critic update.
    observations = torch.ones((2, *env.single_observation_space.shape))
    _, values, _, _ = model.policy(observations, zero_states(model.policy, 2), torch.ones(2), deterministic=True)
    values.square().mean().backward()
    model.policy.optimizer.step()
    critic = [model.policy.vf_features_extractor, model.policy.lstm_critic, model.policy.value_net]
    before = list(map(parameter_hash, critic))
    rates = [group["lr"] for group in model.policy.optimizer.param_groups]
    model._rehearsal.update(model)
    assert list(map(parameter_hash, critic)) == before
    assert [group["lr"] for group in model.policy.optimizer.param_groups] == rates
    assert model.num_timesteps == model._n_updates == 0
    env.close()


@pytest.mark.parametrize("key,value", [("seed", -1), ("sequences", True), ("sequence_length", 0),
    ("updates_per_rollout", 0), ("learning_rate", float("nan")), ("learning_rate", True), ("datasets", [])])
def test_invalid_rehearsal_configuration_fails_early(key, value):
    config = settings("dataset")
    config[key] = value
    with pytest.raises(ValueError):
        validate_rehearsal(config)
