import numpy as np
import pytest
import torch
from sb3_contrib import RecurrentPPO

from test_behavior_cloning import dataset
from test_shared_ppo import fixture_config, fixture_env, save_contract
from soku_rl.rl.behavior_cloning import ObservationContractEnv, fit_demonstrations, load_demonstrations
from soku_rl.rl.ppo import create_ppo, parameter_hash
from soku_rl.rl.recurrent_cloning import demonstration_episodes, episode_chunks, sequence_epoch, zero_states
from soku_rl.rl.storage import PackedObservation
from soku_rl.policy.loader import load_policy


def test_chunk_padding_and_state_columns_preserve_episode_identity():
    episodes = [[(f"{i}:{j}", 0, 0, -1 if j == 0 else 1) for j in range(size)]
        for i, size in enumerate([3, 5, 1])]
    chunks = list(episode_chunks(episodes, [0, 1, 2], 2, 2))
    assert [[row[0] for row in chunk["samples"]] for chunk in chunks] == [
        ["0:0", "0:1", "1:0", "1:1"], ["0:2", "0:2", "1:2", "1:3"], ["1:4"], ["2:0"]]
    assert [chunk["keep"] for chunk in chunks] == [[0, 1], [0, 1], [1], [0]]
    assert [chunk["new_group"] for chunk in chunks] == [True, False, False, True]
    assert chunks[1]["valid"] == [True, False, True, True]
    assert sum(sum(chunk["valid"]) for chunk in chunks) == 9


def test_sequence_scoring_matches_framewise_online_memory_and_resets():
    torch.set_num_threads(1)
    env = fixture_env()
    config = fixture_config("lstm")
    model, _ = create_ppo(ObservationContractEnv(env.interface), env.interface, config,
        {"kind": "fresh"}, "cpu", 17)
    rng = np.random.default_rng(19)
    episodes = [[(PackedObservation.pack(rng.normal(size=env.single_observation_space.shape).astype(np.float32)),
        int(rng.integers(env.single_action_space.n)), float(rng.normal()), -1 if j == 0 else 1)
        for j in range(size)] for size in [3, 7, 5]]
    metrics = []
    model.policy.set_training_mode(False)
    with torch.no_grad():
        for episode in episodes:
            states = zero_states(model.policy, 1)
            for packed, action, target, changed in episode:
                tensor = torch.from_numpy(packed.unpack()).unsqueeze(0)
                predicted, values, _, states = model.policy(tensor, states, torch.zeros(1), deterministic=True)
                distribution = model.policy.action_dist
                metrics.append((-float(distribution.log_prob(torch.tensor([action]))),
                    float(predicted[0] == action), float((values[0, 0] - target) ** 2),
                    float(distribution.entropy()), changed))
    expected = dict(zip(["nll", "accuracy", "value_mse", "entropy"], np.mean(np.asarray(metrics)[:, :4], axis=0)))
    expected["changed_samples"] = 12
    changed_rows = np.asarray([row[:4] for row in metrics if row[4] == 1])
    expected.update(changed_nll=changed_rows[:, 0].mean(), changed_accuracy=changed_rows[:, 1].mean())
    before = parameter_hash(model.policy)
    for length, batch in [(1, 1), (2, 4), (4, 12), (16, 32)]:
        for order in ([0, 1, 2], [2, 0, 1]):
            actual, loss, updates = sequence_epoch(model, episodes, order, batch, length, .5, False)
            assert actual == pytest.approx(expected, abs=1e-6)
            assert loss == pytest.approx(expected["nll"] + .5 * expected["value_mse"], abs=1e-6)
            assert updates == 0
    assert parameter_hash(model.policy) == before
    env.close()


def test_recurrent_fit_loads_as_shared_ppo_and_continues_online(dataset):
    directory, interface, _ = dataset
    torch.set_num_threads(1)
    samples, _, _, _ = load_demonstrations(directory, interface)
    config = fixture_config("lstm") | {"name": "br"}
    config["ppo"]["learning_rate"] = .02
    output = directory / "recurrent-fit"
    output.mkdir()
    result = fit_demonstrations(interface, config, samples, {"epochs": 10, "batch_size": 4,
        "sequence_length": 2, "value_coef": .5, "initial_policy": {"kind": "fresh"}}, "cpu", 7, output)
    assert result["ppo_steps"] == 0 and result["supervised_updates"] == 20
    assert result["history"][-1]["validation"]["nll"] < result["history"][0]["validation"]["nll"]
    model = RecurrentPPO.load(result["final_checkpoint"], device="cpu")
    assert parameter_hash(model.policy) == result["final_policy_hash"]
    initial = RecurrentPPO.load(output / "initial.zip", device="cpu")
    assert not torch.equal(initial.policy.lstm_actor.weight_hh_l0, model.policy.lstm_actor.weight_hh_l0)
    env = fixture_env()
    contract = save_contract(output, env, config)
    specification = {"kind": "sb3_recurrent", "path": result["final_checkpoint"], "training_config": contract}
    policy = load_policy("recurrent-clone", specification, interface, "cpu")
    first, second = policy.spawn(8), policy.spawn(8)
    assert [first.act(row[0].unpack()) for row in samples["validation"]] == [
        second.act(row[0].unpack()) for row in samples["validation"]]
    from soku_rl.rl.opponent_env import OpponentMixtureVecEnv
    from soku_rl.marl.br import OpponentEntry
    from test_demonstrations import ConstantPolicy
    view = OpponentMixtureVecEnv(env, 0, [OpponentEntry("constant", ConstantPolicy(8))], [1.], 17)
    continued, source = create_ppo(view, interface, config,
        {"kind": "weights", "path": result["final_checkpoint"], "training_config": contract}, "cpu", 71)
    assert parameter_hash(continued.policy) == result["final_policy_hash"]
    assert not continued.policy.optimizer.state
    continued.learn(8)
    assert continued.num_timesteps == 8 and source["source_steps"] == 0
    assert parameter_hash(continued.policy) != result["final_policy_hash"]
    view.close()
    env.close()


@pytest.mark.parametrize("rows", [[], [(0, 0, 0, 1)], [(0, 0, 0, -2)], [(0, 0, 0, -1), (0, 0, 0, 2)]])
def test_missing_episode_boundaries_are_rejected(rows):
    with pytest.raises(ValueError):
        demonstration_episodes(rows)


@pytest.mark.parametrize("length,batch", [(0, 4), (True, 4), (3, 4), (8, 4)])
def test_bad_sequence_configuration_fails_before_fitting(dataset, length, batch):
    directory, interface, _ = dataset
    samples, _, _, _ = load_demonstrations(directory, interface)
    with pytest.raises(ValueError, match="sequence_length"):
        fit_demonstrations(interface, fixture_config("lstm"), samples, {"epochs": 1,
            "batch_size": batch, "sequence_length": length, "value_coef": .5,
            "initial_policy": {"kind": "fresh"}}, "cpu", 7, directory)
