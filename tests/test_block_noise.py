"""Correlated gates preserve frame cadence, private RNG and adaptive feedback."""
import copy
from pathlib import Path

from hydra import compose, initialize_config_dir
import numpy as np
from omegaconf import OmegaConf
import pytest

from soku_rl.policy.action_noise import ActionNoisePolicy
from soku_rl.policy.block_noise import BlockActionNoisePolicy
from soku_rl.policy.loader import load_policy
from soku_rl.policy.matchups import opponent_interface
from soku_rl.policy.population import UniformPolicy
from soku_rl.rl.curriculum import create_curriculum
from test_action_noise import CounterPolicy
from test_adaptive_curriculum import settings
from test_shared_ppo import fixture_env


@pytest.mark.parametrize("probability", [0., .125, .7, 1.])
def test_single_decision_blocks_exactly_reproduce_independent_gate(probability):
    base = CounterPolicy()
    old = ActionNoisePolicy("target", base, 4, probability).spawn(31)
    new = BlockActionNoisePolicy("target", base, 4, probability, 1).spawn(31)
    for observation in range(1024):
        assert old.act(observation) == new.act(observation)
        assert old.gate_rng.bit_generator.state == new.gate_rng.bit_generator.state
        assert old.action_rng.bit_generator.state == new.action_rng.bit_generator.state
    assert old.actor.count == new.actor.count == 1027


@pytest.mark.parametrize("probability", [0., 1.])
def test_blocks_preserve_endpoint_actions_and_play_reset_contract(probability):
    base = CounterPolicy()
    old = ActionNoisePolicy("target", base, 4, probability).spawn(31)
    new = BlockActionNoisePolicy("target", base, 4, probability, 16).spawn_play(31)
    assert new.reset_each_round is False
    for observation in range(1024):
        assert old.act(observation) == new.act(observation)
    assert new.actor.actor.count == old.actor.count


def test_gate_is_constant_within_blocks_but_uniform_commands_and_original_actor_advance():
    policy = BlockActionNoisePolicy("target", CounterPolicy(), 576, .25, 16)
    actor, twin, unrelated = policy.spawn(71), policy.spawn(71), policy.spawn(19)
    gate_seed, action_seed = np.random.SeedSequence(71).spawn(2)
    gate_rng, action_rng = np.random.default_rng(gate_seed), np.random.default_rng(action_seed)
    flags, noise_commands = [], []
    for block in range(2048):
        flag = gate_rng.random() < .25
        flags.append(flag)
        for offset in range(16):
            observation = block * 16 + offset
            action = actor.act(observation)
            unrelated.act(observation)
            unrelated.act(observation + 1)
            assert action == twin.act(observation)
            assert actor.replace_block == flag and actor.remaining_decisions == 15 - offset
            expected = int(action_rng.integers(576)) if flag else (71 % 4 + observation + 1 + observation) % 4
            assert action == expected
            if flag:
                noise_commands.append(action)
    assert actor.actor.count == 71 % 4 + 2048 * 16
    assert abs(np.mean(flags) - .25) < .03
    assert len(set(noise_commands)) == 576
    assert any(a != b for a, b in zip(noise_commands, noise_commands[1:]))
    assert policy.fingerprint != BlockActionNoisePolicy("target", CounterPolicy(), 576, .25, 8).fingerprint


@pytest.mark.parametrize("value", [0, -1, 1.5, True, "16", None])
def test_invalid_block_lengths_fail_in_policy_and_curriculum(value):
    with pytest.raises(ValueError, match="block_decisions"):
        BlockActionNoisePolicy("target", CounterPolicy(), 4, .5, value)
    with pytest.raises(ValueError, match="block_decisions"):
        create_curriculum(settings() | {"kind": "adaptive_block_action_noise", "block_decisions": value},
            [UniformPolicy("a", 4)], [1.], 4)


def test_shared_ema_feedback_stays_identical_and_existing_actors_keep_their_probability(tmp_path):
    opponents = [UniformPolicy("a", 4), UniformPolicy("b", 4)]
    base = settings() | {"warmup_episodes": 1, "update_every": 1}
    config = base | {"kind": "adaptive_block_action_noise", "block_decisions": 16}
    old = create_curriculum(base, opponents, [.5, .5], 4)
    new = create_curriculum(config, opponents, [.5, .5], 4)
    actor, initial_context = new.spawn(opponents[0], 29)
    twin, _ = new.spawn(opponents[0], 29)
    before = copy.deepcopy(initial_context)
    for index in range(80):
        opponent = opponents[index % 2]
        _, old_context = old.spawn(opponent, index)
        _, new_context = new.spawn(opponent, index)
        context = {"opponent": opponent.name, "player": index % 2}
        info = {"outcome": "p1_win" if index < 40 else "p2_win"}
        assert old.observe(old_context | context, info) == new.observe(new_context | context, info)
    assert new.snapshot()["states"] == old.snapshot()["states"]
    assert initial_context == before and actor.random_probability == .5
    assert [actor.act(i) for i in range(31)] == [twin.act(i) for i in range(31)]
    checkpoint = tmp_path / "model.zip"
    checkpoint.write_bytes(b"block-feedback-checkpoint")
    new.save(checkpoint, 2048)
    restored = create_curriculum(config, opponents, [.5, .5], 4)
    restored.restore({"kind": "checkpoint", "path": str(checkpoint)})
    assert restored.snapshot() == new.snapshot()
    for instance in (old, create_curriculum(config | {"block_decisions": 8}, opponents, [.5, .5], 4)):
        with pytest.raises(ValueError, match="kind|config"):
            instance.restore({"kind": "checkpoint", "path": str(checkpoint)})


def test_loader_preserves_nested_god_character_validation_and_hydra_contract():
    env = fixture_env()
    try:
        spec = {"kind": "block_action_noise", "random_probability": .125,
                "block_decisions": 16, "policy": {"kind": "uniform"}}
        policy = load_policy("target", spec, env.interface, "cpu")
        assert policy.block_decisions == 16
        with pytest.raises(ValueError, match="block_action_noise"):
            load_policy("target", spec | {"extra": 1}, env.interface, "cpu")
        spec["policy"] = {"kind": "rule", "name": "god", "rules": {"god": {"script": "00_reimu_main.ai"}}}
        with pytest.raises(ValueError, match="script and opponent character"):
            opponent_interface(env.interface, {"character": 1, "palette": 0, "deck": 0},
                {"policy": spec, "setup": {"character": 6, "palette": 0, "deck": 0}})
    finally:
        env.close()
    with initialize_config_dir(config_dir=str(Path(__file__).parents[1] / "config"), version_base="1.3"):
        base = OmegaConf.to_container(compose(config_name="benchmark_address_noise_boundary"), resolve=True)
        block = OmegaConf.to_container(compose(config_name="benchmark_address_block_noise"), resolve=True)
        adaptive = compose(config_name="train", overrides=["algorithm=br", "+curriculum=adaptive_block_noise"])
    assert adaptive.algorithm.curriculum.kind == "adaptive_block_action_noise"
    assert adaptive.algorithm.curriculum.block_decisions == 16
    actual = block["algorithm"]["opponents"][0]["policy"]
    assert actual.pop("block_decisions") == 16
    actual["kind"] = "action_noise"
    block["output"] = base["output"]
    assert block == base
