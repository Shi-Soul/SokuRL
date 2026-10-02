"""Process relocation must not change neural features or their gradients."""
from pathlib import Path

from gymnasium import spaces
from hydra import compose, initialize_config_dir
import numpy as np
from omegaconf import OmegaConf
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")

from soku_rl.env.observation.memory_schema import (
    FIGHTER_NAMES, FIGHTER_WIDTH, MAX_OBJECTS, OBJECT_NAMES, OBJECT_WIDTH,
    PLAYER_WIDTH, PRIVILEGED_FEATURES, WORLD_NAMES)
from soku_rl.env.observation.privileged import encode_values
from soku_rl.rl.address_invariant_features import AddressInvariantCombatFeatures


@pytest.mark.parametrize("history", [1, 2])
def test_relocation_leaves_features_and_parameter_gradients_identical(history):
    torch.set_num_threads(1)
    torch.manual_seed(379)
    space = spaces.Box(-np.inf, np.inf, (history * PRIVILEGED_FEATURES + 8,), np.float32)
    encoder = AddressInvariantCombatFeatures(space, history, 4, 16, 8)
    values = np.zeros((1, space.shape[0]), np.float32)
    indices = []
    for frame in range(history):
        for seat in (0, 1):
            base = frame * PRIVILEGED_FEATURES + 2 * (len(WORLD_NAMES) + seat * PLAYER_WIDTH)
            values[0, base + 2 * FIGHTER_NAMES.index("obj_n") + 1] = 2 / 65536.
            indices.append(base + 2 * FIGHTER_NAMES.index("address"))
            indices.extend(base + 2 * (FIGHTER_WIDTH + obj * OBJECT_WIDTH + OBJECT_NAMES.index("address"))
                           for obj in range(2))
    # Low address bits must not be lost through a float32 uint32 conversion.
    relocated = values.copy()
    for index, position in enumerate(indices):
        values[0, position:position + 2] = encode_values(np.array([0xF0000123 + index * 32]))[0]
        relocated[0, position:position + 2] = encode_values(np.array([0x01000111 + index * 64]))[0]
    original = torch.tensor(values, requires_grad=True)
    before = original.detach().clone()
    output = encoder(original)
    output.sum().backward()
    gradients = {name: parameter.grad.clone() for name, parameter in encoder.named_parameters()}
    for position in indices:
        assert torch.count_nonzero(original.grad[0, position:position + 2]) == 0
    encoder.zero_grad(set_to_none=True)
    second = encoder(torch.tensor(relocated))
    second.sum().backward()
    torch.testing.assert_close(output, second, rtol=0, atol=0)
    for name, parameter in encoder.named_parameters():
        torch.testing.assert_close(gradients[name], parameter.grad, rtol=0, atol=0)
    assert torch.equal(original, before)


@pytest.mark.parametrize("width", [FIGHTER_WIDTH, OBJECT_WIDTH])
def test_only_address_channels_are_removed_and_empty_objects_are_supported(width):
    space = spaces.Box(-np.inf, np.inf, (PRIVILEGED_FEATURES,), np.float32)
    encoder = AddressInvariantCombatFeatures(space, 1, 4, 16, 8)
    raw = np.arange(width, dtype=np.float64) - 40
    records = torch.from_numpy(encode_values(raw).reshape(1, -1))
    before = records.clone()
    features = encoder.numeric_features(records)
    address = FIGHTER_NAMES.index("address")
    keep = np.ones(width, dtype=bool)
    keep[address] = False
    assert torch.equal(features[:, :2 * width].reshape(1, width, 2)[:, keep], records.reshape(1, width, 2)[:, keep])
    np.testing.assert_allclose(features[:, 2 * width:][:, keep],
                               (np.sign(raw) * np.log1p(abs(raw)) / 16.)[None, keep], rtol=1e-6)
    assert torch.equal(records, before)
    assert encoder.numeric_features(records[:0]).shape == (0, 3 * width)


def test_address_invariant_offline_and_online_contracts_match():
    with initialize_config_dir(config_dir=str(Path(__file__).parents[1] / "config"), version_base="1.3"):
        original = compose(config_name="pretrain_recurrent_numeric_combat_demonstrations")
        offline = compose(config_name="pretrain_recurrent_address_invariant_demonstrations")
        online = compose(config_name="train", overrides=["algorithm=br", "rl=recurrent_demonstration_transfer",
            "track=superhuman_address_invariant", "wrappers=superhuman_learning"])
        assert OmegaConf.to_container(original.episode) == OmegaConf.to_container(offline.episode)
        assert OmegaConf.to_container(original.wrappers) == OmegaConf.to_container(offline.wrappers)
        assert OmegaConf.to_container(offline.rl.ppo.policy_kwargs) == OmegaConf.to_container(online.rl.ppo.policy_kwargs)
        assert offline.pretraining.initial_policy.kind == "fresh"
        assert offline.pretraining.action_change_weight == 1.
