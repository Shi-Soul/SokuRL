"""Complete privileged records must reach the shared PPO encoder unchanged."""
import numpy as np
import pytest
from gymnasium import spaces

torch = pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")

from soku_rl.env.observation.memory_schema import (
    FIGHTER_NAMES, FIGHTER_WIDTH, MAX_OBJECTS, OBJECT_WIDTH, PLAYER_WIDTH,
    PRIVILEGED_FEATURES, WORLD_NAMES)
from soku_rl.rl.features import PrivilegedFeatures, NumericPrivilegedFeatures
from soku_rl.env.observation.privileged import encode_values


@pytest.mark.parametrize("encoder_type", [PrivilegedFeatures, NumericPrivilegedFeatures])
def test_every_object_position_affects_features_and_padding_is_ignored(encoder_type):
    torch.set_num_threads(1)
    torch.manual_seed(17)
    space = spaces.Box(-np.inf, np.inf, (PRIVILEGED_FEATURES + 8,), np.float32)
    encoder = encoder_type(space, 1, 4, 16, 8)
    values = torch.zeros(1, space.shape[0], requires_grad=True)
    with torch.no_grad():
        for seat in (0, 1):
            offset = (len(WORLD_NAMES) + seat * PLAYER_WIDTH) * 2
            values[0, offset + FIGHTER_NAMES.index("obj_n") * 2 + 1] = MAX_OBJECTS / 65536.
    encoder(values).sum().backward()
    for seat in (0, 1):
        start = (len(WORLD_NAMES) + seat * PLAYER_WIDTH + FIGHTER_WIDTH) * 2
        gradients = values.grad[0, start:start + MAX_OBJECTS * OBJECT_WIDTH * 2]
        assert torch.all(gradients.reshape(MAX_OBJECTS, -1).abs().sum(1) > 0)
    assert torch.all(values.grad[0, -8:].abs() > 0)
    empty = torch.zeros_like(values)
    changed_padding = empty.clone()
    start = (len(WORLD_NAMES) + FIGHTER_WIDTH) * 2
    changed_padding[0, start:start + OBJECT_WIDTH * 2] = 123
    assert torch.equal(encoder(empty), encoder(changed_padding))
    assert sum(p.numel() for p in encoder.parameters()) < 1_000_000


def test_numeric_features_keep_lossless_parts_and_scale_small_signed_values():
    space = spaces.Box(-np.inf, np.inf, (PRIVILEGED_FEATURES,), np.float32)
    encoder = NumericPrivilegedFeatures(space, 1, 4, 16, 8)
    raw = np.array([-1., 0., 1., 0.125, 10000., 4294967295.])
    parts = torch.from_numpy(encode_values(raw).reshape(1, -1))
    features = encoder.numeric_features(parts)
    assert torch.equal(features[:, :parts.shape[-1]], parts)
    np.testing.assert_allclose(features[0, parts.shape[-1]:].numpy(),
                               np.sign(raw) * np.log1p(np.abs(raw)) / 16., rtol=1e-6)
    assert features[0, parts.shape[-1] + 2] > 1000 * parts[0, 5]
