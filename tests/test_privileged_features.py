"""Complete privileged records must reach the shared PPO encoder unchanged."""
import numpy as np
import pytest
from gymnasium import spaces

torch = pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")

from soku_rl.env.observation.memory_schema import (
    FIGHTER_NAMES, FIGHTER_WIDTH, MAX_OBJECTS, OBJECT_WIDTH, PLAYER_WIDTH,
    PRIVILEGED_FEATURES, WORLD_NAMES)
from soku_rl.rl.features import PrivilegedFeatures


def test_every_object_position_affects_features_and_padding_is_ignored():
    torch.set_num_threads(1)
    torch.manual_seed(17)
    space = spaces.Box(-np.inf, np.inf, (PRIVILEGED_FEATURES + 8,), np.float32)
    encoder = PrivilegedFeatures(space, 1, 4, 16, 8)
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
