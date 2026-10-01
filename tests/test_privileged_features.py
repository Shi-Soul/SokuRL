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
from soku_rl.rl.combat_features import CombatPrivilegedFeatures, FIGHTER_SCALES
from soku_rl.env.observation.privileged import encode_values


@pytest.mark.parametrize("encoder_type", [PrivilegedFeatures, NumericPrivilegedFeatures, CombatPrivilegedFeatures])
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


def test_combat_context_scales_health_and_facing_relative_geometry():
    space = spaces.Box(-np.inf, np.inf, (PRIVILEGED_FEATURES,), np.float32)
    encoder = CombatPrivilegedFeatures(space, 1, 4, 16, 8)
    observations = np.zeros((2, PRIVILEGED_FEATURES), np.float32)
    # Mirrored positions and facing should preserve the learner-relative distance.
    for batch, facing in enumerate((1, -1)):
        for seat in (0, 1):
            fields = {"hp": (8000., 5000.)[seat], "rei": (5000., 2500.)[seat],
                      "x": (300., 620.)[seat] if facing == 1 else (980., 660.)[seat],
                      "y": (0., 128.)[seat], "dir": facing if seat == 0 else -facing,
                      "xspeed": facing * (0., 2.)[seat], "yspeed": (0., -2.)[seat]}
            for name, value in fields.items():
                index = (len(WORLD_NAMES) + seat * PLAYER_WIDTH + FIGHTER_NAMES.index(name)) * 2
                observations[batch, index:index + 2] = encode_values(np.array([value]))[0]
    tensor = torch.from_numpy(observations)
    context = encoder.frame_context(tensor)
    expected = torch.tensor([.25, .1, .1, -.1, .3, .5]).expand(2, -1)
    torch.testing.assert_close(context[:, -6:], expected)
    assert context.shape == (2, len(FIGHTER_SCALES) * 2 + 6)
    torch.testing.assert_close(context[:, 0], torch.full((2,), .8))
    assert encoder(tensor).shape == (2, 8)


def test_combat_context_supports_history_and_checkpoint_roundtrip(tmp_path):
    space = spaces.Box(-np.inf, np.inf, (2 * PRIVILEGED_FEATURES + 8,), np.float32)
    encoder = CombatPrivilegedFeatures(space, 2, 4, 16, 8)
    values = torch.zeros(1, space.shape[0])
    assert encoder(values).shape == (1, 8)
    path = tmp_path / "encoder.pt"
    torch.save(encoder.state_dict(), path)
    reloaded = CombatPrivilegedFeatures(space, 2, 4, 16, 8)
    reloaded.load_state_dict(torch.load(path, weights_only=True))
    torch.testing.assert_close(encoder(values), reloaded(values), rtol=0, atol=0)
