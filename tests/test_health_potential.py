import numpy as np
import pytest

from test_privileged_encoding import observation
from soku_rl.env.observation.memory_schema import PRIVILEGED_FEATURES
from soku_rl.env.observation.privileged import (
    PrivilegedObservation, decode_privileged, encode_privileged)
from soku_rl.env.wrappers.features import health_potential


@pytest.mark.parametrize("hp", [(10000., 0.), (0., 10000.), (65535., 1.),
    (65536., 131071.), (-.125, 1234.5), (0xFFFFFFFF, 0x81234567)])
def test_privileged_health_matches_full_decoder_and_swapped_history(hp):
    source = observation(3)
    players = tuple(player | {"hp": value} for player, value in zip(source.players, hp, strict=True))
    old_frame = encode_privileged(source)
    for order, sign in [(players, 1), (players[::-1], -1)]:
        current = encode_privileged(PrivilegedObservation(source.world, order))
        decoded = decode_privileged(current).players
        expected = (decoded[0]["hp"] - decoded[1]["hp"]) / 10000.
        assert health_potential(np.concatenate((old_frame, current)), "privileged_state") == expected
        assert expected == sign * (hp[0] - hp[1]) / 10000.


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_health_still_rejects_nonfinite_unrelated_fields(invalid):
    values = encode_privileged(observation(0))
    values[-1] = invalid
    with pytest.raises(ValueError, match="shape or values"):
        health_potential(values, "privileged_state")


@pytest.mark.parametrize("shape", [(PRIVILEGED_FEATURES - 1,), (2, PRIVILEGED_FEATURES)])
def test_health_rejects_incomplete_or_matrix_observations(shape):
    with pytest.raises(ValueError, match="shape or values"):
        health_potential(np.zeros(shape, np.float32), "privileged_state")
