"""Check paused game stepping against a fake bridge, without opening a game."""
from types import SimpleNamespace
from unittest.mock import Mock
import sys

import pytest

if sys.platform != "win32":
    pytest.skip("Windows bridge bindings", allow_module_level=True)

from bridge_shared import RawFrameState, calculate_state_hash
from game_runtime.stepping import InputPair, PausedGame, input_tuple


def fake_game(frame, valid_hash):
    state = RawFrameState()
    state.frameId = frame
    state.p1.input.horizontalAxis = -20
    state.p1.input.a = 91
    state.stateHash = calculate_state_hash(state) if valid_hash else 0
    client = Mock()
    client.snapshot.side_effect = [SimpleNamespace(game_frame=0),
        SimpleNamespace(game_frame=frame, run_state_name="PAUSED", latest=state)]
    client.step.return_value = client.step_with_inputs.return_value = 7
    client.wait_for_ack.return_value = SimpleNamespace(ack_seq=7)
    return PausedGame(SimpleNamespace(pid=17), client, 0), state


@pytest.mark.parametrize("native", [False, True])
def test_paused_step_checks_hash_and_returns_independent_frame(native):
    game, state = fake_game(1, True)
    if native:
        restored = game.step_native(3.)
        game.client.step.assert_called_once_with(1)
    else:
        pair = InputPair((1, 0, 1, 0, 0, 0, 0, 0))
        restored = game.step(pair, 3.)
        game.client.step_with_inputs.assert_called_once_with(pair.p1, pair.p2)
    assert input_tuple(restored.p1.input) == (-20, 0, 91, 0, 0, 0, 0, 0)
    state.p1.input.a = 0
    assert restored.p1.input.a == 91
    game.client.drain_frames.assert_called_once_with()


@pytest.mark.parametrize("frame,valid_hash,reason", [(1, False, "hash mismatch"), (2, True, "advanced 2 frames")])
def test_native_step_rejects_bad_state_or_extra_frame(frame, valid_hash, reason):
    game, _ = fake_game(frame, valid_hash)
    with pytest.raises(RuntimeError, match=reason):
        game.step_native(3.)
    game.client.drain_frames.assert_not_called()
