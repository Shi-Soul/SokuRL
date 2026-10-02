"""Compaction preserves ordered valid records, padding, and gradient semantics."""
import copy

from gymnasium import spaces
import numpy as np
import pytest
import torch

from soku_rl.env.observation.memory_schema import (
    FIGHTER_NAMES, FIGHTER_WIDTH, MAX_OBJECTS, OBJECT_WIDTH, PLAYER_WIDTH,
    PRIVILEGED_FEATURES, WORLD_NAMES)
from soku_rl.rl.features import PrivilegedFeatures, NumericPrivilegedFeatures
from soku_rl.rl.combat_features import CombatPrivilegedFeatures, NumericCombatPrivilegedFeatures


@pytest.mark.parametrize('encoder_type', [NumericPrivilegedFeatures, NumericCombatPrivilegedFeatures])
@pytest.mark.parametrize('counts', [(0, 0, 0, 0), (0, 1, 17, MAX_OBJECTS)])
def test_compaction_preserves_outputs_input_and_parameter_gradients(encoder_type, counts):
    torch.set_num_threads(1)
    torch.manual_seed(319)
    encoder = encoder_type(spaces.Box(-np.inf, np.inf, (PRIVILEGED_FEATURES,), np.float32), 1, 4, 8, 8)
    reference = copy.deepcopy(encoder)
    objects = (torch.randn(4, MAX_OBJECTS, OBJECT_WIDTH * 2) / 65536).requires_grad_()
    dense_objects = objects.detach().clone().requires_grad_()
    present = torch.arange(MAX_OBJECTS).unsqueeze(0) < torch.tensor(counts).unsqueeze(1)
    expected = reference.object_encoder(reference.numeric_features(dense_objects)) * present.unsqueeze(-1)
    actual = encoder.encode_present_objects(objects, present)
    torch.testing.assert_close(actual, expected)
    weights = torch.randn_like(actual)
    (expected * weights).sum().backward()
    (actual * weights).sum().backward()
    torch.testing.assert_close(objects.grad, dense_objects.grad)
    assert torch.count_nonzero(objects.grad[~present]) == 0
    if any(counts):
        assert objects.grad[-1, -1].abs().sum() > 0  # highest valid slot survives
    for value, other in zip(encoder.object_encoder.parameters(), reference.object_encoder.parameters(), strict=True):
        assert value.grad is not None  # all-empty batches retain zero gradients
        torch.testing.assert_close(value.grad, other.grad, rtol=1e-4, atol=1e-5)
    changed = objects.detach().clone()
    changed[~present] = 123
    torch.testing.assert_close(encoder.encode_present_objects(changed, present), actual, rtol=0, atol=0)


@pytest.mark.parametrize('encoder_type', [PrivilegedFeatures, NumericPrivilegedFeatures,
    CombatPrivilegedFeatures, NumericCombatPrivilegedFeatures])
@pytest.mark.parametrize('players', [8, 126, 128])
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA batch dispatch requires a GPU')
def test_cuda_dispatch_only_compacts_large_numeric_batches(encoder_type, players):
    torch.set_num_threads(1)
    encoder = encoder_type(spaces.Box(-np.inf, np.inf, (PRIVILEGED_FEATURES,), np.float32),
                           1, 4, 8, 8).to('cuda')
    objects = torch.zeros(players, MAX_OBJECTS, OBJECT_WIDTH * 2, device='cuda')
    present = torch.zeros(players, MAX_OBJECTS, dtype=torch.bool, device='cuda')
    present[:, 0] = True
    inputs = []
    handle = encoder.object_encoder.register_forward_pre_hook(lambda module, args: inputs.append(args[0].shape))
    with torch.no_grad():
        result = encoder.encode_objects(objects, present)
    handle.remove()
    expected_shape = ((players, OBJECT_WIDTH * 3) if encoder.numeric_width == 3 and players >= 128
                      else (players, MAX_OBJECTS, OBJECT_WIDTH * encoder.numeric_width))
    assert inputs == [expected_shape]
    assert result.shape == (players, MAX_OBJECTS, 4)
    assert torch.count_nonzero(result[~present]) == 0


@pytest.mark.parametrize('history_frames', [1, 2])
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA history dispatch requires a GPU')
def test_large_cuda_forward_preserves_history_remainder_and_state_dict(history_frames):
    torch.set_num_threads(1)
    torch.manual_seed(89)
    space = spaces.Box(-np.inf, np.inf, (history_frames * PRIVILEGED_FEATURES + 8,), np.float32)
    encoder = NumericCombatPrivilegedFeatures(space, history_frames, 4, 8, 8).to('cuda')
    reference = copy.deepcopy(encoder)
    reference.encode_objects = lambda objects, present: (
        reference.object_encoder(reference.numeric_features(objects)) * present.unsqueeze(-1))
    observations = torch.zeros(64 // history_frames, space.shape[0], device='cuda')
    observations[:, -8:] = torch.arange(8, device='cuda') / 8
    # Populate frame storage separately: reshape of a strided history/remainder
    # slice can allocate a copy and must not silently discard these test writes.
    frames = torch.zeros(64, PRIVILEGED_FEATURES, device='cuda')
    for seat in (0, 1):
        offset = (len(WORLD_NAMES) + seat * PLAYER_WIDTH) * 2
        frames[:, offset + FIGHTER_NAMES.index('obj_n') * 2 + 1] = (seat + 1) / 65536
        start = offset + FIGHTER_WIDTH * 2
        frames[:, start + 1] = torch.arange(64, device='cuda') / 65536
    observations[:, :encoder.base_width] = frames.reshape(len(observations), -1)
    reloaded = NumericCombatPrivilegedFeatures(space, history_frames, 4, 8, 8).to('cuda')
    reloaded.load_state_dict(reference.state_dict(), strict=True)
    with torch.no_grad():
        torch.testing.assert_close(encoder(observations), reference(observations), rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(reloaded(observations), encoder(observations), rtol=0, atol=0)
