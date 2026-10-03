from copy import deepcopy

from gymnasium import spaces
import numpy as np
import pytest
import torch

from soku_rl.env.observation.memory_schema import (
    FIGHTER_NAMES, FIGHTER_WIDTH, MAX_BOXES, MAX_OBJECTS, OBJECT_WIDTH,
    PLAYER_WIDTH, PRIVILEGED_FEATURES, WORLD_NAMES)
from soku_rl.env.observation.privileged import (
    decode_privileged, encode_privileged, encode_values)
from soku_rl.rl.canonical_features import CanonicalCombatFeatures
from test_privileged_encoding import observation


def scene(count, facing):
    original = observation(count)
    for seat, player in enumerate(original.players):
        player.update(x=360.25 + seat * 400, xspeed=2.5 - seat * 4, dir=facing * (1 - 2 * seat))
        for index, obj in enumerate(player['objects']):
            obj.update(x=500.5 + index, xspeed=-3.25 + index, dir=1-2*seat)
    return original


def reflected(original):
    result = deepcopy(original)
    for player in result.players:
        for entity in (player, *player['objects']):
            entity['x'] = 1280. - entity['x']
            entity['xspeed'] *= -1
            entity['dir'] *= -1
            for field in ('hitarea', 'attackarea'):
                entity[field] = tuple((1280.-right, top, 1280.-left, bottom)
                    for left, top, right, bottom in entity[field])
        keys = list(player['keys'])
        keys[2], keys[3] = keys[3], keys[2]
        player['keys'] = tuple(keys)
    return result


def encoder(history, action_history, device):
    torch.set_num_threads(1)
    space = spaces.Box(-np.inf, np.inf, (history * PRIVILEGED_FEATURES + action_history * 8,), np.float32)
    return CanonicalCombatFeatures(space, history, 4, 16, 8, 1280.).to(device)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('history', [1, 2])
def test_canonical_state_matches_independent_reflection_with_latest_history_frame(device, history):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    model = encoder(history, 2, device)
    states = [scene(2, -1) for _ in range(history)]
    if history == 2:
        states[0].players[0]['dir'] = 1
    commands = np.array([[1, -1, 1, 0, 0, 0, 0, 1], [-1, 0, 0, 1, 0, 1, 0, 0]], np.float32)
    values = np.concatenate([*(encode_privileged(item) for item in states), commands.ravel()])
    opposite_commands = commands.copy()
    opposite_commands[:, 0] *= -1
    opposite = np.concatenate([*(encode_privileged(reflected(item)) for item in states), opposite_commands.ravel()])
    batch = torch.from_numpy(np.stack((values, opposite))).to(device)
    untouched = batch.clone()
    canonical = model.canonical_observations(batch)
    torch.testing.assert_close(canonical[0], canonical[1], rtol=0, atol=0)
    for frame, state in enumerate(states):
        decoded = decode_privileged(canonical[0, frame*PRIVILEGED_FEATURES:(frame+1)*PRIVILEGED_FEATURES].cpu().numpy())
        assert decoded == decode_privileged(encode_privileged(reflected(state)))
    np.testing.assert_array_equal(canonical[0, -16:].cpu(), opposite_commands.ravel())
    output = model(batch)
    # CPU GEMM can differ by a few ulps across batch rows with equal inputs.
    torch.testing.assert_close(output[0], output[1], rtol=1e-6, atol=1e-7)
    assert torch.equal(batch, untouched)
    # Mirrored samples induce the same parameter update, not merely equal logits.
    gradients = []
    for row in batch:
        model.zero_grad(set_to_none=True)
        model(row[None]).sum().backward()
        gradients.append([parameter.grad.clone() for parameter in model.parameters()])
    for left, right in zip(*gradients, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_subtraction_preserves_fine_coordinate_bits_without_mutating_raw_observation():
    model = encoder(1, 0, 'cpu')
    source = observation(1)
    source.players[0]['dir'] = -1
    encoded = torch.from_numpy(encode_privileged(source))[None]
    transformed = model.canonical_observations(encoded)[0].numpy()
    decoded = decode_privileged(transformed)
    assert decoded.players[0]['x'] == 1280. - source.players[0]['x']
    assert decoded.players[0]['x'] != float(np.float32(decoded.players[0]['x']))
    assert decoded.players[0]['fflags'] == 0xFFFFFFFF
    assert decoded.players[0]['address'] == 0xABCDEF01
    assert np.array_equal(encoded[0].numpy(), encode_privileged(source))


def test_padding_absent_objects_and_inactive_boxes_remain_unchanged():
    model = encoder(2, 1, 'cpu')
    current = scene(0, -1)
    for player in current.players:
        player.update(hitarea=(), hitarea_n=0, attackarea=(), attackarea_n=0)
    values = torch.from_numpy(np.concatenate((np.zeros(PRIVILEGED_FEATURES, np.float32),
        encode_privileged(current), np.zeros(8, np.float32))))[None]
    canonical = model.canonical_observations(values)
    assert torch.count_nonzero(canonical[:, :PRIVILEGED_FEATURES]) == 0
    for seat in (0, 1):
        base = PRIVILEGED_FEATURES + 2*(len(WORLD_NAMES) + seat*PLAYER_WIDTH)
        boxes = base + 2*len(FIGHTER_NAMES)
        assert torch.count_nonzero(canonical[:, boxes:boxes+MAX_BOXES*16]) == 0
        objects = base + FIGHTER_WIDTH*2
        assert torch.count_nonzero(canonical[:, objects:objects+MAX_OBJECTS*OBJECT_WIDTH*2]) == 0
    empty = torch.zeros_like(values)
    assert torch.equal(model.canonical_observations(empty), empty)
    assert torch.isfinite(model(empty)).all()


def test_all_present_slots_retain_gradients_order_and_address_invariance():
    model = encoder(1, 1, 'cpu')
    values = torch.zeros(1, PRIVILEGED_FEATURES + 8)
    for seat in (0, 1):
        base = 2*(len(WORLD_NAMES)+seat*PLAYER_WIDTH)
        for field, value in (('dir', -1 if seat==0 else 1), ('obj_n', MAX_OBJECTS)):
            index = base + 2*FIGHTER_NAMES.index(field)
            values[0,index:index+2] = torch.from_numpy(encode_values(np.array([value]))[0])
    values.requires_grad_()
    model(values).sum().backward()
    for seat in (0, 1):
        start = 2*(len(WORLD_NAMES)+seat*PLAYER_WIDTH+FIGHTER_WIDTH)
        gradient = values.grad[0,start:start+MAX_OBJECTS*OBJECT_WIDTH*2].reshape(MAX_OBJECTS,-1)
        assert torch.all(gradient.abs().sum(1)>0)
        address = FIGHTER_NAMES.index('address')*2
        assert torch.count_nonzero(gradient[:,address:address+2]) == 0
    small = torch.from_numpy(np.r_[encode_privileged(scene(2, -1)), np.zeros(8,np.float32)])[None]
    moved = small.clone()
    for seat in (0,1):
        start = 2*(len(WORLD_NAMES)+seat*PLAYER_WIDTH)
        for offset in (0, FIGHTER_WIDTH, FIGHTER_WIDTH+OBJECT_WIDTH):
            index = start+2*(offset+FIGHTER_NAMES.index('address'))
            moved[0,index:index+2] = torch.tensor([.7,.2])
    assert torch.equal(model(small),model(moved))
    start = 2*(len(WORLD_NAMES)+FIGHTER_WIDTH)
    moved = small.clone()
    moved[:,start:start+OBJECT_WIDTH*4] = small[:,start:start+OBJECT_WIDTH*4].reshape(1,2,-1).flip(1).flatten(1)
    assert not torch.equal(model(small),model(moved))


@pytest.mark.parametrize('width', [0, -1, True, float('nan'), float('inf')])
def test_invalid_arena_width_is_rejected(width):
    with pytest.raises(ValueError, match='arena_width'):
        CanonicalCombatFeatures(spaces.Box(-np.inf,np.inf,(PRIVILEGED_FEATURES,),np.float32),1,4,16,8,width)
