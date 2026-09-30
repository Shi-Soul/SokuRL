"""Single-seat commands preserve ownership and reject invalid inputs before publication."""
import ctypes

import pytest

from bridge_shared import BridgeClient, BridgeMapping, COMMAND_STEP_WITH_CONTROLLED_INPUTS, LogicalInput


@pytest.fixture
def client():
    value = BridgeClient.__new__(BridgeClient)
    value._mapping_pointer = ctypes.pointer(BridgeMapping())
    return value


@pytest.mark.parametrize("seats", ((0,), (1,), (0, 1)))
def test_step_identifies_exact_controlled_seats(client, seats):
    keys = (-1, 1, 1, 0, 0, 1, 0, 0)
    sequence = client.step_controlled({seat: keys for seat in seats})
    block = client.block
    assert sequence == block.commandSeq == 1
    assert block.commandType == COMMAND_STEP_WITH_CONTROLLED_INPUTS
    assert block.durationFrames == 1
    assert block.commandArgument == sum(1 << seat for seat in seats)
    for seat, values in enumerate((block.commandInput, block.commandInputP2)):
        assert tuple(getattr(values, name) for name, _ in LogicalInput._fields_) == (keys if seat in seats else (0,) * 8)


@pytest.mark.parametrize("inputs", ({}, {True: (0,) * 8}, {2: (0,) * 8},
    {0: (0,) * 7}, {1: (2,) * 8}, {0: (0,) * 8, 1: (0,) * 7 + (True,)}))
def test_invalid_step_does_not_publish_or_partially_change_request(client, inputs):
    before = bytes(client.block)
    with pytest.raises(ValueError):
        client.step_controlled(inputs)
    assert bytes(client.block) == before


def test_joint_environment_step_uses_the_same_command(client):
    client.step_with_inputs(LogicalInput(), LogicalInput())
    assert client.block.commandType == COMMAND_STEP_WITH_CONTROLLED_INPUTS
    assert client.block.commandArgument == 3
