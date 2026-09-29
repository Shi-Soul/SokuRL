"""Reproduce the signed spirit value recorded during a real guard break."""
from types import SimpleNamespace
from dataclasses import replace

import pytest

from soku_rl.render_state import RenderEntity, RenderSnapshot
from soku_rl.visible_state import observe_visible_states
from soku_rl.observations import observe
from soku_rl.gauges import spirit_fraction
from test_env_timing import VISIBILITY


def recorded_frame(spirit):
    fighters = [SimpleNamespace(characterId=1, hp=2201, spirit=876, maxSpirit=1000),
                SimpleNamespace(characterId=0, hp=4783, spirit=spirit, maxSpirit=1000)]
    for index, fighter in enumerate(fighters):
        fighter.x, fighter.y = 400. + index * 100., 0.
        fighter.actionId, fighter.airborne, fighter.hitstop = 0, False, 0
        fighter.facing = 1 if index == 0 else -1
    raw = SimpleNamespace(frameId=3185, p1=fighters[0], p2=fighters[1],
                          p1ObjectOverflow=False, p2ObjectOverflow=False,
                          p1Objects=(), p2Objects=(), p1ObjectCount=0, p2ObjectCount=0)
    render = RenderSnapshot(-700., 470., 1., 21,
        (RenderEntity(1201., 312.70535, 1., 1, 1),
         RenderEntity(1240., 370.81131, 1., -1, 1)), ((), ()), False)
    return raw, render


def test_guard_break_negative_spirit_is_an_empty_visible_gauge():
    raw, render = recorded_frame(-88)  # Old unsigned ABI exported this as 65448.
    first, second = observe_visible_states(raw, render, VISIBILITY)
    assert first.frame == second.frame == 3185
    assert first.values[14] == second.values[6] == 0.
    assert first.values[6] == second.values[14] == .9


@pytest.mark.parametrize("spirit", [0, -88])
def test_all_broken_spirit_orbs_have_zero_available_fraction(spirit):
    # The engine's orb-break routine at 0x463A00 subtracts 200 from the
    # maximum, permits zero, and assigns the new maximum to current spirit.
    raw, render = recorded_frame(spirit)
    raw.p2.maxSpirit = 0
    assert spirit_fraction(raw.p2.spirit, raw.p2.maxSpirit) == 0.
    first, second = observe_visible_states(raw, render, VISIBILITY)
    assert first.values[14] == second.values[6] == 0.
    assert observe(raw, 0).opponent.spirit_fraction == observe(raw, 1).player.spirit_fraction == 0.
    assert raw.p2.spirit == spirit and raw.p2.maxSpirit == 0


@pytest.mark.parametrize("value,maximum", [(1, 0), (0, -1), (0, float("nan"))])
def test_invalid_capacity_or_spirit_above_zero_capacity_is_rejected(value, maximum):
    with pytest.raises(ValueError, match="spirit"):
        spirit_fraction(value, maximum)


@pytest.mark.parametrize("invalid", [-32769, 32768, 65448, 1001])
def test_malformed_spirit_words_and_positive_overflow_still_fail(invalid):
    raw, render = recorded_frame(invalid)
    with pytest.raises(ValueError, match="spirit|gauge"):
        observe_visible_states(raw, render, VISIBILITY)


def test_guard_break_diagnostic_observation_keeps_rules_usable():
    from soku_rl.env.encoding import encode_observation
    from soku_rl.observed_rules import decode_diagnostic
    from test_tactical_rules import actor

    raw, _ = recorded_frame(-88)
    for player in (0, 1):
        observation = observe(raw, player)
        decoded = decode_diagnostic(encode_observation(observation, 7200), 7200)
        assert (decoded.player, decoded.opponent)[1 - player].spirit_fraction == 0.
        assert (decoded.player, decoded.opponent)[player].spirit_fraction == pytest.approx(.876)
        # Start the rule's state machine at zero with this recorded resource value.
        assert actor("spirit_siege").act(replace(decoded, frame=0)).inputs[3] in (0, 1)
    assert raw.p2.spirit == -88
