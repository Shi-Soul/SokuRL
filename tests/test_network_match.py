"""A KO is not a match result; rematches get fresh policy state."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from network_match import NetworkMatch


def frame(match, round_id, scores, hp, scene, updates):
    return SimpleNamespace(match=match, scores=scores, scene=scene, connected=True,
        updates=updates, in_battle=scene in (13, 14) and updates > 0,
        raw=SimpleNamespace(roundId=round_id, p1=SimpleNamespace(hp=hp[0]),
                            p2=SimpleNamespace(hp=hp[1])))


def kinds(events):
    return [event.kind for event in events]


def test_three_rounds_then_rematch():
    match = NetworkMatch(2)
    assert kinds(match.update(frame(1, 0, (0, 0), (10000, 10000), 13, 1))) == ["match_started", "round_started"]
    assert match.can_act
    assert not match.update(frame(1, 0, (0, 0), (10000, 0), 13, 100))
    assert match.phase == "between_rounds" and not match.can_act
    assert kinds(match.update(frame(1, 0, (1, 0), (10000, 0), 13, 101))) == ["score_changed"]
    assert kinds(match.update(frame(1, 1, (1, 0), (10000, 10000), 13, 200))) == ["round_started"]
    assert kinds(match.update(frame(1, 1, (1, 1), (0, 10000), 13, 300))) == ["score_changed"]
    assert kinds(match.update(frame(1, 2, (1, 1), (10000, 10000), 13, 400))) == ["round_started"]
    final = frame(1, 2, (2, 1), (10000, 0), 13, 500)
    assert kinds(match.update(final)) == ["score_changed", "match_finished"]
    assert match.phase == "match_finished" and not match.can_act
    assert not match.update(final)
    assert not match.update(frame(1, 2, (2, 1), (10000, 0), 8, 500))
    assert match.phase == "menu"
    assert kinds(match.update(frame(2, 0, (0, 0), (10000, 10000), 13, 1))) == ["match_started", "round_started"]


def test_interruption_is_not_a_match_loss():
    match = NetworkMatch(2)
    match.update(frame(1, 0, (0, 0), (10000, 10000), 14, 1))
    assert kinds(match.update(frame(1, 0, (0, 0), (10000, 10000), 2, 2))) == ["match_interrupted"]
    assert match.phase == "disconnected"
    assert not match.update(frame(1, 0, (0, 0), (10000, 10000), 2, 2))


def test_late_reader_can_observe_final_score_in_menu():
    match = NetworkMatch(2)
    match.update(frame(1, 0, (1, 0), (10000, 10000), 13, 1))
    assert kinds(match.update(frame(1, 1, (2, 0), (10000, 0), 8, 200))) == ["score_changed", "match_finished"]


def test_reject_out_of_order_frames():
    match = NetworkMatch(2)
    match.update(frame(1, 0, (0, 0), (10000, 10000), 13, 100))
    with pytest.raises(ValueError, match="counter moved backwards"):
        match.update(frame(1, 0, (0, 0), (10000, 10000), 13, 99))
    with pytest.raises(ValueError, match="positive integer"):
        NetworkMatch(0)
