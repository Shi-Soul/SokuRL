"""A KO is not a match result; rematches get fresh policy state."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from soku_rl.play.match import MatchLifecycle, MatchState


def frame(match, round_id, scores, hp, scene, updates):
    phase = "battle" if scene in (13, 14) else "menu" if scene in (8, 9) else "disconnected"
    return MatchState(match, round_id, updates, scores, hp, phase)


def kinds(events):
    return [event.kind for event in events]


def test_three_rounds_then_rematch():
    match = MatchLifecycle(2)
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
    match = MatchLifecycle(2)
    match.update(frame(1, 0, (0, 0), (10000, 10000), 14, 1))
    assert kinds(match.update(frame(1, 0, (0, 0), (10000, 10000), 2, 2))) == ["match_interrupted"]
    assert match.phase == "disconnected"
    assert not match.update(frame(1, 0, (0, 0), (10000, 10000), 2, 2))


def test_late_reader_can_observe_final_score_in_menu():
    match = MatchLifecycle(2)
    match.update(frame(1, 0, (1, 0), (10000, 10000), 13, 1))
    assert kinds(match.update(frame(1, 1, (2, 0), (10000, 0), 8, 200))) == ["score_changed", "match_finished"]


def test_reject_out_of_order_frames():
    match = MatchLifecycle(2)
    match.update(frame(1, 0, (0, 0), (10000, 10000), 13, 100))
    with pytest.raises(ValueError, match="counter moved backwards"):
        match.update(frame(1, 0, (0, 0), (10000, 10000), 13, 99))
    with pytest.raises(ValueError, match="positive integer"):
        MatchLifecycle(0)


def test_local_battle_starts_at_zero_and_keeps_original_round_scores():
    match = MatchLifecycle(2)
    initial = MatchState(1, 0, 0, (0, 0), (10000, 10000), "battle")
    assert kinds(match.update(initial)) == ["match_started", "round_started"]
    assert not match.update(MatchState(1, 0, 1, (0, 0), (0, 0), "battle"))
    assert match.phase == "between_rounds"
    events = match.update(MatchState(1, 1, 2, (1, 1), (10000, 10000), "battle"))
    assert kinds(events) == ["score_changed", "round_started"]
    events = match.update(MatchState(1, 1, 3, (2, 1), (10000, 0), "battle"))
    assert kinds(events) == ["score_changed", "match_finished"]
    assert not match.can_act
