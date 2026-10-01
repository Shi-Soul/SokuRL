from pathlib import Path
import runpy

import numpy as np
import pytest


analysis = runpy.run_path(str(Path(__file__).parents[1] / "tools/analyze_actions.py"))


def test_command_runs_do_not_join_across_episode_boundaries():
    first = analysis["command_counts"](np.array([256, 256, 259, 259, 512]))
    second = analysis["command_counts"](np.array([256]))
    result = analysis["summarize_commands"]([first, second])
    assert result["decisions"] == 6
    assert result["transitions"] == 4
    assert result["command_changes"] == 2
    assert result["direction_changes"] == 1
    assert result["neutral_commands"] == 3
    assert result["command_repeat_fraction"] == .5
    assert result["direction_change_fraction"] == .25
    assert result["mean_command_run_decisions"] == 1.5
    assert result["mean_direction_run_decisions"] == 2
    assert result["multiple_attack_button_fraction"] == pytest.approx(1 / 3)
    assert result["button_down_fractions"] == {
        "a": pytest.approx(1 / 3), "b": pytest.approx(1 / 3), "c": 0, "d": 0,
        "change_card": 0, "spellcard": 0}
    assert sum(result["command_histogram"]) == 6


def test_single_decision_has_no_invented_transition_rate():
    result = analysis["summarize_commands"]([analysis["command_counts"](np.array([575]))])
    assert "command_repeat_fraction" not in result
    assert result["mean_command_run_decisions"] == 1
    assert all(value == 1 for value in result["button_down_fractions"].values())
    for values in (np.array([]), np.array([576]), np.array([-1]), np.array([1.5]), np.array([[0]])):
        with pytest.raises(ValueError):
            analysis["command_counts"](values)
