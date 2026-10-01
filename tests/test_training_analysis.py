from pathlib import Path
import runpy

import pytest


analysis = runpy.run_path(str(Path(__file__).parents[1] / "tools/analyze_training.py"))


def test_delayed_scalars_follow_actual_resumed_and_early_stopped_updates():
    rollouts = [
        {"steps": 139264, "ppo_n_updates": 50, "update_seconds": 2},
        {"steps": 147456, "ppo_n_updates": 51, "update_seconds": 1},
        {"steps": 155648, "ppo_n_updates": 54, "update_seconds": 2},
        {"steps": 163840, "rollout_seconds": 90},
    ]
    rows = [{"time/total_timesteps": "139264", "train/n_updates": ""}]
    for counter, dump_step, loss in [(50, 147456, -.4), (51, 155648, -.5), (54, 163840, -.6)]:
        rows.append({"time/total_timesteps": str(dump_step), "train/n_updates": str(counter),
            "train/entropy_loss": str(loss), "train/unused": ""})
    updates = analysis["align_update_metrics"](rows, rollouts)
    assert [row["steps"] for row in updates] == [139264, 147456, 155648]
    assert [row["train/entropy_loss"] for row in updates] == [-.4, -.5, -.6]
    assert all("train/unused" not in row and "time/total_timesteps" not in row for row in updates)


@pytest.mark.parametrize("counter", ["49", "50.5", "nan"])
def test_unmatched_or_invalid_counters_do_not_invent_training_steps(counter):
    with pytest.raises(ValueError, match="no recorded completed rollout"):
        analysis["align_update_metrics"]([{"train/n_updates": counter}], [
            {"steps": 139264, "ppo_n_updates": 50, "update_seconds": 2}])


def test_ambiguous_update_counter_is_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        analysis["align_update_metrics"]([], [
            {"steps": step, "ppo_n_updates": 50, "update_seconds": 2}
            for step in [139264, 147456]])


def test_update_without_next_scalar_dump_is_not_fabricated():
    assert analysis["align_update_metrics"]([{"train/n_updates": ""}], [
        {"steps": 8192, "ppo_n_updates": 10, "update_seconds": 2}]) == []
