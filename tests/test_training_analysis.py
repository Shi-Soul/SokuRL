"""Training diagnostics must align each learner to observed update counters."""
import json
from pathlib import Path
import sys

from omegaconf import OmegaConf
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
from analyze_training import snapshot_run


@pytest.mark.parametrize("learner,frequency,steps,count", [("ppo", 128, 1024, 10), ("dqn", 32, 4352, 32)])
def test_update_axis_uses_recorded_counter_not_ppo_epoch_formula(tmp_path, learner, frequency, steps, count):
    source, output = tmp_path / "run", tmp_path / "snapshot"
    (source / "scalars").mkdir(parents=True)
    output.mkdir()
    parameters = {"train_freq" if learner == "dqn" else "n_steps": frequency}
    # Both configurations may contain legacy/default PPO parameters.
    rl = {"learner": learner, "ppo": {"n_steps": 2048, "n_epochs": 10}, learner: parameters}
    OmegaConf.save(OmegaConf.create({"rl": rl, "num_envs": 8, "seed": 7}), source / "config.yaml")
    (source / "timing.json").write_text(json.dumps({"phase": "sampling", "rollouts": [
        {"steps": steps, "rollout_seconds": 2, "update_seconds": 1, learner + "_n_updates": count}]}))
    (source / "progress.json").write_text(json.dumps({"steps": steps, "episodes": []}))
    (source / "scalars/progress.csv").write_text(
        f"time/total_timesteps,train/n_updates,train/loss\n{steps + 128},{count},0.5\n{steps + 256},{count * 2},0.4\n")
    _, updates, _, summary = snapshot_run("one", source, output)
    assert updates == [{"steps": steps, "train/n_updates": count, "train/loss": .5}]
    assert summary["unaligned_train_counters"] == [count * 2]
    assert summary["completed_cycle_steps_per_second"] == frequency * 8 / 3
    assert summary["episodes"] == 0 and summary["snapshot_atomic"] is False
    assert (output / "one/progress.csv").read_bytes() == (source / "scalars/progress.csv").read_bytes()
