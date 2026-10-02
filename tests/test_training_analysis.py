"""Training diagnostics must align each learner to observed update counters."""
import json
from pathlib import Path
import runpy
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
    updates, unaligned = analysis["align_update_metrics"](rows, rollouts, "ppo", False)
    assert unaligned == []
    assert [row["steps"] for row in updates] == [139264, 147456, 155648]
    assert [row["train/entropy_loss"] for row in updates] == [-.4, -.5, -.6]
    assert all("train/unused" not in row and "time/total_timesteps" not in row for row in updates)


@pytest.mark.parametrize("counter", ["49", "50.5", "nan"])
def test_unmatched_or_invalid_counters_do_not_invent_training_steps(counter):
    with pytest.raises(ValueError, match="no recorded completed rollout"):
        analysis["align_update_metrics"]([{"train/n_updates": counter}], [
            {"steps": 139264, "ppo_n_updates": 50, "update_seconds": 2}], "ppo", False)


def test_ambiguous_update_counter_is_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        analysis["align_update_metrics"]([], [
            {"steps": step, "ppo_n_updates": 50, "update_seconds": 2}
            for step in [139264, 147456]], "ppo", False)


def test_update_without_next_scalar_dump_is_not_fabricated():
    assert analysis["align_update_metrics"]([{"train/n_updates": ""}], [
        {"steps": 8192, "ppo_n_updates": 10, "update_seconds": 2}], "ppo", False) == ([], [])


@pytest.mark.parametrize("learner", ["ppo", "dqn"])
def test_episode_mixture_plot_distinguishes_probability_from_actual_policy(tmp_path, monkeypatch, learner):
    import json
    from omegaconf import OmegaConf
    folder = tmp_path / 'episodic'
    folder.mkdir()
    OmegaConf.save(OmegaConf.create({'rl': {'learner': learner, learner: {'n_steps' if learner == 'ppo' else 'train_freq': 64}}, 'num_envs': 2}), folder / 'config.yaml')
    (folder / 'timing.json').write_text(json.dumps({'rollouts': [{'steps': 128}]}))
    state = {'kind': 'adaptive_episode_mixture', 'opponents': [{'name': 'god'}],
        'states': {'god': {'random_probability': .45}},
        'config': {'ema_half_life': 50, 'target_win_rate': .5, 'deadband': .05}}
    records = []
    for index, selected in enumerate(('uniform', 'original')):
        records.append({'end_steps': (index + 1) * 128, 'training_context': {'player': index},
            'curriculum_event': {'opponent': 'god', 'previous_random_probability': .5 if index == 0 else .45,
                'next_random_probability': .45, 'episode_random_probability': .5 if index == 0 else .45,
                'ema_win_rate': 1 if index == 0 else .49, 'selected_policy': selected}})
    (folder / 'progress.json').write_text(json.dumps({'steps': 256, 'episodes': records, 'curriculum': state}))
    saved_draws = []
    save = analysis['plt'].Figure.savefig

    def inspect(figure, *args, **kwargs):
        for collection in figure.axes[0].collections:
            if collection.get_label().startswith('Chosen policy:'):
                saved_draws.append(collection.get_offsets()[:, 1].tolist())
        return save(figure, *args, **kwargs)

    monkeypatch.setattr(analysis['plt'].Figure, 'savefig', inspect)
    analysis['plot_curriculum'](['episodic'], tmp_path)
    assert saved_draws == [[1., 0.], [1., 0.]]
    assert (tmp_path / 'curriculum-episodic-1.png').is_file()
    rows = json.loads((tmp_path / 'curriculum_series.json').read_text())['episodic']['opponents']['god']
    assert json.loads((tmp_path / 'curriculum_series.json').read_text())['episodic']['start_steps'] == 0
    assert [r['selected_policy'] for r in rows] == ['uniform', 'original']
    assert [r['episode_random_probability'] for r in rows] == [.5, .45]
