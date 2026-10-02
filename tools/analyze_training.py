"""Snapshot ongoing learner evidence and render reproducible training diagnostics."""
import csv
import hashlib
import io
import json
from pathlib import Path

import hydra
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
from omegaconf import OmegaConf

from soku_rl.rl.episode_metrics import grouped_episode_metrics


def align_update_metrics(rows, rollouts, learner, allow_unaligned):
    """Match delayed scalars to actual updates; live snapshots may miss newer timing."""
    steps_by_update = {}
    for rollout in rollouts:
        if "update_seconds" not in rollout:
            continue
        update = rollout.get("learner_n_updates", rollout.get(learner + "_n_updates"))
        if update == 0:  # DQN collection before learning_starts has no update.
            continue
        if type(update) is not int or update < 0:
            raise ValueError("completed rollout requires an integer learner update counter")
        if update in steps_by_update:
            raise ValueError(f"duplicate completed learner update counter: {update}")
        steps_by_update[update] = rollout["steps"]
    updates, unaligned = [], []
    for row in rows:
        if not row.get("train/n_updates"):
            continue
        counter = float(row["train/n_updates"])
        if not counter.is_integer():
            raise ValueError(f"scalar update {counter} has no recorded completed rollout")
        count = int(counter)
        if count not in steps_by_update:
            if not allow_unaligned:
                raise ValueError(f"scalar update {counter} has no recorded completed rollout")
            unaligned.append(count)
            continue
        updates.append({"steps": steps_by_update[count],
            **{key: float(value) for key, value in row.items() if key.startswith("train/") and value}})
    return updates, unaligned


def snapshot_run(label, source, output):
    target = output / label
    target.mkdir()
    files = {}
    for name in ("config.yaml", "timing.json", "progress.json", "scalars/progress.csv"):
        data = (source / name).read_bytes()
        (target / Path(name).name).write_bytes(data)
        files[name] = hashlib.sha256(data).hexdigest()
    config = OmegaConf.to_container(OmegaConf.load(target / "config.yaml"), resolve=True)
    timing = json.loads((target / "timing.json").read_text())
    progress = json.loads((target / "progress.json").read_text())
    rows = list(csv.DictReader(io.StringIO((target / "progress.csv").read_text())))
    learner = config["rl"].get("learner", "ppo")
    if learner not in {"ppo", "dqn"}:
        raise ValueError("unsupported learner diagnostics")
    frequency = config["rl"][learner]["train_freq" if learner == "dqn" else "n_steps"]
    if type(frequency) is not int or frequency < 1:
        raise ValueError("diagnostics require a fixed positive step collection frequency")
    rollout_size = frequency * config["num_envs"]
    # CSV may dump inside a later rollout. Align the gradient counter with
    # completed updates actually recorded by the learner callback, not epochs.
    updates, unaligned = align_update_metrics(rows, timing["rollouts"], learner, True)
    completed = [row for row in timing["rollouts"] if "update_seconds" in row]
    summary = {"source": str(source.resolve()), "sha256": files, "phase": timing["phase"],
        "sampled_steps": progress["steps"], "completed_cycles": len(completed),
        "episodes": len(progress["episodes"]), "outcomes": {}, "learner_results": {},
        "rollout_size": rollout_size, "seed": config["seed"], "learner": learner,
        "unaligned_train_counters": unaligned, "snapshot_atomic": False}
    summary["episode_metrics"] = grouped_episode_metrics(progress["episodes"])
    for episode in progress["episodes"]:
        name = episode["outcome"]
        summary["outcomes"][name] = summary["outcomes"].get(name, 0) + 1
        player = episode["training_context"]["player"]
        relative = ("win" if name == f"p{player + 1}_win" else "loss") if name in {
            "p1_win", "p2_win"} else name
        summary["learner_results"][relative] = summary["learner_results"].get(relative, 0) + 1
    if completed:
        rollout_seconds = sum(row["rollout_seconds"] for row in completed)
        update_seconds = sum(row["update_seconds"] for row in completed)
        summary.update(rollout_seconds=rollout_seconds, update_seconds=update_seconds,
            completed_cycle_steps_per_second=len(completed) * rollout_size / (rollout_seconds + update_seconds))
    return rows, updates, timing["rollouts"], summary


def plot_combat(labels, output, window, palette, styles):
    if type(window) is not int or window < 1:
        raise ValueError("combat_window must be a positive episode count")
    metrics = (
        ("own_hp_loss", "Learner HP decreases per episode"),
        ("opponent_hp_loss", "Opponent HP decreases per episode"),
        ("own_hp_loss_frames", "Learner frames with HP decrease"),
        ("opponent_hp_loss_frames", "Opponent frames with HP decrease"),
        ("own_spell_action_entries", "Learner spell action entries (proxy)"),
        ("opponent_spell_action_entries", "Opponent spell action entries (proxy)"))
    figure, axes = plt.subplots(3, 2, figsize=(12, 10), layout="constrained")
    series = {}
    for index, label in enumerate(labels):
        records = json.loads((output / label / "progress.json").read_text())["episodes"]
        series[label] = {}
        for axis, (metric, title) in zip(axes.flat, metrics, strict=True):
            points = []
            for end in range(1, len(records) + 1):
                measured = [record["combat_metrics"][metric] for record in records[max(0, end-window):end]
                    if record.get("combat_metrics", {}).get("available") is True and metric in record["combat_metrics"]]
                point = {"episode": end, "window_episodes": min(end, window), "measured_episodes": len(measured)}
                if measured:
                    point["mean"] = sum(measured) / len(measured)
                points.append(point)
            series[label][metric] = points
            if any(point["measured_episodes"] for point in points):
                axis.plot([point["episode"] for point in points],
                    [point["mean"] if point["measured_episodes"] else np.nan for point in points],
                    color=palette[index], linestyle=styles[index], label=label, marker="o", markersize=3)
            axis.set_title(title, fontsize=10)
            axis.set_xlabel("Completed training episode (not environment steps)")
            axis.grid(alpha=.2)
            axis.spines[["top", "right"]].set_visible(False)
    for axis in axes.flat:
        axis.set_ylim(bottom=0)
        if not axis.lines:
            axis.text(.5, .5, "No recorded measurements", ha="center", transform=axis.transAxes)
    if axes[0, 0].lines:
        axes[0, 0].legend(fontsize=9)
    figure.suptitle(f"Training combat metrics — mean over last {window} completed episodes\n"
        "Missing measurements excluded; HP loss is not attributed attack damage; entries are not confirmed casts\n"
        "Adaptive runs change opponent noise; these are not full god-AI evaluations",
        fontsize=12)
    figure.savefig(output / "combat.png", dpi=150)
    figure.savefig(output / "combat.pdf")
    plt.close(figure)
    (output / "combat_series.json").write_text(json.dumps({"window": window, "runs": series}, indent=2))


def plot_curriculum(labels, output):
    """Keep feedback and applied episode difficulty separate, on the same step axis."""
    series = {}
    for label in labels:
        progress = json.loads((output / label / "progress.json").read_text())
        state = progress.get("curriculum", {"kind": "fixed"})
        if state["kind"] not in {"adaptive_action_noise", "adaptive_episode_mixture"}:
            continue
        episodic = state["kind"] == "adaptive_episode_mixture"
        config = OmegaConf.to_container(OmegaConf.load(output / label / "config.yaml"), resolve=True)
        timing = json.loads((output / label / "timing.json").read_text())
        learner = config["rl"].get("learner", "ppo")
        frequency = config["rl"][learner]["train_freq" if learner == "dqn" else "n_steps"]
        start = timing["rollouts"][0]["steps"] - frequency * config["num_envs"]
        end = progress["steps"]
        points = {row["name"]: [] for row in state["opponents"]}
        for episode in progress["episodes"]:
            event = episode["curriculum_event"]
            points[event["opponent"]].append({"end_steps": episode["end_steps"],
                "learner_seat": episode["training_context"]["player"], **event})
        series[label] = {"start_steps": start, "end_steps": end, "state": state, "opponents": points}
        names = list(points)
        for offset in range(0, len(names), 4):
            selected = names[offset:offset + 4]
            figure, axes = plt.subplots(len(selected), 2, figsize=(12, 3.2 * len(selected) + 1),
                squeeze=False, layout="constrained")
            for row, name in enumerate(selected):
                probability, performance = axes[row]
                events = points[name]
                x = [event["end_steps"] / 1000 for event in events]
                if events:
                    probability.step([start / 1000, *x, end / 1000],
                        [events[0]["previous_random_probability"],
                         *[event["next_random_probability"] for event in events],
                         events[-1]["next_random_probability"]], where="post",
                        color="#2563a6", label="Probability for future games")
                    probability.scatter(x, [event["episode_random_probability"] for event in events],
                        color="#b57427", marker="x", s=28, label="Probability used in finished game", zorder=3)
                    if episodic:
                        probability.scatter(x, [int(event["selected_policy"] == "uniform") for event in events],
                            color="#444444", marker="|", s=40,
                            label="Chosen policy: uniform=1, original=0", zorder=3)
                    performance.plot(x, [event["ema_win_rate"] for event in events],
                        color="#2563a6", marker="o", markersize=3)
                else:
                    probability.plot([start / 1000, end / 1000],
                        [state["states"][name]["random_probability"]] * 2,
                        color="#2563a6", label="Probability for future games")
                    performance.text(.5, .72, "No completed episodes in this run", ha="center",
                        transform=performance.transAxes)
                target, band = state["config"]["target_win_rate"], state["config"]["deadband"]
                performance.axhspan(target - band, target + band, color="#777777", alpha=.12)
                performance.axhline(target, color="#444444", linestyle=":", linewidth=1)
                probability.set_title(f"{name} — uniform probability", fontsize=10)
                performance.set_title(f"EMA win rate — {len(events)} completed games in this run", fontsize=10)
                probability.legend(fontsize=8, loc="lower left")
                for axis in (probability, performance):
                    margin = (end - start) / 1000 * .015
                    axis.set_xlim(start / 1000 - margin, end / 1000 + margin)
                    axis.set_ylim(-.04, 1.04)
                    axis.yaxis.set_major_formatter(PercentFormatter(xmax=1))
                    axis.set_xlabel("Environment steps (thousands)")
                    axis.grid(alpha=.2)
                    axis.spines[["top", "right"]].set_visible(False)
            figure.suptitle(f"{label}: adaptive training curriculum through {end:,} steps\n"
                + ("Per-game strategy selection; " if episodic else "Per-decision action replacement; ") +
                f"EMA half-life {state['config']['ema_half_life']:g} games; shaded band is controller deadband\n"
                "Episode points are placed at completion; training win rate is not full god-AI evaluation",
                fontsize=11)
            stem = f"curriculum-{label}-{offset // 4 + 1}"
            figure.savefig(output / f"{stem}.png", dpi=150)
            figure.savefig(output / f"{stem}.pdf")
            plt.close(figure)
    if series:
        (output / "curriculum_series.json").write_text(json.dumps(series, indent=2))


@hydra.main(version_base="1.3", config_path="../config", config_name="analyze_training")
def main(cfg):
    output = Path(cfg.output)
    output.mkdir(exist_ok=True, parents=True)
    if (output / "summary.json").exists():
        raise ValueError("use a fresh diagnostic output directory")
    palette = ("#b57427", "#2563a6", "#b44579", "#637938")
    styles = ("-", "--", "-.", ":")
    figure, axes = plt.subplots(3, 3, figsize=(15, 11), layout="constrained")
    learner_types = {OmegaConf.load(Path(path) / "config.yaml").rl.get("learner", "ppo")
                     for path in cfg.runs.values()}
    if len(learner_types) != 1:
        raise ValueError("plot PPO and DQN optimizer diagnostics separately; their training metrics differ")
    dqn = learner_types == {"dqn"}
    if not 1 <= len(cfg.runs) <= len(palette):
        raise ValueError("diagnostic figures require one to four runs")
    plots = (
        ("rollout/ep_rew_mean", "Training return (rolling episode mean)"),
        ("rollout/ep_len_mean", "Episode length (decisions, rolling mean)"),
        (("rollout/exploration_rate", "Exploration probability (epsilon)") if dqn else
         ("train/entropy_loss", "Policy entropy (nats)")),
        (("train/loss", "Huber loss") if dqn else ("train/approx_kl", "Approximate KL per update")),
        (("train/td_error", "Mean absolute TD error") if dqn else ("train/clip_fraction", "PPO clipped fraction")),
        (("train/q_mean", "Mean sampled Q value") if dqn else
         ("train/explained_variance", "Value explained variance (symlog)")),
        ("rollout_seconds", "Sampling seconds per rollout"),
        ("update_seconds", "Optimization seconds per rollout"),
        ("throughput", "Completed cycle throughput (steps/s)"))
    summaries, update_series = {}, {}
    return_limits = [-1., 1.]
    for index, (label, path) in enumerate(cfg.runs.items()):
        rows, updates, timing, summary = snapshot_run(label, Path(path), output)
        summaries[label] = summary
        update_series[label] = updates
        for axis, (key, title) in zip(axes.flat, plots, strict=True):
            data = updates if key.startswith("train/") else rows if key.startswith("rollout/") else timing
            pairs = []
            for row in data:
                if key == "throughput" and "update_seconds" in row:
                    value = summary["rollout_size"] / (row["rollout_seconds"] + row["update_seconds"])
                elif key in row and row[key] != "":
                    value = float(row[key]) * (-1 if key == "train/entropy_loss" else 1)
                else:
                    continue
                step = float(row["time/total_timesteps"] if key.startswith("rollout/") else row["steps"])
                if np.isfinite(value):
                    pairs.append((step, value))
                    if key == "rollout/ep_rew_mean":
                        return_limits.append(value)
            if pairs:
                x, y = zip(*pairs)
                axis.plot(np.asarray(x) / 1000, y, label=label, color=palette[index],
                          linestyle=styles[index], marker="o", markersize=3)
            axis.set_title(title, fontsize=10)
            axis.set_xlabel("Environment steps (thousands)")
            axis.grid(alpha=.2)
            axis.spines[["top", "right"]].set_visible(False)
    if not dqn:
        axes[0, 2].axhline(np.log(576), color="#444444", linewidth=1, linestyle=":")
        axes[0, 2].text(.02, .88, "Dotted line: uniform over 576 actions", transform=axes[0, 2].transAxes, fontsize=8)
    else:
        axes[0, 2].set_ylim(0, 1)
    if not dqn:
        axes[1, 2].set_yscale("symlog", linthresh=1)
        axes[1, 1].yaxis.set_major_formatter(PercentFormatter(xmax=1))
    # Do not magnify floating-point noise around an all-loss return of -1.
    axes[0, 0].set_ylim(min(return_limits) - .05, max(return_limits) + .05)
    for axis in (axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1], *axes[2]):
        axis.set_ylim(bottom=0)
    figure.suptitle("Superhuman BR snapshots — one seed per run\n"
                     "Step counter includes checkpoint continuation; weight-only pretraining is excluded", fontsize=14)
    # A newly started run may have timing but no episode/optimizer CSV yet.
    # Select the panel covering most runs so every visible color is identified.
    legend_axis = max(axes.flat, key=lambda axis: len(axis.get_legend_handles_labels()[1]))
    if legend_axis.get_legend_handles_labels()[0]:
        legend_axis.legend(fontsize=9, title="Runs")
    maximum_steps = max(summary["sampled_steps"] for summary in summaries.values())
    for axis in axes.flat:
        axis.set_xlim(0, max(maximum_steps / 1000, .001))
        if not axis.lines:
            axis.text(.5, .5, "No recorded measurements", ha="center", transform=axis.transAxes)
    (output / "training_updates.json").write_text(json.dumps({
        label: snapshot for label, snapshot in update_series.items()}, indent=2))
    figure.savefig(output / "curves.png", dpi=150)
    figure.savefig(output / "curves.pdf")
    plt.close(figure)
    plot_combat(list(cfg.runs), output, cfg.combat_window, palette, styles)
    plot_curriculum(list(cfg.runs), output)
    (output / "summary.json").write_text(json.dumps(summaries, indent=2))
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
