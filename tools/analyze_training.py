"""Snapshot ongoing PPO evidence and render reproducible training diagnostics."""
import csv
import hashlib
import io
import json
from pathlib import Path

import hydra
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

from soku_rl.rl.episode_metrics import grouped_episode_metrics


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
    rollout_size = config["rl"]["ppo"]["n_steps"] * config["num_envs"]
    epochs = config["rl"]["ppo"]["n_epochs"]
    updates = []
    for row in rows:
        if row.get("train/n_updates"):
            # SB3 dumps the previous update at the end of the NEXT rollout.
            # These runs have fixed n_epochs and no target_kl early stopping.
            if config["rl"]["ppo"].get("target_kl") is not None:
                raise ValueError("update-step inference requires fixed epochs without target_kl")
            updates.append({"steps": float(row["train/n_updates"]) / epochs * rollout_size,
                **{key: float(value) for key, value in row.items() if key.startswith("train/") and value}})
    completed = [row for row in timing["rollouts"] if "update_seconds" in row]
    summary = {"source": str(source.resolve()), "sha256": files, "phase": timing["phase"],
        "sampled_steps": progress["steps"], "finished_updates": len(completed),
        "episodes": len(progress["episodes"]), "outcomes": {}, "learner_results": {},
        "rollout_size": rollout_size, "seed": config["seed"]}
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
        "Missing measurements excluded; HP loss is not attributed attack damage; entries are not confirmed casts",
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
        if state["kind"] != "adaptive_action_noise":
            continue
        config = OmegaConf.to_container(OmegaConf.load(output / label / "config.yaml"), resolve=True)
        timing = json.loads((output / label / "timing.json").read_text())
        start = timing["rollouts"][0]["steps"] - config["rl"]["ppo"]["n_steps"] * config["num_envs"]
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
                    axis.set_xlabel("PPO environment steps (thousands)")
                    axis.grid(alpha=.2)
                    axis.spines[["top", "right"]].set_visible(False)
            figure.suptitle(f"{label}: adaptive training curriculum through {end:,} steps\n"
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
    plots = (
        ("rollout/ep_rew_mean", "Training return (rolling episode mean)"),
        ("rollout/ep_len_mean", "Episode length (decisions, rolling mean)"),
        ("train/entropy_loss", "Policy entropy (nats)"),
        ("train/approx_kl", "Approximate KL per update"),
        ("train/clip_fraction", "PPO clipped fraction"),
        ("train/explained_variance", "Value explained variance (symlog)"),
        ("rollout_seconds", "Sampling seconds per rollout"),
        ("update_seconds", "Optimization seconds per rollout"),
        ("throughput", "Completed cycle throughput (steps/s)"))
    summaries = {}
    return_limits = [-1., 1.]
    for index, (label, path) in enumerate(cfg.runs.items()):
        rows, updates, timing, summary = snapshot_run(label, Path(path), output)
        summaries[label] = summary
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
    axes[0, 2].axhline(np.log(576), color="#444444", linewidth=1, linestyle=":")
    axes[0, 2].text(.02, .88, "Dotted line: uniform over 576 actions", transform=axes[0, 2].transAxes, fontsize=8)
    axes[1, 2].set_yscale("symlog", linthresh=1)
    # Do not magnify floating-point noise around an all-loss return of -1.
    axes[0, 0].set_ylim(min(return_limits) - .05, max(return_limits) + .05)
    for axis in (axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1], *axes[2]):
        axis.set_ylim(bottom=0)
    figure.suptitle("Superhuman BR snapshots — one seed per run\n"
                     "PPO step counter includes checkpoint continuation; weight-only pretraining is excluded", fontsize=14)
    axes[0, 0].legend(fontsize=9)
    figure.savefig(output / "curves.png", dpi=150)
    figure.savefig(output / "curves.pdf")
    plt.close(figure)
    plot_combat(list(cfg.runs), output, cfg.combat_window, palette, styles)
    plot_curriculum(list(cfg.runs), output)
    (output / "summary.json").write_text(json.dumps(summaries, indent=2))
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
