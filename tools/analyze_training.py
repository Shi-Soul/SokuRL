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
    for axis in (axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1], *axes[2]):
        axis.set_ylim(bottom=0)
    figure.suptitle("Superhuman BR diagnostics — ongoing, one seed per configuration\n"
                     "Updates aligned to trained steps; mixed uses a different opponent population", fontsize=14)
    axes[0, 0].legend(fontsize=9)
    figure.savefig(output / "curves.png", dpi=150)
    figure.savefig(output / "curves.pdf")
    plt.close(figure)
    (output / "summary.json").write_text(json.dumps(summaries, indent=2))
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
