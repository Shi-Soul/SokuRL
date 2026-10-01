"""Describe submitted commands from completed BR replays, without inferring hits."""
import hashlib
import json
from pathlib import Path

import hydra
import numpy as np
from omegaconf import OmegaConf


def command_counts(actions):
    if actions.ndim != 1 or not len(actions) or actions.dtype.kind not in "iu":
        raise ValueError("nonempty integer command sequence required")
    if np.any(actions < 0) or np.any(actions >= 576):
        raise ValueError("full logical commands must be in [0, 576)")
    axes = actions // 64
    buttons = (actions[:, None] % 64 >> np.arange(6)) & 1
    return {"decisions": len(actions), "transitions": len(actions) - 1,
        "command_changes": int(np.count_nonzero(np.diff(actions))),
        "direction_changes": int(np.count_nonzero(np.diff(axes))),
        "neutral_commands": int(np.count_nonzero(actions == 256)),
        "button_down_counts": buttons.sum(0).tolist(),
        "multiple_attack_button_commands": int(np.count_nonzero(buttons[:, :3].sum(1) >= 2)),
        "histogram": np.bincount(actions.astype(np.int64), minlength=576).tolist()}


def summarize_commands(records):
    if not records:
        raise ValueError("at least one episode is required")
    totals = {key: sum(record[key] for record in records) for key in (
        "decisions", "transitions", "command_changes", "direction_changes", "neutral_commands",
        "multiple_attack_button_commands")}
    buttons = np.sum([record["button_down_counts"] for record in records], axis=0)
    histogram = np.sum([record["histogram"] for record in records], axis=0)
    result = {"games": len(records), **totals,
        "button_down_fractions": dict(zip(("a", "b", "c", "d", "change_card", "spellcard"),
                                          (buttons / totals["decisions"]).tolist(), strict=True)),
        "multiple_attack_button_fraction": totals["multiple_attack_button_commands"] / totals["decisions"],
        "mean_command_run_decisions": totals["decisions"] / (len(records) + totals["command_changes"]),
        "mean_direction_run_decisions": totals["decisions"] / (len(records) + totals["direction_changes"]),
        "command_histogram": histogram.tolist()}
    if totals["transitions"]:
        result["command_repeat_fraction"] = 1 - totals["command_changes"] / totals["transitions"]
        result["direction_change_fraction"] = totals["direction_changes"] / totals["transitions"]
    return result


@hydra.main(version_base="1.3", config_path="../config", config_name="analyze_actions")
def main(cfg):
    output = Path(cfg.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists():
        raise ValueError("use a fresh action-analysis output directory")
    summaries = {}
    for label, directory in cfg.runs.items():
        source, target = Path(directory), output / label
        target.mkdir()
        hashes = {}
        for name in ("config.yaml", "result.json"):
            data = (source / name).read_bytes()
            (target / name).write_bytes(data)
            hashes[name] = hashlib.sha256(data).hexdigest()
        report = json.loads((target / "result.json").read_text())
        config = OmegaConf.to_container(OmegaConf.load(target / "config.yaml"), resolve=True)
        training = config["source_training"]
        if not report["success"] or training["wrappers"]["action_set"] != "full":
            raise ValueError("analysis requires successful full-action BR evaluation")
        records = []
        roles = {role: [] for role in ("learner", "opponent")}
        for game in report["result"]["games"]:
            name = game["replay"]
            if Path(name).name != name:
                raise ValueError("replay must be directly inside the evaluation directory")
            data = (source / name).read_bytes()
            (target / name).write_bytes(data)
            hashes[name] = hashlib.sha256(data).hexdigest()
            with np.load(target / name, allow_pickle=False) as replay:
                actions = replay["actions"]
                if actions.ndim != 2 or actions.shape[1] != 2 or int(replay["seed"]) != game["world_seed"]:
                    raise ValueError("replay dimensions or world seed disagree with result")
                counts = {role: command_counts(actions[:, seat]) for role, seat in (
                    ("learner", game["learner_seat"]), ("opponent", 1 - game["learner_seat"]))}
            records.append({key: game[key] for key in ("trial_id", "learner_seat", "world_seed", "outcome")} | counts)
            for role in roles:
                roles[role].append(counts[role])
        summaries[label] = {"source": str(source.resolve()), "sha256": hashes,
            "decision_frames": training["episode"]["decision_frames"],
            "latency_frames": training["episode"]["latency_frames"],
            "roles": {role: summarize_commands(values) for role, values in roles.items()}, "games": records}
    result = {"measurement": "submitted logical commands, not confirmed engine actions or casts",
        "weighting": "counts pooled within role across games; transitions never cross episode boundaries",
        "runs": summaries}
    (output / "summary.json").write_text(json.dumps(result, indent=2))
    for label, run in summaries.items():
        print(label, {role: {key: value for key, value in values.items() if key != "command_histogram"}
                      for role, values in run["roles"].items()})


if __name__ == "__main__":
    main()
