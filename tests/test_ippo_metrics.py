"""Every independent learner flushes its update metrics, including after resume."""
import csv
import math

import pytest
import torch

from soku_rl.marl.ippo import train_ippo
from test_shared_ppo import fixture_config, fixture_env, save_contract
from test_dqn import contract, dqn_config


@pytest.mark.parametrize("kind", ["mlp", "lstm", "dqn"])
def test_ippo_writes_each_players_update_metrics_and_global_steps(tmp_path, kind):
    torch.set_num_threads(1)
    config = dqn_config() if kind == "dqn" else fixture_config(kind)
    config.update(name="ippo", timesteps_per_player=16)
    env = fixture_env()
    try:
        for run, start in (("fresh", 0), ("resumed", 16)):
            directory = tmp_path / run
            directory.mkdir()
            report = train_ippo(env, config, "cpu", 17, directory)
            source = (contract if kind == "dqn" else save_contract)(directory, env, config)
            assert report["updates"] == 2
            for player in (0, 1):
                with (directory / f"player_{player}/scalars/progress.csv").open() as stream:
                    rows = list(csv.DictReader(stream))
                assert len(rows) == report["updates"]
                assert [int(row["time/total_timesteps"]) for row in rows] == [start + 8, start + 16]
                assert all(math.isfinite(float(row["train/loss"])) for row in rows)
                assert int(rows[1]["train/n_updates"]) > int(rows[0]["train/n_updates"]) > 0
                config["initial_policies"][f"player_{player}"] = {"kind": "checkpoint",
                    "path": str(directory / f"player_{player}/final.zip"), "training_config": source}
    finally:
        env.close()
