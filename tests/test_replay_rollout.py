"""Rollout archives preserve observation bits and the full official replay."""
from dataclasses import replace

import numpy as np
import pytest

from soku_rl.env import EpisodeConfig
from soku_rl.env.match import LEGACY_MATCH
from soku_rl.env.observation.visibility import VisibilityConfig
from soku_rl.replay.rollout import RolloutFrame, export_replay, read_rollout, write_rollout
from test_official_replay import example


def config():
    return EpisodeConfig(600, 2, 3, 5, "diagnostic_state",
                         VisibilityConfig(8, .5, .02, .1, 48., 96., 16., .25), LEGACY_MATCH)


def test_archive_preserves_observations_counters_and_original_multimatch_replay(tmp_path):
    episode = config()
    replay = example()
    header = bytearray(replay.header)
    header[7] = 2
    replay = replace(replay, header=bytes(header), matches=replay.matches * 2)
    observations = np.linspace(-1, 1, 2 * episode.space().shape[0], dtype=np.float32).reshape(2, -1)
    frames = [RolloutFrame(i, observations + i,
                          np.asarray([[-20, 0, 91, 0, 0, 0, 0, 1]] * 2, np.int32),
                          (0., 0.), "ongoing") for i in range(3)]
    archive = tmp_path / "episode.zip"
    assert write_rollout(archive, replay, 1, episode, iter(frames)) == 3
    restored = list(read_rollout(archive))
    for original, actual in zip(frames, restored, strict=True):
        assert original.frame == actual.frame
        assert original.observations.tobytes() == actual.observations.tobytes()
        assert original.engine_inputs.tobytes() == actual.engine_inputs.tobytes()
        assert original.rewards == actual.rewards
        assert original.outcome == actual.outcome
    destination = tmp_path / "restored.rep"
    export_replay(archive, destination)
    assert destination.read_bytes() == replay.encode()


def test_interrupted_archive_is_not_exported_as_complete(tmp_path):
    archive = tmp_path / "incomplete.zip"
    with pytest.raises(ValueError, match="initial frame"):
        write_rollout(archive, example(), 0, config(), ())
    with pytest.raises(KeyError, match="manifest"):
        export_replay(archive, tmp_path / "wrong.rep")
    assert not (tmp_path / "wrong.rep").exists()
