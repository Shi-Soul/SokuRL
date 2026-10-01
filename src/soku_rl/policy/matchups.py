"""Bind a frozen opponent specification to its declared character setup."""
from dataclasses import replace
from pathlib import PurePosixPath

from soku_rl.env.match import MatchConfig, PlayerSetup
from soku_rl.env.wrappers.learning import LearningInterface


def opponent_interface(interface, learner, entry):
    match = MatchConfig(PlayerSetup(**learner), PlayerSetup(**entry["setup"]))
    spec = entry["policy"]
    if spec["kind"] == "rule" and spec["name"] == "god":
        script = spec["rules"]["god"]["script"]
        if script != "character":
            stem = PurePosixPath(script).name
            if not stem[:2].isdigit() or int(stem[:2]) != match.player_1.character:
                raise ValueError("BR god script and opponent character do not match")
    return LearningInterface(replace(interface.episode, match=match), interface.config)
