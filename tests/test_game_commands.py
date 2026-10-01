"""Validate diagnostic command dispatch without starting or reading a game."""
from pathlib import Path
from types import SimpleNamespace
import sys

from hydra import compose, initialize_config_dir
from omegaconf import MissingMandatoryValue
import pytest

from game_runtime.commands import run


def configuration(overrides):
    with initialize_config_dir(config_dir=str(Path(__file__).parents[1] / "config"), version_base="1.3"):
        return compose(config_name="game_control", overrides=overrides)


def test_command_must_be_explicit():
    with pytest.raises(MissingMandatoryValue):
        run(configuration([]))


@pytest.mark.parametrize("command", ["shutdown", "script", "anchor_save"])
def test_commands_that_modify_existing_game_require_pid(command):
    with pytest.raises(ValueError, match="explicit launch.pid"):
        run(configuration([f"launch.command={command}"]))


@pytest.mark.parametrize("override, message", [
    ("launch.command=unknown", "unknown game command"),
    ("launch.timeout=0", "positive"),
    ("launch.pid=-1", "positive integer"),
])
def test_invalid_command_rejected_before_windows_imports(override, message):
    with pytest.raises(ValueError, match=message):
        run(configuration(["launch.command=status", override]))


@pytest.mark.parametrize("command, method, overrides, expected", [
    ("practice", "practice", [], (30., None)),
    ("vs", "versus", ["launch.headless=true", "launch.unlimited=true"], (30., True, True)),
    ("replay", "replay", ["launch.path=record.rep", "launch.frame=42"], (Path("record.rep"), 42, 30.)),
    ("list", "list_instances", [], ()),
    ("status", "status", ["launch.pid=17"], (17,)),
    ("shutdown", "shutdown", ["launch.pid=17", "launch.timeout=5"], (5., 17)),
])
def test_commands_use_configured_directory_and_arguments(monkeypatch, command, method, overrides, expected):
    calls = []

    def dispatch(*args):
        calls.append((method, args))
        return 0

    monkeypatch.setitem(sys.modules, "sokurl", SimpleNamespace(**{method: dispatch}))
    monkeypatch.setitem(sys.modules, "game_runtime.startup", SimpleNamespace(
        configure_game=lambda path: calls.append(("configure", path))))
    config = configuration([f"launch.command={command}", "runtime.game_directory=selected-game", *overrides])
    assert run(config) == 0
    assert calls == [("configure", "selected-game"), (method, expected)]
