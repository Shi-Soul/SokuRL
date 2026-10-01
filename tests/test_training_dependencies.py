"""All training installation options resolve to the same PPO dependency versions."""
from pathlib import Path
import tomllib

from packaging.requirements import Requirement
import pytest

PROJECT = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8"))["project"]


def requirements(extra, active):
    if extra in active:
        raise ValueError("training dependency groups contain a cycle")
    resolved = set()
    for text in PROJECT["optional-dependencies"][extra]:
        requirement = Requirement(text)
        if requirement.name == PROJECT["name"]:
            for dependency in requirement.extras:
                resolved.update(requirements(dependency, active | {extra}))
        else:
            resolved.add(str(requirement))
    return resolved


@pytest.mark.parametrize("extra", ["ppo", "nfsp", "psro", "marl"])
def test_training_installations_share_pinned_ppo_and_environment(extra):
    resolved = {Requirement(text).name: Requirement(text).specifier
                for text in requirements(extra, set())}
    for name, version in (("torch", "2.9.1"), ("stable-baselines3", "2.9.0"), ("sb3-contrib", "2.9.0")):
        assert str(resolved[name]) == f"=={version}"
    assert {"numpy", "gymnasium", "pettingzoo"} <= resolved.keys()


def test_base_launcher_installs_its_hydra_entry_dependency():
    assert "hydra-core" in {Requirement(text).name for text in PROJECT["dependencies"]}
