"""Every shipped script is a selectable opponent with an explicit own character."""
from pathlib import Path

from omegaconf import OmegaConf
import pytest

from soku_rl.play.opponents import opponent_catalog
from soku_rl.policy.god.package import ScriptPackage


@pytest.fixture
def catalog():
    root = Path(__file__).parents[1]
    scripts = root / "third_party/th123_ai/package/th123ai/script"
    if not scripts.is_dir():
        pytest.skip("original script package required")
    package = ScriptPackage(scripts, root / "third_party/th123_ai/source/th123_ai/api.ai")
    rules = OmegaConf.to_container(OmegaConf.load(root / "config/rules/default.yaml"))
    rules["god"] = dict(package=str(scripts), api_source=str(package.root), script="character")
    return opponent_catalog(rules, package, {}), rules, package


def test_all_twenty_characters_and_seven_variants_are_selectable(catalog):
    opponents, rules, package = catalog
    gods = [opponent for opponent in opponents.values() if opponent.script]
    assert len(gods) == 27 and len(opponents) == 42
    assert {opponent.characters[0] for opponent in gods} == set(range(20))
    expected = {name for name in package.files if name.endswith(".ai") and name[:2].isdigit() and "_main" in name}
    assert {opponent.script for opponent in gods} == expected
    for opponent in gods:
        candidate, selected = opponent.configuration("superhuman", opponent.characters[0], rules)
        assert candidate["policy"] == {"kind": "rule", "name": "god"}
        assert selected["god"]["script"] == opponent.script
        assert rules["god"]["script"] == "character"
        with pytest.raises(ValueError, match="tracks"):
            opponent.configuration("human", opponent.characters[0], rules)
        with pytest.raises(ValueError, match="AI characters"):
            opponent.configuration("superhuman", (opponent.characters[0]+1) % 20, rules)


def test_checkpoint_catalog_preserves_shared_loader_specs(catalog):
    _, rules, package = catalog
    kinds = ("sb3", "sb3_recurrent", "nfsp_average", "psro_mixture", "benchmarl_ippo", "onnx_recurrent")
    entries = {kind: {"label": kind, "characters": [1], "tracks": ["human"],
                      "policy": {"kind": kind, "path": "artifact"}} for kind in kinds}
    opponents = opponent_catalog(rules, package, entries)
    for kind in kinds:
        candidate, _ = opponents[kind].configuration("human", 1, rules)
        assert candidate["policy"] == entries[kind]["policy"]
    with pytest.raises(ValueError, match="duplicate"):
        opponent_catalog(rules, package, {"rush": entries["sb3"]})
