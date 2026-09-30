"""MARL settings cannot silently diverge from the single PPO configuration."""
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest

from soku_rl.rl import ppo_settings


@pytest.mark.parametrize("algorithm", ("ppo", "ippo", "nfsp", "psro"))
def test_all_training_methods_share_one_resolved_ppo_configuration(algorithm):
    with initialize_config_dir(config_dir=str(Path(__file__).parents[1] / "config"), version_base="1.3"):
        composed = compose(config_name="train", overrides=[f"algorithm={algorithm}", "rl.ppo.gamma=0.9"])
        config = {name: OmegaConf.to_container(composed[name], resolve=True) for name in ("rl", "algorithm")}
    assert ppo_settings(config) is config["rl"]
    assert config["rl"]["ppo"]["gamma"] == 0.9
    specific = config["algorithm"]["response"] if algorithm == "psro" else config["algorithm"]
    specific["ppo"]["gamma"] = 0.8
    with pytest.raises(ValueError, match="configure rl.ppo"):
        ppo_settings(config)
