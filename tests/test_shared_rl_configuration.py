"""MARL settings cannot silently diverge from the single PPO configuration."""
from pathlib import Path
from types import SimpleNamespace

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest

from soku_rl.rl import ppo_settings, validate_payoff


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


@pytest.mark.parametrize("scale,gamma", [(0., .95), (0., 1.), (.1, 1.), (1., 1.)])
def test_shared_payoff_accepts_supported_shaping_and_discount(scale, gamma):
    interface = SimpleNamespace(config=SimpleNamespace(health_potential_scale=scale))
    validate_payoff(interface, {"timeout_payoff": "zero_at_horizon", "ppo": {"gamma": gamma}})


@pytest.mark.parametrize("gamma", [0., .95, 1.01])
def test_shared_payoff_rejects_discounted_undiscounted_potential(gamma):
    interface = SimpleNamespace(config=SimpleNamespace(health_potential_scale=.1))
    with pytest.raises(ValueError, match="requires gamma=1"):
        validate_payoff(interface, {"timeout_payoff": "zero_at_horizon", "ppo": {"gamma": gamma}})


def test_shared_payoff_rejects_an_unimplemented_horizon_convention():
    interface = SimpleNamespace(config=SimpleNamespace(health_potential_scale=0.))
    with pytest.raises(ValueError, match="finite-horizon payoff"):
        validate_payoff(interface, {"timeout_payoff": "bootstrap", "ppo": {"gamma": 1.}})
