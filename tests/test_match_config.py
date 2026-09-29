"""Match selection must fail before launch and remain part of saved contracts."""
from dataclasses import asdict, replace

import pytest

from soku_rl.env.match import LEGACY_MATCH, MatchConfig, PlayerSetup
from soku_rl.env import EpisodeConfig
from test_env_timing import VISIBILITY


@pytest.mark.parametrize("field,value", [("character", 20), ("character", True),
    ("deck", 4), ("deck", -1), ("palette", 8)])
def test_invalid_selection(field, value):
    with pytest.raises(ValueError, match=field):
        PlayerSetup(**(asdict(LEGACY_MATCH.player_0) | {field: value}))


def test_selections_survive_configuration_and_reset_transport():
    match = MatchConfig(PlayerSetup(19, 7, 3), PlayerSetup(6, 1, 2))
    episode = EpisodeConfig(60, 1, 1, 0, "diagnostic_state", VISIBILITY, match)
    assert EpisodeConfig.from_dict(asdict(episode)) == episode
    assert episode.backend_observation()["match"] == asdict(match)
    assert match.environment() == {
        "SOKURL_VS_P1_CHARACTER": "19", "SOKURL_VS_P1_PALETTE": "7", "SOKURL_VS_P1_DECK": "3",
        "SOKURL_VS_P2_CHARACTER": "6", "SOKURL_VS_P2_PALETTE": "1", "SOKURL_VS_P2_DECK": "2"}
    old = asdict(episode)
    del old["match"]
    assert EpisodeConfig.from_dict(old).match == LEGACY_MATCH
    assert replace(episode, match=LEGACY_MATCH) != episode
