"""Public slots are bounded only after all captured contours affect visibility."""
from dataclasses import replace
from types import SimpleNamespace

from soku_rl.render_state import RenderEntity, RenderSnapshot
from soku_rl.visibility import VisibilityConfig, visible_entities
from soku_rl.visible_state import observe_visible_states, STATE_FEATURES


def test_large_scene_keeps_model_shape_and_uses_all_contours():
    config = VisibilityConfig(1, .5, .02, .1, 48., 96., 8., .25)
    hidden = RenderEntity(0., 0., 0., 1, 1)
    objects = tuple(RenderEntity(32. + (i % 16)*32, 64. + (i // 16)*32, 1., 1, 1)
                    for i in range(65))
    scene = RenderSnapshot(0., 480., 1., 0, (hidden, hidden), (objects, ()), False)
    fighter = SimpleNamespace(hp=10000, spirit=1000, maxSpirit=1000, characterId=0)
    raw = SimpleNamespace(frameId=1, p1=fighter, p2=fighter)
    assert len(visible_entities(scene, config)[1][0]) == 65
    observations = observe_visible_states(raw, scene, config)
    assert STATE_FEATURES == 400
    assert all(len(observation.values) == 400 for observation in observations)
    assert sum(observations[0].values[16:208:3]) == 64
    # A late contour outside the first 64 metadata entries must still occlude.
    occluded = replace(scene, objects=(objects + (objects[0],), ()))
    visible = visible_entities(occluded, config)[1][0]
    assert len(visible) == 64
    assert not any(entity.x == 32. and entity.y == 416. for entity in visible)
    assert observe_visible_states(raw, occluded, config)[0] != observations[0]


def test_capture_overflow_remains_an_error():
    import pytest
    config = VisibilityConfig(1, .5, .02, .1, 48., 96., 8., .25)
    hidden = RenderEntity(0., 0., 0., 1, 1)
    scene = RenderSnapshot(0., 480., 1., 0, (hidden, hidden), ((), ()), True)
    with pytest.raises(RuntimeError, match="overflow"):
        visible_entities(scene, config)
