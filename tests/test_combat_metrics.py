from soku_rl.env.combat_metrics import CombatMetrics, summarize_combat
from soku_rl.env.observation.privileged import PrivilegedObservation


def observation(hp):
    return PrivilegedObservation({}, tuple({"hp": value} for value in hp))


def test_hp_accounting_healing_reset_and_both_seats():
    metrics = CombatMetrics()
    metrics.reset(observation((10000, 10000)))
    for hp in ((9500, 9000), (9600, 9000), (9400, 0), (10000, 10000)):
        metrics.step(observation(hp))
    own, other = metrics.snapshot(0), metrics.snapshot(1)
    assert own["own_hp_loss"] == other["opponent_hp_loss"] == 700
    assert own["opponent_hp_loss"] == other["own_hp_loss"] == 10000
    assert own["own_hp_loss_frames"] == 2
    metrics.reset(observation((10000, 10000)))
    assert metrics.snapshot(0)["own_hp_loss"] == 0
    summary = summarize_combat([own, other, {"available": False}])
    assert summary["measured_episodes"] == 2
    assert summary["means"]["own_hp_loss"] == 5350


def test_unavailable_is_not_zero_damage():
    metrics = CombatMetrics()
    metrics.reset(object())
    metrics.step(object())
    assert metrics.snapshot(0)["available"] is False
