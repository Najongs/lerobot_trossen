"""base_budget.BaseThetaBudget: the per-episode yaw budget.

Loaded by file path so it runs without the robot SDK (the package __init__ pulls
it in); with LEROBOT_BASE_THETA_BUDGET_DEG unset the module creates no governor.
"""

import importlib.util
import math
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "src/lerobot_robot_trossen/base_budget.py"


@pytest.fixture(scope="module")
def bb():
    spec = importlib.util.spec_from_file_location("base_budget_under_test", _PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Loop:
    """Fake record_loop phase tag and a 21 Hz clock."""

    def __init__(self):
        self.phase, self.seq, self.t = None, 0, 0.0

    def start(self, phase):
        self.phase, self.seq = phase, self.seq + 1

    def clock(self):
        return self.t


def make(bb, budget_deg=90.0):
    loop = Loop()
    gov = bb.BaseThetaBudget(budget_deg, lambda: loop.phase, lambda: loop.seq, loop.clock)
    return loop, gov


def run(gov, loop, commands, dt=1 / 21):
    sent = []
    for c in commands:
        sent.append(gov.apply(c))
        loop.t += dt
    return sent


def test_parse(bb):
    assert bb.parse_budget_deg("") is None
    assert bb.parse_budget_deg("  ") is None
    assert bb.parse_budget_deg("105") == 105.0
    for bad in ("0", "-5", "nan", "inf", "abc"):
        with pytest.raises(ValueError):
            bb.parse_budget_deg(bad)


def test_module_off_without_env(bb, monkeypatch):
    monkeypatch.delenv(bb.ENV_VAR, raising=False)
    assert bb._create() is None
    monkeypatch.setattr(bb, "_GOVERNOR", None)
    assert bb.govern_theta(0.7) == 0.7
    bb.finish()  # no-op when off


def test_caps_same_direction_and_passes_x_free_turn_back(bb):
    loop, gov = make(bb, 90.0)
    loop.start("policy")
    w = 0.5  # rad/s -> 90 deg after pi/2 / 0.5 = 3.14 s ~ 66 ticks
    sent = run(gov, loop, [w] * 200)
    assert gov.fired_tick is not None
    assert all(s == w for s in sent[: gov.fired_tick - 1])
    assert all(s == 0.0 for s in sent[gov.fired_tick - 1 :])
    # Net stops within one tick's rotation of the budget.
    assert math.degrees(gov.net_rad) == pytest.approx(90.0, abs=math.degrees(w / 21) + 1e-6)
    assert gov.suppressed == 200 - gov.fired_tick + 1
    # Turning back is allowed and lowers the net.
    back = run(gov, loop, [-w] * 10)
    assert back == [-w] * 10
    assert math.degrees(gov.net_rad) < 90.0 - 1.0


def test_negative_direction(bb):
    loop, gov = make(bb, 45.0)
    loop.start("policy")
    sent = run(gov, loop, [-0.4] * 200)
    assert gov.fired_tick is not None and sent[-1] == 0.0
    assert math.degrees(gov.net_rad) == pytest.approx(-45.0, abs=math.degrees(0.4 / 21) + 1e-6)
    # Positive (turning back) passes.
    assert gov.apply(0.3) == 0.3


def test_reset_and_teleop_untouched_and_episode_restarts(bb):
    loop, gov = make(bb, 30.0)
    loop.start("policy")
    run(gov, loop, [0.5] * 100)
    assert gov.fired_tick is not None
    loop.start("teleop")  # leader-arm reset between episodes
    assert run(gov, loop, [0.9] * 50) == [0.9] * 50
    loop.start("policy")  # next episode: integral from 0, fires again later
    sent = run(gov, loop, [0.5] * 5)
    assert sent == [0.5] * 5 and gov.fired_tick is None
    assert gov.net_rad == pytest.approx(4 * 0.5 / 21)


def test_back_to_back_policy_episodes_reset_on_seq(bb):
    loop, gov = make(bb, 30.0)
    loop.start("policy")
    run(gov, loop, [0.5] * 100)
    loop.start("policy")  # a reset that ran zero ticks: phase string unchanged, seq moves
    assert gov.apply(0.5) == 0.5 and gov.net_rad == 0.0


def test_uses_measured_dt_with_cap(bb):
    loop, gov = make(bb, 1000.0)
    loop.start("policy")
    run(gov, loop, [0.2, 0.2], dt=0.1)  # slow loop: the held command rotates longer
    assert gov.net_rad == pytest.approx(0.2 * 0.1)
    gov.apply(0.2)  # third call adds the second tick's 0.1 s
    assert gov.net_rad == pytest.approx(0.2 * 0.2)
    loop.t += 10.0  # a stall counts at most _MAX_DT_S
    gov.apply(0.0)
    assert gov.net_rad == pytest.approx(0.2 * 0.2 + 0.2 * bb._MAX_DT_S)


def test_overshoot_bounded_by_one_capped_tick(bb):
    # Just under budget, a slow tick lets the passed command run up to _MAX_DT_S.
    loop, gov = make(bb, 10.0)
    loop.start("policy")
    gov.net_rad = math.radians(9.9)
    gov._prev_t = loop.t
    gov._prev_theta = 1.0
    loop.t += 0.01
    assert gov.apply(1.0) == 1.0  # still under budget: passes
    loop.t += 5.0  # stall
    assert gov.apply(1.0) == 0.0  # now over: blocked
    over = gov.net_rad - math.radians(10.0)
    assert 0 < over <= 1.0 * bb._MAX_DT_S + 0.01 + 1e-9


def test_suppressed_tick_adds_no_rotation(bb):
    loop, gov = make(bb, 10.0)
    loop.start("policy")
    run(gov, loop, [1.0] * 100)
    net = gov.net_rad
    run(gov, loop, [1.0] * 100)
    assert gov.net_rad == net


def test_outside_any_loop_untouched(bb):
    loop, gov = make(bb, 1.0)
    assert run(gov, loop, [1.0] * 100) == [1.0] * 100


def test_summary_logged_once(bb, caplog):
    loop, gov = make(bb, 10.0)
    loop.start("policy")
    run(gov, loop, [1.0] * 50)
    with caplog.at_level("INFO"):
        gov.finish_episode()
        gov.finish_episode()
    lines = [r.message for r in caplog.records if "policy 구간 1 끝" in r.message]
    assert len(lines) == 1 and "발동 tick" in lines[0]


def test_finish_flags_a_run_that_never_saw_a_policy_tick(bb, monkeypatch, caplog):
    loop, gov = make(bb, 105.0)
    monkeypatch.setattr(bb, "_GOVERNOR", gov)
    run(gov, loop, [0.5] * 10)  # phase None: never a policy tick
    with caplog.at_level("INFO"):
        bb.finish()
    assert any("policy 구간을 한 번도 보지 못했다" in r.message for r in caplog.records)
    caplog.clear()
    loop.start("policy")
    run(gov, loop, [0.5] * 10)
    with caplog.at_level("INFO"):
        bb.finish()
    assert not any("한 번도" in r.message for r in caplog.records)
    assert any("policy 구간 1 끝" in r.message for r in caplog.records)
