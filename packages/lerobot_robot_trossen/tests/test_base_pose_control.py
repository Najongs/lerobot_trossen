"""Base pose tracking: the tracker against a simulated base.

No hardware: the base is a differential drive whose velocity follows the command
through a first-order lag, integrated at the control tick.

Copied from PR 69 (``refs/pull/69/head``, bf3e0dc) with only the tracker tests
kept. The robot-level tests there need PR 69's pose mode in ``mobileai.py``,
which this branch does not bring; stage_runner's base stages drive the tracker
themselves.
"""

import itertools
import math

import pytest
from lerobot_robot_trossen.base_pose_control import (
    BasePoseGains,
    BasePoseTracker,
    wrap_angle,
)

TICK_S = 1 / 20.40  # the measured acquisition loop period the labels use
LAG_S = 0.13  # command-to-velocity time constant, middle of the 0.10-0.16 s measured


class SimBase:
    """Differential base; velocity lags the command, odometry is exact."""

    def __init__(self, x=0.0, y=0.0, heading=0.0):
        self.x, self.y, self.heading = x, y, heading
        self.v = 0.0
        self.w = 0.0

    def odom(self):
        return (self.x, self.y, wrap_angle(self.heading))

    def step(self, v_cmd, w_cmd, dt):
        alpha = 1.0 - math.exp(-dt / LAG_S)
        self.v += alpha * (v_cmd - self.v)
        self.w += alpha * (w_cmd - self.w)
        self.x += self.v * math.cos(self.heading) * dt
        self.y += self.v * math.sin(self.heading) * dt
        self.heading += self.w * dt


def demo_turn_targets(total_rad=-1.5, rate=0.6, hold_ticks=60, dt=TICK_S):
    """theta.pos labels of a demo-like turn: constant rate, then hold."""
    targets, theta, step = [], 0.0, math.copysign(rate * dt, total_rad)
    while abs(theta) < abs(total_rad):
        theta = theta + step
        if abs(theta) > abs(total_rad):
            theta = total_rad
        targets.append(theta)
    return targets + [total_rad] * hold_ticks


def run(tracker, base, targets_s, targets_theta, tick_s=TICK_S, odom_fn=None):
    tracker.reset(base.odom(), nominal_dt=TICK_S)
    trace = []
    now = 0.0
    for s_target, theta_target in zip(targets_s, targets_theta):
        odom = odom_fn(base) if odom_fn else base.odom()
        cmd = tracker.command(s_target, theta_target, odom, now)
        base.step(cmd.x_vel, cmd.theta_vel, tick_s)
        trace.append(cmd)
        now += tick_s
    return trace


def turned(base, start_heading=0.0):
    return base.heading - start_heading


# ----- tracker against the simulated base -------------------------------------------


def test_turn_settles_on_the_target_heading():
    targets = demo_turn_targets()
    base = SimBase()
    run(BasePoseTracker(BasePoseGains()), base, [0.0] * len(targets), targets)
    assert turned(base) == pytest.approx(-1.5, abs=math.radians(1.0))
    assert abs(base.w) < 0.01  # stopped, not still turning


def test_overshoot_stays_small():
    targets = demo_turn_targets()
    base = SimBase()
    headings = []
    tracker = BasePoseTracker(BasePoseGains())
    tracker.reset(base.odom(), nominal_dt=TICK_S)
    for i, target in enumerate(targets):
        cmd = tracker.command(0.0, target, base.odom(), i * TICK_S)
        base.step(cmd.x_vel, cmd.theta_vel, TICK_S)
        headings.append(base.heading)
    assert min(headings) > -1.5 - math.radians(3.0)


def test_feedforward_keeps_the_ramp_lag_small():
    # Pure P lags a 0.6 rad/s ramp by omega/kp ~ 17 deg; with feed-forward the lag
    # is the base's own time constant, ~ omega * LAG_S ~ 4.5 deg.
    targets = demo_turn_targets()
    base = SimBase()
    tracker = BasePoseTracker(BasePoseGains())
    tracker.reset(base.odom(), nominal_dt=TICK_S)
    worst = 0.0
    for i, target in enumerate(targets[:40]):
        cmd = tracker.command(0.0, target, base.odom(), i * TICK_S)
        base.step(cmd.x_vel, cmd.theta_vel, TICK_S)
        worst = max(worst, abs(target - base.heading))
    assert worst < math.radians(8.0)


@pytest.mark.parametrize("tick_s", [1 / 15.0, 1 / 25.0])
def test_loop_rate_does_not_change_the_final_heading(tick_s):
    # The point of pose actions: a velocity held for a longer tick turns further,
    # a pose target does not. Same labels, slower and faster loop.
    targets = demo_turn_targets()
    base = SimBase()
    run(
        BasePoseTracker(BasePoseGains()),
        base,
        [0.0] * len(targets),
        targets,
        tick_s=tick_s,
    )
    assert turned(base) == pytest.approx(-1.5, abs=math.radians(1.0))

    # Contrast: replaying the same motion as held velocities scales with the tick.
    open_loop = sum((b - a) / TICK_S * tick_s for a, b in zip([0.0] + targets, targets))
    assert abs(open_loop - (-1.5)) > math.radians(10.0)


def test_heading_is_unwrapped_across_pi():
    targets = [min(1.0, 0.03 * (i + 1)) for i in range(100)]
    base = SimBase(heading=2.9)  # turning +1 rad crosses +pi in the odometry
    trace = run(BasePoseTracker(BasePoseGains()), base, [0.0] * 100, targets)
    assert trace[-1].theta == pytest.approx(1.0, abs=math.radians(1.0))
    assert turned(base, 2.9) == pytest.approx(1.0, abs=math.radians(1.0))


def test_forward_distance_is_measured_along_the_heading():
    targets = [min(0.5, 0.02 * (i + 1)) for i in range(120)]
    base = SimBase(x=3.0, y=-2.0, heading=1.0)  # odometry origin is elsewhere
    trace = run(BasePoseTracker(BasePoseGains()), base, targets, [0.0] * 120)
    assert trace[-1].s == pytest.approx(0.5, abs=0.01)
    assert math.hypot(base.x - 3.0, base.y + 2.0) == pytest.approx(0.5, abs=0.01)


def test_first_target_off_the_start_is_not_a_one_tick_jump():
    # A policy's first target on an unseen screen is often a few cm / degrees
    # off the stage start. The first tick must not ask for it in one tick.
    tracker = BasePoseTracker(BasePoseGains())
    tracker.reset((0.0, 0.0, 0.0), nominal_dt=TICK_S)
    cmd = tracker.command(-0.07, 0.05, (0.0, 0.0, 0.0), 0.0)
    assert cmd.x_vel == pytest.approx(2.0 * -0.07)
    assert cmd.theta_vel == pytest.approx(2.0 * 0.05)


def test_averaging_takes_the_wobble_out_and_keeps_the_stop():
    # Same ramp with a +-0.7 deg step-to-step wobble on the targets.
    wobble = [math.radians(0.7) * (1 if i % 2 else -1) for i in range(200)]
    ramp = demo_turn_targets()
    noisy = [t + w for t, w in zip(ramp, wobble)] + [ramp[-1]] * 40

    def run(window):
        base = SimBase()
        tracker = BasePoseTracker(BasePoseGains(target_window=window))
        tracker.reset(base.odom(), nominal_dt=TICK_S)
        commands = []
        for i, target in enumerate(noisy):
            cmd = tracker.command(0.0, target, base.odom(), i * TICK_S)
            base.step(cmd.x_vel, cmd.theta_vel, TICK_S)
            commands.append(cmd.theta_vel)
        steps = [b - a for a, b in itertools.pairwise(commands)]
        return math.sqrt(sum(d * d for d in steps) / len(steps)), base.heading

    raw_jitter, raw_end = run(1)
    avg_jitter, avg_end = run(3)
    assert avg_jitter < 0.5 * raw_jitter
    assert avg_end == pytest.approx(raw_end, abs=math.radians(0.5))
    assert avg_end == pytest.approx(-1.5, abs=math.radians(0.5))


def test_target_window_must_be_positive():
    with pytest.raises(ValueError):
        BasePoseTracker(BasePoseGains(target_window=0))


def test_output_is_saturated():
    gains = BasePoseGains(max_x_vel=0.3, max_theta_vel=0.4, target_window=1)
    tracker = BasePoseTracker(gains)
    tracker.reset((0.0, 0.0, 0.0), nominal_dt=TICK_S)
    tracker.command(0.0, 0.0, (0.0, 0.0, 0.0), 0.0)
    cmd = tracker.command(5.0, -5.0, (0.0, 0.0, 0.0), TICK_S)
    assert (cmd.x_vel, cmd.theta_vel) == (0.3, -0.4)


def test_garbage_odometry_read_stops_the_base_and_is_skipped():
    tracker = BasePoseTracker(BasePoseGains())
    tracker.reset((0.0, 0.0, 0.0), nominal_dt=TICK_S)
    tracker.command(0.0, -0.03, (0.0, 0.0, -0.01), 0.0)
    bad = tracker.command(0.0, -0.06, (1e28, 0.0, 0.0), TICK_S)
    assert (bad.x_vel, bad.theta_vel, bad.odom_ok) == (0.0, 0.0, False)
    nan = tracker.command(0.0, -0.09, (0.0, float("nan"), 0.0), 2 * TICK_S)
    assert not nan.odom_ok
    good = tracker.command(0.0, -0.12, (0.0, 0.0, -0.05), 3 * TICK_S)
    assert good.odom_ok and good.theta == pytest.approx(-0.05)
    assert good.s == pytest.approx(0.0)


def test_odometry_that_jumps_for_good_is_reanchored():
    tracker = BasePoseTracker(BasePoseGains())
    tracker.reset((0.0, 0.0, 0.5), nominal_dt=TICK_S)
    results = [
        tracker.command(0.0, 0.0, (0.0, 0.0, -1.0), i * TICK_S) for i in range(3)
    ]
    assert [r.odom_ok for r in results] == [False, False, True]
    assert results[-1].theta == 0.0  # the jump itself is not counted as a turn


def test_command_before_reset_is_an_error():
    with pytest.raises(RuntimeError):
        BasePoseTracker(BasePoseGains()).command(0.0, 0.0, (0.0, 0.0, 0.0), 0.0)
