"""Track a pose-target base action with the base odometry (opt-in, eval only).

Why this exists
---------------
The base action is normally a velocity (``x.vel``, ``theta.vel``) that is held
until the next control tick, so the angle a policy turns is its output *times*
how long each command is held. A policy that outputs a little too much keeps
adding that excess every tick, and a slow tick adds more.

A policy trained on pose labels instead outputs where the base should be:

- ``x.pos``  -- forward distance travelled since the start of the stage (m),
  the running sum of ``x.vel * dt``;
- ``theta.pos`` -- heading change since the start of the stage (rad), the
  running sum of ``theta.vel * dt``.

Both sums include the current step (label t is the pose *after* step t), and the
stage start is the episode start because the datasets are cut per stage. Turning
those targets into a base command needs a tracker, and stopping is then the
tracker's job, not the policy's.

Controller
----------
Per axis, once per control tick::

    m_t = mean(r_{t-n+1} .. r_t)
    u   = (m_t - m_{t-1}) / dt  +  kp * (m_{t-1} - y)

``r_t`` is this tick's target, ``m_t`` the mean of the last ``n =
target_window`` targets of this episode, ``y`` the odometry read at the top of
this same iteration, and ``dt`` a smoothed estimate of the tick period.
``m_{t-1}`` is where the (smoothed) labels say the base should be *now*. The
first term is the feed-forward: a pure P controller lags a ramp by
``omega / kp`` (about 15 deg at 0.6 rad/s with kp = 2), so it is not optional.
The result is saturated before it goes to ``set_cmd_vel``.

Why the targets are averaged: a policy's targets wobble a little from step to
step (about 0.7 deg on screens it has not seen), and the one-step difference
passes that wobble to the wheels. Averaging the targets themselves, and having
both terms follow the average, removes most of it and only delays the path by
``(n - 1) / 2`` ticks; it does not change where the base stops. Averaging only
the feed-forward slope would leave the P term on the raw targets, and the
lagging slope then pushes past the end of a turn.

The average starts from this episode's first target, so the first tick has no
feed-forward. A policy's first target is often a little off the stage start
(a few degrees, several cm); a difference from the reference would ask for that
whole offset in one tick and saturate, while the P term closes it at its own
pace (time constant ``1 / kp``).

Odometry
--------
``TrossenSlate.get_pose()`` returns ``[x, y, theta]`` from the cache that
``update_state()`` already filled, so reading it adds no serial traffic.
``trossen_slate`` 0.0.3 has no ``reset_odometry()``, so the reference is
subtracted here: heading is unwrapped across +-pi, and forward distance is the
odometry displacement projected on the heading, summed tick by tick. A read that
jumps further than one tick could move the base is dropped as a garbage read
(the velocity path guards against the same failure in ``mobileai.py``).

This module is pure Python so it can be tested without a robot; the robot-side
guards (policy phase only, reference reset per episode) live in ``mobileai.py``.
"""

import math
from collections import deque
from dataclasses import dataclass

# A single tick at the 1 m/s / 1 rad/s hardware clamp moves the base about
# 0.05 m / 0.05 rad, so anything several times larger is not motion.
_MAX_ODOM_JUMP_M = 0.25
_MAX_ODOM_JUMP_RAD = 0.5
# After this many rejected reads in a row, the odometry itself is assumed to have
# jumped (a firmware reset, say) and the tracker re-anchors on the new reading
# instead of rejecting every read for the rest of the episode.
_MAX_REJECTED_READS = 3

# Weight of the newest measured tick period in the smoothed dt.
_DT_SMOOTHING = 0.2


def wrap_angle(angle: float) -> float:
    """Map an angle to [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class BasePoseGains:
    kp_x: float = 2.0  # 1/s
    kp_theta: float = 2.0  # 1/s
    max_x_vel: float = 1.0  # m/s
    max_theta_vel: float = 1.0  # rad/s
    # Targets averaged before tracking; 1 = follow each target as it comes.
    target_window: int = 3


@dataclass
class BasePoseCommand:
    x_vel: float
    theta_vel: float
    # Pose relative to the reference, after this tick's odometry was folded in.
    s: float
    theta: float
    # False when this tick's odometry could not be used; the command is then zero.
    odom_ok: bool


class BasePoseTracker:
    """Follow (x.pos, theta.pos) targets with the base odometry. One per robot."""

    def __init__(self, gains: BasePoseGains):
        if gains.target_window < 1:
            raise ValueError(f"target_window must be >= 1, got {gains.target_window}")
        self.gains = gains
        self._anchored = False
        self._last_raw: tuple[float, float, float] | None = None
        self._rejected = 0
        self.s = 0.0
        self.theta = 0.0
        self._recent: deque[tuple[float, float]] = deque(maxlen=gains.target_window)
        self._prev_mean: tuple[float, float] | None = None
        self._prev_t: float | None = None
        self._dt = 0.05

    @property
    def anchored(self) -> bool:
        return self._anchored

    def reset(self, odom: tuple[float, float, float], nominal_dt: float) -> None:
        """Make ``odom`` the reference pose: the start of the stage."""
        self._anchored = True
        self._last_raw = tuple(float(v) for v in odom)
        self._rejected = 0
        self.s = 0.0
        self.theta = 0.0
        self._recent.clear()
        self._prev_mean = None
        self._prev_t = None
        self._dt = nominal_dt

    def release(self) -> None:
        """Forget the reference; the next pose command must ``reset()`` first."""
        self._anchored = False
        self._last_raw = None

    def _fold_odom(self, odom: tuple[float, float, float] | None) -> bool:
        """Advance (s, theta) by the move since the last accepted read."""
        if odom is None or not all(math.isfinite(v) for v in odom):
            return False
        x, y, heading = (float(v) for v in odom)
        last_x, last_y, last_heading = self._last_raw
        dtheta = wrap_angle(heading - last_heading)
        dx, dy = x - last_x, y - last_y
        if abs(dtheta) > _MAX_ODOM_JUMP_RAD or math.hypot(dx, dy) > _MAX_ODOM_JUMP_M:
            self._rejected += 1
            if self._rejected < _MAX_REJECTED_READS:
                return False
            # The odometry moved for good: take the new reading as the last one
            # without counting the jump as motion.
            self._last_raw = (x, y, heading)
            self._rejected = 0
            return True
        mid = last_heading + 0.5 * dtheta
        self.s += dx * math.cos(mid) + dy * math.sin(mid)
        self.theta += dtheta
        self._last_raw = (x, y, heading)
        self._rejected = 0
        return True

    def command(
        self,
        target_s: float,
        target_theta: float,
        odom: tuple[float, float, float] | None,
        now: float,
    ) -> BasePoseCommand:
        """Velocity command for this tick. Requires ``reset()`` first."""
        if not self._anchored:
            raise RuntimeError("BasePoseTracker.command() before reset()")

        if self._prev_t is not None:
            measured = now - self._prev_t
            # Bounded so one long stall (or a clock hiccup) cannot blow up the
            # feed-forward.
            measured = min(max(measured, 0.5 * self._dt), 3.0 * self._dt)
            self._dt += _DT_SMOOTHING * (measured - self._dt)
        self._prev_t = now
        self._recent.append((target_s, target_theta))
        n = len(self._recent)
        mean_s = sum(r[0] for r in self._recent) / n
        mean_theta = sum(r[1] for r in self._recent) / n
        if self._prev_mean is None:
            # First tick of the episode: no feed-forward (see module docstring).
            prev_s, prev_theta = mean_s, mean_theta
        else:
            prev_s, prev_theta = self._prev_mean
        self._prev_mean = (mean_s, mean_theta)

        if not self._fold_odom(odom):
            return BasePoseCommand(0.0, 0.0, self.s, self.theta, odom_ok=False)

        g = self.gains
        x_vel = (mean_s - prev_s) / self._dt + g.kp_x * (prev_s - self.s)
        theta_vel = (mean_theta - prev_theta) / self._dt + g.kp_theta * (
            prev_theta - self.theta
        )
        x_vel = max(-g.max_x_vel, min(g.max_x_vel, x_vel))
        theta_vel = max(-g.max_theta_vel, min(g.max_theta_vel, theta_vel))
        return BasePoseCommand(x_vel, theta_vel, self.s, self.theta, odom_ok=True)
