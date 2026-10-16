"""Base rotation budget for eval episodes (off by default, opt in per run).

Why this exists
---------------
On task01 the demonstrations turn the base by a net 76..89 degrees, but the ACT
policies keep turning in the real runs: +3..18 degrees past the demo maximum on a
normal run, +30..80 when a new chunk lands on a scene the demos never reached
(docs/eval_najy_results_1015.md, 10/16 sections). The overshoot comes from the
scene, not from one model, so this caps it in the execution layer instead.

What it does
------------
In the policy phase only, it integrates the base yaw command actually sent,
``theta.vel`` (rad/s) held for the measured time until the next ``send_action``,
into a signed net angle that restarts at 0 every episode. Once ``|net|`` has
reached the budget, a ``theta.vel`` that would push ``|net|`` further is replaced
by 0.0; one that turns back toward the start passes unchanged. ``x.vel`` and the
arms are never touched. It only ever lowers a command to zero -- every safety
constant (the +-1.0 clamp before it, the arm clamps, e-stop) is unchanged.

Usage
-----
    LEROBOT_BASE_THETA_BUDGET_DEG=105 uv run lerobot-record ...

``scripts/eval_najy.sh`` sets it from ``BASE_BUDGET_DEG`` (task01 only, never on
a replay run, always passed explicitly); ``eval_chain.sh`` and ``eval_smolvla.sh``
pass it empty, so a stale export cannot reach those runs. Nothing here knows the
stage: the value is a task01 number (demo max 89 + a 10-degree start offset +
margin), and a chain would apply it to all 11 stages.

Logs: one ``[base_budget]`` line at load, one when it fires (tick, net angle),
and one per policy episode at the end with the suppressed tick count (counted
from 1 per process, re-recorded episodes included -- not the dataset
episode_index). ``basevel.csv`` carries the governed command (what the base got);
the recorded dataset action and ``chunks.csv`` carry the policy's ungoverned
output, so judge an on/off comparison from ``basevel.csv`` ``cmd_theta_vel``.

Limits: a tick longer than 0.5 s counts as 0.5 s, and the base's coast after a
zeroed command is not counted, so the real turn can end a few degrees past the
budget. The command passed just under the budget runs a whole tick before the
next check (~2.7 deg at 1 rad/s and 21 Hz, up to ~29 deg on a 0.5 s stall). The
hold after the last policy tick (episode save, until the next send_action) is
not counted either. A failed base write (``set_cmd_vel`` False, warned in
mobileai.py) leaves the base on its previous command while the integral assumes
the new one; once over budget every tick keeps sending 0, so it re-blocks on the
next good write, but the summary undercounts.

Needs the record_loop phase tag (``LEROBOT_LOOP_HZ_LOG``, on by default): without
it no tick is a policy tick and the budget never acts -- the load line says so.
"""

import logging
import math
import os
import time
from typing import Callable

logger = logging.getLogger(__name__)

ENV_VAR = "LEROBOT_BASE_THETA_BUDGET_DEG"
# A tick longer than this (a stall, the first tick after a pause) still counts,
# but only up to this long -- the base was told to hold the command, so the real
# rotation grew too, yet one hiccup should not by itself spend the whole budget.
_MAX_DT_S = 0.5


def parse_budget_deg(raw: str) -> float | None:
    """'' -> None (off). A positive finite number of degrees, else ValueError."""
    raw = raw.strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{ENV_VAR}={raw!r} is not a number of degrees") from None
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{ENV_VAR}={raw!r} must be a positive number of degrees")
    return value


class BaseThetaBudget:
    """Signed net yaw budget per policy episode. SDK-free; phase and clock injected."""

    def __init__(
        self,
        budget_deg: float,
        phase: Callable[[], str | None],
        phase_seq: Callable[[], int],
        clock: Callable[[], float] = time.perf_counter,
    ):
        self.budget_rad = math.radians(budget_deg)
        self._phase = phase
        self._phase_seq = phase_seq
        self._clock = clock
        self._seq = None
        self._in_policy = False
        self._episode = 0  # policy episodes seen by this process
        self._reset_episode()

    def _reset_episode(self) -> None:
        self.net_rad = 0.0
        self._prev_t = None
        self._prev_theta = 0.0
        self.ticks = 0
        self.fired_tick = None
        self.fired_net_rad = None
        self.suppressed = 0

    def summary(self) -> str:
        if self.fired_tick is None:
            fired = "발동 없음"
        else:
            fired = (
                f"발동 tick {self.fired_tick} (net {math.degrees(self.fired_net_rad):+.1f}°), "
                f"막은 tick {self.suppressed}"
            )
        return (
            f"[base_budget] policy 구간 {self._episode} 끝: {fired}, "
            f"끝 net {math.degrees(self.net_rad):+.1f}° / 예산 {math.degrees(self.budget_rad):.0f}°, "
            f"policy tick {self.ticks}"
        )

    def finish_episode(self) -> None:
        """Log the running policy episode's summary once (phase change or disconnect)."""
        if self._in_policy:
            logger.info(self.summary())
            self._in_policy = False

    def apply(self, theta_vel: float) -> float:
        """Called once per loop with the sanitized yaw command; returns the one to send."""
        phase = self._phase()
        seq = self._phase_seq()
        if seq != self._seq:
            # A new record_loop call (episode or reset): close the previous policy
            # episode and start the integral from 0.
            self.finish_episode()
            self._seq = seq
            if phase == "policy":
                self._episode += 1
                self._in_policy = True
                self._reset_episode()
        if phase != "policy" or not self._in_policy:
            return theta_vel

        now = self._clock()
        if self._prev_t is not None:
            # The base held the previous command since the previous call.
            dt = min(max(now - self._prev_t, 0.0), _MAX_DT_S)
            self.net_rad += self._prev_theta * dt
        self._prev_t = now
        self.ticks += 1

        sent = theta_vel
        over = abs(self.net_rad) >= self.budget_rad
        # Same sign as the net turn (or exactly at 0 net, which cannot be over budget).
        pushes_further = theta_vel * self.net_rad > 0
        if over and pushes_further:
            sent = 0.0
            self.suppressed += 1
            if self.fired_tick is None:
                self.fired_tick = self.ticks
                self.fired_net_rad = self.net_rad
                logger.warning(
                    f"[base_budget] 발동 — policy 구간 {self._episode} tick {self.ticks}: "
                    f"net {math.degrees(self.net_rad):+.1f}° ≥ 예산 "
                    f"{math.degrees(self.budget_rad):.0f}°, 같은 방향 theta.vel 을 0 으로 (x.vel·팔 그대로)"
                )
        self._prev_theta = sent
        return sent


def _create() -> BaseThetaBudget | None:
    budget_deg = parse_budget_deg(os.getenv(ENV_VAR, ""))
    if budget_deg is None:
        return None
    from lerobot_robot_trossen import loop_rate_log

    governor = BaseThetaBudget(
        budget_deg, loop_rate_log.current_phase, loop_rate_log.current_phase_seq
    )
    if not loop_rate_log.LOOP_HZ_LOG_ENABLED:
        logger.error(
            f"[base_budget] {ENV_VAR}={budget_deg:g} 인데 LEROBOT_LOOP_HZ_LOG=0 이라 "
            "phase 태그가 없다 — 예산이 한 번도 걸리지 않는다. 이 회차는 「끔」 으로 기록하라."
        )
    else:
        logger.warning(
            f"[base_budget] 켜짐: policy 구간 에피소드마다 순회전 |net| ≤ {budget_deg:g}° "
            "(넘는 같은 방향 theta.vel → 0, 되돌리는 방향·x.vel·팔은 그대로)"
        )
    return governor


_GOVERNOR = _create()


def govern_theta(theta_vel: float) -> float:
    """Pass-through unless LEROBOT_BASE_THETA_BUDGET_DEG is set."""
    if _GOVERNOR is None:
        return theta_vel
    return _GOVERNOR.apply(theta_vel)


def finish() -> None:
    """Log the last policy episode's summary (called from disconnect)."""
    if _GOVERNOR is None:
        return
    _GOVERNOR.finish_episode()
    if _GOVERNOR._episode == 0:
        # Switched on but never saw a policy tick (phase tagging off or failed):
        # the run must not read as "budget on, never fired".
        logger.error(
            "[base_budget] 켜졌지만 policy 구간을 한 번도 보지 못했다 — 예산이 걸리지 않은 회차다. 「끔」 으로 기록하라."
        )
