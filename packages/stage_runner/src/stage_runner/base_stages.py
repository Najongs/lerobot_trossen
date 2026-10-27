"""The ``base_replay`` and ``base_align`` stages: the 10/24 field loop as chain stages.

``executors.run_base_replay_stage`` / ``run_base_align_stage`` run the shared
flag helpers and then call :func:`replay` / :func:`align` here. This module
never imports ``executors`` (the dispatch table imports it), and it keeps its
top-level imports free of ``landmark_align``: ``executors`` is imported by every
run, and a YAML without a base_align stage has no use for the estimator, so it
is imported inside the align path only. (``cv2`` itself is no reason: lerobot's
camera modules already import it on every run.)

What lives here: the constants every piece shares -- the executors' timing and
safety thresholds, and the stage ceiling floors that preflight and base_assets
check a YAML against, so the gate and the loop cannot disagree about how long a
stage may need -- and the two executors, :func:`replay` and :func:`align`.

WHAT EVERY BASE STAGE DOES THE SAME WAY, wherever in the chain it sits and
however often it runs:

* it anchors the odometry at its OWN start (the tracker's reset), so its targets
  mean the same motion whatever the previous stages did to the base;
* it runs in base_loop.BaseTickLoop, so every tick is one recorded frame under
  the stage's task string -- the episode stays one continuous chain;
* on every exit it zeroes the base directly first (no new observation, no
  recorded frame), then restores robot.base, then writes its side files and its
  one stage event. The runner's boundary stop_base follows as for any stage;
* it reads its targets from context.base_assets, loaded before connect -- no
  dataset I/O between stages.

The pose tracker (lerobot_robot_trossen.base_pose_control, PR 69's) is imported
inside the executors: stage_runner does not declare the Trossen package, and
preflight.check_base_stages refuses a base stage when it cannot be imported.

PROVENANCE of the numbers. "field" = the value stage1_align.py / heading_align.py
/ base_pose_step_test.run_track ran with on the robot on 2026-10-24 (vault
.claude/scripts/heading-align/). "new guard" = a check the field script did not
have; its threshold is a first guess, kept as a named constant so the first
robot run can tune it from the stage events, where every such abort is logged
with the value that tripped it.
"""

from __future__ import annotations

import csv
import logging
import math
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from stage_runner import record_adapter, transitions
from stage_runner.base_loop import (
    KEY_RERECORD,
    KEY_RIGHT,
    KEY_STOP,
    BaseTickLoop,
    stop_image_writer_after,
)
from stage_runner.events import (
    EVENT_STAGE_BASE_ALIGN,
    EVENT_STAGE_BASE_ALIGN_STEP,
    EVENT_STAGE_BASE_REPLAY,
    wall_clock_iso,
)
from stage_runner.results import (
    TERMINATED_BY_BASE_FAILED,
    TERMINATED_BY_CONVERGED,
    TERMINATED_BY_ERROR,
    TERMINATED_BY_MANUAL,
    TERMINATED_BY_RERECORD_REQUESTED,
    TERMINATED_BY_STOP_RECORDING,
    TERMINATED_BY_TIMEOUT,
    StageResult,
)

if TYPE_CHECKING:
    from stage_runner.config import StageConfig
    from stage_runner.context import StageContext

logger = logging.getLogger(__name__)

# The module the executors drive the base with. Named once, so preflight's
# import check and the executors' import cannot drift apart.
TRACKER_MODULE: str = "lerobot_robot_trossen.base_pose_control"

# Side files go under run_directory / BASE_SIDE_DIR.
BASE_SIDE_DIR: str = "base"

# --------------------------------------------------------------------------
# Shared base-motion thresholds (both executors).

# A base is "still" when |measured forward velocity| < 0.005 m/s and |measured
# turn rate| < 0.01 rad/s. field: stage1_align's replay hold test.
STILL_LINEAR_M_S: float = 0.005
STILL_ANGULAR_RAD_S: float = 0.01

# More than this many CONSECUTIVE set_cmd_vel refusals ends the stage
# (base_refused). field: stage1_align `failures > 10` -> "base refuses commands
# (e-stop?)". Consecutive, not total: one dropped serial write is not a fault.
CMD_REFUSED_LIMIT: int = 10

# More than this many consecutive ticks without a usable odometry read ends the
# stage (no_odometry). new guard: the field only warned. The tracker commands
# zero on a tick with no odometry, so a short gap is harmless; a long one means
# the base is driving blind on its last command chain, which the guard stops.
# 10 ticks is about 0.5 s at the measured 20 Hz.
ODOM_BAD_LIMIT: int = 10

# Tracking error that, held continuously for TRACK_ERROR_HOLD_S, ends the stage
# (tracking_lost). new guard. A stalled base (e-stop pressed, wheel blocked)
# leaves the target running away while the P term grows; released at that
# point the base would lunge to catch up. The error is |raw target - tracker
# pose|, the definition the 10/24 run_track logs use, and it is mostly the
# base's dead time times the target's rate, so it grows with the move.
# Measured on 10/24 (hlog/20261024-heading-align): the stage-1 replays peaked
# at 6.4 deg, so TRACK_ERROR_DEG (10) leaves the replay about 3.6 deg; the
# align turns peaked at 9.1 deg (turn +12.44 and -12.98), so the same 10 deg
# would leave them under 1 deg -- which is why align moves use their own,
# larger MOVE_TRACK_ERROR_DEG. 0.5 s rides out one bad read and the short
# lag peak at a ramp's start; it, not the angle, is what keeps a slightly
# slower base from tripping the guard. Tune from the max_tracking_error_*
# fields of the first robot run, and do NOT tighten either angle below what a
# normal run reaches plus a few degrees.
TRACK_ERROR_M: float = 0.10
TRACK_ERROR_DEG: float = 10.0
TRACK_ERROR_HOLD_S: float = 0.5
# The angular limit for base_align moves (turns and drives). Align turns start
# as a velocity step at TURN_RATE_RAD_S (0.6 rad/s = 34 deg/s), so every tick
# of base dead time costs about 1.7 deg of lag: the 10/24 turns reached 9.1
# deg, and a delay-plus-first-order model fitted to those logs (1 tick delay,
# tau 0.21 s) saturates at 9.5 deg for the largest turns now allowed (20 deg
# fine, 30 deg coarse), and exceeds 10 deg for 0.24-0.44 s with one more tick
# of delay or tau 0.3 s. 15 deg keeps about 5 deg above the field peak and
# still catches a stalled base on any turn larger than 15 deg about 0.9 s
# into the move, before the ramp-plus-hold ends; a stall on a smaller turn
# never reaches it and is caught as move_failed after the hold instead.
MOVE_TRACK_ERROR_DEG: float = 15.0

# --------------------------------------------------------------------------
# base_replay.

# Tracker gains for the replay. field: stage1_align.replay's BasePoseGains(
# kp_x=2.0, kp_theta=2.0, max_x_vel=1.0, max_theta_vel=1.0, target_window=3),
# the pose-mode eval's gains. Plain numbers here, not a BasePoseGains: the
# tracker module is imported inside the executor (see the module doc).
REPLAY_KP_X: float = 2.0
REPLAY_KP_THETA: float = 2.0
REPLAY_MAX_X_VEL: float = 1.0
REPLAY_MAX_THETA_VEL: float = 1.0
REPLAY_TARGET_WINDOW: int = 3
# The fastest a replay's TARGETS may move: max |row-to-row step| / row_period_s
# of the loaded rows, checked in base_assets before connect. new guard. The
# target speed scales with 1 / row_period_s, so a typo there (or a pose dataset
# denser than one row per ~21 Hz tick) would turn the replay into a full-speed
# drive and turn, clipped only by the tracker's 1.0 m/s / 1.0 rad/s clamp, and
# the only thing left to stop it would be tracking_lost 0.5 s later. The
# fastest replay run on the robot (10/24 f1 / f3, episode 0 rows 0..297 at
# 1/21 s) peaked at 0.49 m/s and 0.57 rad/s; these limits are about 1.2x that,
# below the clamp. A demo whose own base moved faster needs a longer
# row_period_s (the refusal names the one that fits), or a deliberate raise
# here after it has been watched on the robot.
REPLAY_TARGET_MAX_X_VEL_M_S: float = 0.6
REPLAY_TARGET_MAX_THETA_VEL_RAD_S: float = 0.7

# Hold ticks recorded before the first motion tick, from which the odometry
# anchor and the nominal tick period are taken. Replaces the field's 2 s
# countdown: the chain is already running and the operator is at the e-stop.
PREROLL_S: float = 0.5
# Past the last target row the stage holds the final target and ends "settled"
# once the base has been still for REPLAY_STILL_S, or "not_settled" after
# REPLAY_SETTLE_MAX_S. field: stage1_align `still >= 0.5 * REPLAY_HZ` and
# `hold_max = 4.0 * REPLAY_HZ`.
REPLAY_STILL_S: float = 0.5
REPLAY_SETTLE_MAX_S: float = 4.0
# Final |target - reached| above either ends the stage "missed_target". new
# guard: a replay that stops far from the demo's end hands the align stage a
# view it may not recover from. 3 deg / 5 cm are first guesses.
REPLAY_MAX_FINAL_ERROR_RAD: float = math.radians(3.0)
REPLAY_MAX_FINAL_ERROR_M: float = 0.05
# Seconds of ceiling beyond the computed need, so a few slow ticks do not turn
# a good replay or align into a stage_ceiling abort.
CEILING_MARGIN_S: float = 2.0

# --------------------------------------------------------------------------
# base_align.

# Align moves are ramps at these rates. field: stage1_align.correct() (turn
# rate 0.6 rad/s, drive rate 0.1 m/s, kp 2.0, max_vel 1.0) via run_track.
TURN_RATE_RAD_S: float = 0.6
DRIVE_RATE_M_S: float = 0.1
MOVE_KP_X: float = 2.0
MOVE_MAX_VEL: float = 1.0
MOVE_TARGET_WINDOW: int = 3
# Fixed hold after the ramp, then stop. field: correct(settle_seconds=1.0).
MOVE_HOLD_S: float = 1.0
# Longest ramp a single move can need: the largest turn the estimates allow
# (the coarse limit, 30 deg, above the fine 20 deg) at TURN_RATE_RAD_S, or the
# largest drive (0.10 m) at DRIVE_RATE_M_S -- max(0.87 s, 1.0 s). Hard-coded so
# this module does not import landmark_align (and cv2); a test pins it against
# landmark_align's limits.
MOVE_RAMP_MAX_S: float = 1.0
# A capture waits until the base has been still for CAPTURE_DELAY_S, and gives
# up ("not_still") after CAPTURE_STILL_MAX_S. Replaces the field's fixed drop
# of 8 camera frames (about 0.27 s) after the move's hold.
CAPTURE_DELAY_S: float = 0.3
CAPTURE_STILL_MAX_S: float = 3.0
# Allowance per measurement in the ceiling floor only. One measurement took
# about 0.5 s in the office; on the robot it is unmeasured, hence 3 s.
MEASURE_BUDGET_S: float = 3.0
# Ceiling-floor allowance for the hold ticks before the first capture.
ALIGN_ENTRY_S: float = 0.5
# A correction that grows the error by more than this ends the stage
# ("diverging"). field: stage1_align `abs(yaw) > abs(last) + 0.5` and
# `abs(move) > abs(last) + 0.005`.
DIVERGE_YAW_DEG: float = 0.5
DIVERGE_MOVE_M: float = 0.005
# A move that achieved less than this fraction of its command ends the stage
# ("move_failed": e-stop or base torque off). new guard; a negative fraction is
# reported as "moved opposite to command".
MIN_MOVE_FRACTION: float = 0.2
# With min_turn_rate_rad_s on, the rate floor applies only while the turn is
# more than this short of its target, so it cannot hunt around the target.
MIN_RATE_DEADBAND_RAD: float = math.radians(0.5)
# A coarse heading estimate turns the base and is measured again; after more
# than this many coarse turns the stage ends "estimate_unusable". field:
# stage1_align `coarse_turns > 2` -> STOPPED_UNRELIABLE. Each coarse turn also
# counts toward max_moves, as in the field.
MAX_COARSE_TURNS: int = 2
# A move smaller than this (turn) or this (drive) is not judged by the
# MIN_MOVE_FRACTION / sign rule, only logged. Only a COARSE turn can be that
# small -- a fine turn is larger than yaw_tol_deg and a drive larger than
# dist_tol_cm, or no move is made -- and the coarse estimate turns even when it
# reads inside tolerance (field rule). Odometry noise on a 0.3 deg turn can read
# as a negative fraction; the fine measurement after it is the real check.
MOVE_JUDGE_MIN_RAD: float = math.radians(1.0)
MOVE_JUDGE_MIN_M: float = 0.005


def replay_ceiling_floor_s(rows: int, row_period_s: float) -> float:
    """The shortest ``terminator.timeout_s`` a replay of ``rows`` target rows can live with.

    Pre-roll, the target rows at their period, the settle hold, and the margin.
    A ceiling below it would cut a healthy replay off as stage_ceiling, which
    aborts the trial -- so base_assets refuses the YAML instead.
    """
    return (
        PREROLL_S
        + max(rows - 1, 0) * row_period_s
        + REPLAY_SETTLE_MAX_S
        + CEILING_MARGIN_S
    )


def align_ceiling_floor_s(max_moves: int) -> float:
    """The shortest ``terminator.timeout_s`` an align with ``max_moves`` can live with.

    Every one of the max_moves + 1 measurements may wait CAPTURE_STILL_MAX_S for
    stillness and take MEASURE_BUDGET_S, and every one of the max_moves moves
    takes at most MOVE_RAMP_MAX_S plus MOVE_HOLD_S. 56.5 s at max_moves 6.
    """
    return (
        ALIGN_ENTRY_S
        + (max_moves + 1) * (CAPTURE_STILL_MAX_S + MEASURE_BUDGET_S)
        + max_moves * (MOVE_RAMP_MAX_S + MOVE_HOLD_S)
        + CEILING_MARGIN_S
    )


# --------------------------------------------------------------------------
# Outcomes. Closed, because a reader buckets on them: the ``outcome`` field of
# the stage event says WHICH of the terminator's causes it was (converged is a
# settled replay; base_failed is any of the guards). Shared names are shared
# meanings across both executors.

OUTCOME_SETTLED: str = "settled"
OUTCOME_NOT_SETTLED: str = "not_settled"
OUTCOME_NO_ODOMETRY: str = "no_odometry"
OUTCOME_BASE_REFUSED: str = "base_refused"
OUTCOME_BASE_ESTOP: str = "base_estop"
OUTCOME_TRACKING_LOST: str = "tracking_lost"
OUTCOME_MISSED_TARGET: str = "missed_target"
OUTCOME_STAGE_CEILING: str = "stage_ceiling"
OUTCOME_MANUAL: str = "manual"
OUTCOME_ABORTED: str = "aborted"
OUTCOME_INTERRUPTED: str = "interrupted"

REPLAY_OUTCOMES: tuple[str, ...] = (
    OUTCOME_SETTLED,
    OUTCOME_NOT_SETTLED,
    OUTCOME_NO_ODOMETRY,
    OUTCOME_BASE_REFUSED,
    OUTCOME_BASE_ESTOP,
    OUTCOME_TRACKING_LOST,
    OUTCOME_MISSED_TARGET,
    OUTCOME_STAGE_CEILING,
    OUTCOME_MANUAL,
    OUTCOME_ABORTED,
    OUTCOME_INTERRUPTED,
)

# terminated_by for every outcome but "aborted" (which key) and "interrupted"
# (the exception propagates; the runner records "error").
# not_settled is a timeout, not a failure: the align stage after it measures
# where the base actually is, as the field did after a replay that did not
# settle in its 4 s.
_REPLAY_TERMINATED_BY: dict[str, str] = {
    OUTCOME_SETTLED: TERMINATED_BY_CONVERGED,
    OUTCOME_NOT_SETTLED: TERMINATED_BY_TIMEOUT,
    OUTCOME_NO_ODOMETRY: TERMINATED_BY_BASE_FAILED,
    OUTCOME_BASE_REFUSED: TERMINATED_BY_BASE_FAILED,
    OUTCOME_BASE_ESTOP: TERMINATED_BY_BASE_FAILED,
    OUTCOME_TRACKING_LOST: TERMINATED_BY_BASE_FAILED,
    OUTCOME_MISSED_TARGET: TERMINATED_BY_BASE_FAILED,
    OUTCOME_STAGE_CEILING: TERMINATED_BY_BASE_FAILED,
    OUTCOME_MANUAL: TERMINATED_BY_MANUAL,
}
_TERMINATED_BY_KEY: dict[str, str] = {
    KEY_STOP: TERMINATED_BY_STOP_RECORDING,
    KEY_RERECORD: TERMINATED_BY_RERECORD_REQUESTED,
}

OUTCOME_ALIGNED: str = "aligned"
OUTCOME_HEADING_ONLY: str = "heading_only"
OUTCOME_ESTIMATE_UNUSABLE: str = "estimate_unusable"
OUTCOME_DIVERGING: str = "diverging"
OUTCOME_NOT_CONVERGED: str = "not_converged"
OUTCOME_MOVE_FAILED: str = "move_failed"
OUTCOME_NOT_STILL: str = "not_still"
OUTCOME_INTRINSICS_UNAVAILABLE: str = "intrinsics_unavailable"

ALIGN_OUTCOMES: tuple[str, ...] = (
    OUTCOME_ALIGNED,
    OUTCOME_HEADING_ONLY,
    OUTCOME_ESTIMATE_UNUSABLE,
    OUTCOME_DIVERGING,
    OUTCOME_NOT_CONVERGED,
    OUTCOME_MOVE_FAILED,
    OUTCOME_NOT_STILL,
    OUTCOME_NO_ODOMETRY,
    OUTCOME_BASE_REFUSED,
    OUTCOME_BASE_ESTOP,
    OUTCOME_TRACKING_LOST,
    OUTCOME_INTRINSICS_UNAVAILABLE,
    OUTCOME_STAGE_CEILING,
    OUTCOME_MANUAL,
    OUTCOME_ABORTED,
    OUTCOME_INTERRUPTED,
)

# heading_only is converged, not a failure: heading inside tolerance with a
# distance that cannot be read is the field's RESULT=HEADING_ONLY (exit 0), and
# the chain goes on (user decision A3; pi0 went 2/3 after it on 10/24). The
# three MEASUREMENT failures also go on, from wherever the base stopped, with
# the outcome and reason kept in stage_base_align (user decision 2026-10-25: in
# front of the fridge a real 3-4 deg error read as "diverging" and stopped t05;
# the operator can still end the trial with ESC). The safety failures below
# still stop the chain before the next policy runs on a base nobody trusts.
_ALIGN_TERMINATED_BY: dict[str, str] = {
    OUTCOME_ALIGNED: TERMINATED_BY_CONVERGED,
    OUTCOME_HEADING_ONLY: TERMINATED_BY_CONVERGED,
    OUTCOME_ESTIMATE_UNUSABLE: TERMINATED_BY_CONVERGED,
    OUTCOME_DIVERGING: TERMINATED_BY_CONVERGED,
    OUTCOME_NOT_CONVERGED: TERMINATED_BY_CONVERGED,
    OUTCOME_MOVE_FAILED: TERMINATED_BY_BASE_FAILED,
    OUTCOME_NOT_STILL: TERMINATED_BY_BASE_FAILED,
    OUTCOME_NO_ODOMETRY: TERMINATED_BY_BASE_FAILED,
    OUTCOME_BASE_REFUSED: TERMINATED_BY_BASE_FAILED,
    OUTCOME_BASE_ESTOP: TERMINATED_BY_BASE_FAILED,
    OUTCOME_TRACKING_LOST: TERMINATED_BY_BASE_FAILED,
    OUTCOME_INTRINSICS_UNAVAILABLE: TERMINATED_BY_BASE_FAILED,
    OUTCOME_STAGE_CEILING: TERMINATED_BY_BASE_FAILED,
    OUTCOME_MANUAL: TERMINATED_BY_MANUAL,
}

# The align stage's states (the `state` column of its CSV).
STATE_WAIT_CAPTURE: str = "wait_capture"
STATE_MEASURING: str = "measuring"
STATE_MOVING: str = "moving"

# The two kinds of align move.
MOVE_TURN: str = "turn"
MOVE_DRIVE: str = "drive"

# Columns of run_directory/base/<stage_id>_align.csv, one row per recorded
# frame: the state the tick was in, the move's ramp target and the tracker's
# command, and the base's own readings -- what a person tuning the new guards
# after the first robot run needs next to the step events.
ALIGN_CSV_COLUMNS: tuple[str, ...] = (
    "t",
    "state",
    "step",
    "move",
    "target_s",
    "target_theta",
    "cmd_s",
    "cmd_theta",
    "cmd_x_vel",
    "cmd_theta_vel",
    "meas_x_vel",
    "meas_theta_vel",
    "odom_x",
    "odom_y",
    "odom_theta",
    "odom_ok",
    "update_ok",
    "cmd_ok",
)

# Columns of run_directory/base/<stage_id>_replay.csv, one row per tick. The
# replay's odometry record: rate-meter and base_vel_log rows from this loop
# carry phase None, so per-stage analysis reads this file.
REPLAY_CSV_COLUMNS: tuple[str, ...] = (
    "t",
    "phase",
    "r",
    "target_s",
    "target_theta",
    "cmd_s",
    "cmd_theta",
    "cmd_x_vel",
    "cmd_theta_vel",
    "meas_x_vel",
    "meas_theta_vel",
    "odom_x",
    "odom_y",
    "odom_theta",
    "odom_ok",
    "update_ok",
    "cmd_ok",
    "e_s",
    "e_theta",
)


def import_tracker() -> Any:
    """The pose tracker module, or ImportError naming what to install."""
    import importlib

    try:
        return importlib.import_module(TRACKER_MODULE)
    except ImportError as error:
        raise ImportError(
            f"the base stages drive the base with {TRACKER_MODULE} "
            "(lerobot_robot_trossen, PR 69's tracker); it is not importable "
            f"here: {error}"
        ) from error


class _Ended(Exception):
    """Raised inside a stage's tick loop to leave it with an outcome. Never escapes."""

    def __init__(self, outcome: str, reason: str, key: str | None = None) -> None:
        super().__init__(reason)
        self.outcome = outcome
        self.reason = reason
        self.key = key


class _Replay:
    """One base_replay stage's run: the tick loop, the tracker and everything logged.

    A class only to keep the state the tick loop, the exit path and the event
    share in one place; it lives for one call of :func:`replay`.
    """

    def __init__(self, context: StageContext, stage: StageConfig, targets: Any) -> None:
        self.context = context
        self.stage = stage
        self.replay = stage.replay
        self.s_m = [float(value) for value in targets.s_m]
        self.theta_rad = [float(value) for value in targets.theta_rad]
        self.rows = len(self.s_m)
        self.source = dict(targets.source)
        self.row_period_s = float(self.replay.row_period_s)
        self.plan = record_adapter.plan_stage(stage, context.config.defaults)
        self.tracker: Any = None
        self.loop: BaseTickLoop | None = None
        self.csv_rows: list[list[Any]] = []
        self.anchor: tuple[float, float, float] | None = None
        self.nominal_dt: float | None = None
        self.t0: float | None = None
        self.hold_started: float | None = None
        self.ended_at: float | None = None
        self.still_since: float | None = None
        self.track_bad_since: float | None = None
        self.odom_bad_run = 0
        self.max_odom_bad_run = 0
        self.odom_bad_ticks = 0
        self.max_error_m = 0.0
        self.max_error_deg = 0.0
        self.last_error: tuple[float, float] | None = None

    # ---------------------------------------------------------------- ticks

    def run(self) -> None:
        """Pre-roll, anchor, motion and hold; ends by raising _Ended."""
        tracker_module = import_tracker()
        loop = self.loop
        last_odom: tuple[float, float, float] | None = None
        while True:
            self._keys()
            observation, odom, now = loop.observe()
            self._guards(now)
            if odom is not None:
                last_odom = odom
            if now - loop.started >= PREROLL_S:
                break
            loop.act(observation, 0.0, 0.0)
            self._row(now, "preroll", None, None, None, None, odom, None)
        # The tick that crossed PREROLL_S is the first motion tick: it has
        # observed but not acted yet.
        if last_odom is None:
            # Sent nothing for this tick; the exit path zeroes the base.
            raise _Ended(
                OUTCOME_NO_ODOMETRY,
                f"no usable odometry in {loop.ticks} pre-roll ticks "
                f"({PREROLL_S:.1f}s): update_state failed or get_pose was not "
                "finite, so there is no anchor to replay from",
            )
        self.anchor = last_odom
        self.nominal_dt = loop.median_period_s()
        self.tracker = tracker_module.BasePoseTracker(
            tracker_module.BasePoseGains(
                kp_x=REPLAY_KP_X,
                kp_theta=REPLAY_KP_THETA,
                max_x_vel=REPLAY_MAX_X_VEL,
                max_theta_vel=REPLAY_MAX_THETA_VEL,
                target_window=REPLAY_TARGET_WINDOW,
            )
        )
        self.tracker.reset(self.anchor, self.nominal_dt)
        self.t0 = now
        logger.info(
            f"stage {self.stage.id!r}: replaying {self.rows} target rows at "
            f"{self.row_period_s * 1e3:.1f} ms/row (to "
            f"{self.s_m[-1]:+.3f} m, {math.degrees(self.theta_rad[-1]):+.1f} deg), "
            f"anchored at {self._fmt(self.anchor)}, nominal_dt "
            f"{self.nominal_dt * 1e3:.1f} ms"
        )
        first = True
        while True:
            if not first:
                self._keys()
                observation, odom, now = loop.observe()
                self._guards(now)
            first = False
            self._motion_tick(observation, odom, now)

    def _motion_tick(
        self,
        observation: dict[str, Any],
        odom: tuple[float, float, float] | None,
        now: float,
    ) -> None:
        loop = self.loop
        r = (now - self.t0) / self.row_period_s
        target_s, target_theta = self._target(r)
        cmd = self.tracker.command(target_s, target_theta, odom, now)
        if cmd.odom_ok:
            self.odom_bad_run = 0
        else:
            self.odom_bad_ticks += 1
            self.odom_bad_run += 1
            self.max_odom_bad_run = max(self.max_odom_bad_run, self.odom_bad_run)
            if self.odom_bad_run > ODOM_BAD_LIMIT:
                raise _Ended(
                    OUTCOME_NO_ODOMETRY,
                    f"{self.odom_bad_run} consecutive ticks without usable "
                    f"odometry (limit {ODOM_BAD_LIMIT}); the tracker was "
                    "commanding zero on each",
                )
        e_s = e_theta = None
        if cmd.odom_ok:
            e_s = abs(target_s - cmd.s)
            e_theta = abs(_wrap(target_theta - cmd.theta))
            self.last_error = (e_s, e_theta)
            self.max_error_m = max(self.max_error_m, e_s)
            self.max_error_deg = max(self.max_error_deg, math.degrees(e_theta))
            if e_s > TRACK_ERROR_M or math.degrees(e_theta) > TRACK_ERROR_DEG:
                if self.track_bad_since is None:
                    self.track_bad_since = now
                elif now - self.track_bad_since >= TRACK_ERROR_HOLD_S:
                    raise _Ended(
                        OUTCOME_TRACKING_LOST,
                        f"tracking error {e_s * 100:.1f} cm / "
                        f"{math.degrees(e_theta):.1f} deg held for "
                        f"{now - self.track_bad_since:.2f}s (limits "
                        f"{TRACK_ERROR_M * 100:.0f} cm / {TRACK_ERROR_DEG:.0f} deg "
                        f"for {TRACK_ERROR_HOLD_S:.1f}s): the base is not "
                        "following -- stalled, e-stopped or torque off?",
                    )
            else:
                self.track_bad_since = None
        loop.act(observation, cmd.x_vel, cmd.theta_vel)
        holding = r >= self.rows - 1
        self._row(
            now,
            "hold" if holding else "motion",
            r,
            (target_s, target_theta),
            cmd,
            (e_s, e_theta),
            odom,
            loop.probe.last_cmd_ok,
        )
        if not holding:
            return
        if self.hold_started is None:
            self.hold_started = now
        velocity = loop.read_base_velocity()
        still = (
            velocity is not None
            and abs(velocity[0]) < STILL_LINEAR_M_S
            and abs(velocity[1]) < STILL_ANGULAR_RAD_S
        )
        if not still:
            self.still_since = None
        elif self.still_since is None:
            self.still_since = now
        if self.still_since is not None and now - self.still_since >= REPLAY_STILL_S:
            raise _Ended(
                OUTCOME_SETTLED,
                f"still for {now - self.still_since:.2f}s after "
                f"{now - self.hold_started:.2f}s of hold",
            )
        if now - self.hold_started >= REPLAY_SETTLE_MAX_S:
            raise _Ended(
                OUTCOME_NOT_SETTLED,
                f"not still {REPLAY_SETTLE_MAX_S:.1f}s after the last target row "
                f"(last measured velocity {velocity})",
            )

    def _target(self, r: float) -> tuple[float, float]:
        """The target at fractional row ``r``, linearly interpolated in time.

        CHOICE: by time, not by tick. This loop ticks at the hardware rate (about
        20 Hz measured), not at a paced 21 Hz, so indexing a row per tick would
        tie the replay's speed to the loop rate.
        """
        last = self.rows - 1
        if r >= last:
            return self.s_m[last], self.theta_rad[last]
        r = max(r, 0.0)
        index = int(r)
        fraction = r - index
        return (
            self.s_m[index] + fraction * (self.s_m[index + 1] - self.s_m[index]),
            self.theta_rad[index]
            + fraction * (self.theta_rad[index + 1] - self.theta_rad[index]),
        )

    def _keys(self) -> None:
        _check_keys(self.loop, "the replay")

    def _guards(self, now: float) -> None:
        _check_guards(self.loop, now, "the replay settled")

    def _row(
        self,
        now: float,
        phase: str,
        r: float | None,
        target: tuple[float, float] | None,
        cmd: Any,
        error: tuple[float | None, float | None] | None,
        odom: tuple[float, float, float] | None,
        cmd_ok: bool | None,
    ) -> None:
        loop = self.loop
        velocity = loop.read_base_velocity()
        self.csv_rows.append(
            [
                now - loop.started,
                phase,
                r,
                *(target or (None, None)),
                *((cmd.s, cmd.theta, cmd.x_vel, cmd.theta_vel) if cmd else (None,) * 4),
                *(velocity or (None, None)),
                *(odom or (None, None, None)),
                None if cmd is None else cmd.odom_ok,
                loop.probe.last_update_ok,
                loop.probe.last_cmd_ok if cmd_ok is None else cmd_ok,
                *(error or (None, None)),
            ]
        )

    @staticmethod
    def _fmt(pose: tuple[float, float, float]) -> str:
        return f"({pose[0]:+.3f} m, {pose[1]:+.3f} m, {math.degrees(pose[2]):+.1f} deg)"

    # ----------------------------------------------------------------- exit

    def final_pose(self) -> tuple[float, float] | None:
        """(s, theta) the tracker last folded in, relative to the anchor."""
        if self.tracker is None:
            return None
        return float(self.tracker.s), float(self.tracker.theta)

    def write_csv(self) -> str | None:
        """Write the per-tick log; the run-relative path, or None. Never raises.

        A full disk must not turn a replay that already stopped the base into
        an exception that ends the trial.
        """
        relative = Path(BASE_SIDE_DIR) / f"{self.stage.id}_replay.csv"
        try:
            path = self.context.run_directory / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(REPLAY_CSV_COLUMNS)
                writer.writerows(self.csv_rows)
        except Exception:
            logger.exception(f"stage {self.stage.id!r}: could not write {relative}")
            return None
        return relative.as_posix()

    def fields(self) -> dict[str, Any]:
        """Every field of the stage_base_replay event except outcome and reason."""
        loop = self.loop
        final = self.final_pose()
        target_s, target_theta = self.s_m[-1], self.theta_rad[-1]
        ended = self.ended_at
        motion_s = hold_s = None
        if self.t0 is not None and ended is not None:
            motion_end = self.hold_started if self.hold_started is not None else ended
            motion_s = motion_end - self.t0
            hold_s = ended - self.hold_started if self.hold_started is not None else 0.0
        fields: dict[str, Any] = {
            "dataset": self.replay.dataset,
            "root": self.source.get("root"),
            "episode": self.replay.episode,
            "from_row": self.replay.from_row,
            "until_row": self.replay.until_row,
            "rows": self.rows,
            "row_period_s": self.row_period_s,
            "control_time_s": self.plan.control_time_s,
            "anchor_odom": list(self.anchor) if self.anchor is not None else None,
            "nominal_dt": self.nominal_dt,
            "final_target_s_m": target_s,
            "final_target_theta_deg": math.degrees(target_theta),
            "final_s_m": final[0] if final else None,
            "final_theta_deg": math.degrees(final[1]) if final else None,
            "final_error_m": abs(target_s - final[0]) if final else None,
            "final_error_deg": (
                math.degrees(abs(_wrap(target_theta - final[1]))) if final else None
            ),
            "motion_s": motion_s,
            "hold_s": hold_s,
            "ticks": loop.ticks if loop else 0,
            "odom_bad_ticks": self.odom_bad_ticks,
            "max_odom_bad_run": self.max_odom_bad_run,
            "max_tracking_error_m": self.max_error_m,
            "max_tracking_error_deg": self.max_error_deg,
            "base_state_last": loop.base_state_name() if loop else None,
            "held_arm_positions": loop.held_arm_positions if loop else None,
            "gripper_hold_source": loop.gripper_hold_source if loop else None,
        }
        if loop is not None:
            counters = loop.probe.counters()
            fields.update(
                update_failed_ticks=counters["update_failed_ticks"],
                cmd_refused_ticks=counters["cmd_refused_ticks"],
                max_cmd_refused_run=counters["max_cmd_refused_run"],
            )
            stats = loop.tick_stats()
            fields.update(
                tick_mean_s=stats["tick_mean_s"],
                tick_p95_s=stats["tick_p95_s"],
                tick_max_s=stats["tick_max_s"],
                slow_ticks=stats["slow_ticks"],
            )
        return fields


def _check_keys(loop: BaseTickLoop, during: str) -> None:
    """End the stage on an operator key: Esc / left arrow abort, right arrow accepts."""
    key = loop.poll_keys()
    if key is None:
        return
    if key == KEY_RIGHT:
        raise _Ended(
            OUTCOME_MANUAL,
            "right arrow: the operator accepted the base where it is",
        )
    raise _Ended(
        OUTCOME_ABORTED,
        f"{'esc' if key == KEY_STOP else 'left arrow'} pressed during {during}",
        key,
    )


def _check_guards(loop: BaseTickLoop, now: float, before: str) -> None:
    """The probe guards and the ceiling, checked right after each observation.

    Shared by both executors and run on EVERY tick, holds included: the zero
    commands of a hold go through set_cmd_vel too, and an e-stop pressed while
    the base stands still must end the stage before the next move starts.
    """
    fault = loop.base_state_fault()
    if fault is not None:
        raise _Ended(
            OUTCOME_BASE_ESTOP,
            f"the base reports {fault}; stopped before the release can turn "
            "the tracker's error into a lunge",
        )
    if loop.probe.cmd_refused_run > CMD_REFUSED_LIMIT:
        raise _Ended(
            OUTCOME_BASE_REFUSED,
            f"{loop.probe.cmd_refused_run} consecutive set_cmd_vel refusals "
            f"(limit {CMD_REFUSED_LIMIT}): base refuses commands (e-stop?)",
        )
    if loop.ceiling_reached(now):
        raise _Ended(
            OUTCOME_STAGE_CEILING,
            f"the stage ceiling terminator.timeout_s={loop.ceiling_s:.1f}s was "
            f"reached before {before}",
        )


def _wrap(angle: float) -> float:
    """[-pi, pi), the tracker's wrap_angle (kept local: no tracker import at module level)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _stage_asset(context: StageContext, stage: StageConfig) -> Any:
    try:
        return context.base_assets[stage.id]
    except KeyError:
        # cli loads every base stage's assets before connect; reaching this
        # means a caller built a StageContext without them.
        raise RuntimeError(
            f"stage {stage.id!r}: no base assets were loaded for it "
            "(base_assets.load_base_assets runs in cli before connect)"
        ) from None


def _zero_base(context: StageContext, stage: StageConfig) -> str:
    """Zero the base directly, logging a failure. Returns the problem or ""."""
    problem = transitions.zero_base_directly(context.robot)
    if problem:
        logger.error(
            f"stage {stage.id!r}: COULD NOT ZERO THE BASE DIRECTLY ({problem}); the "
            "runner's boundary stop_base follows"
        )
    return problem


def replay(context: StageContext, stage: StageConfig) -> StageResult:
    """Drive the base along a demo's base path with the arms held, recording every tick.

    1. PRE-ROLL: PREROLL_S of recorded hold ticks; the last usable odometry is
       the anchor and the median tick period the tracker's nominal_dt.
    2. MOTION: each tick's target is the demo path interpolated at
       (now - t0) / row_period_s, followed by PR 69's BasePoseTracker with the
       field's gains (stage1_align.replay).
    3. HOLD: past the last row the last target is held until the base has been
       still for REPLAY_STILL_S ("settled", converged) or for at most
       REPLAY_SETTLE_MAX_S ("not_settled", timeout -- the align stage after it
       still measures, as in the field). A settle that ends further than
       REPLAY_MAX_FINAL_ERROR_* from the last target is "missed_target".

    Guards, each ending the stage base_failed with the value that tripped it:
    the base reporting an e-stop or error state, more than CMD_REFUSED_LIMIT
    consecutive refused commands (the field's rule), more than ODOM_BAD_LIMIT
    consecutive ticks without odometry, a tracking error held past
    TRACK_ERROR_HOLD_S, and the stage ceiling. Esc / left arrow abort the trial;
    the right arrow accepts the base where it is (manual).

    Every exit zeroes the base directly first, then restores robot.base, then
    writes run_directory/base/<stage_id>_replay.csv and one stage_base_replay
    event. An exception does the same (minus the CSV), stops the image writer
    as record_loop's decorator would, and propagates unchanged.
    """
    if stage.replay is None:
        # preflight.check_stage_definitions makes this unreachable from cli.
        raise RuntimeError(f"stage {stage.id!r}: base_replay without a replay: block")
    run = _Replay(context, stage, _stage_asset(context, stage))

    def emit(outcome: str, reason: str, terminator: str | None, **extra: Any) -> None:
        context.log.emit(
            EVENT_STAGE_BASE_REPLAY,
            stage_id=stage.id,
            frame_idx=context.buffered_frame_count(),
            terminator=terminator,
            reason=reason,
            outcome=outcome,
            **run.fields(),
            **extra,
        )

    try:
        run.loop = BaseTickLoop(context, stage, ceiling_s=run.plan.control_time_s)
        try:
            run.run()
        except _Ended as ended:
            outcome, reason, key = ended.outcome, ended.reason, ended.key
        run.ended_at = time.perf_counter()
        elapsed_s = run.loop.elapsed_s(run.ended_at)
    except BaseException as error:
        # HARDWARE FIRST, then the probe, then the record. No observation and
        # no frame: the tick that raised may be half done.
        zero_problem = _zero_base(context, stage)
        if run.tracker is not None:
            run.tracker.release()
        if run.loop is not None:
            run.loop.close()
        stop_image_writer_after(error, context.dataset)
        run.ended_at = time.perf_counter()
        try:
            emit(
                OUTCOME_INTERRUPTED,
                f"the replay raised {error!r}",
                TERMINATED_BY_ERROR,
                base_zero_error=zero_problem or None,
                csv=None,
            )
        except Exception:
            logger.exception(f"stage {stage.id!r}: could not record the interruption")
        raise

    zero_problem = _zero_base(context, stage)
    if run.tracker is not None:
        run.tracker.release()
    run.loop.close()

    final = run.final_pose()
    if outcome in (OUTCOME_SETTLED, OUTCOME_NOT_SETTLED) and final is not None:
        error_m = abs(run.s_m[-1] - final[0])
        error_rad = abs(_wrap(run.theta_rad[-1] - final[1]))
        if error_m > REPLAY_MAX_FINAL_ERROR_M or error_rad > REPLAY_MAX_FINAL_ERROR_RAD:
            reason = (
                f"{outcome} {error_m * 100:.1f} cm / {math.degrees(error_rad):.1f} deg "
                f"from the last target (limits {REPLAY_MAX_FINAL_ERROR_M * 100:.0f} "
                f"cm / {math.degrees(REPLAY_MAX_FINAL_ERROR_RAD):.0f} deg); {reason}"
            )
            outcome = OUTCOME_MISSED_TARGET
    terminated_by = (
        _TERMINATED_BY_KEY[key]
        if outcome == OUTCOME_ABORTED
        else _REPLAY_TERMINATED_BY[outcome]
    )
    csv_path = run.write_csv()
    emit(
        outcome,
        reason,
        terminated_by,
        base_zero_error=zero_problem or None,
        csv=csv_path,
    )
    frames = run.loop.frames
    log = logger.info if terminated_by != TERMINATED_BY_BASE_FAILED else logger.error
    log(
        f"stage {stage.id!r} ended: {outcome} ({terminated_by}) -- {reason} "
        f"({frames} frames in {elapsed_s:.2f}s)"
    )
    return StageResult(
        terminated_by=terminated_by,
        elapsed_s=elapsed_s,
        frames=frames,
        reason=f"{outcome}: {reason}",
    )


def device_intrinsics(camera: Any) -> dict[str, Any]:
    """The colour stream's intrinsics, read off a connected RealSense camera.

    lerobot's RealSenseCamera keeps the started pipeline's profile as
    ``rs_profile``; the colour stream's intrinsics are the ones the image the
    stage measures was taken with (camera_realsense.py builds that stream at
    the configured size, unrotated -- preflight refuses a rotation). Raises
    whatever goes wrong; the caller turns it into intrinsics_unavailable.
    ``pyrealsense2`` is imported here so a run without an align stage never
    loads it.
    """
    import pyrealsense2 as rs

    profile = getattr(camera, "rs_profile", None)
    if profile is None:
        raise RuntimeError(
            f"camera {type(camera).__name__} has no started RealSense profile"
        )
    intrinsics = (
        profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    )
    return {
        "camera_matrix": [
            [float(intrinsics.fx), 0.0, float(intrinsics.ppx)],
            [0.0, float(intrinsics.fy), float(intrinsics.ppy)],
            [0.0, 0.0, 1.0],
        ],
        # OpenCV reads the first five as Brown-Conrady k1 k2 p1 p2 k3; the
        # model is logged because RealSense's own model may differ (spec risk
        # 11 -- check on the first robot run).
        "dist_coeffs": [float(value) for value in list(intrinsics.coeffs)[:5]],
        "model": str(intrinsics.model),
        "width": int(intrinsics.width),
        "height": int(intrinsics.height),
    }


class _Move:
    """One align turn or drive: a ramp to ``delta`` at ``rate``, then a fixed hold."""

    def __init__(
        self,
        kind: str,
        delta: float,
        step: int,
        heading_mode: str | None,
        frame_idx_start: int,
    ) -> None:
        self.kind = kind
        # Radians for a turn, metres for a drive; signed.
        self.delta = float(delta)
        self.step = step
        self.heading_mode = heading_mode
        self.rate = TURN_RATE_RAD_S if kind == MOVE_TURN else DRIVE_RATE_M_S
        self.ramp_s = abs(self.delta) / self.rate
        self.frame_idx_start = frame_idx_start
        self.tracker: Any = None
        self.t0: float | None = None
        self.odom_bad_ticks = 0
        self.odom_bad_run = 0
        self.track_bad_since: float | None = None
        self.max_error_m = 0.0
        self.max_error_deg = 0.0
        self.min_rate_ticks = 0

    def target(self, now: float) -> tuple[float, float]:
        """(s, theta) of the ramp at ``now``; a drive holds the heading at 0."""
        reached = min(self.rate * (now - self.t0), abs(self.delta))
        signed = math.copysign(reached, self.delta)
        return (0.0, signed) if self.kind == MOVE_TURN else (signed, 0.0)

    def achieved(self) -> float:
        """How far the base got along this move, relative to its anchor."""
        if self.tracker is None:
            return 0.0
        return float(self.tracker.theta if self.kind == MOVE_TURN else self.tracker.s)


class _Align:
    """One base_align stage's run: the state machine, the worker and everything logged.

    A class only to keep the state the tick loop, the exit path and the events
    share in one place; it lives for one call of :func:`align`.
    """

    def __init__(
        self, context: StageContext, stage: StageConfig, reference: Any
    ) -> None:
        self.context = context
        self.stage = stage
        self.align = stage.align
        self.reference = reference
        self.camera = self.align.camera
        self.plan = record_adapter.plan_stage(stage, context.config.defaults)
        self.save_dir = context.run_directory / BASE_SIDE_DIR / stage.id
        self.loop: BaseTickLoop | None = None
        self.worker: ThreadPoolExecutor | None = None
        self.landmark: Any = None
        self.tracker_module: Any = None
        self.params: Any = None
        self.intrinsics: dict[str, Any] | None = None
        self.field_intrinsics_delta: dict[str, float | None] | None = None
        self.camera_up: list[float] | None = None
        self.auto_roi: list[int] | None = None

        self.state = STATE_WAIT_CAPTURE
        self.wait_started: float | None = None
        self.still_since: float | None = None
        self.future: Future | None = None
        self.capture: dict[str, Any] | None = None
        self.measurements: list[dict[str, Any]] = []
        self.moves: list[dict[str, Any]] = []
        self.move: _Move | None = None
        self.move_count = 0
        self.coarse_turns = 0
        self.last_measurement: Any = None
        # Field rule (stage1_align.py): the error before the last correction of
        # each kind, for the divergence test. A drive does NOT reset the turn
        # baseline (spec §4 step 4; the reset is a proposed deviation, §11).
        self.last_turn_yaw: float | None = None
        self.last_drive_move: float | None = None
        self.measuring_periods: list[float] = []
        self.measure_s_total = 0.0
        self.csv_rows: list[list[Any]] = []

    # ---------------------------------------------------------------- setup

    def setup(self) -> None:
        """Tracker, estimator, intrinsics, parameters and the worker; no tick yet."""
        self.tracker_module = import_tracker()
        from stage_runner import landmark_align

        self.landmark = landmark_align
        self.intrinsics = self._intrinsics()
        camera_matrix = np.asarray(self.intrinsics["camera_matrix"], dtype=float)
        dist_coeffs = np.asarray(self.intrinsics["dist_coeffs"], dtype=float)
        self.field_intrinsics_delta = {
            "max_abs_camera_matrix": float(
                np.max(np.abs(camera_matrix - landmark_align.FIELD_K))
            ),
            "max_abs_dist_coeffs": (
                float(np.max(np.abs(dist_coeffs - landmark_align.FIELD_DIST)))
                if dist_coeffs.shape == landmark_align.FIELD_DIST.shape
                else None
            ),
        }
        up = np.asarray(self.align.camera_up, dtype=float)
        up = up / np.linalg.norm(up)
        self.camera_up = [float(value) for value in up]
        roi = tuple(int(value) for value in self.align.landmark_roi)
        self.params = landmark_align.AlignParams(
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            up=up,
            landmark_roi=roi,
            landmark_distance_m=float(self.align.landmark_distance_m),
            width_fallback=bool(self.align.distance_width_fallback),
            coarse_heading=bool(self.align.coarse_heading),
        )
        auto = landmark_align.dark_blob_box(self.reference.image)
        self.auto_roi = None if auto is None else [int(value) for value in auto]
        if self.auto_roi != list(roi):
            # The field never passed an ROI: it scaled the box dark_blob_box
            # found in the reference on every run. A different ROI here means a
            # different scale, and with it every distance move.
            logger.warning(
                f"stage {self.stage.id!r}: landmark_roi {list(roi)} differs from "
                f"the box the field would have used, dark_blob_box(reference) = "
                f"{self.auto_roi}; distance moves follow the YAML ROI"
            )
        logger.info(
            f"stage {self.stage.id!r}: aligning {self.camera} to "
            f"{self.reference.source.get('path') or self.reference.source.get('repo_id')} "
            f"(roi {list(roi)}, auto {self.auto_roi}, landmark "
            f"{self.align.landmark_distance_m:.2f} m, tolerances "
            f"{self.align.yaw_tol_deg:.1f} deg / {self.align.dist_tol_cm:.1f} cm, "
            f"max {self.align.max_moves} moves, coarse heading "
            f"{'on' if self.align.coarse_heading else 'off'}, intrinsics from "
            f"{self.intrinsics['source']})"
        )
        self.worker = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"align-{self.stage.id}"
        )
        # The first job: the reference beside the live frames it is compared
        # with. The worker writes every side file, never the main loop.
        self.worker.submit(
            landmark_align.write_png,
            self.save_dir / "reference.png",
            self.reference.image,
        )

    def _intrinsics(self) -> dict[str, Any]:
        """K and D from the YAML, or off the device; _Ended(intrinsics_unavailable) otherwise."""
        align = self.align
        if align.camera_matrix is not None:
            return {
                "camera_matrix": [
                    [float(v) for v in row] for row in align.camera_matrix
                ],
                "dist_coeffs": [float(v) for v in (align.dist_coeffs or [])],
                "model": None,
                "source": "yaml",
                "width": None,
                "height": None,
            }
        camera = (getattr(self.context.robot, "cameras", None) or {}).get(self.camera)
        try:
            read = device_intrinsics(camera)
        except Exception as error:
            # Any failure (no pyrealsense2, a camera that is not a RealSense, a
            # pipeline that is not streaming) means the same thing here.
            logger.warning(
                f"stage {self.stage.id!r}: no device intrinsics", exc_info=True
            )
            raise _Ended(
                OUTCOME_INTRINSICS_UNAVAILABLE,
                f"could not read the intrinsics of {self.camera} off the device "
                f"({error!r}); give align.camera_matrix and dist_coeffs",
            ) from None
        height, width = self.reference.image.shape[:2]
        if (read["width"], read["height"]) != (width, height):
            raise _Ended(
                OUTCOME_INTRINSICS_UNAVAILABLE,
                f"{self.camera} streams {read['width']}x{read['height']} but the "
                f"reference is {width}x{height}; the intrinsics would be for a "
                "different image",
            )
        return {**read, "source": "device"}

    # ---------------------------------------------------------------- ticks

    def run(self) -> None:
        """The state machine, one recorded frame per tick; ends by raising _Ended."""
        loop = self.loop
        while True:
            _check_keys(loop, "the align")
            observation, odom, now = loop.observe()
            _check_guards(loop, now, "the base was aligned")
            if self.wait_started is None:
                self.wait_started = now
            state = self.state
            step = len(self.measurements)
            x_vel, theta_vel, target, cmd = self._tick(observation, odom, now)
            loop.act(observation, x_vel, theta_vel)
            self._row(now, state, step, target, cmd, odom, (x_vel, theta_vel))

    def _tick(
        self,
        observation: dict[str, Any],
        odom: tuple[float, float, float] | None,
        now: float,
    ) -> tuple[float, float, tuple[float, float] | None, Any]:
        """This tick's base command (x_vel, theta_vel), ramp target and tracker command."""
        if self.state == STATE_MEASURING:
            if self.loop.last_period_s is not None:
                self.measuring_periods.append(self.loop.last_period_s)
            if not self.future.done():
                return 0.0, 0.0, None, None
            # A worker exception re-raises here and ends the stage interrupted.
            measurement = self.future.result()
            self.future = None
            return self._decide(measurement, odom, now)
        if self.state == STATE_MOVING:
            return self._move_tick(odom, now)
        return self._wait_tick(observation, now)

    def _wait_tick(
        self, observation: dict[str, Any], now: float
    ) -> tuple[float, float, None, None]:
        """WAIT_CAPTURE: hold zero until the base has been still for CAPTURE_DELAY_S."""
        velocity = self.loop.read_base_velocity()
        still = (
            velocity is not None
            and abs(velocity[0]) < STILL_LINEAR_M_S
            and abs(velocity[1]) < STILL_ANGULAR_RAD_S
        )
        if not still:
            self.still_since = None
        elif self.still_since is None:
            self.still_since = now
        if self.still_since is not None and now - self.still_since >= CAPTURE_DELAY_S:
            self._capture(observation, now)
            return 0.0, 0.0, None, None
        if now - self.wait_started >= CAPTURE_STILL_MAX_S:
            raise _Ended(
                OUTCOME_NOT_STILL,
                f"the base was not still for {CAPTURE_DELAY_S:.1f}s within "
                f"{CAPTURE_STILL_MAX_S:.1f}s before capture {len(self.measurements)} "
                f"(last measured velocity {velocity}); a frame taken while it moves "
                "would measure motion blur and a pose it is leaving",
            )
        return 0.0, 0.0, None, None

    def _capture(self, observation: dict[str, Any], now: float) -> None:
        """Copy this tick's frame for the worker and hand it the measurement."""
        step = len(self.measurements)
        # A COPY: the frame itself goes into the episode by reference (the
        # image writer saves it later), and the worker must not share it.
        live = np.array(observation[self.camera], dtype=np.uint8, copy=True)
        self.capture = {
            "step": step,
            # The frame this tick records is the captured image.
            "capture_frame_idx": self.context.buffered_frame_count(),
            "capture_wall_clock_iso": wall_clock_iso(),
            "still_wait_s": now - self.wait_started,
            "_submitted": now,
        }
        self.future = self.worker.submit(
            self.landmark.measure,
            self.reference.image,
            live,
            self.params,
            self.save_dir,
            step,
        )
        self.state = STATE_MEASURING

    def _decide(
        self,
        measurement: Any,
        odom: tuple[float, float, float] | None,
        now: float,
    ) -> tuple[float, float, tuple[float, float] | None, Any]:
        """Record the measurement, then act on it in this same tick."""
        self.last_measurement = measurement
        self.measure_s_total += float(measurement.measure_s)
        decision = self._decision(measurement)
        capture = dict(self.capture or {})
        submitted = capture.pop("_submitted", now)
        record: dict[str, Any] = {
            "kind": "measure",
            **capture,
            **measurement.as_event(),
            # From capture to the tick that saw the result: measure_s plus the
            # GIL contention with this loop and up to one tick of polling.
            "wait_s": now - submitted,
        }
        for key in ("live_png", "overlay_png"):
            record[key] = self._relative(record.get(key))
        if isinstance(decision, _Ended):
            record["decision"] = decision.outcome
        else:
            kind, delta = decision
            record["decision"] = kind
            record["commanded"] = (
                math.degrees(delta) if kind == MOVE_TURN else float(delta)
            )
        self.measurements.append(record)
        self._emit_step(record)
        logger.info(
            f"stage {self.stage.id!r} step {record['step']}: heading "
            f"{self._fmt_deg(measurement.heading_yaw_deg)} "
            f"[{measurement.heading_mode}] (fine "
            f"{self._fmt_deg(measurement.yaw_deg)}, {measurement.inliers}/"
            f"{measurement.matches} inliers), distance "
            f"{self._fmt_cm(measurement.move_m)} "
            f"[{measurement.scale_source}, {measurement.scale_inliers} inliers] "
            f"-> {record['decision']} ({measurement.measure_s:.2f}s)"
        )
        self.capture = None
        if isinstance(decision, _Ended):
            raise decision
        kind, delta = decision
        self.move = _Move(
            kind,
            delta,
            record["step"],
            measurement.heading_mode if kind == MOVE_TURN else None,
            self.context.buffered_frame_count(),
        )
        self.move_count += 1
        self.state = STATE_MOVING
        return self._move_tick(odom, now)

    def _decision(self, m: Any) -> _Ended | tuple[str, float]:
        """The field's decision order (stage1_align.py), with the coarse fallback.

        Heading first, distance only once the heading is inside tolerance
        (landmark_align trap 3). A COARSE estimate turns -- even when it reads
        inside tolerance -- and is measured again: it folds any sideways offset
        into the angle, so it can never be the evidence that the base is
        aligned. Each coarse turn counts toward max_moves, and more than
        MAX_COARSE_TURNS of them end the stage estimate_unusable.

        One exception (robot PC 2026-10-25, r09): a coarse heading inside
        tolerance with a usable distance outside dist_tol_cm DRIVES instead of
        turning. Far from the arrival the view is smaller and the fine estimate
        stays under MIN_INLIERS (13-14 against 36-70 in the runs that aligned),
        so turning again only spent the coarse turns and the stage ended with
        the base still ~10 cm away. Closer, the fine estimate comes back; the
        drive is a correction like any other (divergence, max_moves), and only
        a fine heading can still end the stage aligned.
        """
        align = self.align
        if not m.heading_usable:
            fine = (
                "no fine estimate"
                if m.yaw_deg is None
                else f"fine {m.yaw_deg:+.2f} deg with {m.inliers}/{m.matches} inliers "
                f"(needs >= {self.landmark.MIN_INLIERS} and <= "
                f"{self.landmark.MAX_ABS_YAW_DEG:.0f} deg)"
            )
            if not align.coarse_heading:
                coarse = "coarse heading off"
            elif m.coarse_yaw_deg is None:
                coarse = "no rack blob for the coarse estimate"
            else:
                coarse = (
                    f"coarse {m.coarse_yaw_deg:+.2f} deg beyond "
                    f"{self.landmark.MAX_ABS_COARSE_YAW_DEG:.0f} deg"
                )
            return _Ended(
                OUTCOME_ESTIMATE_UNUSABLE,
                f"heading not usable at step {len(self.measurements)}: {fine}; "
                f"{coarse}. Align by hand",
            )
        yaw = float(m.heading_yaw_deg)
        coarse = m.heading_mode == self.landmark.HEADING_COARSE
        coarse_far = (
            coarse
            and abs(yaw) <= align.yaw_tol_deg
            and m.distance_valid
            and abs(float(m.move_m)) * 100.0 > align.dist_tol_cm
        )
        if (coarse and not coarse_far) or abs(yaw) > align.yaw_tol_deg:
            if (
                self.last_turn_yaw is not None
                and abs(yaw) > abs(self.last_turn_yaw) + DIVERGE_YAW_DEG
            ):
                return _Ended(
                    OUTCOME_DIVERGING,
                    f"heading {self.last_turn_yaw:+.2f} -> {yaw:+.2f} deg after a "
                    f"turn (worse by more than {DIVERGE_YAW_DEG} deg): a wrong sign, "
                    "or the landmark moved?",
                )
            if coarse:
                self.coarse_turns += 1
                if self.coarse_turns > MAX_COARSE_TURNS:
                    return _Ended(
                        OUTCOME_ESTIMATE_UNUSABLE,
                        f"still only a coarse heading ({yaw:+.2f} deg) after "
                        f"{MAX_COARSE_TURNS} coarse turns. Align by hand",
                    )
            if self.move_count >= align.max_moves:
                return _Ended(
                    OUTCOME_NOT_CONVERGED,
                    f"heading {yaw:+.2f} deg [{m.heading_mode}] still outside "
                    f"+-{align.yaw_tol_deg:.1f} deg after {self.move_count} moves",
                )
            self.last_turn_yaw = yaw
            return MOVE_TURN, math.radians(-yaw)
        if not m.distance_valid:
            return _Ended(
                OUTCOME_HEADING_ONLY,
                f"heading {yaw:+.2f} deg inside +-{align.yaw_tol_deg:.1f} deg; distance "
                f"not usable ({self._fmt_cm(m.move_m)}, {m.scale_inliers} scale "
                "inliers), so it was not corrected",
            )
        move_m = float(m.move_m)
        if abs(move_m) * 100.0 > align.dist_tol_cm:
            if (
                self.last_drive_move is not None
                and abs(move_m) > abs(self.last_drive_move) + DIVERGE_MOVE_M
            ):
                return _Ended(
                    OUTCOME_DIVERGING,
                    f"distance {self.last_drive_move * 100:+.1f} -> "
                    f"{move_m * 100:+.1f} cm after a drive (worse by more than "
                    f"{DIVERGE_MOVE_M * 100:.1f} cm)",
                )
            if self.move_count >= align.max_moves:
                return _Ended(
                    OUTCOME_NOT_CONVERGED,
                    f"distance {move_m * 100:+.1f} cm still outside "
                    f"+-{align.dist_tol_cm:.1f} cm after {self.move_count} moves",
                )
            self.last_drive_move = move_m
            return MOVE_DRIVE, move_m
        return _Ended(
            OUTCOME_ALIGNED,
            f"heading {yaw:+.2f} deg, distance {move_m * 100:+.1f} cm after "
            f"{self.move_count} moves",
        )

    def _move_tick(
        self, odom: tuple[float, float, float] | None, now: float
    ) -> tuple[float, float, tuple[float, float] | None, Any]:
        """MOVING: follow the ramp; after ramp_s + MOVE_HOLD_S stop and judge the move."""
        move = self.move
        if move.tracker is None:
            # Anchor at this move's own start. A tick without odometry waits
            # (holding zero) rather than ending the stage on one dropped read;
            # ODOM_BAD_LIMIT of them in a row is no_odometry, as mid-move.
            if odom is None:
                self._odom_bad(move)
                return 0.0, 0.0, None, None
            move.odom_bad_run = 0
            move.tracker = self.tracker_module.BasePoseTracker(
                self.tracker_module.BasePoseGains(
                    kp_x=MOVE_KP_X,
                    kp_theta=float(self.align.turn_kp),
                    max_x_vel=MOVE_MAX_VEL,
                    max_theta_vel=MOVE_MAX_VEL,
                    target_window=MOVE_TARGET_WINDOW,
                )
            )
            move.tracker.reset(odom, self.loop.median_period_s())
            move.t0 = now
        target = move.target(now)
        cmd = move.tracker.command(target[0], target[1], odom, now)
        if cmd.odom_ok:
            move.odom_bad_run = 0
            e_s = abs(target[0] - cmd.s)
            e_theta = math.degrees(abs(_wrap(target[1] - cmd.theta)))
            move.max_error_m = max(move.max_error_m, e_s)
            move.max_error_deg = max(move.max_error_deg, e_theta)
            if e_s > TRACK_ERROR_M or e_theta > MOVE_TRACK_ERROR_DEG:
                if move.track_bad_since is None:
                    move.track_bad_since = now
                elif now - move.track_bad_since >= TRACK_ERROR_HOLD_S:
                    raise _Ended(
                        OUTCOME_TRACKING_LOST,
                        f"{move.kind} tracking error {e_s * 100:.1f} cm / "
                        f"{e_theta:.1f} deg held for "
                        f"{now - move.track_bad_since:.2f}s (limits "
                        f"{TRACK_ERROR_M * 100:.0f} cm / "
                        f"{MOVE_TRACK_ERROR_DEG:.0f} deg "
                        f"for {TRACK_ERROR_HOLD_S:.1f}s): the base is not "
                        "following -- stalled, e-stopped or torque off?",
                    )
            else:
                move.track_bad_since = None
        else:
            self._odom_bad(move)
        if now - move.t0 >= move.ramp_s + MOVE_HOLD_S:
            # The field's run_track: ramp ticks, then settle_seconds of hold,
            # then stop -- with no stillness test (the capture gate after it is
            # where stillness is required). This tick sends the stop.
            self._finish_move(now)
            return 0.0, 0.0, target, cmd
        theta_vel = cmd.theta_vel
        floor = float(self.align.min_turn_rate_rad_s)
        if (
            floor > 0.0
            and move.kind == MOVE_TURN
            and abs(move.delta - cmd.theta) > MIN_RATE_DEADBAND_RAD
            and 0.0 < abs(theta_vel) < floor
        ):
            theta_vel = math.copysign(floor, theta_vel)
            move.min_rate_ticks += 1
        return cmd.x_vel, theta_vel, target, cmd

    def _odom_bad(self, move: _Move) -> None:
        move.odom_bad_ticks += 1
        move.odom_bad_run += 1
        if move.odom_bad_run > ODOM_BAD_LIMIT:
            raise _Ended(
                OUTCOME_NO_ODOMETRY,
                f"{move.odom_bad_run} consecutive ticks without usable odometry "
                f"during a {move.kind} (limit {ODOM_BAD_LIMIT})",
            )

    def _move_record(self, now: float, ended_by: str | None) -> dict[str, Any]:
        move = self.move
        achieved = move.achieved()
        judged = abs(move.delta) >= (
            MOVE_JUDGE_MIN_RAD if move.kind == MOVE_TURN else MOVE_JUDGE_MIN_M
        )
        ratio = achieved / move.delta if move.delta else None
        velocity = self.loop.read_base_velocity()
        to_unit = math.degrees if move.kind == MOVE_TURN else float
        elapsed = None if move.t0 is None else now - move.t0
        return {
            "kind": "move",
            "step": move.step,
            "move": move.kind,
            "unit": "deg" if move.kind == MOVE_TURN else "m",
            "heading_mode": move.heading_mode,
            "commanded": to_unit(move.delta),
            "achieved": to_unit(achieved),
            "ratio": ratio,
            "judged": judged,
            "ramp_s": move.ramp_s,
            "hold_s": None if elapsed is None else max(0.0, elapsed - move.ramp_s),
            "still_at_exit": (
                velocity is not None
                and abs(velocity[0]) < STILL_LINEAR_M_S
                and abs(velocity[1]) < STILL_ANGULAR_RAD_S
            ),
            "odom_bad_ticks": move.odom_bad_ticks,
            "max_tracking_error_m": move.max_error_m,
            "max_tracking_error_deg": move.max_error_deg,
            "min_rate_ticks": move.min_rate_ticks,
            "frame_idx_start": move.frame_idx_start,
            "frame_idx_end": self.context.buffered_frame_count(),
            "ended_by": ended_by,
        }

    def _finish_move(self, now: float) -> None:
        """Record the finished move, judge it, and go back to WAIT_CAPTURE."""
        record = self._move_record(now, None)
        self._close_move(record)
        ratio = record["ratio"]
        if record["judged"] and ratio is not None:
            what = (
                f"{record['move']} of {record['commanded']:+.3f} {record['unit']} "
                f"achieved {record['achieved']:+.3f} {record['unit']}"
            )
            if ratio < 0.0:
                raise _Ended(OUTCOME_MOVE_FAILED, f"moved opposite to command: {what}")
            if ratio < MIN_MOVE_FRACTION:
                raise _Ended(
                    OUTCOME_MOVE_FAILED,
                    f"base did not move: e-stop or torque off? {what} (ratio "
                    f"{ratio:.2f} < {MIN_MOVE_FRACTION})",
                )
        self.state = STATE_WAIT_CAPTURE
        self.wait_started = now
        self.still_since = None

    def _close_move(self, record: dict[str, Any]) -> None:
        if self.move is not None and self.move.tracker is not None:
            self.move.tracker.release()
        self.move = None
        self.moves.append(record)
        self._emit_step(record)

    def interrupt_move(self, outcome: str) -> None:
        """At exit: record a move a guard or key cut short. Never raises."""
        if self.move is None:
            return
        try:
            self._close_move(self._move_record(time.perf_counter(), outcome))
        except Exception:
            logger.exception(f"stage {self.stage.id!r}: could not record the cut move")
            self.move = None

    # ------------------------------------------------------------------ logs

    def _emit_step(self, record: dict[str, Any]) -> None:
        self.context.log.emit(
            EVENT_STAGE_BASE_ALIGN_STEP,
            stage_id=self.stage.id,
            frame_idx=self.context.buffered_frame_count(),
            **record,
        )

    def _relative(self, path: str | None) -> str | None:
        """A side-file path relative to the run directory (as the replay CSV's)."""
        if path is None:
            return None
        try:
            return Path(path).relative_to(self.context.run_directory).as_posix()
        except ValueError:
            return str(path)

    def _row(
        self,
        now: float,
        state: str,
        step: int,
        target: tuple[float, float] | None,
        cmd: Any,
        odom: tuple[float, float, float] | None,
        sent: tuple[float, float],
    ) -> None:
        loop = self.loop
        velocity = loop.read_base_velocity()
        move = self.move.kind if self.move is not None else None
        self.csv_rows.append(
            [
                now - loop.started,
                state,
                step,
                move,
                *(target or (None, None)),
                *((cmd.s, cmd.theta) if cmd else (None, None)),
                *sent,
                *(velocity or (None, None)),
                *(odom or (None, None, None)),
                None if cmd is None else cmd.odom_ok,
                loop.probe.last_update_ok,
                loop.probe.last_cmd_ok,
            ]
        )

    def write_csv(self) -> str | None:
        """Write the per-tick log; the run-relative path, or None. Never raises."""
        relative = Path(BASE_SIDE_DIR) / f"{self.stage.id}_align.csv"
        try:
            path = self.context.run_directory / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(ALIGN_CSV_COLUMNS)
                writer.writerows(self.csv_rows)
        except Exception:
            logger.exception(f"stage {self.stage.id!r}: could not write {relative}")
            return None
        return relative.as_posix()

    @staticmethod
    def _fmt_deg(value: float | None) -> str:
        return "none" if value is None else f"{value:+.2f} deg"

    @staticmethod
    def _fmt_cm(value: float | None) -> str:
        return "none" if value is None else f"{value * 100:+.1f} cm"

    def fields(self) -> dict[str, Any]:
        """Every field of the stage_base_align event except outcome and reason."""
        align = self.align
        loop = self.loop
        last = self.last_measurement
        fields: dict[str, Any] = {
            "reference": dict(self.reference.source),
            "camera": self.camera,
            "intrinsics": self.intrinsics,
            "field_intrinsics_delta": self.field_intrinsics_delta,
            "camera_up": self.camera_up,
            "landmark_roi": list(align.landmark_roi),
            "auto_roi": self.auto_roi,
            "landmark_distance_m": align.landmark_distance_m,
            "yaw_tol_deg": align.yaw_tol_deg,
            "dist_tol_cm": align.dist_tol_cm,
            "max_moves": align.max_moves,
            "turn_kp": align.turn_kp,
            "min_turn_rate_rad_s": align.min_turn_rate_rad_s,
            "distance_width_fallback": align.distance_width_fallback,
            "coarse_heading": align.coarse_heading,
            "control_time_s": self.plan.control_time_s,
            "measurements": self.measurements,
            "moves": self.moves,
            "move_count": self.move_count,
            "coarse_turns": self.coarse_turns,
            "final_yaw_deg": None if last is None else last.heading_yaw_deg,
            "final_heading_mode": None if last is None else last.heading_mode,
            "final_move_cm": (
                None if last is None or last.move_m is None else last.move_m * 100.0
            ),
            "final_distance_valid": None if last is None else last.distance_valid,
            "measure_s_total": self.measure_s_total,
            "ticks": loop.ticks if loop else 0,
            "base_state_last": loop.base_state_name() if loop else None,
            "held_arm_positions": loop.held_arm_positions if loop else None,
            "gripper_hold_source": loop.gripper_hold_source if loop else None,
        }
        if loop is not None:
            counters = loop.probe.counters()
            fields.update(
                update_failed_ticks=counters["update_failed_ticks"],
                cmd_refused_ticks=counters["cmd_refused_ticks"],
                max_cmd_refused_run=counters["max_cmd_refused_run"],
            )
            stats = loop.tick_stats()
            fields.update(
                tick_mean_s=stats["tick_mean_s"],
                tick_p95_s=stats["tick_p95_s"],
                tick_max_s=stats["tick_max_s"],
                slow_ticks=stats["slow_ticks"],
            )
            # What the worker thread cost this loop: the tick periods while a
            # measurement ran (spec risk 7 -- the pure-Python RANSAC takes GIL
            # slices from the main loop).
            measuring = loop.tick_stats(self.measuring_periods)
            fields.update(
                tick_p95_s_while_measuring=measuring["tick_p95_s"],
                tick_max_s_while_measuring=measuring["tick_max_s"],
            )
        return fields


def _shutdown_worker(worker: ThreadPoolExecutor | None) -> None:
    """Stop taking jobs and drop queued ones, WITHOUT waiting. Never raises.

    A measurement already running (about 0.5 s in the office, longer on the
    robot) finishes harmlessly in its thread and its result is dropped: the
    stage has already decided, and the base is already zeroed.
    """
    if worker is None:
        return
    try:
        worker.shutdown(wait=False, cancel_futures=True)
    except Exception:
        logger.exception("could not shut the align worker down")


def align(context: StageContext, stage: StageConfig) -> StageResult:
    """Turn and drive the base until cam_high matches the demo's arrival view.

    The 10/24 field loop (stage1_align.py / heading_align.py) as a chain stage:

    1. WAIT_CAPTURE: hold zero until the base has been still for
       CAPTURE_DELAY_S (at most CAPTURE_STILL_MAX_S, else "not_still"), then
       copy this tick's camera frame for the worker.
    2. MEASURING: the worker thread runs landmark_align.measure on the copy
       while this loop keeps recording hold frames at fps.
    3. DECIDE, in the field's order: heading first -- a turn of -yaw while it
       is outside yaw_tol_deg, or whenever it is only the COARSE rack-blob
       estimate (A2: the fine SIFT estimate is unusable, the rack blob is not;
       a coarse estimate never counts as aligned) -- then the distance, which
       is only read once the heading is inside tolerance. "aligned" when both
       are inside; "heading_only" (converged, the chain goes on: A3) when the
       heading is inside and the distance cannot be read.
    4. MOVING: a ramp at the field's rate followed by PR 69's tracker, a fixed
       MOVE_HOLD_S hold, then stop and judge the move (signed fraction
       achieved), and back to 1.

    Failures, each base_failed with its value: estimate_unusable (neither
    estimate usable, or more than MAX_COARSE_TURNS coarse turns), diverging (a
    correction made its error worse), not_converged (max_moves used), and the
    guards shared with the replay -- e-stop state, refused commands, no
    odometry, tracking lost -- plus move_failed, not_still,
    intrinsics_unavailable and the stage ceiling. Esc / left arrow abort; the
    right arrow accepts the base where it is (manual).

    Every exit zeroes the base directly first, shuts the worker down without
    waiting, restores robot.base, then writes
    run_directory/base/<stage_id>_align.csv and one stage_base_align event
    (step events were written as each measurement and move finished; PNGs are
    under run_directory/base/<stage_id>/). An exception does the same (minus
    the CSV), stops the image writer and propagates unchanged.
    """
    if stage.align is None:
        # preflight.check_stage_definitions makes this unreachable from cli.
        raise RuntimeError(f"stage {stage.id!r}: base_align without an align: block")
    run = _Align(context, stage, _stage_asset(context, stage))

    def emit(outcome: str, reason: str, terminator: str | None, **extra: Any) -> None:
        context.log.emit(
            EVENT_STAGE_BASE_ALIGN,
            stage_id=stage.id,
            frame_idx=context.buffered_frame_count(),
            terminator=terminator,
            reason=reason,
            outcome=outcome,
            **run.fields(),
            **extra,
        )

    try:
        run.loop = BaseTickLoop(context, stage, ceiling_s=run.plan.control_time_s)
        try:
            run.setup()
            run.run()
        except _Ended as ended:
            outcome, reason, key = ended.outcome, ended.reason, ended.key
        ended_at = time.perf_counter()
        elapsed_s = run.loop.elapsed_s(ended_at)
    except BaseException as error:
        # HARDWARE FIRST, then the worker and the probe, then the record.
        zero_problem = _zero_base(context, stage)
        _shutdown_worker(run.worker)
        run.interrupt_move(OUTCOME_INTERRUPTED)
        if run.loop is not None:
            run.loop.close()
        stop_image_writer_after(error, context.dataset)
        try:
            emit(
                OUTCOME_INTERRUPTED,
                f"the align raised {error!r}",
                TERMINATED_BY_ERROR,
                base_zero_error=zero_problem or None,
                csv=None,
            )
        except Exception:
            logger.exception(f"stage {stage.id!r}: could not record the interruption")
        raise

    zero_problem = _zero_base(context, stage)
    _shutdown_worker(run.worker)
    run.interrupt_move(outcome)
    run.loop.close()
    terminated_by = (
        _TERMINATED_BY_KEY[key]
        if outcome == OUTCOME_ABORTED
        else _ALIGN_TERMINATED_BY[outcome]
    )
    csv_path = run.write_csv()
    emit(
        outcome,
        reason,
        terminated_by,
        base_zero_error=zero_problem or None,
        csv=csv_path,
    )
    frames = run.loop.frames
    log = logger.info if terminated_by != TERMINATED_BY_BASE_FAILED else logger.error
    log(
        f"stage {stage.id!r} ended: {outcome} ({terminated_by}) -- {reason} "
        f"({run.move_count} moves, {len(run.measurements)} measurements, "
        f"{frames} frames in {elapsed_s:.2f}s)"
    )
    return StageResult(
        terminated_by=terminated_by,
        elapsed_s=elapsed_s,
        frames=frames,
        reason=f"{outcome}: {reason}",
    )
