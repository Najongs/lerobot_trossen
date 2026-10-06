"""The boundary reset: a minimum-jerk pose ramp shaped as a policy.

WHY A RESET EXISTS AT ALL. The demonstrations were recorded one stage at a time,
each from its own hand-built start pose, so stage k's END pose and stage k+1's
START pose are 0.25-1.28 rad apart at all ten boundaries -- 1.6 to 14 times the
within-stage spread (§91). The transition between two stages is simply not in the
training data, so no policy can produce it. The chain therefore drives it
itself: 12 arm joints, from where stage k left them to stage k+1's designated
start pose.

WHY A POLICY AND NOT A DIRECT CALL. Every command in this package goes through
ONE path:

    make_robot_action -> robot_action_processor -> robot.send_action
      -> arm key filter (mobileai.py:472)
      -> ensure_safe_goal_position, max_relative_target 0.1 rad
      -> velocity pacing, safety factor 0.4
      -> set_all_positions(blocking=False)

``WidowXAIFollower.staged_positions`` / ``set_all_positions(blocking=True)``
would move the arms while BYPASSING the relative-target clamp and the pacing, and
this module does not call them. Wearing the shape of a policy
(``select_action`` returning a tensor) is what keeps the ramp on the audited
path, and it also means the reset ticks are RECORDED -- as ``task="reset:NN"``
frames in the same episode -- so a boundary can be analysed afterwards instead of
being a gap in the data.

WHAT IT DOES NOT MOVE. The grippers (each arm's 7th value) are commanded to their
MEASURED position every tick, because the arm may be carrying something, and the
base is commanded to zero. Neither is interpolated.

    ⚠️ Whether re-commanding the measured gripper position preserves the GRIP
    FORCE on a held object is unverified [추정]: it is a position command to a
    carriage that is squeezing. This is a gate before bring-up step ③ (reset
    while holding an object), not something this module can settle.

THE RAMP. ``f(u) = 10u^3 - 15u^4 + 6u^5``, the same quintic the fork already uses
to blend chunk seams (chunk_execution_patch.py:325). ``f(0)=0``, ``f(1)=1``,
``f'(0)=f'(1)=0`` -- the endpoints are exact and the boundary velocities are
zero, so the ramp neither jumps at the start nor overshoots at the end.
``max f' = 1.875`` at ``u=0.5``, which bounds the per-tick step at
``1.875 * delta_max / N``: for the worst measured boundary gap of 1.28 rad over
T = 2.44 s at 21 Hz that is 0.047 rad/tick, under half of the 0.1 rad relative
clamp. The clamp and the pacing stay as a SAFETY NET, never as the mechanism --
if either fires during a reset, the ramp was wrong and the run says so
(``clamped_ticks`` in the stage_end event).

TICK-BASED, NOT WALL-CLOCK-BASED. ``u = tick / N`` with ``N = ceil(T * fps)``.
A loop that runs slower than fps makes the ramp take LONGER in wall-clock time,
which is the safe direction: a wall-clock ramp would compensate for a slow loop
by taking a bigger step, i.e. by moving the arm faster exactly when the robot is
already struggling.
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_STATE

from stage_runner.chain_params import ARM_JOINT_NAMES
from stage_runner.completion import (
    BASE_VELOCITY_KEYS,
    POSITION_SUFFIX,
    PROGRESS_KEY,
)

logger = logging.getLogger(__name__)

# The phase tag loop_rate_log reads off the policy object. It makes the reset
# ticks show up as `phase=reset` in basevel.csv (so eval_base_stats' policy
# filter and the per-stage base integral stay correct), lets base_serial_rearm
# run in them (same record_loop, same 21 fps), and turns on pose_guide's
# once-a-second "distance to target" line.
RESET_PHASE: str = "reset"


class ResetAbort(RuntimeError):
    """The ramp was refused BEFORE anything moved.

    Raised by :func:`plan_reset` only, which runs in the executor before
    ``record_loop`` is entered. It is NEVER raised from inside the loop: an
    exception there unwinds through ``record_loop``'s
    ``@safe_stop_image_writer``, the runner takes its error path, and the whole
    episode -- every stage already recorded in it -- is discarded unsaved.
    """


@dataclass(frozen=True)
class ResetSettings:
    """The YAML's ``reset:`` block."""

    # Floor on the ramp duration, so a tiny correction still takes a visible,
    # gentle second and a half rather than a single 21 Hz step.
    t_min_s: float = 1.5
    # 0.524 rad/s = 30 deg/s, the AVERAGE speed the duration is sized for. The
    # quintic's peak is 1.875x that.
    v_des_rad_s: float = 0.524
    # A gap larger than this is refused rather than ramped. It is far outside
    # the 0.25-1.28 rad the ten measured boundaries span (§91), so reaching it
    # means the arm is not where the previous stage was supposed to leave it --
    # the one case where a slow, correct-looking sweep across the workspace is
    # the wrong answer.
    max_jump_rad: float = 1.5
    # The same refusal for the INITIAL reset, and deliberately much tighter.
    # The ten BOUNDARY gaps are measured (0.25-1.28 rad) and the previous stage
    # is supposed to have left the arm at one end of one of them, so 1.5 rad
    # there is a "something is wrong" threshold. The initial reset starts from
    # wherever a HUMAN left the arms, where there is no measured distribution at
    # all -- so the answer to a large gap is not a slow correct-looking sweep
    # across the workspace, it is a person moving the arms by hand first.
    initial_max_jump_rad: float = 0.6
    # Arrival: max |measured - target| over the 12 joints. Same NORM as the
    # completion monitor's thresholds (inf-norm, per joint, radians) so
    # "arrived" and "still" are measured the same way; the VALUES are separate
    # knobs (see CompletionSettings.stall_track_rad / stall_arm_rad).
    tol_rad: float = 0.05
    # How long arrival must hold before the reset is declared reached.
    settle_s: float = 1.0
    # control_time_s = (T + settle_s) * this. The ceiling exists so a reset that
    # never arrives ends instead of running to a 300 s manual ceiling; reaching
    # it is a CHAIN FAILURE with no retry.
    #
    # 3.0, raised from 2.0 on 2026-10-06, and raising it is the SAFE direction:
    # the ramp is tick-based (u = tick / N), so a bigger ceiling gives a slow
    # loop more wall-clock time to finish the SAME trajectory at the SAME
    # per-tick step -- it never makes the arm move faster. The bound it buys:
    # the ramp plus the settle window costs (T + settle_s) * fps ticks, which at
    # an achieved loop rate r takes (T + settle_s) * fps / r seconds, so arrival
    # is only POSSIBLE while
    #
    #     r >= fps / ceiling_factor
    #
    # i.e. 7.0 Hz at fps 21 with factor 3 (it was 10.5 Hz with factor 2). Below
    # that the reset reports not_reached no matter how correct the ramp is, and
    # the thing to fix is the loop rate (README "Control loop rate"), not this
    # number.
    ceiling_factor: float = 3.0
    # Pre-entry stillness check: the arm must be at rest before a ramp anchors
    # on its position, or the anchor is a point the arm is already leaving.
    settle_check_tries: int = 3
    settle_check_gap_s: float = 0.1
    settle_check_tol_rad: float = 0.02


@dataclass(frozen=True)
class ResetPlan:
    """Everything decided before the loop is entered. Nothing here is measured later."""

    stage_number: int
    anchor: Mapping[str, float]
    target: Mapping[str, float]
    delta_max: float
    duration_s: float
    ticks: int
    settle_ticks: int
    control_time_s: float
    max_step_rad: float
    fps: int
    # Whether this is the initial reset (from wherever a human left the arms).
    initial: bool = False
    # The gap threshold that WAS applied -- initial_max_jump_rad for an initial
    # reset and max_jump_rad for a boundary one. Carried on the plan rather than
    # re-derived in ResetPolicy, so the mid-entry backstop and the pre-entry
    # refusal can never disagree about which limit this ramp was approved under.
    jump_limit_rad: float = 1.5

    def as_detail(self) -> dict[str, Any]:
        return {
            "reset_T": round(self.duration_s, 3),
            "reset_dmax": round(self.delta_max, 4),
            "reset_ticks": self.ticks,
            "reset_max_step_rad": round(self.max_step_rad, 4),
            "reset_settle_ticks": self.settle_ticks,
            "reset_jump_limit_rad": round(self.jump_limit_rad, 3),
            "reset_initial": self.initial,
        }


def quintic(u: float) -> float:
    """``10u^3 - 15u^4 + 6u^5`` clamped to ``[0, 1]``.

    Exact at the endpoints by construction, which is the property the test
    asserts: a ramp that lands at 0.999 of the target leaves a 1 mrad bias on
    every joint at every boundary, and ten boundaries of that is a drift.
    """
    if u <= 0.0:
        return 0.0
    if u >= 1.0:
        return 1.0
    return u * u * u * (10.0 + u * (-15.0 + 6.0 * u))


# max |d/du quintic(u)| = 15/8, at u = 0.5.
QUINTIC_PEAK_SLOPE: float = 1.875

# How much of the trailing `settle_ticks` window must be inside `tol_rad` for
# the reset to count as arrived. See ResetPolicy._check_arrival.
ARRIVAL_WINDOW_RATIO: float = 0.9


def plan_reset(
    *,
    stage_number: int,
    anchor: Mapping[str, float],
    target: Mapping[str, float],
    fps: int,
    settings: ResetSettings,
    arm_joint_names: Sequence[str] = ARM_JOINT_NAMES,
    initial: bool = False,
) -> ResetPlan:
    """Size the ramp, or refuse it. Called BEFORE ``record_loop``.

    ``initial=True`` is the first ramp of a chain, whose anchor is wherever a
    HUMAN left the arms rather than where the previous stage ended. It is
    refused at ``initial_max_jump_rad`` (0.6) instead of ``max_jump_rad`` (1.5),
    and the message says what to do about it, because the fix is a person
    moving the arms -- not a 1.5 rad sweep from a pose no demonstration has.

    Refuses on a missing joint, a non-finite value, or a gap over the applicable
    limit. A non-finite value has to be caught HERE because it does
    not fail loudly downstream: ``WidowXAIFollower``'s pacing compares
    ``abs(goal - present) > limit``, every comparison with NaN is False, so the
    clamp silently does not fire and the NaN goal reaches ``set_all_positions``
    (the fork recorded the same hazard on the base side: "max(-MAX, NaN) returns
    -MAX -- a NaN command reaches the base as full-speed reverse",
    mobileai.py:54-55).
    """
    problems: list[str] = []
    delta_max = 0.0
    for name in arm_joint_names:
        if name not in anchor:
            problems.append(f"{name!r} is missing from the measured pose")
            continue
        if name not in target:
            problems.append(f"{name!r} is missing from the target pose")
            continue
        start, end = float(anchor[name]), float(target[name])
        if not (math.isfinite(start) and math.isfinite(end)):
            problems.append(
                f"{name!r}: measured {start!r} -> target {end!r} is not finite"
            )
            continue
        delta_max = max(delta_max, abs(end - start))

    if problems:
        raise ResetAbort(
            f"reset before stage {stage_number} refused (nothing moved):\n  - "
            + "\n  - ".join(problems)
        )
    limit = (
        settings.initial_max_jump_rad if initial else settings.max_jump_rad
    )
    limit_name = "initial_max_jump_rad" if initial else "max_jump_rad"
    if delta_max > limit:
        if initial:
            raise ResetAbort(
                f"INITIAL reset before stage {stage_number} refused (nothing "
                f"moved): the largest joint gap is {delta_max:.3f} rad, over "
                f"{limit_name}={limit:.3f}. This is the ramp from wherever the "
                f"arms were LEFT BY HAND to task{stage_number:02d}'s start "
                f"pose, and there is no measured distribution for it -- so put "
                f"the arms near task{stage_number:02d}'s start pose (within "
                "0.3 rad) by hand and start again. The `POSE` line of a "
                "teleoperate session, or the per-stage table in "
                "docs/eval_najy.md, gives the pose in degrees. Raising "
                f"`chain.reset.{limit_name}` instead means asking the runner "
                "to sweep the arms across the workspace from a pose nothing has "
                "validated."
            )
        raise ResetAbort(
            f"reset before stage {stage_number} refused (nothing moved): the "
            f"largest joint gap is {delta_max:.3f} rad, over {limit_name}="
            f"{limit:.3f}. The ten measured stage boundaries span "
            "0.25-1.28 rad, so this means the arm is not where the previous stage "
            "was supposed to leave it -- check the pose by hand before retrying."
        )

    duration_s = max(settings.t_min_s, delta_max / settings.v_des_rad_s)
    ticks = max(1, math.ceil(duration_s * fps))
    settle_ticks = max(1, math.ceil(settings.settle_s * fps))
    return ResetPlan(
        stage_number=stage_number,
        anchor={name: float(anchor[name]) for name in arm_joint_names},
        target={name: float(target[name]) for name in arm_joint_names},
        delta_max=delta_max,
        duration_s=duration_s,
        ticks=ticks,
        settle_ticks=settle_ticks,
        control_time_s=(duration_s + settings.settle_s) * settings.ceiling_factor,
        # The bound, not a measurement: the largest step the ramp can take.
        max_step_rad=QUINTIC_PEAK_SLOPE * delta_max / ticks,
        fps=int(fps),
        initial=bool(initial),
        jump_limit_rad=float(limit),
    )


def wait_until_still(
    robot,
    *,
    arm_joint_names: Sequence[str] = ARM_JOINT_NAMES,
    settings: ResetSettings | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[dict[str, float], bool]:
    """Read the arm twice, ``settle_check_gap_s`` apart, until it is still.

    Returns ``(the second reading, whether it was still)``. The reading is the
    ramp's ANCHOR, and anchoring on a moving arm means the ramp starts from a
    point the arm has already left -- the first interpolated command is then a
    step backwards, which is exactly what the 0.1 rad clamp and the pacing would
    have to absorb.

    Not still after ``settle_check_tries`` is returned, not raised: the caller
    decides, and it has a better answer than an exception (a chain failure with
    the measurement in the log).
    """
    settings = settings or ResetSettings()
    previous: dict[str, float] | None = None
    reading: dict[str, float] = {}
    for attempt in range(max(1, settings.settle_check_tries)):
        observation = robot.get_observation()
        reading = {
            name: float(observation[f"{name}{POSITION_SUFFIX}"])
            for name in arm_joint_names
            if f"{name}{POSITION_SUFFIX}" in observation
        }
        if previous is not None and len(reading) == len(arm_joint_names):
            drift = max(
                abs(reading[name] - previous[name])
                for name in arm_joint_names
                if name in previous
            )
            if drift < settings.settle_check_tol_rad:
                return reading, True
            logger.info(
                f"arm still moving before the reset ({drift:.4f} rad between two "
                f"reads, attempt {attempt + 1}/{settings.settle_check_tries})"
            )
        previous = reading
        sleep(settings.settle_check_gap_s)
    return reading, False


class PassthroughProcessor:
    """Identity stand-in for a ``PolicyProcessorPipeline``.

    ``record_loop`` requires policy, preprocessor and postprocessor to be ALL
    non-None: with one of them missing it falls through to the no-action branch,
    which ``continue``s WITHOUT advancing ``timestamp`` and therefore never
    reaches ``control_time_s`` -- a silent unbounded busy loop with no error and
    no frames (record_adapter.call_record_loop says the same thing).

    A reset needs no normalization: :class:`ResetPolicy` works in radians and
    emits the action in robot units directly, so there is no checkpoint whose
    stats could be applied. ``predict_action`` calls ``preprocessor(observation)``
    and ``postprocessor(action)`` and ``.reset()`` on both, and nothing else.
    """

    def __call__(self, value: Any) -> Any:
        return value

    def reset(self) -> None:
        return None


@dataclass
class ResetPolicyConfig:
    """The subset of ``PreTrainedConfig`` that ``record_loop`` reads.

    ``device`` is "cpu" because ``get_safe_torch_device`` answers anything
    starting with "cuda" through a bare ``assert torch.cuda.is_available()``
    (utils/utils.py:60-61). The ramp is twelve multiplications; there is nothing
    to accelerate.
    """

    device: str = "cpu"
    use_amp: bool = False
    n_action_steps: int | None = 1
    temporal_ensemble_coeff: float | None = None
    pretrained_path: str | None = None
    input_features: dict = None  # type: ignore[assignment]
    output_features: dict = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.input_features is None:
            self.input_features = {}
        if self.output_features is None:
            self.output_features = {}


class ResetPolicy:
    """Emits the ramp. Duck-typed against ``PreTrainedPolicy``; no weights, no hub.

    ``record_loop`` does no isinstance check on the policy -- it tests
    ``is not None`` and then reaches for ``config.device``, ``config.use_amp``,
    ``reset()`` and ``select_action(batch)`` through ``predict_action``.

    ``record_loop_phase = "reset"`` is read by ``loop_rate_log``'s wrapper off
    this object, which is what tags the ticks for ``base_serial_rearm``,
    ``base_vel_log`` and ``pose_guide``.
    """

    record_loop_phase: str = RESET_PHASE

    def __init__(
        self,
        plan: ResetPlan,
        events: dict[str, bool],
        *,
        action_names: Sequence[str],
        state_names: Sequence[str],
        settings: ResetSettings | None = None,
        arm_joint_names: Sequence[str] = ARM_JOINT_NAMES,
    ) -> None:
        self.plan = plan
        self.events = events
        self.settings = settings or ResetSettings()
        # input/output_features are DECLARED, not left empty. Two readers:
        # `policies.bundle_descriptor` puts them in the trial_start event (an
        # empty dict reports state_dim/action_dim null for every reset, so the
        # log could not say what the ramp was driving), and anything upstream
        # that shapes a batch from `policy.config.input_features` -- which is
        # how `policies.warm_up_bundle` works and how a future fast-path could.
        # MockPolicyConfig declares them for the same reason.
        self.config = ResetPolicyConfig(
            pretrained_path=f"reset://{plan.stage_number}",
            input_features={
                OBS_STATE: PolicyFeature(
                    type=FeatureType.STATE, shape=(len(state_names),)
                )
            },
            output_features={
                ACTION: PolicyFeature(
                    type=FeatureType.ACTION, shape=(len(action_names),)
                )
            },
        )
        self.arm_joint_names = tuple(arm_joint_names)
        self.action_names = tuple(action_names)
        self.state_names = tuple(state_names)
        # Resolved by NAME, never by index: observation.state drops the base keys
        # when include_base_in_state is false while the action never does, so the
        # two vectors are not the same layout (mobileai.py:337-339).
        self._state_index: dict[str, int] = {
            name: index for index, name in enumerate(self.state_names)
        }
        # Outcome, read by the executor after the loop.
        self.reached: bool = False
        self.refused: str = ""
        self.worst_error_rad: float = float("inf")
        self.tick: int = 0
        # The last `settle_ticks` arrival answers, newest last. A RATIO over
        # this window, not a consecutive count -- see _check_arrival.
        self._arrival_window: deque[bool] = deque(
            maxlen=max(1, plan.settle_ticks)
        )
        self._last_action: list[float] | None = None

    # ``record_loop`` calls this on entry to every stage.
    def reset(self) -> None:
        self.tick = 0
        self._arrival_window.clear()
        self._last_action = None
        self.reached = False
        self.worst_error_rad = float("inf")

    def select_action(self, batch: dict[str, Any]) -> torch.Tensor:
        state = batch[OBS_STATE].squeeze(0).tolist()
        measured = {
            name: float(state[index])
            for name, index in self._state_index.items()
            if index < len(state)
        }

        # Backstop only, and ON THE FIRST TICK ONLY (`self.tick == 0`, i.e.
        # before this policy has commanded anything). plan_reset already refused
        # an impossible gap before the loop was entered; this covers the case
        # where the arm MOVED between the anchor read and the first tick by more
        # than the whole budget -- something only a fault can do. It holds
        # position instead of raising, because raising here discards the episode
        # (see ResetAbort).
        #
        # WITHOUT the tick guard it ran on every tick, and then it was not a
        # backstop at all but a second, wrong, interpretation of the same
        # number: `_anchor_drift` grows monotonically as the ramp does its job
        # (by the last tick it IS delta_max), so the check measured the ramp's
        # own progress and refused it near the END -- a reset that had just
        # arrived, reported as `not_reached`, which is a chain failure. The
        # docstring always said "between the anchor read and the first tick";
        # the code did not.
        if not self.refused and self.tick == 0:
            drift = self._anchor_drift(measured)
            if drift is not None and drift > self.plan.jump_limit_rad:
                self.refused = (
                    f"the arm moved {drift:.3f} rad away from the anchor between "
                    f"the pre-entry read and the first tick, over the "
                    f"{self.plan.jump_limit_rad:.3f} rad limit this ramp was "
                    "planned under; holding position"
                )
                logger.error(f"reset refused mid-entry: {self.refused}")
                self.events["exit_early"] = True

        self.tick += 1
        if self.refused:
            values = self._hold(measured)
        else:
            values = self._ramp(measured)
        self._last_action = values

        if not self.refused:
            self._check_arrival(measured)
        # [1, action_dim]: make_robot_action squeezes the batch dimension back off
        # and zips what is left against ds_features[ACTION]["names"].
        return torch.tensor([values], dtype=torch.float32)

    # ------------------------------------------------------------------ internals

    def _anchor_drift(self, measured: Mapping[str, float]) -> float | None:
        worst: float | None = None
        for name in self.arm_joint_names:
            key = f"{name}{POSITION_SUFFIX}"
            if key not in measured:
                return None
            value = measured[key]
            if not math.isfinite(value):
                return float("inf")
            gap = abs(value - self.plan.anchor[name])
            worst = gap if worst is None else max(worst, gap)
        return worst

    def _ramp(self, measured: Mapping[str, float]) -> list[float]:
        fraction = quintic(self.tick / self.plan.ticks)
        values: list[float] = []
        for name in self.action_names:
            if name == PROGRESS_KEY:
                # The 17th slot of a `tph` checkpoint's action. Zero during a
                # reset: the reset is not part of any stage's progress, and the
                # robot ignores the key anyway (mobileai.py:472-480). It is
                # RECORDED in the frame, which is what makes a reset segment
                # identifiable in the dataset afterwards alongside task="reset:NN".
                values.append(0.0)
                continue
            if name in BASE_VELOCITY_KEYS:
                values.append(0.0)
                continue
            joint = name[: -len(POSITION_SUFFIX)] if name.endswith(POSITION_SUFFIX) else name
            if joint in self.plan.target:
                start = self.plan.anchor[joint]
                values.append(start + (self.plan.target[joint] - start) * fraction)
                continue
            # A gripper, or any other commandable position that is not one of the
            # 12 reset joints: HOLD THE MEASURED VALUE. Never 0.0 -- a gripper
            # commanded to 0.0 would open or close on an object the arm is
            # carrying. If the measurement is unavailable (it should not be: the
            # executor checks every .pos action name against the observation
            # before entering the loop), repeat the last command rather than
            # inventing a position.
            values.append(self._hold_value(name, measured, len(values)))
        return values

    def _hold(self, measured: Mapping[str, float]) -> list[float]:
        """Every joint at its measured position, base zero. The refusal action."""
        values: list[float] = []
        for name in self.action_names:
            if name == PROGRESS_KEY or name in BASE_VELOCITY_KEYS:
                values.append(0.0)
                continue
            values.append(self._hold_value(name, measured, len(values)))
        return values

    def _hold_value(
        self, name: str, measured: Mapping[str, float], index: int
    ) -> float:
        if name in measured:
            return measured[name]
        if self._last_action is not None and index < len(self._last_action):
            return self._last_action[index]
        # Nothing measured and nothing previously commanded. Refuse rather than
        # guess: the hold path still commands every other joint to its measured
        # position, so the arm does not move, and the executor reports
        # not-reached.
        if not self.refused:
            self.refused = (
                f"action feature {name!r} has no counterpart in observation.state "
                f"and no previous command to repeat"
            )
            logger.error(f"reset refused: {self.refused}")
            self.events["exit_early"] = True
        return 0.0

    def _check_arrival(self, measured: Mapping[str, float]) -> None:
        """Count this tick as arrived or not, and declare arrival on the window.

        A RATIO over the trailing ``settle_ticks``, not a consecutive run. The
        consecutive version reset the counter to zero on a single tick outside
        ``tol_rad``, so one noisy encoder read -- or one tick where the follower's
        velocity pacing happened to land just outside 0.05 rad -- cost the whole
        settle window and started it again. At 21 Hz the window is 21 ticks, and
        requiring all 21 in a row of a real arm settling against gravity made
        `not_reached` a coin flip; `not_reached` is a CHAIN FAILURE with no
        retry, so the cost of that flip is the rest of the run.

        90% is the threshold, and it is still a window every tick of which was
        MEASURED: the window must be FULL (``settle_ticks`` answers present)
        before arrival can be declared, so this never shortens the settle time,
        it only tolerates up to 10% of it being outside tol.
        """
        if self.tick < self.plan.ticks:
            # Still ramping. Arrival is only meaningful once the trajectory has
            # commanded the target at least once.
            return
        worst = 0.0
        for name in self.arm_joint_names:
            key = f"{name}{POSITION_SUFFIX}"
            if key not in measured:
                return
            worst = max(worst, abs(measured[key] - self.plan.target[name]))
        self.worst_error_rad = worst
        self._arrival_window.append(worst < self.settings.tol_rad)
        window = self._arrival_window
        if len(window) < (window.maxlen or 1):
            return
        inside = sum(1 for value in window if value)
        needed = math.ceil(ARRIVAL_WINDOW_RATIO * len(window))
        if inside >= needed and not self.reached:
            self.reached = True
            self.events["exit_early"] = True
            logger.info(
                f"reset before stage {self.plan.stage_number} reached: worst joint "
                f"error {worst:.4f} rad, {inside}/{len(window)} of the last "
                f"{self.plan.settle_ticks} ticks within tol "
                f"{self.settings.tol_rad:.3f} (needed {needed}); "
                f"{self.tick} ticks total, T={self.plan.duration_s:.2f}s"
            )


def make_reset_bundle(
    stage_id: str,
    plan: ResetPlan,
    events: dict[str, bool],
    *,
    action_names: Sequence[str],
    state_names: Sequence[str],
    settings: ResetSettings | None = None,
    arm_joint_names: Sequence[str] = ARM_JOINT_NAMES,
):
    """Wrap a :class:`ResetPolicy` as a ``PolicyBundle`` so record_loop accepts it.

    ``PolicyBundle``'s field annotations describe the production types and are
    documentation, not enforcement -- ``record_loop`` runs no isinstance check on
    any of the three. Going through the bundle means ``call_record_loop`` needs
    no second signature, and the reset is driven by THE one call site of
    ``record_loop`` in this repo.

    ``PolicyBundle`` is imported inside the function: ``policies`` is the module
    that owns it and importing it at module scope here would be a cycle whenever
    policies is loaded first (``mock_policy`` resolves the same cycle the same
    way).
    """
    from stage_runner.policies import PolicyBundle

    policy = ResetPolicy(
        plan,
        events,
        action_names=action_names,
        state_names=state_names,
        settings=settings,
        arm_joint_names=arm_joint_names,
    )
    return PolicyBundle(
        stage_id=stage_id,
        policy_path=f"reset://{plan.stage_number}",
        config=policy.config,
        policy=policy,
        preprocessor=PassthroughProcessor(),
        postprocessor=PassthroughProcessor(),
        # Nothing to warm: twelve multiplications on the CPU, no kernels.
        warmed_up=False,
    )
