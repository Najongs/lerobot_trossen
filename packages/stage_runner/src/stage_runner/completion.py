"""When is a stage DONE? One ProcessorStep, no thread.

The chain has to decide, at 21 Hz and without a human, that stage k has finished
so the boundary reset may start. The three candidate signals each fail alone:

* **time** cannot: the demonstrations spread 1.3-2.7x between p10 and p90 (§93).
* **"the output stopped"** cannot: 23-35% of the demonstrations of t02·06·08
  pause for over two seconds MID-TASK, and M1 does not stop at the end scene at
  all for t04 (0.56-0.79 rad of continued motion), because ACT's loss excludes
  the chunk steps past the episode end through ``action_is_pad``
  (modeling_act.py:144) and leaves the end scene unsupervised.
* **the progress output** alone cannot: it is a learned scalar and it wobbles in
  exactly those mid-task pauses.

So completion is their CONJUNCTION, and the conjunction is what this module
computes:

    done  =  THE STAGE HAS DEPARTED (see below)
          AND p held >= p_done for p_hold_s      (17-D models only)
          AND the command has stopped for stall_s
          AND the elapsed time has passed p10_s

and for a 16-D model with no progress output, the second clause is replaced by
"the measured arm is within end_pose_tol_rad of one of the demonstrated end
poses".

WHY THE DEPARTURE LATCH IS A PREMISE OF BOTH. Without it the 16-D rule fires on
a stage that never started. In ``configs/chain/stage_params.json`` the
designated START pose of t03, t04 and t10 is within 0.0008 rad of one of that
same stage's twenty END poses, and t01/t05/t07's are inside their own
``end_pose_tol_rad`` as well. (Those three distances are MEASURED off the file.
WHY they are that small -- a stage that mostly drives the base leaves the arm
where it found it -- is [추정]: it is read off the task NAMES, and t04 is
``task04_pour_liquid_from_...``, which is not a moving stage.) P2 measured that
20-40% of M1's holdout starts
predict a stop on the first chunk -- so an arm standing still at its start pose
satisfied "stalled AND near an end pose" and the stage fell to ``complete``
after p10_s having done nothing. Eleven of those in a row is a chain that
reports a full 1->11 run and never moved.

``departed`` is therefore a LATCH on the stage, set once and never cleared,
and it gates the 17-D path too: a progress head that reads 1.0 out of the start
scene is the same failure with a different sensor. It is set by either

* the measured arm being outside ``departure_arm_rad`` (0.10 rad, inf-norm) of
  THE POSE IT STARTED THAT STAGE AT -- the first measurement, not the file's
  designated pose, because with ``reset.initial: false`` and ``from_stage > 1``
  nothing has driven the arm to the designated pose and departure from a pose the
  arm was never at is either already true on tick 1 or unreachable. 0.10 rad is
  twice the reset's own arrival tolerance (``tol_rad`` 0.05), so a reset that
  merely landed at the edge of ``tol_rad`` can never latch it. Required for
  ``DEPARTURE_ARM_TICKS`` CONSECUTIVE ticks: the latch is permanent, and one
  glitched encoder read is indistinguishable from one tick of departure; or
* the COMMANDED base integrating past ``departure_base_rot_rad`` (0.17 rad, 10
  degrees) or ``departure_base_fwd_m`` (0.10 m). Some stages are expected to
  turn or drive with very little arm motion -- t01, t03 and t10 by their names
  (``move_to_tube_rack``, ``turn_to_face_beaker``, ``move_to_beaker_shelf``),
  which is [추정]; the arm clause alone would then never latch for them.

WHY A ProcessorStep AND NOT A WATCHER THREAD. ``robot_action_processor`` receives
``(action, observation)`` as ONE transition every tick
(processor/converters.py:214-238), which is the only place in the loop where the
commanded arm, the commanded base, the progress output and the MEASURED arm are
all in hand at the same instant. A thread reading
``dataset.episode_buffer["observation.state"]`` -- the mechanism
``record_adapter.plan_stage``'s docstring proposed -- sees the measurement and
not the command, samples at its own rate, and races the buffer that
``save_episode`` pops. There is no lock to take here and no rate to pick.

HOW IT STOPS THE LOOP. ``events["exit_early"] = True``, which ``record_loop``
breaks on at the top of its next iteration (lerobot_record.py:343-344). The tick
that fired still completes -- its action is sent and its frame recorded -- which
is deliberate: the action was already computed from an observation taken before
the decision, so suppressing it would mean sending nothing for one tick and
leaving the base holding its previous command for 1/fps longer.

THIS STEP NEVER CHANGES THE ACTION. It returns the transition it was given,
unmodified and not copied. The dict inside it is the same object
``record_loop`` passes to ``build_dataset_frame`` as ``action_values``
(lerobot_record.py:395, :410), so popping ``progress`` out of it here -- the
tempting way to keep a 17-D action off the robot -- is a KeyError on the dataset
write one line later. The robot ignores the extra key by itself:
``MobileAIRobot.send_action`` filters the arm keys by membership in
``arms.action_features`` (mobileai.py:472-474) and reads the base with ``.get``
(:478-480).
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from lerobot.processor.pipeline import ProcessorStep

from stage_runner.chain_params import ARM_JOINT_NAMES, StageParams

logger = logging.getLogger(__name__)

# The 17th action slot: the progress scalar the `tph` recipe trains. Named, not
# indexed -- `make_robot_action` builds the dict by zipping the dataset's action
# NAMES against the tensor (policies/utils.py:197-200), so this key exists
# exactly when `record_adapter.build_dataset_features` declared it.
PROGRESS_KEY: str = "progress"

POSITION_SUFFIX: str = ".pos"

# Base velocity keys, always present in action_features regardless of
# include_base_in_state (mobileai.py:337-339). Named individually as well,
# because the departure latch integrates them SEPARATELY (a rotation threshold
# in radians and a forward one in metres) while the stall rule only needs the
# larger magnitude of the two.
BASE_X_VELOCITY_KEY: str = "x.vel"
BASE_THETA_VELOCITY_KEY: str = "theta.vel"
BASE_VELOCITY_KEYS: tuple[str, ...] = (
    BASE_X_VELOCITY_KEY,
    BASE_THETA_VELOCITY_KEY,
)

# `reason` on a CompletionResult. Two, because the two models are judged by
# different evidence and a chain report that could not tell them apart would
# compare a progress-gated completion against a pose-gated one as if they were
# the same measurement.
REASON_PROGRESS: str = "progress"
REASON_STALL_NN: str = "stall_nn"

# Prefixed onto the `reason` of a stage that ran out its ceiling without the
# departure latch ever setting. It is a DIFFERENT fact from an ordinary timeout
# -- the model never left the start scene, so nothing about the stage's own
# difficulty was measured -- and the chain report counts it separately.
REASON_NEVER_DEPARTED: str = "never_departed"

# How many CONSECUTIVE ticks the arm must be outside `departure_arm_rad` before
# the latch sets. ONE tick used to be enough, and one tick is also what a single
# glitched encoder read looks like: the latch is permanent and it is the premise
# of every completion, so a 1-tick spike handed the rest of the stage to the
# 16-D "stalled near an end pose" rule on a stage that never moved -- which is
# the exact failure the latch exists to prevent (module docstring). THREE at 21 Hz
# is 143 ms, far below the 0.25-1.28 rad a real departure covers, and the
# reset's own arrival test already tolerates up to 10% of its window being
# outside tolerance for the same reason.
#
# The BASE clause is deliberately NOT streaked: it is an INTEGRAL of the
# commanded velocity, so a one-tick spike contributes one tick's worth of
# radians and cannot cross 0.17 rad by itself. Integration is its own filter.
DEPARTURE_ARM_TICKS: int = 3

# How many CONSECUTIVE non-finite ticks make the monitor blind. A blind monitor
# can never declare completion, so the stage ends on its timeout or on the right
# arrow -- which is what "no evidence" has to mean. One NaN is a glitch and is
# simply not counted as evidence either way (the sample is marked non-finite and
# every window containing it refuses); a run of them means the signal is gone.
NON_FINITE_BLIND_TICKS: int = 3

# How far the file's DESIGNATED start pose may sit from the stage's first
# MEASURED arm pose before the monitor says so. It is a warning and nothing else:
# the departure origin is always the first measurement (see
# `_update_departure`), so a disagreement does not change any judgement -- it
# means the arm is not where the reset was supposed to leave it, which is a thing
# to look at in the start scene. 0.3 rad is the same number the initial reset's
# refusal message tells the operator to get the arms within by hand.
START_POSE_WARN_RAD: float = 0.3


@dataclass(frozen=True)
class CompletionSettings:
    """The YAML's ``completion:`` block.

    Defaults are the design's (§3.1). ``stall_s`` of 3 s is about 63 ticks at 21
    Hz, i.e. two full 30-step chunks, so a window where every tick is both
    tracking its command and not travelling already contains what
    ``end_stationary_check`` measures offline -- there is no separate
    "end-of-chunk displacement" test to run.

    ``stall_track_rad`` and ``stall_arm_rad`` were ONE knob until 2026-10-06 and
    had to be split: the first is a TRACKING error (how far the arm lags the
    command it is being given) and the second is a MOVEMENT (how far the
    measured arm travelled across the window). They are different quantities in
    different directions -- a steady-state following error of 0.06 rad under
    load is a perfectly stopped arm, while 0.06 rad of travel is not -- and one
    number could only ever be right for one of them. A single knob set tight
    enough to catch slow travel (0.05) declared a loaded, stationary arm "still
    moving" and the stage then ran to its timeout.
    """

    p_done: float = 0.95
    p_hold_s: float = 1.0
    stall_s: float = 3.0
    # |commanded - measured| per joint: the arm is FOLLOWING its command.
    stall_track_rad: float = 0.08
    # Peak-to-peak of the MEASURED joint across the window: the arm is NOT
    # TRAVELLING. Also the inf-norm radius the 16-D end-pose NN is judged in.
    stall_arm_rad: float = 0.05
    stall_base: float = 0.05
    # Departure latch (see the module docstring). 0.10 rad is twice
    # ResetSettings.tol_rad, so a reset that only just arrived cannot latch it.
    departure_arm_rad: float = 0.10
    # |integral of the COMMANDED theta.vel| over the stage, in radians. 0.17 is
    # about 10 degrees.
    departure_base_rot_rad: float = 0.17
    # |integral of the COMMANDED x.vel| over the stage, in metres.
    departure_base_fwd_m: float = 0.10


@dataclass(frozen=True)
class CompletionResult:
    """Why the monitor declared the stage finished, with the numbers behind it.

    Every field lands in the ``stage_end`` event's ``reason_detail``. A completion
    whose numbers are not recorded cannot be told from a mislabelled one months
    later, which is the mistake ``record_adapter.classify_termination`` already
    refuses to make for the manual terminator.
    """

    reason: str
    elapsed_s: float
    ticks: int
    stall_s: float
    p_last: float | None = None
    p_hold_s: float | None = None
    nn_dist: float | None = None
    nn_index: int | None = None
    # Elapsed seconds at which the departure latch set. Never None on a
    # CompletionResult -- the latch is a premise of every completion -- but
    # typed optional because `Departure` carries None for a stage that never
    # departed and the two are read through the same key in the report.
    departed_s: float | None = None

    def as_detail(self) -> dict[str, Any]:
        return {
            "completion_reason": self.reason,
            "elapsed_s": round(self.elapsed_s, 3),
            "ticks": self.ticks,
            "stall_s": round(self.stall_s, 3),
            "p_last": None if self.p_last is None else round(self.p_last, 4),
            "p_hold_s": None if self.p_hold_s is None else round(self.p_hold_s, 3),
            "nn_dist": None if self.nn_dist is None else round(self.nn_dist, 4),
            "nn_index": self.nn_index,
            "departed": self.departed_s is not None,
            "departed_s": (
                None if self.departed_s is None else round(self.departed_s, 3)
            ),
        }


@dataclass(frozen=True)
class Departure:
    """Whether (and when) the stage that just ended ever left its start scene.

    Read off the monitor AFTER ``end_stage``, because the interesting case is
    the one where no CompletionResult exists: a stage that timed out having
    never departed is a different finding from one that timed out mid-task, and
    only the monitor saw the difference.
    """

    departed: bool
    at_s: float | None
    arm_rad: float
    base_rot_rad: float
    base_fwd_m: float
    # Why the monitor went blind, or "". A blind stage that did not latch is
    # "unknown", NOT "did not depart": the arm may never have been readable at
    # all, and reporting `arm_rad: 0.000` as if it had been measured sends the
    # operator to the start scene when the fault is in the monitor's input.
    blind: str = ""
    # How far the stage's FIRST MEASURED arm pose was from the file's designated
    # start pose, or None when the file carried no comparable pose. Reported only:
    # the departure origin is the measurement, never the designated pose.
    start_pose_offset_rad: float | None = None

    @property
    def unknown(self) -> bool:
        return bool(self.blind) and not self.departed

    def as_detail(self) -> dict[str, Any]:
        # `_meas_` in the names, because `departure_arm_rad` and friends are
        # the SETTINGS and these are the measurements against them. The two
        # landing in one flat `reason_detail` under the same name is how a
        # threshold gets read months later as a measurement.
        return {
            "departed": self.departed,
            "departed_s": None if self.at_s is None else round(self.at_s, 3),
            "departure_unknown": self.unknown,
            "departure_meas_arm_rad": round(self.arm_rad, 4),
            "departure_meas_base_rot_rad": round(self.base_rot_rad, 4),
            "departure_meas_base_fwd_m": round(self.base_fwd_m, 4),
            "start_pose_offset_rad": (
                None
                if self.start_pose_offset_rad is None
                else round(self.start_pose_offset_rad, 4)
            ),
        }


@dataclass
class _Sample:
    t: float
    # max |commanded - measured| over the 12 reset joints.
    track_error: float
    # max |x.vel|, |theta.vel| as COMMANDED.
    base: float
    # The 17th action slot, or None for a 16-D model.
    progress: float | None
    # Measured positions of the 12 reset joints, in ARM_JOINT_NAMES order.
    measured: tuple[float, ...]
    # Was EVERY number behind this tick finite -- the 12 commanded and 12 measured
    # joints, both base velocities, and the progress slot when there is one?
    #
    # False means NO EVIDENCE, in either direction. Every rule that reads a window
    # refuses outright if one sample in it is non-finite, rather than letting the
    # value through: `max()` and `min()` both RETURN THE OTHER ARGUMENT when one is
    # NaN, so a NaN travelled through the peak-to-peak test as 0.0 of travel and
    # through `_inf_norm` as 0.0 of distance -- i.e. a lost signal read as "the arm
    # is perfectly still, exactly on an end pose", which is the completion
    # condition itself.
    finite: bool = True


@dataclass
class _Stage:
    params: StageParams
    has_progress: bool
    started: float
    samples: list[_Sample] = field(default_factory=list)
    ticks: int = 0
    result: CompletionResult | None = None
    # The pose departure is measured FROM: ALWAYS the first finite tick's
    # measurement, never the file's designated pose. In ARM_JOINT_NAMES order,
    # like _Sample.measured. See `_update_departure` for why.
    start_pose: tuple[float, ...] | None = None
    # The file's designated start pose, for the one-time comparison against
    # `start_pose` above. It is NOT the origin of the departure measurement.
    declared_start_pose: tuple[float, ...] | None = None
    # How far the two were apart on the first finite tick, or None if there was
    # no designated pose to compare against. Reported, never acted on.
    start_pose_offset_rad: float | None = None
    # Consecutive ticks the measured arm has been outside `departure_arm_rad`.
    # DEPARTURE_ARM_TICKS of them latch `departed`; anything less is a glitch.
    arm_departure_ticks: int = 0
    # Consecutive non-finite ticks. NON_FINITE_BLIND_TICKS of them blind the
    # monitor. Reset by any finite tick -- it is a run, not a total.
    non_finite_ticks: int = 0
    # The latch. Set once, never cleared, and a premise of every completion.
    # NOT derived from `samples`: _prune drops anything older than the longest
    # window, so by the time a stage could complete the ticks that proved it
    # moved are long gone.
    departed: bool = False
    departed_at_s: float | None = None
    # Largest inf-norm distance from start_pose seen so far, for the log.
    worst_arm_rad: float = 0.0
    # SIGNED integrals of the commanded base velocity. The threshold is on
    # |integral| as the design states it: a stage that turned ten degrees and
    # turned back has a net rotation of zero and is, for this purpose, still
    # standing where it started.
    base_rot_rad: float = 0.0
    base_fwd_m: float = 0.0
    # Clock reading of the previous tick, for dt. None on the first tick, which
    # therefore integrates nothing.
    last_t: float | None = None
    # Set once when a key the monitor needs is missing from the action or the
    # observation. While it is set the monitor can never declare completion, so
    # the stage ends on the timeout or on the operator's right arrow -- it must
    # never guess "done" from data it could not read.
    blind: str = ""


class CompletionMonitorStep(ProcessorStep):
    """Watches one stage at a time and sets ``exit_early`` when it is done.

    Lives for the whole trial and is told which stage it is in:
    :meth:`begin_stage` before the stage's ``record_loop`` and
    :meth:`end_stage` after it. Between stages -- during a boundary reset, and
    during the gap where ``stop_base`` runs -- it is INERT and returns every
    transition untouched.

    ``record_loop`` resets the policy and BOTH POLICY processors on entry
    (lerobot_record.py:332-335) but never the ``robot_action_processor`` this
    step lives in, so :meth:`begin_stage` is the only thing that clears the
    window. Forgetting it would carry the previous stage's three seconds of
    stalled samples into the next stage and declare it complete on its first
    tick.
    """

    def __init__(
        self,
        events: dict[str, bool],
        *,
        settings: CompletionSettings | None = None,
        arm_joint_names: Sequence[str] = ARM_JOINT_NAMES,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.events = events
        self.settings = settings or CompletionSettings()
        self.arm_joint_names = tuple(arm_joint_names)
        # Injectable so the synthetic sequences in the tests are deterministic
        # rather than dependent on how fast the machine ran them. perf_counter
        # by default, which is the SAME clock as the runner's elapsed_s and as
        # the `t_mono` column of basevel.csv (base_vel_log.py:69) -- that shared
        # clock is what lets eval_chain_report integrate the base over one
        # stage's exact range.
        self.clock = clock
        self._stage: _Stage | None = None
        # The departure of the stage that most recently ended. Stashed by
        # end_stage because it drops the _Stage, and the executor needs it for
        # the stages where completion did NOT fire -- which is the whole point
        # of recording it.
        self.last_departure: Departure | None = None

    # ------------------------------------------------------------------ control

    def begin_stage(
        self,
        params: StageParams,
        *,
        has_progress: bool,
        started: float | None = None,
    ) -> None:
        """Arm the monitor for one policy stage. Clears all window state.

        ``started`` is the executor's own ``perf_counter`` reading, passed so
        the monitor's ``elapsed_s`` and the StageResult's are THE SAME NUMBER.
        Letting the monitor take its own reading put the two a
        ``begin_stage`` + bookkeeping apart, which is small but means the
        ``elapsed_s >= p10_s`` the event log shows is not the comparison the
        monitor actually made.
        """
        self._stage = _Stage(
            params=params,
            has_progress=bool(has_progress),
            started=self.clock() if started is None else float(started),
            # NOT `start_pose`: the origin is the first MEASURED pose and nothing
            # else. This one is only ever compared against it.
            declared_start_pose=self._designated_start_pose(params),
        )
        self.last_departure = None

    def _designated_start_pose(self, params: StageParams) -> tuple[float, ...] | None:
        """``params.start_pose`` in arm order, or None when the file lacks a joint.

        By NAME, never by index -- the same rule the whole chain follows. None
        when the file does not carry every joint this monitor was given, which
        the loader already refuses; the tolerance exists so a hand-built
        StageParams in a test (or a future 6-joint rig) still works.

        THIS IS NOT THE DEPARTURE ORIGIN. It used to be, and that was wrong for
        every stage the arm does not actually start at: with `reset.initial:
        false` and `from_stage > 1` -- the configuration for resuming a chain
        mid-way -- nothing has driven the arm to this pose, so the first
        measurement can be anywhere, and measuring departure from a pose the arm
        was never at either latches on the first tick (the arm "has already left")
        or never latches (it started on the far side and moved towards it).
        """
        pose = getattr(params, "start_pose", None) or {}
        if not all(name in pose for name in self.arm_joint_names):
            return None
        return tuple(float(pose[name]) for name in self.arm_joint_names)

    def end_stage(self) -> CompletionResult | None:
        """Disarm, and return the result if completion fired. None means it did not.

        None is the signal the caller turns into "timeout" or "manual": the
        monitor is the only thing that knows the difference between a stage that
        ran out of time and one a human ended, because ``record_loop``
        self-clears ``exit_early`` on its way out and leaves no other evidence.
        """
        stage = self._stage
        self._stage = None
        if stage is None:
            self.last_departure = None
            return None
        self.last_departure = Departure(
            departed=stage.departed,
            at_s=stage.departed_at_s,
            arm_rad=stage.worst_arm_rad,
            base_rot_rad=stage.base_rot_rad,
            base_fwd_m=stage.base_fwd_m,
            blind=stage.blind,
            start_pose_offset_rad=stage.start_pose_offset_rad,
        )
        if not stage.departed and stage.blind:
            logger.warning(
                f"stage {stage.params.number} ({stage.params.name}) departure is "
                "UNKNOWN, not false: the monitor was blind "
                f"({stage.blind}), so the arm may never have been readable. "
                "Do not read the departure numbers as a measurement."
            )
        elif not stage.departed:
            logger.warning(
                f"stage {stage.params.number} ({stage.params.name}) NEVER "
                f"DEPARTED: the measured arm stayed within "
                f"{stage.worst_arm_rad:.3f} rad of the pose it STARTED THIS STAGE "
                f"AT (threshold {self.settings.departure_arm_rad:.3f} for "
                f"{DEPARTURE_ARM_TICKS} ticks in a row) and the "
                f"commanded base integrated {stage.base_rot_rad:+.3f} rad / "
                f"{stage.base_fwd_m:+.3f} m. Completion was therefore refused "
                "for the whole stage -- the model did not leave the start scene."
            )
        if stage.blind:
            logger.warning(
                f"completion monitor was blind for stage {stage.params.number} "
                f"({stage.blind}); it could not declare completion, so this "
                "stage ended on the timeout or on the right arrow"
            )
        return stage.result

    @property
    def armed(self) -> bool:
        return self._stage is not None

    # ------------------------------------------------------------- ProcessorStep

    def __call__(self, transition):
        stage = self._stage
        if stage is None or stage.result is not None:
            # Inert between stages, and inert for the rest of a stage that has
            # already fired: the loop breaks on its next iteration, and a second
            # decision from the same stage would overwrite the numbers that
            # explain the first.
            return transition
        try:
            self._observe(stage, transition)
        except Exception:
            # NEVER kill the run from the monitor. It is a judgement about when
            # to stop, and the fallbacks (the timeout and the operator's right
            # arrow) both still work. An exception here would unwind through
            # record_loop's @safe_stop_image_writer and discard the episode.
            if not stage.blind:
                stage.blind = "an exception escaped the monitor; see the traceback"
                logger.exception("completion monitor tick failed; monitor disabled")
        return transition

    def transform_features(self, features):
        # This step reads and never writes, so the pipeline's feature description
        # is unchanged. (It is also never serialised: the robot_action_processor
        # is built fresh by make_default_processors every run.)
        return features

    def get_config(self) -> dict[str, Any]:
        return {
            "p_done": self.settings.p_done,
            "p_hold_s": self.settings.p_hold_s,
            "stall_s": self.settings.stall_s,
            "stall_track_rad": self.settings.stall_track_rad,
            "stall_arm_rad": self.settings.stall_arm_rad,
            "stall_base": self.settings.stall_base,
            "departure_arm_rad": self.settings.departure_arm_rad,
            "departure_base_rot_rad": self.settings.departure_base_rot_rad,
            "departure_base_fwd_m": self.settings.departure_base_fwd_m,
        }

    # ------------------------------------------------------------------ internals

    def _observe(self, stage: _Stage, transition) -> None:
        from lerobot.processor.core import TransitionKey

        action = transition.get(TransitionKey.ACTION)
        observation = transition.get(TransitionKey.OBSERVATION)
        if not isinstance(action, dict) or not isinstance(observation, dict):
            self._blind(stage, "the transition carried no action/observation dict")
            return

        commanded: list[float] = []
        measured: list[float] = []
        for name in self.arm_joint_names:
            key = f"{name}{POSITION_SUFFIX}"
            if key not in action or key not in observation:
                self._blind(
                    stage,
                    f"{key!r} is missing from the "
                    f"{'action' if key not in action else 'observation'}",
                )
                return
            commanded.append(_as_float(action[key]))
            measured.append(_as_float(observation[key]))

        now = self.clock()
        stage.ticks += 1
        x_velocity = _as_float(action.get(BASE_X_VELOCITY_KEY, 0.0))
        theta_velocity = _as_float(action.get(BASE_THETA_VELOCITY_KEY, 0.0))

        # Everything the departure latch and the stall rule read. Checked HERE
        # and once, because the comparisons downstream cannot check it for
        # themselves: `max(a, nan)` is `a` and `nan >= x` is False, so a NaN that
        # reaches them is not an error, it is a silent vote for "nothing moved".
        arm_base_finite = all(
            math.isfinite(value)
            for value in (*commanded, *measured, x_velocity, theta_velocity)
        )

        # BEFORE the progress-key check, before _decide, and before _prune could
        # drop the evidence. The ORDER against the progress check is the point:
        # a 17-D model whose action is missing `progress` blinds the monitor, and
        # if the departure were updated after that check a blind stage would
        # report "never departed, worst arm 0.000 rad" -- a fabricated
        # measurement pointing the operator at the start scene when the real
        # cause is that the monitor could not read its signal. The arm and the
        # base ARE readable here; only the progress slot is not.
        #
        # SKIPPED ENTIRELY on a non-finite tick: the origin must never be a NaN
        # (it is kept for the whole stage) and the base integrals must never be
        # poisoned by one (`nan + x` is nan forever, which would then latch the
        # base clause on the `abs(...) >= threshold` test never being true again
        # -- or rather, never being true at all, which is the silent direction).
        if arm_base_finite:
            self._update_departure(stage, now, measured, x_velocity, theta_velocity)
        else:
            # A glitched tick is not evidence that the arm stood still, so the
            # consecutive-tick streak restarts rather than carrying across the gap.
            stage.arm_departure_ticks = 0
        # Always, finite or not: the interval spanning a skipped tick is then
        # simply not integrated, which under-counts the base and therefore
        # latches LATER. Refusing completion for longer is the safe direction.
        stage.last_t = now

        progress: float | None = None
        if stage.has_progress:
            if PROGRESS_KEY not in action:
                self._blind(
                    stage,
                    f"the model was declared 17-D but the action has no "
                    f"{PROGRESS_KEY!r} key",
                )
                return
            progress = _as_float(action[PROGRESS_KEY])

        finite = arm_base_finite and (progress is None or math.isfinite(progress))
        if finite:
            stage.non_finite_ticks = 0
        else:
            stage.non_finite_ticks += 1
            logger.warning(
                f"stage {stage.params.number} ({stage.params.name}) tick "
                f"{stage.ticks} carried a non-finite value "
                f"(arm/base finite: {arm_base_finite}, progress: {progress!r}). "
                "It counts as NO EVIDENCE: every window containing it refuses, in "
                "both directions. "
                f"{stage.non_finite_ticks}/{NON_FINITE_BLIND_TICKS} in a row."
            )
            if stage.non_finite_ticks >= NON_FINITE_BLIND_TICKS:
                self._blind(
                    stage,
                    f"{stage.non_finite_ticks} consecutive ticks carried a "
                    "non-finite value (NaN or inf) in the arm, the base command "
                    "or the progress slot",
                )

        stage.samples.append(
            _Sample(
                t=now,
                track_error=(
                    max(abs(c - m) for c, m in zip(commanded, measured))
                    if arm_base_finite
                    else float("nan")
                ),
                base=(
                    max(abs(x_velocity), abs(theta_velocity))
                    if arm_base_finite
                    else float("nan")
                ),
                progress=progress,
                measured=tuple(measured),
                finite=finite,
            )
        )
        self._prune(stage, now)
        if stage.blind:
            return
        result = self._decide(stage, now)
        if result is None:
            return
        stage.result = result
        # The loop breaks at the top of its NEXT iteration; this tick's action is
        # still sent and still recorded. See the module docstring.
        self.events["exit_early"] = True
        logger.info(
            f"stage {stage.params.number} ({stage.params.name}) complete by "
            f"{result.reason}: {result.as_detail()}"
        )

    def _update_departure(
        self,
        stage: _Stage,
        now: float,
        measured: Sequence[float],
        x_velocity: float,
        theta_velocity: float,
    ) -> None:
        """Integrate the base, measure the arm, and latch ``departed`` once.

        Called as soon as the arm and the base have been read, which is BEFORE
        the progress-key check -- so a 17-D model with a missing ``progress``
        slot still gets a real departure measurement. It is NOT called when the
        arm itself was unreadable, nor when any of the numbers was non-finite;
        then the departure is unknown and ``Departure.blind`` says so, because a
        fabricated 0.000 rad would point the operator at the start scene instead
        of at the monitor's input.

        THE ORIGIN IS THE FIRST MEASURED POSE, ALWAYS -- never the file's
        designated start pose. Those two are the same thing only when a reset has
        just driven the arm there, which is the ordinary chain and was the only
        case the first version considered. They are NOT the same with
        ``reset.initial: false`` and ``from_stage > 1``: nothing drove the arm
        anywhere, the operator placed it by hand, and a departure measured from a
        pose the arm was never at is either already satisfied on tick 1 (so a
        stage that never moves can complete -- the failure the latch exists to
        prevent) or unsatisfiable (the arm starts on the far side and the latch
        reads its motion TOWARDS the designated pose as getting closer to home).
        Measuring from where the stage actually began is the only reading that
        means "this stage moved", which is the question the latch asks.

        The designated pose is still used, for ONE warning: if the two are more
        than ``START_POSE_WARN_RAD`` apart the arm is not where the chain expects
        it, which is worth saying once and is not worth changing a judgement over.
        """
        if stage.start_pose is None:
            # First finite tick: its measurement IS the origin, so this tick is
            # 0.0 away from it by construction.
            stage.start_pose = tuple(measured)
            declared = stage.declared_start_pose
            if declared is not None:
                offset = max(
                    (abs(value - origin) for value, origin in zip(measured, declared)),
                    default=0.0,
                )
                stage.start_pose_offset_rad = offset
                if offset >= START_POSE_WARN_RAD:
                    logger.warning(
                        f"stage {stage.params.number} ({stage.params.name}) did "
                        f"NOT start at its designated start pose: the first "
                        f"measured arm pose is {offset:.3f} rad away from it "
                        f"(threshold {START_POSE_WARN_RAD:.2f}). Departure is "
                        "measured from where the stage ACTUALLY began, so this "
                        "changes no judgement -- but the policy is rolling out "
                        "from a pose the demonstrations do not have. Check the "
                        "preceding reset's arrival error, or whether "
                        "`chain.reset.initial: false` skipped the ramp that was "
                        "supposed to put the arm there."
                    )

        delta = 0.0 if stage.last_t is None else max(0.0, now - stage.last_t)
        stage.base_fwd_m += x_velocity * delta
        stage.base_rot_rad += theta_velocity * delta

        worst = 0.0
        for value, origin in zip(measured, stage.start_pose):
            worst = max(worst, abs(value - origin))
        stage.worst_arm_rad = max(stage.worst_arm_rad, worst)

        if stage.departed:
            return
        settings = self.settings
        # The STREAK, not this tick alone. See DEPARTURE_ARM_TICKS: the latch is
        # permanent and a single glitched encoder read looks exactly like one tick
        # of departure, which then hands the whole stage to the rule the latch is
        # there to gate.
        if worst >= settings.departure_arm_rad:
            stage.arm_departure_ticks += 1
        else:
            stage.arm_departure_ticks = 0
        why = ""
        if stage.arm_departure_ticks >= DEPARTURE_ARM_TICKS:
            why = (
                f"the arm has been {worst:.3f} rad from the pose it started this "
                f"stage at for {stage.arm_departure_ticks} ticks in a row"
            )
        elif abs(stage.base_rot_rad) >= settings.departure_base_rot_rad:
            why = f"the base has turned {stage.base_rot_rad:+.3f} rad"
        elif abs(stage.base_fwd_m) >= settings.departure_base_fwd_m:
            why = f"the base has driven {stage.base_fwd_m:+.3f} m"
        if not why:
            return
        stage.departed = True
        stage.departed_at_s = now - stage.started
        logger.info(
            f"stage {stage.params.number} ({stage.params.name}) departed at "
            f"{stage.departed_at_s:.2f}s: {why}. Completion is now allowed to "
            "fire; before this it could not."
        )

    def _blind(self, stage: _Stage, why: str) -> None:
        if not stage.blind:
            stage.blind = why
            logger.warning(
                f"completion monitor cannot judge stage {stage.params.number}: "
                f"{why}. The stage will end on its timeout or on the right arrow; "
                "it will NOT be declared complete."
            )

    def _prune(self, stage: _Stage, now: float) -> None:
        # Keep a little more than the longest window any rule looks at, so the
        # window is always fully covered when the rule asks for it. Unbounded
        # growth matters: a 32 s stage at 21 Hz is 670 samples, and a 300 s
        # ceiling would be 6,300 -- small, but the list is walked every tick.
        horizon = max(self.settings.stall_s, self.settings.p_hold_s) + 1.0
        cutoff = now - horizon
        if stage.samples and stage.samples[0].t < cutoff:
            stage.samples = [s for s in stage.samples if s.t >= cutoff]

    def _decide(self, stage: _Stage, now: float) -> CompletionResult | None:
        elapsed = now - stage.started

        # DEPARTURE FIRST, for BOTH models. A stage that never left its start
        # scene has not finished, whatever the progress head says and however
        # near a demonstrated end pose the arm is standing -- for t03, t04 and
        # t10 the designated start pose IS one of the end poses to within 0.001
        # rad. See the module docstring.
        if not stage.departed:
            return None

        # FLOOR, and it is a floor on the ELAPSED TIME, not on the number
        # of ticks: a mid-task pause of over two seconds happens in 23-35% of the
        # demonstrations of t02·06·08 (§93), and without this the monitor would
        # read the first such pause as the end of the stage.
        if elapsed < stage.params.p10_s:
            return None

        stalled = self._stalled(stage, now)
        if stalled is None:
            return None

        if stage.has_progress:
            held = self._progress_held(stage, now)
            if held is None:
                return None
            return CompletionResult(
                reason=REASON_PROGRESS,
                elapsed_s=elapsed,
                ticks=stage.ticks,
                stall_s=stalled,
                p_last=stage.samples[-1].progress,
                p_hold_s=held,
                departed_s=stage.departed_at_s,
            )

        # 16-D model: no progress output exists, so "stopped" has to be
        # corroborated by WHERE it stopped. M1 does not stop at the end scene for
        # every stage (§93, t04), so this is a weaker signal than progress and
        # that is exactly why `allow_manual_complete` is on for M1 too.
        distance, index = self._nearest_end_pose(stage)
        if distance is None or distance > stage.params.end_pose_tol_rad:
            return None
        return CompletionResult(
            reason=REASON_STALL_NN,
            elapsed_s=elapsed,
            ticks=stage.ticks,
            stall_s=stalled,
            nn_dist=distance,
            nn_index=index,
            departed_s=stage.departed_at_s,
        )

    def _stalled(self, stage: _Stage, now: float) -> float | None:
        """Length of the trailing window in which nothing moved, or None.

        Three conditions on EVERY tick of the trailing ``stall_s`` seconds:
        the arm is tracking its command (``|cmd - meas| < stall_track_rad``),
        the base is commanded to a stop (``max|vel| < stall_base``), and the
        MEASURED arm has not travelled across the window (per-joint
        peak-to-peak < ``stall_arm_rad``).

        The third is not redundant with the first. A policy that commands a slow
        steady ramp is tracked perfectly at every instant -- ``|cmd - meas|``
        stays tiny -- while the arm crosses the workspace. Peak-to-peak over the
        window is what sees that.

        THE FIRST AND THE THIRD READ DIFFERENT KNOBS, and that is the point. A
        tracking error is a lag against a command; travel is displacement of the
        measurement. One threshold for both has to be set tight enough for
        travel (0.05 rad), and at that value a stationary arm holding a load
        against gravity -- which lags its command by a steady 0.06 rad and is
        not moving at all -- is read as "still travelling" forever.
        """
        settings = self.settings
        window = self._covered_window(stage, now, settings.stall_s)
        if window is None:
            return None
        for sample in window:
            # NO EVIDENCE is not evidence of stillness. The three tests below all
            # read a NaN as "under the threshold": `nan >= x` is False, and
            # `max(...) - min(...)` over a list containing a NaN is whatever the
            # finite values gave, because both builtins return the other argument
            # when one side is NaN. So a lost signal looked exactly like a
            # perfectly stopped arm -- the completion condition itself.
            if not sample.finite:
                return None
            if sample.track_error >= settings.stall_track_rad:
                return None
            if sample.base >= settings.stall_base:
                return None
        for index in range(len(self.arm_joint_names)):
            values = [sample.measured[index] for sample in window]
            if max(values) - min(values) >= settings.stall_arm_rad:
                return None
        return now - window[0].t

    def _covered_window(
        self, stage: _Stage, now: float, length_s: float
    ) -> list[_Sample] | None:
        """The trailing ``length_s`` of samples, or None if it is not FULLY covered.

        Covered means a sample exists at or before the cutoff, which proves the
        record spans the whole window. Two cases this rejects, both of which
        would otherwise make "every tick in the window was still" trivially true:

        * the stage has not run for ``length_s`` yet -- a stage that began
          already stopped (P2: 20-40% of M1's holdout starts predict a stop on
          the first chunk) would otherwise satisfy "every tick in the window was
          still" on its first tick. This is a bound on the EVIDENCE, not a
          premise about motion; the premise that the stage actually moved is the
          departure latch, and that one is what finally closed the P2 case;
        * the loop froze -- a RealSense ``async_read`` blocking for seconds
          leaves one sample inside the window and nothing behind it, so the
          window would contain a single tick.
        """
        if not stage.samples:
            return None
        cutoff = now - length_s
        if stage.samples[0].t > cutoff:
            return None
        window = [sample for sample in stage.samples if sample.t >= cutoff]
        return window or None

    def _progress_held(self, stage: _Stage, now: float) -> float | None:
        """Length of the trailing window in which ``p >= p_done``, or None.

        A non-finite sample refuses the window outright, the same way
        :meth:`_stalled` does. ``nan < p_done`` is False, so without the check a
        NaN progress scalar would have PASSED the "p is high enough" test.
        """
        settings = self.settings
        window = self._covered_window(stage, now, settings.p_hold_s)
        if window is None:
            return None
        for sample in window:
            if not sample.finite:
                return None
            if sample.progress is None or not math.isfinite(sample.progress):
                return None
            if sample.progress < settings.p_done:
                return None
        return now - window[0].t

    def _nearest_end_pose(self, stage: _Stage) -> tuple[float | None, int | None]:
        """Inf-norm distance from the latest measured pose to the closest end pose.

        INF-NORM, not L2: the tolerance in the file is ``end_pose_tol_rad``, a
        per-joint allowance in radians, and an L2 over 12 joints would let one
        joint be 0.5 rad out while the sum stayed under a 0.15 budget. It is also
        the same norm the reset's own arrival test uses
        (``max|arm - target| < tol``), so "arrived" means one thing in this
        package.
        """
        if not stage.samples:
            return (None, None)
        measured = stage.samples[-1].measured
        best: float | None = None
        best_index: int | None = None
        for index, pose in enumerate(stage.params.end_poses):
            distance = _inf_norm(measured, pose, self.arm_joint_names)
            if distance is None:
                continue
            if best is None or distance < best:
                best, best_index = distance, index
        return (best, best_index)


def _as_float(value: Any) -> float:
    """``float(value)``, or NaN for anything that is not a number.

    NaN rather than a raise, because every caller already treats non-finite as
    "no evidence" and a raise here would blind the monitor through its
    ``except Exception`` with a traceback instead of the one-line warning that
    names the tick. A bool is NOT a number for this purpose: ``float(True)`` is
    1.0 radians, which nothing in this package ever legitimately means.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return float("nan")
    return float(value)


def _inf_norm(
    measured: Sequence[float], pose: Mapping[str, float], names: Sequence[str]
) -> float | None:
    """Inf-norm distance, or None when anything involved is missing or non-finite.

    None FOR A NaN, not a distance. ``max(worst, abs(value - nan))`` returns
    ``worst``, so a NaN joint used to vanish from the norm entirely and a pose
    with eleven matching joints and one unreadable one came out as "0.00 rad from
    a demonstrated end pose" -- the strongest evidence the 16-D rule accepts,
    produced by the absence of a measurement.
    """
    worst = 0.0
    for value, name in zip(measured, names):
        if name not in pose:
            return None
        difference = abs(value - pose[name])
        if not math.isfinite(difference):
            return None
        worst = max(worst, difference)
    return worst
