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

    done  =  p held >= p_done for p_hold_s      (17-D models only)
          AND the command has stopped for stall_s
          AND the elapsed time has passed p10_s

and for a 16-D model with no progress output, the first clause is replaced by
"the measured arm is within end_pose_tol_rad of one of the demonstrated end
poses".

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
# include_base_in_state (mobileai.py:337-339).
BASE_VELOCITY_KEYS: tuple[str, ...] = ("x.vel", "theta.vel")

# `reason` on a CompletionResult. Two, because the two models are judged by
# different evidence and a chain report that could not tell them apart would
# compare a progress-gated completion against a pose-gated one as if they were
# the same measurement.
REASON_PROGRESS: str = "progress"
REASON_STALL_NN: str = "stall_nn"


@dataclass(frozen=True)
class CompletionSettings:
    """The YAML's ``completion:`` block.

    Defaults are the design's (§3.1). ``stall_s`` of 3 s is about 63 ticks at 21
    Hz, i.e. two full 30-step chunks, so a window where every tick has
    ``|cmd - meas| < stall_arm_rad`` already contains what
    ``end_stationary_check`` measures offline -- there is no separate
    "end-of-chunk displacement" test to run.
    """

    p_done: float = 0.95
    p_hold_s: float = 1.0
    stall_s: float = 3.0
    stall_arm_rad: float = 0.05
    stall_base: float = 0.05


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


@dataclass
class _Stage:
    params: StageParams
    has_progress: bool
    started: float
    samples: list[_Sample] = field(default_factory=list)
    ticks: int = 0
    result: CompletionResult | None = None
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

    # ------------------------------------------------------------------ control

    def begin_stage(
        self,
        params: StageParams,
        *,
        has_progress: bool,
        started: float | None = None,
    ) -> None:
        """Arm the monitor for one policy stage. Clears all window state."""
        self._stage = _Stage(
            params=params,
            has_progress=bool(has_progress),
            started=self.clock() if started is None else float(started),
        )

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
            return None
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
            "stall_arm_rad": self.settings.stall_arm_rad,
            "stall_base": self.settings.stall_base,
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
            commanded.append(float(action[key]))
            measured.append(float(observation[key]))

        progress: float | None = None
        if stage.has_progress:
            if PROGRESS_KEY not in action:
                self._blind(
                    stage,
                    f"the model was declared 17-D but the action has no "
                    f"{PROGRESS_KEY!r} key",
                )
                return
            progress = float(action[PROGRESS_KEY])

        now = self.clock()
        stage.ticks += 1
        stage.samples.append(
            _Sample(
                t=now,
                track_error=max(
                    abs(c - m) for c, m in zip(commanded, measured)
                ),
                base=max(
                    abs(float(action.get(key, 0.0))) for key in BASE_VELOCITY_KEYS
                ),
                progress=progress,
                measured=tuple(measured),
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

        # FLOOR FIRST, and it is a floor on the ELAPSED TIME, not on the number
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
        )

    def _stalled(self, stage: _Stage, now: float) -> float | None:
        """Length of the trailing window in which nothing moved, or None.

        Three conditions on EVERY tick of the trailing ``stall_s`` seconds:
        the arm is tracking its command (``|cmd - meas| < stall_arm_rad``), the
        base is commanded to a stop (``max|vel| < stall_base``), and the MEASURED
        arm has not travelled across the window (per-joint peak-to-peak <
        ``stall_arm_rad``).

        The third is not redundant with the first. A policy that commands a slow
        steady ramp is tracked perfectly at every instant -- ``|cmd - meas|``
        stays tiny -- while the arm crosses the workspace. Peak-to-peak over the
        window is what sees that.
        """
        settings = self.settings
        window = self._covered_window(stage, now, settings.stall_s)
        if window is None:
            return None
        for sample in window:
            if sample.track_error >= settings.stall_arm_rad:
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
          the first chunk) would be declared complete before it ever moved;
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
        """Length of the trailing window in which ``p >= p_done``, or None."""
        settings = self.settings
        window = self._covered_window(stage, now, settings.p_hold_s)
        if window is None:
            return None
        for sample in window:
            if sample.progress is None or sample.progress < settings.p_done:
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


def _inf_norm(
    measured: Sequence[float], pose: Mapping[str, float], names: Sequence[str]
) -> float | None:
    worst = 0.0
    for value, name in zip(measured, names):
        if name not in pose:
            return None
        worst = max(worst, abs(value - pose[name]))
    return worst
