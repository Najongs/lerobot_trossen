"""Function-dispatch table for the ``executor:`` field, and the one P1 executor."""

import logging
import time
from collections.abc import Callable

from stage_runner import record_adapter
from stage_runner.config import EXECUTOR_LEROBOT_POLICY, StageConfig
from stage_runner.context import StageContext
from stage_runner.results import (
    TERMINATED_BY_RERECORD_REQUESTED,
    TERMINATED_BY_STOP_RECORDING,
    StageResult,
)

logger = logging.getLogger(__name__)

# A typing alias, not a Protocol and not an ABC: promotion waits until a second
# implementation exists and shows what the abstraction actually has to hold. The
# argument order (context, stage) is frozen -- every future executor is one dict
# entry plus one function with this signature.
StageExecutor = Callable[[StageContext, StageConfig], StageResult]


def run_lerobot_policy_stage(context: StageContext, stage: StageConfig) -> StageResult:
    """Run one stage as a policy rollout through lerobot's record_loop.

    Calls ``save_episode`` NEVER -- the whole chain is one episode and only
    ``runner.run_trial`` saves it, once. Catches nothing either: an exception
    propagates to run_trial, which owns the error StageResult and the trial_end
    event.
    """
    # READ THE ABORT FLAGS BEFORE ANYTHING CLEARS THEM. An abort pressed in the
    # inter-stage window -- after record_loop returned from the previous stage
    # and before this one starts, which covers classify_termination, the
    # stage_end emit, stop_base's get_observation + send_action and the
    # transition emit -- has no loop running to consume it. Clearing the flags at
    # stage entry (what this function used to do) threw that press away: Esc and
    # the left arrow ALSO set exit_early, so record_loop would break on its first
    # iteration, classify_termination would then see no flags and elapsed_s ~ 0
    # and return "manual" -- which is NOT in ABORTING_TERMINATORS -- and the
    # runner would drive the robot on into the next stage after the operator
    # asked it to stop.
    #
    # The flags are never cleared -- not here and not in run_trial. They are
    # trial-level intent, the runner breaks out of the stage loop on an aborting
    # terminator, and init_keyboard_listener hands out a fresh False for each, so
    # there is nothing stale to clear. See the note in runner.run_trial.
    if context.events.get("stop_recording"):
        return _aborted_before_start(
            stage, "stop_recording", TERMINATED_BY_STOP_RECORDING, "esc"
        )
    if context.events.get("rerecord_episode"):
        return _aborted_before_start(
            stage, "rerecord_episode", TERMINATED_BY_RERECORD_REQUESTED, "left arrow"
        )

    plan = record_adapter.plan_stage(stage, context.config.defaults)
    bundle = context.bundles[stage.id]

    # DELIBERATE ADDITION beyond the frozen step 16.c.i, which names only the two
    # flags above. record_loop self-clears events["exit_early"] only when IT
    # breaks on it (lerobot_record.py:343-345), so a right arrow pressed in the
    # same inter-stage window survives into this stage and breaks the loop on its
    # first iteration: 0 frames, elapsed ~0.001 s, classified "manual" against a
    # 12.9 s control time, and the chain records a stage that never ran while the
    # trial still exits 0. Discarding it is the lesser error -- the press meant
    # "end the stage that just ended", which had already ended -- but it is
    # logged, because a keypress that disappears without a trace is how an
    # operator stops trusting the log.
    if context.events.get("exit_early"):
        logger.warning(
            f"stage {stage.id!r}: a right-arrow press landed in the inter-stage "
            "gap, where no record_loop was running to consume it. Discarded, so "
            "this stage is not cut short to zero frames."
        )
        context.events["exit_early"] = False

    frames_before = context.buffered_frame_count()
    started = time.perf_counter()
    # record_loop's entry does policy.reset() + preprocessor.reset() +
    # postprocessor.reset() (lerobot_record.py:331-335), so re-entering it per
    # stage flushes the previous stage's stale ACT chunk queue for free. That is
    # the transition semantics we want, not an accident of the loop.
    record_adapter.call_record_loop(
        robot=context.robot,
        events=context.events,
        fps=context.config.dataset.fps,
        processors=context.processors,
        dataset=context.dataset,
        bundle=bundle,
        control_time_s=plan.control_time_s,
        single_task=stage.instruction or stage.name,
        display_data=context.config.display_data,
    )
    elapsed_s = time.perf_counter() - started
    frames = context.buffered_frame_count() - frames_before

    terminated_by, reason = record_adapter.classify_termination(
        events=context.events,
        plan=plan,
        elapsed_s=elapsed_s,
        fps=context.config.dataset.fps,
    )
    logger.info(
        f"stage {stage.id!r} ended: {terminated_by} "
        f"({frames} frames in {elapsed_s:.2f}s)"
    )
    return StageResult(
        terminated_by=terminated_by,
        elapsed_s=elapsed_s,
        frames=frames,
        reason=reason,
    )


def _aborted_before_start(
    stage: StageConfig, flag: str, terminated_by: str, key_name: str
) -> StageResult:
    """The result for a stage the operator aborted before it could start.

    Zero frames and zero elapsed time are the truth: record_loop was never
    entered, so nothing was recorded and nothing moved under this stage's
    policy. The reason says where the press landed, because the same terminator
    reached any other way (pressed DURING the stage) means something different
    when the log is read months later.
    """
    reason = (
        f"events[{flag!r}] was already set when the stage started ({key_name} "
        "pressed during the previous transition); record_loop was not entered"
    )
    logger.warning(f"stage {stage.id!r} aborted before it started: {reason}")
    return StageResult(
        terminated_by=terminated_by, elapsed_s=0.0, frames=0, reason=reason
    )


# Keyed by the constant, not by a copy of its value: the same string is the
# default of both StageConfig.executor and DefaultsConfig.executor, and a rename
# that missed this dict would reject every existing YAML with "unknown executor"
# and blame the operator's config for a code-side rename.
EXECUTORS: dict[str, StageExecutor] = {
    EXECUTOR_LEROBOT_POLICY: run_lerobot_policy_stage,
}
