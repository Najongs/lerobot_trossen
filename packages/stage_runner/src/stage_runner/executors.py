"""Function-dispatch table for the ``executor:`` field, and the executors it names."""

import logging
import time
from collections.abc import Callable

from stage_runner import base_stages, leader_reset, record_adapter
from stage_runner.config import (
    EXECUTOR_BASE_ALIGN,
    EXECUTOR_BASE_REPLAY,
    EXECUTOR_LEADER_RESET,
    EXECUTOR_LEROBOT_POLICY,
    StageConfig,
)
from stage_runner.context import StageContext
from stage_runner.events import EVENT_STAGE_RESET
from stage_runner.results import (
    TERMINATED_BY_ERROR,
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
    aborted = _aborted_by_flags(context, stage)
    if aborted is not None:
        return aborted

    plan = record_adapter.plan_stage(stage, context.config.defaults)
    bundle = context.bundles[stage.id]

    _discard_stray_right_arrow(context, stage)

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
        teleop=context.teleop,
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


def run_leader_reset_stage(context: StageContext, stage: StageConfig) -> StageResult:
    """Let the operator drive the arms with the leader arm. No policy, no frames.

    lerobot-record's reset section as a stage of its own, so a YAML can put one
    anywhere in the chain -- typically first, to put the tube in the gripper the
    next stage was trained to start holding. In order:

    1. the leader is aligned to the follower (leader_reset.align_leader_to_follower),
       so the follower's first copied command is not a jump;
    2. base torque is released, so the base can be pushed by hand -- only on a
       robot connected with base torque;
    3. record_loop runs with the leader and NO dataset, the base command forced
       to 0.0, until the right arrow or the stage's ceiling;
    4. base torque is restored, whatever ended the loop.

    THE CEILING DOES NOT ABORT (decided 2026-10-19): a reset that reaches it ends
    "timeout" like any timed stage and the chain moves on, exactly as
    lerobot-record's reset_time_s does. Esc and the left arrow abort the trial as
    they do in a policy stage. A base whose torque could not be restored ends the
    trial instead of running the next stage: a policy that commands a base which
    does not respond is not a measurement.

    frames is 0 by construction, which aggregate reads as "not a policy stage"
    through the executor recorded on stage_start rather than as a failed one.
    The last gripper command travels on StageResult.gripper_goals, so a start
    pose move after this stage keeps the grip the operator set.
    """
    aborted = _aborted_by_flags(context, stage)
    if aborted is not None:
        return aborted
    teleop = context.teleop
    if teleop is None:
        # preflight.check_leader_reset_stages makes this unreachable from cli.
        raise RuntimeError(
            f"stage {stage.id!r} is a {EXECUTOR_LEADER_RESET} stage but no leader "
            "arm was connected (`teleop:` is missing)"
        )

    plan = record_adapter.plan_stage(stage, context.config.defaults)
    frames_before = context.buffered_frame_count()
    fields: dict[str, object] = {"control_time_s": plan.control_time_s}

    def emit(outcome: str, reason: str, **extra: object) -> None:
        context.log.emit(
            EVENT_STAGE_RESET,
            stage_id=stage.id,
            frame_idx=context.buffered_frame_count(),
            reason=reason,
            outcome=outcome,
            **fields,
            **extra,
        )

    started = time.perf_counter()
    try:
        aligned = leader_reset.align_leader_to_follower(context.robot, teleop)
    except Exception as error:
        # Nothing has been handed over: the follower still holds its last command
        # and the base is still powered and stopped. The trial ends here.
        reason = f"could not align the leader arm to the follower: {error!r}"
        logger.exception(f"stage {stage.id!r}: {reason}")
        emit("error", reason, base_torque=leader_reset.BASE_TORQUE_NOT_RELEASED)
        return StageResult(
            terminated_by=TERMINATED_BY_ERROR,
            elapsed_s=time.perf_counter() - started,
            frames=0,
            reason=reason,
        )
    fields.update(
        align_goal_time_s=aligned.goal_time_s,
        align_max_arm_error_rad=aligned.max_arm_error_rad,
        leader_before=aligned.leader_before,
        align_target=aligned.target,
    )

    # Esc during the alignment: hand nothing over.
    aborted = _aborted_by_flags(context, stage)
    if aborted is not None:
        emit(
            "aborted", aborted.reason, base_torque=leader_reset.BASE_TORQUE_NOT_RELEASED
        )
        return aborted
    _discard_stray_right_arrow(context, stage)

    released = leader_reset.release_base_torque(context.robot)
    action_filter = leader_reset.ResetActionFilter()
    logger.info(
        f"stage {stage.id!r}: leader arm in control -- press the right arrow when "
        f"the robot is ready (ceiling {plan.control_time_s:.0f}s)"
    )
    loop_started = time.perf_counter()
    try:
        record_adapter.call_reset_loop(
            robot=context.robot,
            events=context.events,
            fps=context.config.dataset.fps,
            processors=context.processors,
            teleop=teleop,
            teleop_action_processor=action_filter,
            control_time_s=plan.control_time_s,
            single_task=stage.instruction or stage.name,
            display_data=context.config.display_data,
        )
    except BaseException as error:
        # Restore before anything else, then let run_trial's handler stop the
        # base and close the log. The line is best-effort.
        restore_problem = (
            leader_reset.restore_base_torque(context.robot) if released else ""
        )
        try:
            emit(
                "interrupted",
                f"the reset loop raised {error!r}",
                base_torque=(
                    leader_reset.BASE_TORQUE_RESTORE_FAILED
                    if restore_problem
                    else leader_reset.BASE_TORQUE_RELEASED
                    if released
                    else leader_reset.BASE_TORQUE_NOT_RELEASED
                ),
                base_torque_error=restore_problem or None,
                sent_actions=action_filter.sent_count,
            )
        except Exception:
            logger.exception("could not record the interrupted reset")
        raise
    elapsed_s = time.perf_counter() - loop_started
    restore_problem = (
        leader_reset.restore_base_torque(context.robot) if released else ""
    )

    terminated_by, reason = record_adapter.classify_termination(
        events=context.events,
        plan=plan,
        elapsed_s=elapsed_s,
        fps=context.config.dataset.fps,
    )
    base_torque = (
        leader_reset.BASE_TORQUE_RELEASED
        if released
        else leader_reset.BASE_TORQUE_NOT_RELEASED
    )
    if restore_problem:
        base_torque = leader_reset.BASE_TORQUE_RESTORE_FAILED
        terminated_by = TERMINATED_BY_ERROR
        reason = f"base torque was not restored after the reset: {restore_problem}"
        logger.error(f"stage {stage.id!r}: {reason}")
    gripper_goals = action_filter.gripper_goals()
    emit(
        terminated_by,
        reason,
        base_torque=base_torque,
        base_torque_error=restore_problem or None,
        sent_actions=action_filter.sent_count,
        gripper_goals=gripper_goals,
        loop_s=elapsed_s,
    )
    frames = context.buffered_frame_count() - frames_before
    logger.info(
        f"stage {stage.id!r} ended: {terminated_by} after {elapsed_s:.1f}s of "
        f"leader control ({action_filter.sent_count} commands, base torque "
        f"{base_torque})"
    )
    return StageResult(
        terminated_by=terminated_by,
        elapsed_s=elapsed_s,
        frames=frames,
        reason=reason,
        gripper_goals=gripper_goals,
    )


def run_base_replay_stage(context: StageContext, stage: StageConfig) -> StageResult:
    """Drive the base along a demo's base path, arms held, recording every tick.

    A thin wrapper: the two flag rules every stage keeps live here, where the
    other executors keep them, and the stage itself is base_stages.replay. The
    split is what keeps base_stages from importing this module (this module
    imports it for the dispatch table).
    """
    aborted = _aborted_by_flags(context, stage)
    if aborted is not None:
        return aborted
    _discard_stray_right_arrow(context, stage)
    return base_stages.replay(context, stage)


def run_base_align_stage(context: StageContext, stage: StageConfig) -> StageResult:
    """Turn and drive the base until cam_high matches the demo's arrival frame.

    The same thin wrapper as run_base_replay_stage, around base_stages.align.
    """
    aborted = _aborted_by_flags(context, stage)
    if aborted is not None:
        return aborted
    _discard_stray_right_arrow(context, stage)
    return base_stages.align(context, stage)


def _aborted_by_flags(context: StageContext, stage: StageConfig) -> StageResult | None:
    """The result for an abort key already pressed, or None to run the stage."""
    if context.events.get("stop_recording"):
        return _aborted_before_start(
            stage, "stop_recording", TERMINATED_BY_STOP_RECORDING, "esc"
        )
    if context.events.get("rerecord_episode"):
        return _aborted_before_start(
            stage, "rerecord_episode", TERMINATED_BY_RERECORD_REQUESTED, "left arrow"
        )
    return None


def _discard_stray_right_arrow(context: StageContext, stage: StageConfig) -> None:
    """Drop a right arrow pressed in the inter-stage gap.

    DELIBERATE ADDITION beyond the frozen step 16.c.i, which names only the two
    abort flags. record_loop self-clears events["exit_early"] only when IT
    breaks on it (lerobot_record.py:343-345), so a right arrow pressed in the
    same inter-stage window survives into this stage and breaks the loop on its
    first iteration: 0 frames, elapsed ~0.001 s, classified "manual" against a
    12.9 s control time, and the chain records a stage that never ran while the
    trial still exits 0. Discarding it is the lesser error -- the press meant
    "end the stage that just ended", which had already ended -- but it is
    logged, because a keypress that disappears without a trace is how an
    operator stops trusting the log.
    """
    if context.events.get("exit_early"):
        logger.warning(
            f"stage {stage.id!r}: a right-arrow press landed in the inter-stage "
            "gap, where no record_loop was running to consume it. Discarded, so "
            "this stage is not cut short to zero frames."
        )
        context.events["exit_early"] = False


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
    EXECUTOR_LEADER_RESET: run_leader_reset_stage,
    EXECUTOR_BASE_REPLAY: run_base_replay_stage,
    EXECUTOR_BASE_ALIGN: run_base_align_stage,
}
