"""Function-dispatch table for the ``executor:`` field, and the one P1 executor."""

import logging
import math
import sys
import time
from collections.abc import Callable

from stage_runner import record_adapter
from stage_runner.config import (
    EXECUTOR_CHAIN_POLICY,
    EXECUTOR_CHAIN_RESET,
    EXECUTOR_LEROBOT_POLICY,
    StageConfig,
)
from stage_runner.context import StageContext
from stage_runner.results import (
    TERMINATED_BY_COMPLETE,
    TERMINATED_BY_MANUAL,
    TERMINATED_BY_NOT_REACHED,
    TERMINATED_BY_REACHED,
    TERMINATED_BY_RERECORD_REQUESTED,
    TERMINATED_BY_STOP_RECORDING,
    TERMINATED_BY_TIMEOUT,
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


def _discard_stale_exit_early(context: StageContext, stage: StageConfig) -> None:
    """Consume a right arrow that landed between two stages, and say so.

    ``record_loop`` self-clears ``events["exit_early"]`` only when IT breaks on
    it (lerobot_record.py:343-345). Two ways the flag survives into the next
    stage and would break its loop on the FIRST iteration -- zero frames,
    elapsed ~0.001 s:

    1. an operator press in the inter-stage gap (the case
       :func:`run_lerobot_policy_stage` documents);
    2. **the completion monitor or the reset firing on the last tick before
       ``control_time_s``.** The loop then exits through its ``while`` condition
       without ever reaching the ``if events["exit_early"]`` at the top, so
       nothing clears the flag. In a chain the next stage is a RESET, and a reset
       broken on its first tick reports not-reached -- a chain failure caused by
       the previous stage having succeeded.

    Both are logged. A keypress that disappears without a trace is how an
    operator stops trusting the log, and case 2 would otherwise look like one.
    """
    if not context.events.get("exit_early"):
        return
    logger.warning(
        f"stage {stage.id!r}: events['exit_early'] was already set at entry "
        "(a right-arrow press in the inter-stage gap, or the previous stage's "
        "own completion firing on its last tick before control_time_s, where "
        "record_loop exits through the while condition and never clears the "
        "flag). Discarded, so this stage is not cut short to zero frames."
    )
    context.events["exit_early"] = False


def _chain_runtime(context: StageContext, stage: StageConfig):
    chain = context.chain
    if chain is None:
        raise RuntimeError(
            f"stage {stage.id!r} uses the {stage.executor!r} executor, but this "
            "run has no chain runtime. A chain needs `version: 2` and a "
            "`chain:` block; cli.main assembles the runtime from it."
        )
    return chain


def run_chain_policy_stage(context: StageContext, stage: StageConfig) -> StageResult:
    """One stage of the chain: re-point the one-hot, arm the monitor, roll out.

    The ORDER of the three things before ``record_loop`` is load-bearing:

    1. read the abort flags (an Esc in the gap must not be overtaken);
    2. ``set_stage`` on the one-hot, because the preprocessor is reset on loop
       ENTRY and a reset does not touch the index (TaskOneHotStep.reset is a
       no-op, deliberately);
    3. ``begin_stage`` on the monitor, which clears the window -- the robot
       action pipeline is the ONE pipeline record_loop never resets, so without
       this the previous stage's three seconds of stalled samples would declare
       this stage complete on its first tick.
    """
    if context.events.get("stop_recording"):
        return _aborted_before_start(
            stage, "stop_recording", TERMINATED_BY_STOP_RECORDING, "esc"
        )
    if context.events.get("rerecord_episode"):
        return _aborted_before_start(
            stage, "rerecord_episode", TERMINATED_BY_RERECORD_REQUESTED, "left arrow"
        )
    _discard_stale_exit_early(context, stage)

    chain = _chain_runtime(context, stage)
    params = chain.params.stage(stage.stage_number)
    plan = record_adapter.plan_stage(stage, context.config.defaults)
    bundle = context.bundles[stage.id]

    if chain.onehot is not None and stage.onehot_index is not None:
        chain.onehot.set_stage(stage.onehot_index)
    elif chain.onehot is None and stage.onehot_index is not None:
        # A YAML that asks for a one-hot index against a checkpoint with no
        # one-hot step. preflight refuses this combination, so reaching it is a
        # defect, not an operator error -- but it must not run silently on the
        # wrong conditioning.
        raise RuntimeError(
            f"stage {stage.id!r} asks for one-hot index {stage.onehot_index} but "
            "no one-hot step is installed (chain.model.onehot_k is null)"
        )

    chain.monitor.begin_stage(params, has_progress=chain.has_progress)
    _set_pose_guide(None)

    clamped_before = record_adapter.clamped_arm_ticks()
    frames_before = context.buffered_frame_count()
    started = time.perf_counter()
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
    completion = chain.monitor.end_stage()

    terminated_by, reason, detail = _classify_chain_policy(
        context=context,
        stage=stage,
        plan=plan,
        elapsed_s=elapsed_s,
        completion=completion,
        allow_manual=chain.allow_manual_complete,
    )
    detail.update(
        {
            "p10_s": params.p10_s,
            "p90_s": params.p90_s,
            "clamped_ticks": _clamped_delta(clamped_before),
        }
    )
    logger.info(
        f"stage {stage.id!r} (task{stage.stage_number:02d}) ended: {terminated_by} "
        f"({frames} frames in {elapsed_s:.2f}s, p10 {params.p10_s:.1f} / p90 "
        f"{params.p90_s:.1f})"
    )
    return StageResult(
        terminated_by=terminated_by,
        elapsed_s=elapsed_s,
        frames=frames,
        reason=reason,
        detail=detail,
    )


def _classify_chain_policy(
    *,
    context: StageContext,
    stage: StageConfig,
    plan,
    elapsed_s: float,
    completion,
    allow_manual: bool,
) -> tuple[str, str, dict]:
    """Why a chain policy stage ended. PRECEDENCE MATTERS; see the order below.

    1. **abort flags** -- Esc and the left arrow also set ``exit_early``, so a
       stage the operator stopped looks exactly like one the monitor stopped
       from the loop's point of view. Reading the flags first is what stops the
       chain driving on into the next stage after the operator asked it to stop.
    2. **the monitor's result** -- an object, not an inference. This is the only
       positive evidence a completion ever leaves: ``record_loop`` self-clears
       ``exit_early``.
    3. **the elapsed-time heuristic** -> ``manual``. The right arrow leaves no
       other trace, which is why ``classify_termination`` names the numbers in
       its reason, and why this does too.
    4. **timeout**.

    ``classify_termination`` ALONE would label every completion "manual",
    because a completion also ends the stage early. That is the mislabel this
    function exists to prevent.
    """
    margin_s = record_adapter.manual_detection_margin_s(context.config.dataset.fps)
    threshold_s = plan.control_time_s - margin_s
    measured = (
        f"elapsed_s={elapsed_s:.3f}, control_time_s={plan.control_time_s:.3f}, "
        f"margin_s={margin_s:.3f}, threshold_s={threshold_s:.3f}"
    )

    if context.events.get("stop_recording"):
        return (
            TERMINATED_BY_STOP_RECORDING,
            f"events['stop_recording'] was set; {measured}",
            {},
        )
    if context.events.get("rerecord_episode"):
        return (
            TERMINATED_BY_RERECORD_REQUESTED,
            f"events['rerecord_episode'] was set; {measured}",
            {},
        )
    if completion is not None:
        return (
            TERMINATED_BY_COMPLETE,
            f"completion monitor fired by {completion.reason}; {measured}",
            completion.as_detail(),
        )
    if elapsed_s < threshold_s:
        if not allow_manual:
            # The stage ended early with no monitor result and the YAML did not
            # allow a manual completion. Reporting it as `manual` would make the
            # required_terminator check pass and the chain walk on; reporting
            # the truth breaks the chain, which is correct -- something ended the
            # stage and it was not the signal the run is measuring.
            return (
                TERMINATED_BY_MANUAL,
                "the stage ended early with no completion result and "
                f"allow_manual_complete is false; {measured}",
                {},
            )
        return (
            TERMINATED_BY_MANUAL,
            "ended early with no completion result, so a right-arrow press is "
            f"the only explanation left; {measured}",
            {},
        )
    return (
        TERMINATED_BY_TIMEOUT,
        f"ran to control_time_s with no completion; {measured}",
        {},
    )


def run_chain_reset_stage(context: StageContext, stage: StageConfig) -> StageResult:
    """Drive the 12 arm joints to the next stage's designated pose. No retry, ever.

    Everything that can refuse happens BEFORE ``record_loop`` is entered: the
    stillness check, the anchor read, the ``.pos`` coverage check and
    ``plan_reset``'s gap/finiteness refusal. An exception from inside the loop
    would unwind through ``@safe_stop_image_writer``, the runner would take its
    error path, and the WHOLE EPISODE -- every stage already recorded in it --
    would be discarded unsaved. A refusal here is a StageResult instead, the
    episode survives, and the chain fails cleanly.
    """
    if context.events.get("stop_recording"):
        return _aborted_before_start(
            stage, "stop_recording", TERMINATED_BY_STOP_RECORDING, "esc"
        )
    if context.events.get("rerecord_episode"):
        return _aborted_before_start(
            stage, "rerecord_episode", TERMINATED_BY_RERECORD_REQUESTED, "left arrow"
        )
    _discard_stale_exit_early(context, stage)

    from stage_runner.reset_policy import (
        ResetAbort,
        make_reset_bundle,
        plan_reset,
        wait_until_still,
    )

    chain = _chain_runtime(context, stage)
    params = chain.params.stage(stage.stage_number)
    # The monitor must be inert through a reset. It is armed and disarmed by the
    # POLICY executor, so this is an assertion about the sequence, not a
    # mechanism -- but the sequence is what a future executor could break.
    if chain.monitor.armed:
        logger.warning(
            "the completion monitor was still armed entering a reset; disarming"
        )
        chain.monitor.end_stage()

    anchor, still = wait_until_still(
        context.robot,
        arm_joint_names=chain.params.arm_joint_order,
        settings=chain.reset,
    )
    if not still:
        return _reset_refused(
            stage,
            "the arm was still moving after "
            f"{chain.reset.settle_check_tries} reads "
            f"{chain.reset.settle_check_gap_s:.2f}s apart; a ramp anchored on a "
            "moving arm starts from a point the arm has already left",
            {},
        )

    missing = [
        name
        for name in context.chain.action_names
        if name.endswith(".pos") and name not in chain.state_names
    ]
    if missing:
        return _reset_refused(
            stage,
            f"commandable position(s) {missing} have no counterpart in "
            "observation.state, so the ramp cannot hold them at their measured "
            "value -- and a gripper commanded to anything else may release or "
            "crush what the arm is carrying",
            {},
        )

    try:
        reset_plan = plan_reset(
            stage_number=stage.stage_number,
            anchor=anchor,
            target=params.start_pose,
            fps=context.config.dataset.fps,
            settings=chain.reset,
            arm_joint_names=chain.params.arm_joint_order,
        )
    except ResetAbort as error:
        return _reset_refused(stage, str(error), {})

    bundle = make_reset_bundle(
        stage.id,
        reset_plan,
        context.events,
        action_names=chain.action_names,
        state_names=chain.state_names,
        settings=chain.reset,
        arm_joint_names=chain.params.arm_joint_order,
    )
    _set_pose_guide(stage.stage_number, chain)

    clamped_before = record_adapter.clamped_arm_ticks()
    frames_before = context.buffered_frame_count()
    started = time.perf_counter()
    record_adapter.call_record_loop(
        robot=context.robot,
        events=context.events,
        fps=context.config.dataset.fps,
        processors=context.processors,
        dataset=context.dataset,
        bundle=bundle,
        control_time_s=reset_plan.control_time_s,
        single_task=stage.instruction or stage.name,
        display_data=context.config.display_data,
    )
    elapsed_s = time.perf_counter() - started
    frames = context.buffered_frame_count() - frames_before
    policy = bundle.policy

    detail = reset_plan.as_detail()
    detail.update(
        {
            "reset_ceiling_s": round(reset_plan.control_time_s, 3),
            "reach_err": (
                None
                if not math.isfinite(policy.worst_error_rad)
                else round(policy.worst_error_rad, 4)
            ),
            "reset_tol_rad": chain.reset.tol_rad,
            "clamped_ticks": _clamped_delta(clamped_before),
        }
    )

    if context.events.get("stop_recording"):
        terminated_by = TERMINATED_BY_STOP_RECORDING
        reason = "events['stop_recording'] was set during the reset"
    elif context.events.get("rerecord_episode"):
        terminated_by = TERMINATED_BY_RERECORD_REQUESTED
        reason = "events['rerecord_episode'] was set during the reset"
    elif policy.refused:
        terminated_by = TERMINATED_BY_NOT_REACHED
        reason = f"the ramp refused mid-entry: {policy.refused}"
    elif policy.reached:
        terminated_by = TERMINATED_BY_REACHED
        reason = (
            f"arrived within {chain.reset.tol_rad:.3f} rad and held it for "
            f"{reset_plan.settle_ticks} ticks "
            f"(worst joint {policy.worst_error_rad:.4f} rad)"
        )
    else:
        terminated_by = TERMINATED_BY_NOT_REACHED
        reason = (
            f"ran out the {reset_plan.control_time_s:.1f}s ceiling without "
            f"holding the pose: worst joint error "
            f"{policy.worst_error_rad:.4f} rad against tol "
            f"{chain.reset.tol_rad:.3f}. NO RETRY -- a second ramp at the same "
            "pose would repeat the motion that already failed."
        )
    logger.info(
        f"stage {stage.id!r} ended: {terminated_by} ({frames} frames in "
        f"{elapsed_s:.2f}s, {detail})"
    )
    return StageResult(
        terminated_by=terminated_by,
        elapsed_s=elapsed_s,
        frames=frames,
        reason=reason,
        detail=detail,
    )


def _reset_refused(stage: StageConfig, reason: str, detail: dict) -> StageResult:
    """The result for a reset that was refused BEFORE record_loop was entered."""
    logger.error(f"stage {stage.id!r}: reset refused, nothing moved -- {reason}")
    return StageResult(
        terminated_by=TERMINATED_BY_NOT_REACHED,
        elapsed_s=0.0,
        frames=0,
        reason=f"refused before the loop was entered (nothing moved): {reason}",
        detail=detail,
    )


def _clamped_delta(before: int | None) -> int | None:
    after = record_adapter.clamped_arm_ticks()
    if before is None or after is None:
        return None
    return after - before


def _set_pose_guide(stage_number: int | None, chain=None) -> None:
    """Point ``pose_guide`` at this reset's target, or turn it off for a rollout.

    Reached through ``sys.modules`` and never imported: the module lives in
    ``lerobot_robot_trossen``, importing that package imports the Trossen SDK,
    and on the robot PC ``register_plugins()`` has already imported it. Off the
    robot this is a no-op.

    The TARGET is passed, not just the stage number, because
    ``pose_guide.TARGETS_DEG`` is the per-joint MEDIAN of the demonstrations'
    first frames and t02·03·04·05·10 are bimodal -- measuring a ramp that is
    driving to the DESIGNATED pose against the median would print a nonzero
    distance at the moment the ramp has arrived.
    """
    module = sys.modules.get("lerobot_robot_trossen.pose_guide")
    setter = getattr(module, "set_stage", None)
    if not callable(setter):
        return
    try:
        if stage_number is None or chain is None:
            setter(None)
            return
        setter(stage_number, chain.params.start_pose_deg_pairs(stage_number))
    except Exception:  # a readout must never cost a run
        logger.debug("could not point pose_guide at the reset target", exc_info=True)


# Registered by mutating the table rather than by a second literal, so there is
# exactly one EXECUTORS dict and `preflight.check_stage_definitions` ("unknown
# executor") keeps listing every executor that exists.
EXECUTORS[EXECUTOR_CHAIN_POLICY] = run_chain_policy_stage
EXECUTORS[EXECUTOR_CHAIN_RESET] = run_chain_reset_stage
