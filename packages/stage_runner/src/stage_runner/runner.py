"""The trial body: one function, one episode, one pass over the stages."""

import logging
import math
import time

# Module scope, not the call-time import this used to be: the old form was
# there because the package __init__ imported this module eagerly, so
# __version__ did not exist yet while runner was loading. __init__ is lazy now
# (PEP 562) and imports nothing at package-import time, so the cycle is gone.
from stage_runner import __version__ as stage_runner_version
from stage_runner import record_adapter, reposition, transitions
from stage_runner.config import StageConfig
from stage_runner.context import StageContext
from stage_runner.events import (
    EVENT_STAGE_END,
    EVENT_STAGE_REPOSITION,
    EVENT_STAGE_START,
    EVENT_TRANSITION,
    EVENT_TRIAL_END,
    EVENT_TRIAL_START,
)
from stage_runner.executors import EXECUTORS
from stage_runner.policies import bundle_descriptor
from stage_runner.preflight import (
    check_lerobot_version,
    robot_action_dimension,
    robot_state_dimension,
)
from stage_runner.results import (
    ABORTING_TERMINATORS,
    TERMINATED_BY_ERROR,
    TRIAL_REASON_ABORTED,
    TRIAL_REASON_ABORTED_EMPTY,
    TRIAL_REASON_COMPLETED,
    TRIAL_REASON_EXCEPTION,
    StageResult,
    TrialOutcome,
)
from stage_runner.transitions import StopBaseOutcome

logger = logging.getLogger(__name__)


def _stop_base_never_raises(robot) -> StopBaseOutcome:
    """stop_base for the failure path, where nothing may escape.

    ``transitions.stop_base`` already handles ordinary failures itself, but it
    re-raises a KeyboardInterrupt (after zeroing the base) so that an operator
    hammering Ctrl+C on the NORMAL path is not ignored. On the failure path that
    re-raise would jump out before trial_end reaches disk, which is the exact
    hole this round is closing -- and the base is already stopped by then, so
    there is nothing left for the interrupt to protect. The original exception is
    re-raised by the caller regardless, so the process still unwinds.

    ``reraise_interrupt=False`` rather than catching that interrupt here, because
    catching it LOSES THE OUTCOME: the except below cannot see which path ran, so
    it filed every escaping interrupt as stop_base_path="failed" -- the label that
    means "the base may still be moving". An operator Ctrl+C during a stage takes
    the fallback, which stops the base, and reporting that as a failed stop makes
    the indicator fire on the most ordinary abort there is. The except stays for
    what it was always for: a defect in stop_base itself, where "failed" is true.
    """
    try:
        return transitions.stop_base(robot, reraise_interrupt=False)
    except BaseException as error:  # nothing may escape here: see the docstring
        logger.exception("stop_base itself escaped on the failure path")
        return StopBaseOutcome(
            path=transitions.STOP_PATH_FAILED, action=None, error=repr(error)
        )


def run_trial(context: StageContext) -> TrialOutcome:
    """Run every stage of one trial into ONE episode and return what happened.

    The return is a :class:`TrialOutcome`, not the bare stage results, because
    one thing the caller needs is not a stage measurement: whether every
    stop_base actually stopped the base. cli.main turns a False there into its
    own exit code, so a batch script sees the one condition that means the robot
    may still be driving without parsing events.jsonl.

    One run = one trial (option A). Everything that is not the trial itself --
    plugin registration, parsing, preflight, dataset creation, policy preload,
    connect, disconnect -- lives in cli.main, which is what makes moving to
    option B (a trial loop) a matter of wrapping this call rather than editing it.

    Four decisions here deviate from the input designs:

    1. ``transitions.stop_base`` runs after EVERY stage, INCLUDING THE LAST, not
       only between stages. ``save_episode`` flushes video encoders and can take
       seconds, and the base holds its last velocity command for all of them.
    2. The ``transition`` EVENT is emitted at EVERY boundary too, including the
       one after the last stage, after an abort and after an error, with
       ``to_stage_id=null`` where the chain did not actually cross. It is the
       only line that says whether the base was stopped, so gating it made a
       silently-failed stop_base byte-identical to a clean run at exactly the
       two boundaries where it matters most. See :func:`_emit_boundary`.
    3. On OPERATOR ABORT (Esc / left arrow) the partial episode IS saved, with
       ``trial_end.completed=false`` and ``reason="aborted"``. A partial chain is
       data, the aggregator excludes it via ``completed``, and discarding is
       irreversible. On EXCEPTION the episode is NOT saved -- the buffer state is
       unknown at that point -- and ``trial_end.episode_saved=false`` says so
       rather than leaving the reader to infer it from an absence.
    4. An abort at frame 0 leaves an empty buffer, which ``validate_episode_buffer``
       rejects, so the save is guarded on ``buffered_frame_count() > 0`` and
       trial_end carries ``reason="aborted_empty"``.

    A stage with a ``start_pose`` first has its arm joints moved there
    (:func:`_reposition_before_stage`), AFTER the previous boundary's stop_base and
    transition emit and BEFORE its own stage_start, so stage_start still marks the
    instant the policy takes over. Stage 0 included. A pose that is not reached
    ends the trial as an abort (see the reposition_failure branch below), never
    by starting the policy from the wrong pose.

    HARDWARE FIRST, BOOKKEEPING AFTER, the same rule cli.main's finally follows.
    The base stop happens as soon as the executor returns, before the stage_end
    emit, and the whole trial body is inside one ``except BaseException`` so that
    leaving this function by ANY route -- the executor raising, an OSError out of
    log.emit on a full disk, a Ctrl+C anywhere at all -- still stops the base and
    still resolves the episode in the log. Nothing is swallowed; the caller gets
    its exception back.

    Deliberately absent, all of which upstream ``record()`` has: the trial-repeat
    loop, the re-record branch, ``dataset.finalize()`` in a finally of our own
    (VideoEncodingManager.__exit__ already does it), and ``push_to_hub``.
    Upstream's teleop reset section is not absent but MOVED: it is a stage of its
    own (executor ``leader_reset``) that a YAML places where it is needed --
    before stage 4, which must START holding the tube -- rather than a fixed
    section between episodes.
    """
    config = context.config
    log = context.log
    stages = config.stages
    results: list[StageResult] = []

    # Failure-path bookkeeping. Every one of these exists so the handler can tell
    # what is already on disk: emitting a second stage_end for a stage that
    # already has one makes build_trial raise on the pair check, and a second
    # trial_end makes the aggregator read one trial as two.
    stage_index = -1
    stage_id: str | None = None
    stage_in_flight: str | None = None
    stage_started = time.perf_counter()
    frames_before = 0
    boundary_emitted = True
    trial_end_emitted = False
    # Sticky, never reset: one boundary where BOTH stop paths failed means the
    # base was left driving during this trial, and a later boundary that stopped
    # it cleanly does not undo that. This is the flag cli.main turns into an exit
    # code -- without it, the run that the review measured (final stop_base
    # failed, base at 0.4 m/s through save_episode) exited 0.
    base_is_stopped = True

    trial_started = time.perf_counter()
    try:
        log.emit(
            EVENT_TRIAL_START,
            frame_idx=0,
            stage_ids=[stage.id for stage in stages],
            config_version=config.version,
            # check_lerobot_version is the single place that reads
            # lerobot.__version__. cli already ran it as a preflight gate, so
            # here it cannot raise and is only the string source -- no second
            # reader to drift from the pin.
            lerobot_version=check_lerobot_version(),
            stage_runner_version=stage_runner_version,
            robot_type=context.robot.name,
            robot_state_dim=robot_state_dimension(context.robot),
            robot_action_dim=robot_action_dimension(context.robot),
            dataset_repo_id=config.dataset.repo_id,
            # The RESOLVED root, not config.dataset.root, which is null in every
            # config that lets lerobot pick the cache path -- and then the log
            # would not answer "where did this trial's data land".
            dataset_root=str(context.dataset.root) if context.dataset.root else None,
            fps=config.dataset.fps,
            policies=[bundle_descriptor(bundle) for bundle in context.bundles.values()],
        )
        logger.info(f"trial started: {len(stages)} stages -> {config.dataset.repo_id}")

        # NOTHING CLEARS events["stop_recording"] / ["rerecord_episode"], here or
        # in the executor, and that is the fix for the abort this runner used to
        # swallow. They are trial-level intent: once the operator presses Esc or
        # the left arrow the run is over, and the stage loop breaks out on the
        # aborting terminator, so no later stage can misread a leftover. Clearing
        # them at stage entry (the old behaviour) discarded a press made in an
        # inter-stage gap; clearing them HERE instead would discard one made while
        # the arms were powering up in robot.connect(), which is exactly when an
        # operator reaches for the keyboard. init_keyboard_listener hands us a
        # fresh False for each, so there is nothing stale to clear in the first
        # place.
        # Option B (a trial loop around this call) has to decide for itself what a
        # press between two trials means; it must not be decided here by accident.

        # The outcome of the boundary just crossed, None before the first stage.
        # Read by the start pose move, which refuses to drive the arms for
        # seconds while the base may still be holding a velocity.
        last_boundary: StopBaseOutcome | None = None
        # The gripper goals this runner last sent, None until something sent
        # one (connect()'s staged pose is not ours to know). Tracked ONLY when a
        # stage has a start_pose, so a config without one runs and logs exactly
        # as before. Read at a boundary followed by a start pose, so the hold
        # there keeps the gripper's previous command instead of re-anchoring it
        # to its measured position -- see _recorded_gripper_goals.
        uses_start_poses = any(stage.start_pose is not None for stage in stages)
        last_gripper_goals: dict[str, float] | None = None
        gripper_goal_source = reposition.GRIPPER_GOAL_UNCHANGED
        for stage_index, stage in enumerate(stages):
            # Before the start-pose move, so the operator hears which stage is
            # next before the arms start moving.
            record_adapter.announce_stage(stage, config.play_sounds)
            # BEFORE `stage_id = stage.id`, on purpose: until this stage's
            # stage_start is written, the stage that owns the open boundary is
            # still the previous one, and _record_failed_trial keys any boundary
            # line it writes off stage_id. A move that raises here (Ctrl+C) is
            # therefore filed at that boundary, not as a stage that never
            # started -- no stage_start, no stage_end, no extra transition.
            reposition_failure: str | None = None
            if stage.start_pose is not None:
                reposition_failure = _reposition_before_stage(
                    context,
                    stage,
                    stage_index=stage_index,
                    boundary=last_boundary,
                    gripper_goal_source=gripper_goal_source,
                )
            stage_id = stage.id
            plan = record_adapter.plan_stage(stage, config.defaults)
            log.emit(
                EVENT_STAGE_START,
                stage_id=stage.id,
                frame_idx=context.buffered_frame_count(),
                terminator=plan.planned_terminator,
                stage_index=stage_index,
                executor=stage.executor,
                policy_path=stage.policy_path,
                single_task=stage.instruction or stage.name,
                control_time_s=plan.control_time_s,
            )
            stage_in_flight = stage.id
            boundary_emitted = False
            logger.info(
                f"stage {stage_index} {stage.id!r}: {stage.executor} "
                f"({plan.planned_terminator}, {plan.control_time_s:.1f}s)"
            )

            frames_before = context.buffered_frame_count()
            stage_started = time.perf_counter()
            if reposition_failure is not None:
                # The arms are NOT at the pose this stage's policy was trained
                # to start from, so the policy is not run. Recorded the way
                # executors._aborted_before_start records a stage that never
                # entered record_loop -- 0 frames, elapsed_s exactly 0.0, which
                # aggregate reads as "never ran" rather than as a slow stage --
                # and with an ABORTING terminator, so the loop below ends the
                # trial. NOT an exception: the episode buffer's state is known
                # (nothing was added during the move), so the earlier stages'
                # frames are saved as a partial chain exactly like an operator
                # abort, instead of being discarded with an unknown buffer.
                result = StageResult(
                    terminated_by=TERMINATED_BY_ERROR,
                    elapsed_s=0.0,
                    frames=0,
                    reason=(
                        f"start pose not reached, so record_loop was not entered: "
                        f"{reposition_failure}"
                    ),
                )
                logger.error(f"stage {stage.id!r} not started: {reposition_failure}")
            else:
                result = EXECUTORS[stage.executor](context, stage)
            results.append(result)

            # HARDWARE FIRST: the base is zeroed before ANY of the bookkeeping
            # below. Nothing between record_loop's last send_action and this call
            # should be I/O -- an OSError out of the stage_end emit (a full disk
            # on the robot PC while three cameras write video) used to escape
            # with the base still holding the last policy velocity.
            #
            # The one exception is in-memory and never raises: which gripper goal
            # the hold must carry has to be known BEFORE the hold is built.
            next_stage = (
                stages[stage_index + 1]
                if stage_index < len(stages) - 1
                and result.terminated_by not in ABORTING_TERMINATORS
                else None
            )
            held_gripper: dict[str, float] | None = None
            gripper_goal_source = reposition.GRIPPER_GOAL_UNCHANGED
            if uses_start_poses:
                if result.gripper_goals is not None:
                    # A leader_reset stage: its commands never reached the
                    # episode buffer, so it hands its last gripper goal over
                    # directly -- the grip the operator just set.
                    last_gripper_goals = dict(result.gripper_goals)
                elif result.frames > 0:
                    # This stage's policy sent the latest gripper goal; a stage
                    # that recorded nothing sent none, so the previous one stands.
                    last_gripper_goals = _recorded_gripper_goals(context)
                if next_stage is not None and next_stage.start_pose is not None:
                    held_gripper = last_gripper_goals
            transition_started = time.perf_counter()
            outcome = transitions.stop_base(context.robot, held_positions=held_gripper)
            stop_base_s = time.perf_counter() - transition_started
            if uses_start_poses and outcome.path == transitions.STOP_PATH_PRIMARY:
                # The fallback path commands no arm joint, so after it the goals
                # tracked above are still the ones in force.
                gripper_goal_source = (
                    reposition.GRIPPER_GOAL_PREVIOUS_COMMAND
                    if held_gripper is not None
                    else reposition.GRIPPER_GOAL_MEASURED
                )
                last_gripper_goals = {
                    key: value
                    for key, value in (outcome.action or {}).items()
                    if reposition.is_gripper_key(key)
                } or last_gripper_goals
            # Read BEFORE the emit, and `and` rather than assignment: the emit
            # below can raise (a full disk), and the exit code must still carry
            # a stop that already failed.
            base_is_stopped = base_is_stopped and outcome.base_is_stopped
            last_boundary = outcome

            log.emit(
                EVENT_STAGE_END,
                stage_id=stage.id,
                frame_idx=context.buffered_frame_count(),
                terminator=result.terminated_by,
                reason=result.reason,
                stage_index=stage_index,
                elapsed_s=result.elapsed_s,
                frames=result.frames,
                hertz=result.hertz,
            )
            stage_in_flight = None

            aborting = result.terminated_by in ABORTING_TERMINATORS
            _emit_boundary(
                context,
                stage_index=stage_index,
                stage_id=stage.id,
                # The chain only CROSSED when a next stage actually starts. An
                # abort at a non-final stage ends the trial here, so to_stage_id
                # is null and no stage_start follows -- which is what the frozen
                # aggregator fixture's aborted trial t03 already models.
                to_stage_id=(
                    stages[stage_index + 1].id
                    if not aborting and stage_index < len(stages) - 1
                    else None
                ),
                outcome=outcome,
                stop_base_s=stop_base_s,
            )
            boundary_emitted = True

            if aborting:
                logger.warning(
                    f"trial aborted at stage {stage.id!r}: {result.terminated_by}"
                )
                break

        # Read once, BEFORE saving: save_episode pops "size" out of the buffer
        # (lerobot_dataset.py:1249) and then clear_episode_buffer replaces the
        # whole buffer (:1604), so this same call after the save reports 0 and
        # trial_end would claim an empty episode.
        frame_count = context.buffered_frame_count()
        aborted = any(
            result.terminated_by in ABORTING_TERMINATORS for result in results
        )

        episode_saved = False
        if frame_count > 0:
            # ONCE, after the last stage -- never per stage. One episode is the
            # whole point: a per-stage save would cut the chain into separate
            # episodes and destroy the transition the trial exists to measure.
            try:
                context.dataset.save_episode()
                episode_saved = True
            except BaseException as error:
                # A trial's own record of what happened must not be contingent on
                # the save succeeding. Without this, a rejected episode buffer, a
                # full disk or an ffmpeg failure ends the run with NO trial_end
                # line at all, and the aggregator then reports "the process was
                # killed before it could finish the trial" about a trial whose
                # every stage completed -- with the real cause, the failed save,
                # nowhere in the log. BaseException so a Ctrl+C during a
                # multi-second encode gets the same line; the exception is
                # re-raised either way and the handler below adds nothing except
                # a second base stop.
                log.emit(
                    EVENT_TRIAL_END,
                    frame_idx=frame_count,
                    reason=TRIAL_REASON_EXCEPTION,
                    completed=False,
                    stage_count=len(results),
                    elapsed_s=time.perf_counter() - trial_started,
                    episode_saved=False,
                    buffered_frames=frame_count,
                    # Additive field: the closed trial-reason vocabulary has no
                    # "save_failed" member, so the distinction lives here rather
                    # than in a value the aggregator does not know.
                    save_error=repr(error),
                )
                trial_end_emitted = True
                logger.exception("save_episode failed after every stage completed")
                raise

        if not aborted:
            reason = TRIAL_REASON_COMPLETED
        elif frame_count > 0:
            reason = TRIAL_REASON_ABORTED
        else:
            reason = TRIAL_REASON_ABORTED_EMPTY

        log.emit(
            EVENT_TRIAL_END,
            frame_idx=frame_count,
            reason=reason,
            completed=not aborted,
            stage_count=len(results),
            elapsed_s=time.perf_counter() - trial_started,
            episode_saved=episode_saved,
            buffered_frames=frame_count,
        )
        trial_end_emitted = True
        logger.info(
            f"trial ended: {reason} ({len(results)} stages, {frame_count} frames)"
        )
        if not base_is_stopped:
            # Loud, and on the console as well as in the JSONL: this line and the
            # exit code are what an operator who is not reading events.jsonl has.
            logger.error(
                "A STAGE BOUNDARY LEFT THE BASE UNSTOPPED -- both stop paths "
                "failed at least once this trial. CHECK THE ROBOT. The failing "
                "boundary is the transition line with stop_base_path='failed'"
            )
        return TrialOutcome(results=results, base_is_stopped=base_is_stopped)

    except BaseException as error:
        # BaseException, not Exception: Ctrl+C is the operator's reflex stop and
        # KeyboardInterrupt does not derive from Exception, so an
        # `except Exception` would let precisely the emergency case skip the base
        # stop below. The try covers the WHOLE trial body, not just the executor
        # call, because an escape from anywhere in the loop -- an OSError from
        # log.emit, a KeyboardInterrupt between two statements -- left the base
        # driving and the episode unresolved exactly the same way.
        #
        # Stopping the base is the FIRST statement. The arms are position
        # controlled and hold where they are, but the base keeps executing its
        # last velocity command, and the unwind that follows this handler --
        # image-writer join, then encoder flush -- runs for seconds before
        # cli.main's finally reaches robot.disconnect(). Calling stop_base a
        # second time when the loop already stopped the base is deliberate and
        # cheap: one more get_observation and send_action, versus reasoning about
        # which statement the exception came from while the robot is moving.
        emergency_started = time.perf_counter()
        outcome = _stop_base_never_raises(context.robot)
        stop_base_s = time.perf_counter() - emergency_started
        try:
            _record_failed_trial(
                context,
                error=error,
                outcome=outcome,
                stop_base_s=stop_base_s,
                stage_index=stage_index,
                stage_id=stage_id,
                stage_in_flight=stage_in_flight,
                stage_started=stage_started,
                frames_before=frames_before,
                boundary_emitted=boundary_emitted,
                trial_end_emitted=trial_end_emitted,
                results=results,
                trial_started=trial_started,
            )
        except Exception:
            # The log is best-effort from here. The original exception is what
            # the caller has to see -- masking it with "no space left on device"
            # from the very emit that was trying to record it would destroy the
            # only clue to what actually went wrong.
            logger.exception("could not record the failed trial in events.jsonl")
        # save_episode is deliberately NOT called: how far add_frame got is
        # unknown. VideoEncodingManager.__exit__ cleans the interrupted episode's
        # image directories, and trial_end above says episode_saved=false so the
        # discard is a recorded fact rather than an inference from an absence.
        # robot.disconnect() in cli.main's finally sends set_cmd_vel(0.0, 0.0)
        # first (mobileai.py:569) and remains the backstop, but it is seconds of
        # unwinding away, which is why the base is stopped at the top of this
        # handler rather than left to it.
        raise


def _recorded_gripper_goals(context: StageContext) -> dict[str, float] | None:
    """The gripper goals of the last action record_loop sent, or None. Never raises.

    record_loop records ``action_values`` and sends
    ``robot_action_processor((action_values, obs))`` (lerobot_record.py:396-412),
    and the robot_action processor stage_runner hands it is make_default_processors'
    identity (record_adapter.make_processors) -- so the last recorded action IS the
    last action sent, rounded to the dataset's float32 (sub-micrometre on a
    gripper carriage in metres). build_dataset_frame lays the values out in
    ``features["action"]["names"]`` order, so they are read back BY NAME.

    None means "unknown", and the boundary hold then falls back to the measured
    position -- the hold every boundary sent before start poses existed -- with
    a warning, and gripper_goal_source "measured" on the stage_reposition event
    says so. Never raising is the point: this runs between record_loop and
    stop_base, and nothing there may cost the base stop.
    """
    try:
        names = list(context.dataset.features["action"]["names"])
        row = context.dataset.episode_buffer["action"][-1]
        values = dict(zip(names, (float(value) for value in row), strict=True))
        goals = {
            key: values[key]
            for key in context.robot.action_features
            if reposition.is_gripper_key(key)
        }
        if not goals or not all(math.isfinite(value) for value in goals.values()):
            raise ValueError(f"no finite gripper goal in the last action: {goals}")
        return goals
    except Exception:
        logger.warning(
            "could not read the gripper goals of the last recorded action; the "
            "next boundary holds the grippers at their MEASURED position, which "
            "may loosen a grip during the start pose move",
            exc_info=True,
        )
        return None


def _reposition_before_stage(
    context: StageContext,
    stage: StageConfig,
    *,
    stage_index: int,
    boundary: StopBaseOutcome | None,
    gripper_goal_source: str = reposition.GRIPPER_GOAL_UNCHANGED,
) -> str | None:
    """Move the arms to ``stage.start_pose`` and log it. None = start the stage.

    Returns a reason string when the stage must NOT run its policy (the pose was
    not reached, the move failed, or the base was not confirmed stopped). Returns
    None when the arms are at the pose -- or when the move was SKIPPED because the
    operator already pressed Esc / left arrow, in which case the executor's own
    abort-flag check ends the trial exactly as it would without a start pose. A
    KeyboardInterrupt (or SystemExit) is logged and re-raised: run_trial's handler
    stops the base and closes the log, and Ctrl+C reaches the operator.

    Why each refusal is a refusal:

    * Esc pressed -> skipped. Moving the arms for seconds after the operator
      asked the run to stop is the opposite of what the key means.
    * The boundary's stop_base did not stop the base -> refused. Both stop paths
      failed, so the base may still be holding the previous stage's velocity; a
      blocking multi-second arm move would hold the robot in exactly that state
      with no loop running. Ending the trial instead runs one more stop_base at
      this stage's boundary at once, then cli's disconnect, which zeroes the
      base directly again. Not applicable to stage 0: nothing has commanded the
      base since connect().
    * Any exception from the move -> error. The controller may have refused a
      goal; whatever the arms did, they are not known to be at the pose.

    The gripper is never commanded and the base never moved by the move itself
    (reposition.py says why). The gripper goal in force DURING the move is the
    one the boundary's hold left: the previous stage's last command when the
    runner could read it (gripper_goal_source "previous_command"), the measured
    position when it could not ("measured"), and whatever was already standing
    at stage 0 or after a fallback stop that commanded no arm ("unchanged"). No
    frame is recorded: this sits between two record_loop calls, so frame_idx is
    the same on both sides of the event.

    reposition_s IS NOT TRANSITION TIME. It lands between the previous stage_end
    and this stage_start, i.e. inside the wall-clock window aggregate.build_trial
    derives the stage_end -> stage_start residual from; aggregate subtracts it
    back out, so the residual and boundary_cost_s keep measuring the seam's
    bookkeeping exactly as they did before this feature, and the move's own cost
    is reported separately as stage_reposition_s.

    reposition_s INCLUDES THE EXPIRY WAIT: move_arms_to_start_pose first sleeps
    until the boundary hold's trajectory has run out (reposition.py says why),
    and that sleep starts after ``started``. Kept inside the move's own time on
    purpose -- outside it, it would land in the stage_end -> stage_start
    residual and read as seam bookkeeping. expiry_wait_s reports it alone.
    """
    start_pose = stage.start_pose
    source = start_pose.source
    fields = {
        "stage_index": stage_index,
        "target": dict(start_pose.positions),
        "min_goal_time_s": start_pose.min_goal_time_s,
        "position_tolerance_rad": start_pose.position_tolerance_rad,
        "source": {
            "dataset_repo_id": source.dataset_repo_id,
            "dataset_revision": source.dataset_revision,
            "dataset_commit": source.dataset_commit,
            "policy_path": source.policy_path,
            "episode_index": source.episode_index,
            "n_episodes": source.n_episodes,
            "method": source.method,
            "distance": source.distance,
        },
        # The boundary this move follows, so a reader can see the move happened
        # with the base confirmed stopped. Null at stage 0.
        "boundary_stop_base_path": None if boundary is None else boundary.path,
        # Said out loud rather than left to a reader's knowledge of reposition.py:
        # a stage that begins holding an object depends on it.
        "gripper_commanded": False,
        # The gripper goal standing while the arms move, and where it came from
        # -- the question a dropped beaker asks first.
        "gripper_goal_source": gripper_goal_source,
        "gripper_goal": (
            {
                key: value
                for key, value in (boundary.action or {}).items()
                if reposition.is_gripper_key(key)
            }
            if boundary is not None
            and gripper_goal_source != reposition.GRIPPER_GOAL_UNCHANGED
            else None
        ),
    }

    def emit(outcome: str, reason: str, **extra: object) -> None:
        context.log.emit(
            EVENT_STAGE_REPOSITION,
            stage_id=stage.id,
            frame_idx=context.buffered_frame_count(),
            reason=reason,
            outcome=outcome,
            **fields,
            **extra,
        )

    if context.events.get("stop_recording") or context.events.get("rerecord_episode"):
        reason = "an abort key was pressed before this stage; the arms were not moved"
        logger.warning(f"stage {stage.id!r}: start pose skipped -- {reason}")
        emit(
            reposition.REPOSITION_SKIPPED,
            reason,
            goal_time_s=None,
            expiry_wait_s=None,
            reposition_s=0.0,
        )
        return None
    if boundary is not None and not boundary.base_is_stopped:
        reason = (
            f"the previous boundary's stop_base path was {boundary.path!r}, so the "
            "base may still be moving; the arms were not moved"
        )
        emit(
            reposition.REPOSITION_REFUSED,
            reason,
            goal_time_s=None,
            expiry_wait_s=None,
            reposition_s=0.0,
        )
        return reason

    started = time.perf_counter()
    try:
        moved = reposition.move_arms_to_start_pose(
            context.robot,
            start_pose.positions,
            min_goal_time_s=start_pose.min_goal_time_s,
            position_tolerance_rad=start_pose.position_tolerance_rad,
        )
    except Exception as error:
        reason = f"start pose move failed: {error!r}"
        logger.exception(f"stage {stage.id!r}: {reason}")
        emit(
            reposition.REPOSITION_ERROR,
            reason,
            goal_time_s=None,
            expiry_wait_s=None,
            reposition_s=time.perf_counter() - started,
        )
        return reason
    except BaseException as interrupt:
        # Ctrl+C mid-move. The trajectory already handed to the controller runs
        # to its goal (the arms are position-controlled; nothing here can recall
        # it), and the base was confirmed stopped before the move began. The
        # line is best-effort -- a failing emit must not replace the interrupt.
        try:
            emit(
                reposition.REPOSITION_INTERRUPTED,
                f"interrupted during the start pose move: {interrupt!r}",
                goal_time_s=None,
                expiry_wait_s=None,
                reposition_s=time.perf_counter() - started,
            )
        except Exception:
            logger.exception("could not record the interrupted start pose move")
        raise
    reposition_s = time.perf_counter() - started

    outcome = (
        reposition.REPOSITION_REACHED
        if moved.reached
        else reposition.REPOSITION_NOT_REACHED
    )
    emit(
        outcome,
        moved.detail,
        goal_time_s=moved.goal_time_s,
        expiry_wait_s=moved.expiry_wait_s,
        expiry_wait_source=moved.expiry_wait_source,
        previous_goal_time_s=moved.previous_goal_time_s,
        reposition_s=reposition_s,
        max_abs_error_rad=moved.max_abs_error,
        start_positions=moved.start_positions,
        end_positions=moved.end_positions,
    )
    logger.info(
        f"stage {stage.id!r}: start pose {outcome} in {reposition_s:.2f}s "
        f"(episode {source.episode_index} of {source.dataset_repo_id or '?'})"
    )
    return None if moved.reached else moved.detail


def _emit_boundary(
    context: StageContext,
    *,
    stage_index: int,
    stage_id: str,
    to_stage_id: str | None,
    outcome: StopBaseOutcome,
    stop_base_s: float,
) -> None:
    """Write the transition event for one boundary. EVERY boundary gets one.

    This line is the only place a stop_base outcome reaches disk, so gating it on
    "a next stage exists and we are not aborting" meant a failed base stop could
    only ever be recorded at the boundaries where it does not matter. The two it
    hid are the two that do: after the LAST stage, where the base would hold its
    velocity through the seconds-long save_episode and encoder flush, and after an
    ABORT, where the operator hit Esc precisely because something is wrong. A run
    whose final stop_base silently failed must never be byte-identical to a clean
    one.

    ``to_stage_id`` is null when the chain did not cross, which is additive:
    aggregate.build_trial derives the stage_end -> stage_start RESIDUAL from wall
    clocks and reads only ``stop_base_s`` off this event, so a boundary that leads
    nowhere contributes a stop_base duration and no residual. Note that
    stop_base_durations_s therefore samples N boundaries per N-stage trial, not
    N-1. The residual is NOT the whole boundary: this call has already finished
    by the time stage_end is emitted, so the two are disjoint halves and
    aggregate sums them per boundary.
    """
    context.log.emit(
        EVENT_TRANSITION,
        stage_id=stage_id,
        frame_idx=context.buffered_frame_count(),
        # "stop_base" | "stop_base_fallback" | "stop_base_failed", derived from
        # the path in transitions.REASON_BY_STOP_PATH so the two cannot drift.
        reason=outcome.reason,
        from_stage_id=stage_id,
        to_stage_id=to_stage_id,
        stage_index=stage_index,
        # None whenever the primary path did not run to completion, so it says
        # "no hold action was sent" and nothing more. Which mechanism zeroed the
        # base -- or whether anything did -- is stop_base_path, and a reader
        # deciding whether the base was stopped must use that field: a null
        # stop_action covers BOTH the fallback (base stopped, arms uncommanded)
        # and the failure (base possibly still moving).
        stop_action=outcome.action,
        stop_base_path=outcome.path,
        stop_base_error=outcome.error or None,
        # stop_base_s: the FIRST half of the boundary, not the whole of it. On
        # the primary path this covers one get_observation plus one send_action
        # and nothing else; on the other two it is how long the failing call took
        # before the fallback ran, which is why aggregate counts those boundaries
        # separately instead of averaging them in. Either way it excludes the
        # next stage's record_loop entry cost, which is the dominant term. The
        # SECOND half -- the stage_end -> stage_start residual -- is derived in
        # aggregate.py from wall clocks, which is why both carry
        # millisecond-resolution timestamps. The two are disjoint, because this
        # call runs to completion BEFORE the stage_end emit (a base still driving
        # must not wait on bookkeeping), so aggregate sums them per boundary
        # rather than reading either one as the seam's cost.
        stop_base_s=stop_base_s,
    )


def _record_failed_trial(
    context: StageContext,
    *,
    error: BaseException,
    outcome: StopBaseOutcome,
    stop_base_s: float,
    stage_index: int,
    stage_id: str | None,
    stage_in_flight: str | None,
    stage_started: float,
    frames_before: int,
    boundary_emitted: bool,
    trial_end_emitted: bool,
    results: list[StageResult],
    trial_started: float,
) -> None:
    """Close the log on the failure path: stage_end, boundary, trial_end.

    Each line is written only if it is missing, because the failure can arrive
    anywhere: build_trial raises on a stage_end with no stage_start and reads a
    second trial_end as a second trial, so a duplicate is worse than the gap it
    would fill.
    """
    frame_count = context.buffered_frame_count()

    if stage_in_flight is not None and stage_id is not None:
        # The error stage_end carries the same numeric fields as every other
        # stage_end (elapsed_s, frames, hertz) because the aggregator reads them
        # positionally off the schema, not with .get -- a stage_end missing them
        # would break parsing of the whole run, including the stages that DID
        # complete.
        elapsed_s = time.perf_counter() - stage_started
        frames = frame_count - frames_before
        context.log.emit(
            EVENT_STAGE_END,
            stage_id=stage_id,
            frame_idx=frame_count,
            terminator=TERMINATED_BY_ERROR,
            reason=repr(error),
            stage_index=stage_index,
            elapsed_s=elapsed_s,
            frames=frames,
            hertz=frames / elapsed_s if elapsed_s > 0 else 0.0,
        )

    # The error path is a boundary like any other: stop_base ran, so its outcome
    # gets a line, with to_stage_id null because the chain ends here. The second
    # condition covers the case where the loop already wrote this stage's
    # boundary and the emergency stop then FAILED -- a base that is still moving
    # during the unwind is a new fact, and the cost of recording it is one extra
    # stop_base_s sample in an aggregate over a trial that already failed.
    if stage_id is not None and (not boundary_emitted or not outcome.base_is_stopped):
        _emit_boundary(
            context,
            stage_index=stage_index,
            stage_id=stage_id,
            to_stage_id=None,
            outcome=outcome,
            stop_base_s=stop_base_s,
        )

    if not trial_end_emitted:
        context.log.emit(
            EVENT_TRIAL_END,
            frame_idx=frame_count,
            reason=TRIAL_REASON_EXCEPTION,
            completed=False,
            stage_count=len(results) if stage_in_flight is None else len(results) + 1,
            elapsed_s=time.perf_counter() - trial_started,
            # NOT saved, and said out loud. The buffer's state is unknown after
            # an exception, so the frames are discarded by
            # VideoEncodingManager.__exit__ -- an operator counting episodes has
            # to be able to see that here instead of inferring it from silence.
            episode_saved=False,
            buffered_frames=frame_count,
        )
    if frame_count > 0:
        logger.warning(
            f"{frame_count} buffered frames are discarded unsaved: the episode "
            f"buffer's state is unknown after {error!r}"
        )
