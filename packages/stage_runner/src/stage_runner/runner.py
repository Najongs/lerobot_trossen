"""The trial body: one function, one episode, one pass over the stages."""

import logging
import math
import time
from collections.abc import Mapping

from lerobot.utils.control_utils import is_headless

# Module scope, not the call-time import this used to be: the old form was
# there because the package __init__ imported this module eagerly, so
# __version__ did not exist yet while runner was loading. __init__ is lazy now
# (PEP 562) and imports nothing at package-import time, so the cycle is gone.
from stage_runner import __version__ as stage_runner_version
from stage_runner import record_adapter, transitions
from stage_runner.context import StageContext
from stage_runner.events import (
    EVENT_STAGE_END,
    EVENT_STAGE_START,
    EVENT_TELEOP_END,
    EVENT_TELEOP_START,
    EVENT_TRANSITION,
    EVENT_TRIAL_END,
    EVENT_TRIAL_START,
)
from stage_runner.executors import EXECUTORS, _set_pose_guide
from stage_runner.policies import bundle_descriptor
from stage_runner.preflight import (
    check_lerobot_version,
    robot_action_dimension,
    robot_state_dimension,
)
from stage_runner.results import (
    ABORTING_TERMINATORS,
    TERMINATED_BY_ERROR,
    TERMINATED_BY_SIGNAL,
    TRIAL_REASON_ABORTED,
    TRIAL_REASON_ABORTED_EMPTY,
    TRIAL_REASON_CHAIN_FAILED,
    TRIAL_REASON_COMPLETED,
    TRIAL_REASON_EXCEPTION,
    TRIAL_REASON_SIGNAL,
    StageResult,
    TrialOutcome,
    signal_exit_name,
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


TELEOP_ENDED_BY_ARROW: str = "arrow"
TELEOP_ENDED_BY_RESTART: str = "left_arrow_restart"
TELEOP_ENDED_BY_ESC: str = "esc"
TELEOP_ENDED_BY_TIMEOUT: str = "timeout"
TELEOP_ENDED_BY_ESC_BEFORE_START: str = "esc_before_start"
TELEOP_ENDED_BY_LEFT_BEFORE_START: str = "left_arrow_before_start"
TELEOP_ENDED_BY_EXCEPTION: str = "exception"
TELEOP_ENDED_BY_ARROW_NOT_READY: str = "arrow_not_ready"


def _initial_gap_rad(context: StageContext, first) -> tuple[float, str] | None:
    """(largest per-joint gap in rad, joint name) between the arm now and the first
    stage's designated pose -- the number the initial reset's jump gate will see.
    None when it cannot be measured (no chain, no pose, keys missing)."""
    chain = context.chain
    if chain is None or first is None or first.stage_number is None:
        return None
    try:
        stage_params = chain.params.stage(first.stage_number)
        target = getattr(stage_params, "start_pose", None) or stage_params.start_pose_rad_arm12
        observation = context.robot.get_observation()
        worst, worst_name = -1.0, ""
        for index, name in enumerate(chain.params.arm_joint_order):
            # ChainParams keeps the pose by joint NAME (chain_params.py: "why names and
            # not indices"); a plain sequence in arm_joint_order is accepted too.
            goal = target[name] if isinstance(target, Mapping) else target[index]
            value = observation.get(f"{name}.pos")
            if value is None:
                logger.warning(f"initial gap not measured: observation has no {name}.pos -- the arrow is accepted unchecked")
                return None
            gap = abs(float(value) - float(goal))
            if gap > worst:
                worst, worst_name = gap, name
        return (worst, worst_name) if worst >= 0.0 else None
    except Exception:  # a readout must never cost the run
        logger.warning("initial gap could not be measured -- the arrow is accepted unchecked", exc_info=True)
        return None
# SIGTERM/SIGHUP landed inside the window (cli.SignalLatch's SystemExit) -- an
# operator's stop, not a defect; kept apart from `exception` like the stage and
# trial values (results.TERMINATED_BY_SIGNAL / TRIAL_REASON_SIGNAL).
TELEOP_ENDED_BY_SIGNAL: str = "signal"
# The NaN/Inf action gate tripped inside the window. It raises stop_recording,
# the same flag Esc does, so without this value a model/stats defect would be
# filed as "the operator pressed Esc" (review 10/07). The exit code was already
# 4 (cli checks the gate first); only the window's label was wrong.
TELEOP_ENDED_BY_NAN_GATE: str = "nan_gate"


def _run_teleop_phase(context: StageContext, stages) -> bool:
    """리더암 텔레옵 구간 -- 첫 단계(최초 리셋) 전에, lerobot-record 의 리셋 구간처럼.

    ``context.teleop`` 이 None 이면 아무것도 하지 않는다(옛 동작). 있으면:
    사람이 리더암으로 팔을 끌어 물체를 쥐게 하고 시작 자세를 잡는다. ``POSE`` 줄은 첫
    단계의 지정 자세를 목표로 1초마다 찍힌다(pose_guide, phase=teleop).
    ``→`` 로 끝내면 최초 리셋으로 넘어가고, ``←`` 는 구간을 처음부터 다시 돌며(타이머
    리셋), ESC 는 플래그를 남겨 첫 단계가 「시작 전 중단」 으로 끝나게 한다. **타임아웃도
    중단이다** -- 사람 확인 없이 램프가 시작되지 않게(리뷰 10/07).
    구간은 녹화되지 않는다. 끝날 때마다 ``transitions.stop_base`` 로 베이스를 세운다 --
    리더 action 에 x.vel/theta.vel 이 실려 오므로(기본은 0 으로 덮지만, 설정으로 켤 수 있다).
    루프에 들어가기 **전에** 이미 켜진 ESC/``←`` 는 구간을 돌지 않고 그대로 첫 단계의 「시작 전
    중단」 으로 넘긴다 -- connect 중에 누른 ESC 로 리더가 300 s 동안 팔을 끄는 일이 없게.

    Returns whether every base stop in the window reported the base stopped (the
    trial-level ``base_is_stopped`` folds it in, so a failed stop here is exit 3 too).
    """
    teleop = context.teleop
    if teleop is None:
        return True
    config = context.config
    events = context.events
    log = context.log
    first = stages[0] if stages else None
    gate_rad = context.chain.reset.initial_max_jump_rad if context.chain is not None else None
    if first is not None and context.chain is not None:
        _set_pose_guide(first.stage_number, context.chain, gate_rad=gate_rad)
    control_time_s = float(config.teleop_time_s)
    margin = record_adapter.manual_detection_margin_s(config.dataset.fps)
    task = (first.instruction or first.name) if first is not None else "teleop"
    first_id = first.id if first is not None else None
    if is_headless():
        logger.warning(
            "teleop phase: headless environment -- no keyboard, so the phase can only "
            f"end by its {control_time_s:.0f} s ceiling (which ABORTS the trial)"
        )

    def _pre_set() -> str | None:
        if events.get("stop_recording"):
            return TELEOP_ENDED_BY_ESC_BEFORE_START
        if events.get("rerecord_episode"):
            return TELEOP_ENDED_BY_LEFT_BEFORE_START
        return None

    steps_before = list(context.processors.teleop_action.steps)
    if not config.teleop_base_from_leader:
        context.processors.teleop_action.steps = steps_before + [record_adapter.TeleopBaseZeroStep()]
    base_is_stopped = True
    attempt = 0
    try:
        while True:
            attempt += 1
            pre = _pre_set()
            if pre is not None:
                # Not cleared: the first stage's executor turns it into an abort before it moves.
                log.emit(
                    EVENT_TELEOP_END,
                    frame_idx=context.buffered_frame_count(),
                    t_mono=time.perf_counter(),
                    attempt=attempt,
                    elapsed_s=0.0,
                    ended_by=pre,
                    stop_base_path=None,
                    stop_base_direct_ok=None,
                )
                logger.warning(
                    f"teleop phase skipped: {pre} (a flag set before the window, e.g. during "
                    "connect) -- the first stage will end as an abort without moving"
                )
                return True
            # A stale right-arrow from before the loop must not end the phase on its first tick.
            # Said out loud, like executors._discard_stale_exit_early does for a stage: a
            # press during connect that vanished without a line is a mystery at analysis.
            if events.get("exit_early"):
                logger.warning(
                    "teleop phase: a RIGHT ARROW pressed before the window (e.g. during "
                    "connect) is discarded -- press it again once the leader arms are live"
                )
            events["exit_early"] = False
            log.emit(
                EVENT_TELEOP_START,
                frame_idx=context.buffered_frame_count(),
                t_mono=time.perf_counter(),
                attempt=attempt,
                control_time_s=control_time_s,
                teleop_type=getattr(teleop, "name", type(teleop).__name__),
                first_stage_id=first_id,
                base_from_leader=bool(config.teleop_base_from_leader),
            )
            logger.info(
                f"teleop phase (attempt {attempt}): leader arms drive the followers for up to "
                f"{control_time_s:.0f} s. Grasp / pose for {first_id or '?'}, then RIGHT ARROW to "
                "start; LEFT ARROW restarts this phase; ESC aborts; the ceiling aborts too."
            )
            started = time.perf_counter()
            try:
                record_adapter.call_teleop_loop(
                    robot=context.robot,
                    teleop=teleop,
                    events=events,
                    fps=config.dataset.fps,
                    processors=context.processors,
                    control_time_s=control_time_s,
                    single_task=task,
                    display_data=config.display_data,
                )
            except BaseException as error:
                # Recorded here, then re-raised: run_trial's except BaseException stops the
                # base and writes trial_end; without this line the window would vanish
                # from events.jsonl.
                log.emit(
                    EVENT_TELEOP_END,
                    frame_idx=context.buffered_frame_count(),
                    t_mono=time.perf_counter(),
                    attempt=attempt,
                    elapsed_s=time.perf_counter() - started,
                    ended_by=(
                        TELEOP_ENDED_BY_SIGNAL
                        if signal_exit_name(error)
                        else TELEOP_ENDED_BY_EXCEPTION
                    ),
                    error=signal_exit_name(error) or repr(error),
                    stop_base_path=None,
                    stop_base_direct_ok=None,
                )
                raise
            elapsed = time.perf_counter() - started
            gap = None
            gate = context.finite_gate
            if gate is not None and getattr(gate, "tripped", False):
                ended_by = TELEOP_ENDED_BY_NAN_GATE  # checked before Esc: same flag
            elif events.get("stop_recording"):
                ended_by = TELEOP_ENDED_BY_ESC
            elif events.get("rerecord_episode"):
                events["rerecord_episode"] = False
                events["exit_early"] = False
                ended_by = TELEOP_ENDED_BY_RESTART
            elif elapsed < control_time_s - margin:
                ended_by = TELEOP_ENDED_BY_ARROW
                # The arrow is accepted only when the initial reset would be too: measured
                # against the same per-joint gate, so the operator never has to read it off
                # the log (the POSE line shows [→ 가능 ✔] for the same condition).
                gap = _initial_gap_rad(context, first)
                if gap is not None and gate_rad is not None and gap[0] > gate_rad:
                    ended_by = TELEOP_ENDED_BY_ARROW_NOT_READY
                    logger.error(
                        f"RIGHT ARROW IGNORED: {gap[1]} is {gap[0]:.2f} rad "
                        f"({math.degrees(gap[0]):.0f}°) from the {first_id} start pose, over the "
                        f"initial reset gate {gate_rad:.2f} rad ({math.degrees(gate_rad):.0f}°). "
                        "Keep driving with the leader until the POSE line says [→ 가능 ✔], then press again."
                    )
            else:
                ended_by = TELEOP_ENDED_BY_TIMEOUT
                events["stop_recording"] = True  # the ceiling is an abort, not a start
            # Normal-path stop_base: a Ctrl+C here is re-raised like at any boundary, and
            # the outcome counts toward the trial's base_is_stopped (exit 3 on failure).
            outcome = transitions.stop_base(context.robot)
            base_is_stopped = base_is_stopped and outcome.base_is_stopped
            log.emit(
                EVENT_TELEOP_END,
                frame_idx=context.buffered_frame_count(),
                t_mono=time.perf_counter(),
                attempt=attempt,
                elapsed_s=elapsed,
                ended_by=ended_by,
                stop_base_path=outcome.path,
                stop_base_direct_ok=getattr(outcome, "direct_ok", None),
                gap_rad=(round(gap[0], 4) if ended_by == TELEOP_ENDED_BY_ARROW_NOT_READY else None),
                gap_joint=(gap[1] if ended_by == TELEOP_ENDED_BY_ARROW_NOT_READY else None),
                gate_rad=gate_rad,
            )
            logger.info(
                f"teleop phase (attempt {attempt}) ended by {ended_by} after {elapsed:.1f} s; "
                f"stop_base {outcome.path}"
            )
            if ended_by in (TELEOP_ENDED_BY_RESTART, TELEOP_ENDED_BY_ARROW_NOT_READY):
                continue
            return base_is_stopped
    finally:
        context.processors.teleop_action.steps = steps_before


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

    HARDWARE FIRST, BOOKKEEPING AFTER, the same rule cli.main's finally follows.
    The base stop happens as soon as the executor returns, before the stage_end
    emit, and the whole trial body is inside one ``except BaseException`` so that
    leaving this function by ANY route -- the executor raising, an OSError out of
    log.emit on a full disk, a Ctrl+C anywhere at all -- still stops the base and
    still resolves the episode in the log. Nothing is swallowed; the caller gets
    its exception back.

    Deliberately absent, all of which upstream ``record()`` has: the trial-repeat
    loop, the inter-trial teleop reset section (stage 4 must START holding the
    tube, so the grasp pose is built by hand with the leader arm between runs),
    the re-record branch, ``dataset.finalize()`` in a finally of our own
    (VideoEncodingManager.__exit__ already does it), and ``push_to_hub``.
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
    # The chain broke. Not sticky-by-accident like base_is_stopped: the loop
    # breaks on it immediately, so it can only be set once.
    chain_failed = False
    chain_failed_stage_id: str | None = None

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
        base_is_stopped = _run_teleop_phase(context, stages) and base_is_stopped

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

        for stage_index, stage in enumerate(stages):
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
                # ADDITIVE chain fields. All four are null in a version 1 run,
                # and aggregate.build_trial reads this event with .get for
                # everything except the base fields, so adding them changes no
                # existing reader.
                #
                # `kind` and `stage_number` exist because `stage_index` is NOT
                # the stage number once resets are interleaved -- stage 11 sits
                # at index 21 -- and every consumer (the report, the per-stage
                # base integral, the success table) has to drop the resets.
                #
                # `t_mono` is perf_counter, THE SAME CLOCK as the `t_mono`
                # column of basevel.csv (base_vel_log.py:69). That shared clock
                # is the only thing that lets eval_chain_report integrate the
                # base command over one stage's exact range instead of over the
                # whole run; wall_clock_iso cannot do it (it is local time with
                # millisecond resolution and a different origin).
                kind=stage.kind,
                stage_number=stage.stage_number,
                onehot_index=stage.onehot_index,
                t_mono=time.perf_counter(),
            )
            stage_in_flight = stage.id
            boundary_emitted = False
            logger.info(
                f"stage {stage_index} {stage.id!r}: {stage.executor} "
                f"({plan.planned_terminator}, {plan.control_time_s:.1f}s)"
            )

            frames_before = context.buffered_frame_count()
            stage_started = time.perf_counter()
            result = EXECUTORS[stage.executor](context, stage)
            results.append(result)

            # HARDWARE FIRST: the base is zeroed before ANY of the bookkeeping
            # below. Nothing between record_loop's last send_action and this call
            # should be I/O -- an OSError out of the stage_end emit (a full disk
            # on the robot PC while three cameras write video) used to escape
            # with the base still holding the last policy velocity.
            transition_started = time.perf_counter()
            outcome = transitions.stop_base(context.robot)
            stop_base_s = time.perf_counter() - transition_started
            # Read BEFORE the emit, and `and` rather than assignment: the emit
            # below can raise (a full disk), and the exit code must still carry
            # a stop that already failed.
            base_is_stopped = base_is_stopped and outcome.base_is_stopped

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
                # ADDITIVE, see the stage_start emit. `reason_detail` is the
                # machine-readable half of `reason`: p_last, stall_s, nn_dist
                # for a completion; reset_T, reset_dmax, reach_err for a ramp;
                # clamped_ticks either way. A completion whose numbers are not
                # recorded cannot be told from a mislabelled one months later.
                kind=stage.kind,
                stage_number=stage.stage_number,
                t_mono=time.perf_counter(),
                reason_detail=result.detail or None,
                required_terminator=list(stage.required_terminator) or None,
            )
            stage_in_flight = None

            aborting = result.terminated_by in ABORTING_TERMINATORS
            # THE CHAIN BREAK. A stage that ended any way its StageConfig does
            # not allow -- a policy stage that timed out instead of completing,
            # a reset that never arrived -- ends the run. There is NO RETRY and
            # no code path that could add one: retrying a reset would repeat the
            # motion that already failed, and retrying a policy stage would roll
            # it out again from wherever the failed attempt left the arms, which
            # is not a start pose any demonstration has.
            if (
                not aborting
                and stage.required_terminator
                and result.terminated_by not in stage.required_terminator
            ):
                chain_failed = True
                chain_failed_stage_id = stage.id
                logger.error(
                    f"CHAIN BROKEN at stage {stage.id!r}: ended "
                    f"{result.terminated_by!r}, which is not one of "
                    f"{list(stage.required_terminator)}. {result.reason} "
                    "The remaining stages are NOT run."
                )
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
                    if not aborting
                    and not chain_failed
                    and stage_index < len(stages) - 1
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
            if chain_failed:
                # AFTER the boundary emit, like the abort above: stop_base has
                # already run and its outcome is the one line that says whether
                # the base is still driving. Breaking before it would hide a
                # failed stop at exactly the boundary where something already
                # went wrong.
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
                    reason=(
                        TRIAL_REASON_SIGNAL
                        if signal_exit_name(error)
                        else TRIAL_REASON_EXCEPTION
                    ),
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

        if chain_failed:
            # Checked FIRST: a chain failure is the model's, an abort is the
            # operator's, and merging them would count a model failure as a
            # human decision in the chain report. The episode IS saved either
            # way -- a partial chain is data, and the aggregator excludes it via
            # `completed`.
            reason = TRIAL_REASON_CHAIN_FAILED
        elif not aborted:
            reason = TRIAL_REASON_COMPLETED
        elif frame_count > 0:
            reason = TRIAL_REASON_ABORTED
        else:
            reason = TRIAL_REASON_ABORTED_EMPTY

        log.emit(
            EVENT_TRIAL_END,
            frame_idx=frame_count,
            reason=reason,
            completed=not (aborted or chain_failed),
            stage_count=len(results),
            elapsed_s=time.perf_counter() - trial_started,
            episode_saved=episode_saved,
            buffered_frames=frame_count,
            # ADDITIVE: which stage broke the chain, and how many of the
            # configured stages ever ran. `stage_count` alone cannot say it --
            # a chain that failed at stage 3 and one that was configured for
            # three stages both report 5.
            chain_failed=chain_failed,
            chain_failed_stage_id=chain_failed_stage_id,
            stages_configured=len(stages),
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
        return TrialOutcome(
            results=results,
            base_is_stopped=base_is_stopped,
            chain_failed=chain_failed,
            chain_failed_stage_id=chain_failed_stage_id,
        )

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
        # The ONLY delivery ack in the whole stop path: what
        # base.set_cmd_vel(0.0, 0.0) returned. True = the base accepted it,
        # False = it refused or raised, null = this robot has no base to command
        # (the mock, a single-arm rig) -- which is not a failure. A reader asking
        # "did the base actually stop" uses stop_base_path; this field says what
        # the base itself answered, which is the evidence behind that path.
        stop_base_direct_ok=outcome.direct_ok,
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
            # A SIGTERM/SIGHUP unwinding through the stage is the operator's
            # stop, not the stage's defect: `signal` with the signal's name,
            # so the chain report does not file it under crashes (④', 10/07).
            terminator=(
                TERMINATED_BY_SIGNAL if signal_exit_name(error) else TERMINATED_BY_ERROR
            ),
            reason=signal_exit_name(error) or repr(error),
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
            reason=TRIAL_REASON_SIGNAL if signal_exit_name(error) else TRIAL_REASON_EXCEPTION,
            completed=False,
            stage_count=len(results) if stage_in_flight is None else len(results) + 1,
            # ADDITIVE (review 10/07): the emergency stop's outcome. When the
            # unwind happened before any stage ran -- the teleop window, the
            # only place the robot moves before the first stage_start -- there
            # is no stage boundary to carry it (stage_id is None above), and a
            # base that may still be moving left no line in events.jsonl.
            emergency_stop_base_path=outcome.path,
            emergency_stop_base_is_stopped=outcome.base_is_stopped,
            emergency_stop_base_error=outcome.error,
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
