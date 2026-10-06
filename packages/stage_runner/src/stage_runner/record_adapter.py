"""Seam 1 of 2 to installed lerobot: the single call site of ``record_loop``, plus
the robot, processor, dataset and keyboard construction that feeds it.

If an upstream signature moves, this file is the only one that changes. Nothing
else in the package imports ``lerobot.scripts.lerobot_record``.
"""

import logging
import sys
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import (
    aggregate_pipeline_dataset_features,
    create_initial_features,
)
from lerobot.datasets.utils import combine_feature_dicts
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.processor import (
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)
from lerobot.robots import Robot, RobotConfig, make_robot_from_config
from lerobot.scripts import lerobot_record
from lerobot.utils.constants import ACTION
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.import_utils import register_third_party_plugins

from stage_runner.config import (
    TERMINATOR_COMPLETION,
    TERMINATOR_MANUAL,
    TERMINATOR_REACHED,
    DefaultsConfig,
    StageConfig,
    StageRunnerConfig,
)
from stage_runner.policies import PolicyBundle
from stage_runner.results import (
    TERMINATED_BY_MANUAL,
    TERMINATED_BY_RERECORD_REQUESTED,
    TERMINATED_BY_STOP_RECORDING,
    TERMINATED_BY_TIMEOUT,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RecordProcessors:
    """The three pipelines ``record_loop`` takes, as one value.

    They are constructed together by ``make_default_processors`` and the dataset
    features are derived from two of them, so passing them around separately is
    how a feature dict stops matching the pipeline that produced it -- which
    surfaces as a KeyError inside ``build_dataset_frame``, one frame into the
    control loop.
    """

    teleop_action: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ]
    robot_action: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ]
    robot_observation: RobotProcessorPipeline[RobotObservation, RobotObservation]


@dataclass(frozen=True)
class StagePlan:
    """What a stage was set up to do, decided before it runs.

    ``planned_terminator`` is logged on stage_start and compared against the
    ACTUAL terminator on stage_end; a plan and an outcome that disagree is the
    whole point of writing both.
    """

    control_time_s: float
    planned_terminator: str


# Set to "1" to skip the third-party plugin scan. The ONE reason it exists: on a
# machine with no robot, importing `lerobot_robot_trossen` pulls in `mobileai`
# and through it the `trossen_slate` SDK (packages/.../__init__.py:5), and the
# repo rule is that code importing a robot SDK is not executed off the robot PC
# (CLAUDE.md 실기 안전). The mock robot does NOT come from this scan --
# config.parse_config imports stage_runner.mock_robot itself -- so the whole
# hardware-free path works with the scan off.
#
# It is never set on the robot PC, and setting it there would make
# `type: mobileai_robot` unresolvable in draccus, i.e. a loud failure at parse
# time rather than a quiet one later.
NO_PLUGINS_ENV: str = "STAGE_RUNNER_NO_PLUGINS"


def register_plugins() -> None:
    """Import every installed lerobot_{robot,camera,teleoperator,policy}_* distribution.

    MUST run before any config is parsed. This is what puts ``mobileai_robot``
    into RobotConfig's draccus choice registry; without it draccus dies in type
    resolution on the YAML's ``robot: type:`` key. Upstream's own main() does
    exactly this before calling record() (lerobot_record.py:606-608).

    Skipped when :data:`NO_PLUGINS_ENV` is set -- see the comment above it.
    """
    import os

    if os.environ.get(NO_PLUGINS_ENV, "").strip() not in ("", "0", "false", "no"):
        logger.warning(
            f"{NO_PLUGINS_ENV} is set: the third-party plugin scan is SKIPPED, so "
            "`type: mobileai_robot` and the one-hot/loop-rate/rearm patches are "
            "NOT available. This is the hardware-free path; the real robot needs "
            "the variable unset."
        )
        return
    register_third_party_plugins()


def clamped_arm_ticks() -> int | None:
    """Arm-ticks clamped by ``max_relative_target`` so far, or None if unmeasured.

    Read out of ``sys.modules`` rather than imported: the counter lives in
    ``lerobot_robot_trossen.loop_rate_log``, importing that package pulls in the
    Trossen SDK, and on the robot PC ``register_plugins()`` has already imported
    it so the lookup succeeds. Off the robot it returns None, which is the
    truth -- nothing counted.

    ``_clamped["total"]`` is the RUN total and is never reset (the per-phase
    counter IS reset, by ``_reset_phase``'s finally, before a caller could read
    it after ``record_loop`` returns), so a per-stage count is the difference of
    two reads around the stage. A nonzero count during a RESET means the ramp
    asked for more than 0.1 rad in a tick, i.e. the ramp was wrong -- the clamp
    is a safety net here, never the mechanism.
    """
    module = sys.modules.get("lerobot_robot_trossen.loop_rate_log")
    if module is None:
        return None
    getter = getattr(module, "clamped_total", None)
    if not callable(getter):
        return None
    try:
        return int(getter())
    except Exception:
        return None


def install_action_monitor(processors: RecordProcessors, step: Any) -> None:
    """Append a ProcessorStep to the ROBOT ACTION pipeline, in place.

    ``robot_action_processor`` is the pipeline that receives ``(action,
    observation)`` as one transition every tick, which is the only place the
    commanded arm, the commanded base, the progress output and the measured arm
    are in hand together.

    APPENDED, not inserted: upstream builds it with a single
    ``IdentityProcessorStep`` (processor/factory.py:38-45), and running after
    that identity means the monitor sees exactly what ``send_action`` will get.

    It must NOT go in ``teleop_action`` -- that is the pipeline
    ``build_dataset_features`` derives the dataset's ACTION features from
    (lerobot_record.py:449-456), so a step there would have to answer
    ``transform_features`` correctly or silently change the recorded schema.
    """
    steps = list(processors.robot_action.steps)
    steps.append(step)
    processors.robot_action.steps = steps


def make_robot(robot_config: RobotConfig) -> Robot:
    """Construct, NOT connect.

    ``observation_features`` and ``action_features`` are config-derived, so the
    dimension preflight can read them with the arms unpowered.
    """
    return make_robot_from_config(robot_config)


def make_processors() -> RecordProcessors:
    """The three identity pipelines upstream ``record()`` builds, in its order."""
    teleop_action, robot_action, robot_observation = make_default_processors()
    return RecordProcessors(
        teleop_action=teleop_action,
        robot_action=robot_action,
        robot_observation=robot_observation,
    )


def build_dataset_features(
    robot: Robot,
    processors: RecordProcessors,
    use_videos: bool,
    extra_action_names: Sequence[str] = (),
) -> dict[str, dict]:
    """Reproduce record()'s feature derivation exactly (lerobot_record.py:449-462).

    ``record_loop`` calls ``build_dataset_frame(dataset.features, ...)`` every
    iteration, so a feature dict derived any other way KeyErrors inside the
    control loop rather than at construction.

    ``extra_action_names`` IS THE WHOLE 17-D MECHANISM, and it is a declaration
    rather than a patch. ``make_policy`` overwrites ``cfg.output_features``
    from the recording dataset UNCONDITIONALLY (factory.py:470) -- the input
    side is guarded with ``if not cfg.input_features``, the output side is not
    -- so a `tph` checkpoint whose action head is 17 wide gets a 16-wide
    output_features, builds a 16-wide head, and dies in
    ``load_state_dict`` with a safetensors size mismatch several hundred MB into
    ``from_pretrained``. Declaring the 17th action feature HERE makes that
    overwrite land on 17 and the weights load.

    Three consequences, all of them wanted:

    * ``make_robot_action`` zips the 17 names against the tensor and produces a
      ``progress`` key (policies/utils.py:197-200). ``MobileAIRobot.send_action``
      filters arm keys by membership in ``arms.action_features``
      (mobileai.py:472-474) and reads the base with ``.get`` (:478-480), so the
      extra key reaches nothing and commands nothing.
    * ``build_dataset_frame`` writes it, so p is RECORDED per frame -- the
      post-hoc evidence for whether the progress output meant anything.
    * the completion monitor reads it off the same dict.

    ``extra_action_names=()`` reproduces the pre-chain behaviour byte for byte,
    which is the guarantee M1 (16-D) depends on.
    """
    features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=processors.teleop_action,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=use_videos,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=processors.robot_observation,
            initial_features=create_initial_features(
                observation=robot.observation_features
            ),
            use_videos=use_videos,
        ),
    )
    if not extra_action_names:
        return features
    action = features[ACTION]
    names = list(action["names"])
    collisions = sorted(set(names) & set(extra_action_names))
    if collisions:
        raise ValueError(
            f"extra action feature(s) {collisions} are already produced by the "
            f"robot ({robot.name}); adding them again would make "
            "make_robot_action write the same key twice and the dataset action "
            "width disagree with its names"
        )
    names.extend(extra_action_names)
    # shape, not just names: dataset_to_policy_features reads the SHAPE into the
    # PolicyFeature make_policy assigns to output_features, so a name list that
    # grew without the shape would leave the head at 16 and change nothing.
    action["names"] = names
    action["shape"] = (len(names),)
    logger.info(
        f"dataset action widened to {len(names)}-D by declaring "
        f"{list(extra_action_names)}; the robot ignores the extra key(s) and the "
        "frames record them"
    )
    return features


def create_dataset(
    config: StageRunnerConfig,
    robot: Robot,
    processors: RecordProcessors,
    extra_action_names: Sequence[str] = (),
) -> LeRobotDataset:
    """Create the one dataset this trial records into.

    ``create()`` mkdirs the metadata root with exist_ok=False, which is exactly
    why every trial needs a fresh ``dataset.repo_id`` in the YAML (a
    ``--dataset.repo_id=`` override parses too, but does not survive into
    config.source.yaml). We never call ``push_to_hub``: upstream's finally
    calls it unguarded, so a hub failure there replaces whatever exception
    actually ended the run.
    """
    # Robot does not declare `cameras` on the ABC; upstream guards it with
    # hasattr on the resume path and then assumes it on the create path. Our
    # mock robot has none, and 0 threads is the correct answer for it.
    camera_count = len(getattr(robot, "cameras", {}) or {})
    return LeRobotDataset.create(
        config.dataset.repo_id,
        config.dataset.fps,
        root=config.dataset.root,
        robot_type=robot.name,
        features=build_dataset_features(
            robot, processors, config.dataset.video, extra_action_names
        ),
        use_videos=config.dataset.video,
        image_writer_processes=config.dataset.num_image_writer_processes,
        image_writer_threads=(
            config.dataset.num_image_writer_threads_per_camera * camera_count
        ),
    )


def video_encoding_manager(dataset: LeRobotDataset) -> AbstractContextManager[None]:
    """Upstream's VideoEncodingManager, entered around the whole trial.

    Its ``__exit__`` flushes the encoders, calls ``dataset.finalize()`` and
    cleans an interrupted episode's image directories -- which is why cli never
    calls finalize itself (upstream record() calls it twice; we decline to copy
    that). The value it yields is deliberately unused.
    """
    return VideoEncodingManager(dataset)


def make_keyboard_events() -> tuple[Any | None, dict[str, bool]]:
    """Core's own listener and its three flags. We add no listener of our own.

    Right arrow -> exit_early, left arrow -> rerecord_episode, esc ->
    stop_recording, all already owned by core. In a headless environment the
    listener is None and the flags simply never fire, which is what
    preflight.check_manual_terminator_is_reachable exists to catch before a
    manual stage silently degrades into a ceiling-length timeout run.
    """
    listener, events = init_keyboard_listener()
    return listener, events


def plan_stage(stage: StageConfig, defaults: DefaultsConfig) -> StagePlan:
    """Turn a stage's terminator into the number record_loop actually compares against.

    A manual stage still needs a NUMBER: ``while timestamp < control_time_s``
    compares an int against it on the first iteration, so None is a guaranteed
    TypeError. ``defaults.manual_ceiling_s`` doubles as the runaway guard for a
    missed keypress.

    PLUG POINT, and the chain is the thing that plugged in. ``completion`` and
    ``reached`` are the two terminator types the chain adds, and both resolve to
    a number here the same way ``manual`` does. What fires them is NOT a watcher
    thread reading ``dataset.episode_buffer["observation.state"]`` -- the
    mechanism this docstring used to propose, which sees the measurement but not
    the command and races the buffer ``save_episode`` pops. It is
    ``completion.CompletionMonitorStep``, a ProcessorStep in the robot action
    pipeline, and ``reset_policy.ResetPolicy``, which both set
    ``events["exit_early"]`` from inside the tick that decided.

    ``completion``: ``timeout_s`` is the stage's p90 x ``timeout_factor``, set by
    ``config.expand_chain``. Reaching it is a chain failure.

    ``reached``: ``timeout_s`` is the WORST-CASE ceiling
    (``ResetConfig.worst_case_ceiling_s``), because the real one depends on the
    measured anchor and the arm has not been read when this runs. The reset
    executor computes the actual ceiling from its ``ResetPlan`` and uses THAT;
    this number is the bound the stage_start event can honestly state in
    advance, and the actual duration is in the stage_end ``reason_detail``
    (``reset_T``).
    """
    terminator = stage.terminator
    if terminator.type == TERMINATOR_MANUAL and terminator.timeout_s is None:
        control_time_s = defaults.manual_ceiling_s
    elif (
        terminator.type in (TERMINATOR_COMPLETION, TERMINATOR_REACHED)
        and terminator.timeout_s is None
    ):
        # expand_chain always fills it; a hand-written chain stage might not,
        # and record_loop's `while timestamp < control_time_s` raises TypeError
        # against None on the first iteration -- with the robot connected.
        control_time_s = defaults.manual_ceiling_s
    else:
        control_time_s = terminator.timeout_s
    return StagePlan(
        control_time_s=float(control_time_s), planned_terminator=terminator.type
    )


def manual_detection_margin_s(fps: int) -> float:
    """How far short of control_time_s a stage must stop to be read as a keypress.

    Two frames, floored at 0.25 s so a slow loop does not shrink the window to
    nothing. This is the accepted imprecision of inferring a manual break: a
    right arrow pressed inside this margin is recorded as a timeout.
    """
    return max(2.0 / fps, 0.25)


def call_record_loop(
    *,
    robot: Robot,
    events: dict[str, bool],
    fps: int,
    processors: RecordProcessors,
    dataset: LeRobotDataset,
    bundle: PolicyBundle,
    control_time_s: float,
    single_task: str,
    display_data: bool = False,
) -> None:
    """THE call site of lerobot's record_loop. Nothing else in this repo calls it.

    Every argument goes by keyword, and ``dataset=`` by keyword specifically:
    ``@safe_stop_image_writer`` reads ``kwargs.get("dataset")``
    (image_writer.py:26-38), so a positionally-passed dataset makes the
    decorator silently skip stopping the image writer on an exception and
    leaves writer threads hanging.

    It takes a whole ``PolicyBundle`` rather than three loose arguments because
    record_loop's policy branch requires all three to be non-None: with one
    missing it falls through to the no-action branch, which ``continue``s
    WITHOUT updating ``timestamp`` and therefore never reaches control_time_s.
    That is a silent unbounded busy loop, with no error and no frames.

    record_loop resets the policy and both processors on entry
    (lerobot_record.py:332-335), so the previous stage's ACT chunk queue is
    flushed for free -- re-entering per stage IS the transition semantics we
    want, not an accident.
    """
    # RESOLVED ON THE MODULE, AT CALL TIME. `from ... import record_loop` would
    # bind the function at IMPORT time, and three fork plugins REBIND
    # `lerobot_record.record_loop` from `register_plugins()`:
    # loop_rate_log (the phase tag every downstream consumer needs),
    # chunk_execution_patch and, transitively, anything they wrap. Those patches
    # install when `cli.main` calls `register_plugins()`, which is AFTER this
    # module was imported.
    #
    # loop_rate_log does sweep sys.modules and rebind the name in any module
    # holding the original, so an import-time binding happens to survive today
    # -- but only because this module is already imported by then, and only
    # while that sweep exists. If it ever did not, `_phase` would stay None for
    # every stage, which is not a visible failure: base_serial_rearm would
    # refuse with "no record_loop phase tag" (so the chain runs with the 20 ms
    # serial wait and the ("policy", "reset") allowance would never fire),
    # basevel.csv's `phase` column would be empty (so eval_base_stats' policy
    # filter and the report's per-stage integral would find nothing), and
    # pose_guide would print no POSE line. One attribute lookup per stage buys
    # all of that back unconditionally.
    lerobot_record.record_loop(
        robot=robot,
        events=events,
        fps=fps,
        teleop_action_processor=processors.teleop_action,
        robot_action_processor=processors.robot_action,
        robot_observation_processor=processors.robot_observation,
        dataset=dataset,
        policy=bundle.policy,
        preprocessor=bundle.preprocessor,
        postprocessor=bundle.postprocessor,
        control_time_s=control_time_s,
        single_task=single_task,
        display_data=display_data,
    )


def classify_termination(
    *, events: Mapping[str, bool], plan: StagePlan, elapsed_s: float, fps: int
) -> tuple[str, str]:
    """Why the stage ended, plus the comparison that decided it.

    record_loop self-clears ``events["exit_early"]`` on its way out
    (lerobot_record.py:343-344), so elapsed time is the only evidence a manual
    break leaves behind. That makes 'manual' a heuristic, not a signal -- which
    is why the returned reason always names the numbers, so a mislabel is
    visible in the JSONL instead of being an unexplained string.

    The other two flags are NOT self-cleared, so they are read directly here,
    and NOTHING IN THIS PACKAGE EVER CLEARS THEM -- this docstring used to claim
    "the caller clears the flag" and no caller ever did. We do not implement
    re-record: a left arrow aborts the trial. Whoever wraps a P2 trial loop
    around run_trial has to decide for itself what a press between two trials
    means (see runner.run_trial); a reader who believed the old sentence writes
    that loop without clearing, and trial 2 is then aborted before its first
    stage by a key pressed during trial 1.
    """
    margin_s = manual_detection_margin_s(fps)
    threshold_s = plan.control_time_s - margin_s
    measured = (
        f"elapsed_s={elapsed_s:.3f}, control_time_s={plan.control_time_s:.3f}, "
        f"margin_s={margin_s:.3f}, threshold_s={threshold_s:.3f}"
    )
    if events.get("stop_recording"):
        return (
            TERMINATED_BY_STOP_RECORDING,
            f"events['stop_recording'] was set; {measured}",
        )
    if events.get("rerecord_episode"):
        return (
            TERMINATED_BY_RERECORD_REQUESTED,
            f"events['rerecord_episode'] was set; {measured}",
        )
    if elapsed_s < threshold_s:
        return TERMINATED_BY_MANUAL, f"elapsed_s < threshold_s; {measured}"
    return TERMINATED_BY_TIMEOUT, f"elapsed_s >= threshold_s; {measured}"
