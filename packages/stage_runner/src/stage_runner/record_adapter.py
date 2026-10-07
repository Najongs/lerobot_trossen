"""Seam 1 of 2 to installed lerobot: the single call site of ``record_loop``, plus
the robot, processor, dataset and keyboard construction that feeds it.

If an upstream signature moves, this file is the only one that changes. Nothing
else in the package imports ``lerobot.scripts.lerobot_record``.
"""

import logging
import sys
from collections.abc import Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Any

import torch
from lerobot.processor.pipeline import ProcessorStep
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
from lerobot.utils import control_utils
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


# ---------------------------------------------------------------- reset fast path

# `predict_path` in a reset stage's reason_detail. MEASURED, never declared: the
# swap below counts its own calls and the executor writes whichever of these the
# count supports -- see `ResetPredictSwap.predict_path`.
PREDICT_PATH_STATE_ONLY: str = "state_only"
PREDICT_PATH_UPSTREAM: str = "upstream"
PREDICT_PATH_MIXED: str = "mixed"
# Installed, and then nothing was predicted at all: an Esc or a stale
# ``exit_early`` breaks ``record_loop`` at the TOP of its first iteration
# (lerobot_record.py:343-345), before ``predict_action``. Distinct from
# ``upstream`` on purpose -- "nothing ran" is not "the swap did not take", and
# labelling it ``upstream`` would make the report flag a reset that never moved.
PREDICT_PATH_NOT_CALLED: str = "not_called"


def is_image_observation_key(name: str) -> bool:
    """True for the observation keys whose conversion is the cost being skipped.

    THE SAME TEST UPSTREAM USES. ``prepare_observation_for_inference`` decides
    per key with ``if "image" in name`` (policies/utils.py:129) and that branch
    is the whole expense: uint8 -> float32, a divide by 255 and a full
    ``.permute(2,0,1).contiguous()`` copy, per camera, on the CPU. Matching the
    predicate rather than the ``observation.images.`` prefix means a dataset that
    names a single camera ``observation.image`` (OBS_IMAGE, constants.py:24) is
    covered too, and no key that upstream would convert is left behind.
    """
    return "image" in name


@dataclass
class ResetPredictSwap:
    """What the reset window actually did with ``predict_action``.

    Read back by the executor AFTER ``record_loop`` returns, and the reason the
    counters exist at all: the repo rule is 「줬다가 아니라 먹었다」. A
    ``predict_path`` written from the intent would read ``state_only`` even in
    the two cases where the lightweight function never ran -- an upstream that
    moved ``predict_action`` off ``lerobot_record`` (``installed`` False), or a
    plugin that replaced the policy object for this loop
    (``chunk_execution_patch`` does exactly that when ``REPLAY_DATASET`` is set,
    and the swap then delegates). The operator would see the old 12.5 Hz under a
    label claiming the fix was in effect.
    """

    installed: bool = False
    state_only_calls: int = 0
    upstream_calls: int = 0
    # Why the swap could not be installed, for the log. Empty when it was.
    refused: str = field(default="")

    @property
    def predict_path(self) -> str:
        # ``installed`` FIRST. Keyed on the counters alone, a reset that was
        # broken before its first tick (Esc at the top of record_loop) would
        # report ``upstream`` -- the one label that means "the swap did not
        # take" -- about a window where the swap was in place and simply never
        # asked. 「없음 ≠ 안 걸림」.
        if not self.installed:
            return PREDICT_PATH_UPSTREAM
        if self.state_only_calls and not self.upstream_calls:
            return PREDICT_PATH_STATE_ONLY
        if self.state_only_calls:
            return PREDICT_PATH_MIXED
        if self.upstream_calls:
            return PREDICT_PATH_UPSTREAM
        return PREDICT_PATH_NOT_CALLED


@contextmanager
def reset_predict_action(policy: Any) -> Iterator[ResetPredictSwap]:
    """Swap ``lerobot_record.predict_action`` for a state-only one, for a reset only.

    WHY. ``record_loop`` calls ``predict_action`` every tick
    (lerobot_record.py:358) and ``prepare_observation_for_inference`` inside it
    converts EVERY observation key, cameras included. The reset bundle runs on
    ``device="cpu"`` (reset_policy.ResetPolicyConfig) with identity processors,
    so the three 480x640x3 uint8 frames are converted on the CPU -- measured
    51-59 ms of the reset loop's per-frame ``other=`` term on the robot PC
    (docs/eval_najy_results_1007.md), i.e. 11.9-13.1 Hz against a target of 21.
    ``fast_obs_patch`` does not help here: it moves the conversion to ``device``,
    and ``device`` IS the CPU for a reset. ``ResetPolicy.select_action`` reads
    ``batch[OBS_STATE]`` and NOTHING else, so every one of those bytes is waste.

    WHY THE MODULE ATTRIBUTE ON ``lerobot_record`` AND NOT ``control_utils``.
    ``record_loop`` resolves ``predict_action`` as a global of the module it is
    defined in (``from ... import predict_action`` at lerobot_record.py:135-138,
    called at :358), so the binding that matters is
    ``lerobot_record.predict_action``. Rebinding ``control_utils.predict_action``
    would not be seen by the loop. The two fork plugins that rebind
    ``record_loop`` (``loop_rate_log``, ``chunk_execution_patch``) install
    ``functools.wraps`` WRAPPERS that call the original function object, whose
    ``__globals__`` is still ``lerobot_record.__dict__`` -- so the swap reaches
    the loop on the robot PC too, where those wrappers are in place. This is the
    same late-binding argument as ``call_record_loop``'s, read the other way
    round.

    SCOPE. Whatever was bound at entry is restored in ``finally`` -- on the
    normal exit, on an exception out of ``record_loop`` and on the Esc path, all
    of which leave through the same block. Policy stages and plain
    ``lerobot-record`` therefore never see this function: the window is one
    reset's ``call_record_loop`` and nothing else.

    GUARDED BY POLICY IDENTITY. The lightweight path runs only when the loop is
    driving ``policy`` -- the reset bundle's :class:`ResetPolicy`. Anything else
    (``chunk_execution_patch`` substitutes a replay policy into ``kwargs`` when
    ``REPLAY_DATASET`` is set, including for a reset) is delegated to the
    function that was bound at entry, unmodified, and counted separately.
    """
    swap = ResetPredictSwap()
    original = getattr(lerobot_record, "predict_action", None)
    if not callable(original):
        swap.refused = (
            "lerobot.scripts.lerobot_record.predict_action is missing or not "
            "callable; the reset runs through whatever record_loop resolves"
        )
        logger.warning(f"리셋 구간: 영상 변환 생략을 걸 수 없다 -- {swap.refused}")
        yield swap
        return

    expected_policy = policy

    def state_only_predict_action(
        observation: dict,
        policy: Any = None,
        device: Any = None,
        preprocessor: Any = None,
        postprocessor: Any = None,
        use_amp: bool = False,
        task: str | None = None,
        robot_type: str | None = None,
    ):
        # `policy` SHADOWS the enclosing name on purpose: record_loop passes it
        # by keyword (lerobot_record.py:360) and the signature has to match
        # upstream's. The reset bundle's policy is reached through the closure
        # cell `expected_policy` instead.
        if policy is not expected_policy:
            swap.upstream_calls += 1
            return original(
                observation=observation,
                policy=policy,
                device=device,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=use_amp,
                task=task,
                robot_type=robot_type,
            )
        swap.state_only_calls += 1
        # A NEW DICT, never the caller's. record_loop reuses `observation_frame`
        # for the dataset row it writes one block later
        # (`frame = {**observation_frame, **action_frame, ...}`,
        # lerobot_record.py:411-413), and `prepare_observation_for_inference`
        # REPLACES every value in the dict it is handed. Upstream protects the
        # caller with `copy(observation)` (control_utils.py:100); the filter here
        # allocates a new dict anyway, which is the same protection -- so the
        # recorded frame still carries the camera arrays, at full uint8 fidelity,
        # and `build_dataset_frame` is untouched.
        state_only = {
            name: value
            for name, value in observation.items()
            if not is_image_observation_key(name)
        }
        with (
            torch.inference_mode(),
            torch.autocast(device_type=device.type)
            if getattr(device, "type", None) == "cuda" and use_amp
            else nullcontext(),
        ):
            batch = control_utils.prepare_observation_for_inference(
                state_only, device, task, robot_type
            )
            batch = preprocessor(batch)
            action = policy.select_action(batch)
            # RETURNED UNSQUEEZED AND ON ITS DEVICE, exactly as upstream does it.
            # control_utils.predict_action's docstring claims step 5 removes the
            # batch dimension and moves to the CPU; the installed 0.4.4 code does
            # neither (control_utils.py:112-115 -- `return action` straight off
            # the postprocessor). `make_robot_action` is what squeezes and moves
            # (policies/utils.py:194-195), and it is the next call in record_loop
            # either way, so matching the CODE is what keeps the action identical.
            return postprocessor(action)

    # 대입부터 try 안에 둔다 -- 대입 직후 try 에 들어가기 전에 SystemExit(시그널 래치)·
    # KeyboardInterrupt 가 오면 finally 를 못 타고 스왑이 남는 창이 있었다 (codex 10/07).
    try:
        lerobot_record.predict_action = state_only_predict_action
        swap.installed = True
        logger.info(
            "리셋 구간: 영상 변환 생략 -- 리셋은 observation.state 만 읽으므로 카메라 "
            "프레임은 텐서로 바꾸지 않는다 (predict_path=state_only). 정책 구간은 "
            "그대로 전체 변환을 쓴다."
        )
        yield swap
    finally:
        lerobot_record.predict_action = original


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


def make_teleop(teleop_config: Any) -> Any:
    """lerobot's factory, imported late: the teleoperator plugin is registered by
    ``register_plugins`` and resolving the config type before that fails inside
    draccus, not with a message about a missing plugin."""
    from lerobot.teleoperators.utils import make_teleoperator_from_config

    # The hardware-free teleop is resolved here, not by lerobot's factory: that
    # factory finds the device class by NAMING CONVENTION (``MockTeleopConfig``
    # -> module ``stage_runner.mockteleop``), which this package does not follow.
    from stage_runner.mock_teleop import MockTeleop, MockTeleopConfig

    if isinstance(teleop_config, MockTeleopConfig):
        return MockTeleop(teleop_config)
    return make_teleoperator_from_config(teleop_config)


def call_teleop_loop(
    *,
    robot: Robot,
    teleop: Any,
    events: dict[str, bool],
    fps: int,
    processors: RecordProcessors,
    control_time_s: float,
    single_task: str,
    display_data: bool = False,
) -> None:
    """lerobot-record 의 「Reset the environment」 구간과 같은 호출.

    policy 없음·dataset 없음(녹화 안 함), 리더암(``teleop``)이 팔을 끈다. 끝나는 길은
    셋: ``→`` (exit_early), ``←`` (rerecord_episode + exit_early), ESC (stop_recording +
    exit_early), 또는 ``control_time_s`` 타임아웃. 플래그 해석은 호출자
    (``runner._run_teleop_phase``)가 한다 -- record_loop 는 exit_early 만 스스로 지운다.
    ``robot_action_processor`` 는 그대로 타므로 팔 한 틱 상한(max_relative_target)과
    NaN 게이트는 텔레옵 action 에도 걸린다; 완료 감시자는 begin_stage 전이라 꺼져 있다.
    """
    lerobot_record.record_loop(
        robot=robot,
        events=events,
        fps=fps,
        teleop_action_processor=processors.teleop_action,
        robot_action_processor=processors.robot_action,
        robot_observation_processor=processors.robot_observation,
        dataset=None,
        teleop=teleop,
        policy=None,
        preprocessor=None,
        postprocessor=None,
        control_time_s=control_time_s,
        single_task=single_task,
        display_data=display_data,
    )


from stage_runner.completion import BASE_VELOCITY_KEYS  # noqa: E402


class TeleopBaseZeroStep(ProcessorStep):
    """``teleop_action`` 파이프라인용: 리더 action 의 베이스 속도 두 키를 0 으로 덮는다.

    텔레옵 구간 동안만 설치된다(``runner._run_teleop_phase`` 가 넣고 빼므로 정책·리셋 단계와
    ``build_dataset_features`` 에는 보이지 않는다). 키를 더하거나 빼지 않으므로
    ``transform_features`` 는 항등이다.
    """

    def __call__(self, transition):
        from lerobot.processor.core import TransitionKey

        action = transition.get(TransitionKey.ACTION)
        if not isinstance(action, Mapping) or not any(k in action for k in BASE_VELOCITY_KEYS):
            return transition
        zeroed = dict(action)
        for key in BASE_VELOCITY_KEYS:
            if key in zeroed:
                zeroed[key] = 0.0
        patched = dict(transition)
        patched[TransitionKey.ACTION] = zeroed
        return patched

    def transform_features(self, features):
        return features

    def get_config(self) -> dict[str, Any]:
        return {"base_velocity_keys": sorted(BASE_VELOCITY_KEYS)}
