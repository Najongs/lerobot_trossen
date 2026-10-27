"""The whole YAML/CLI surface, plus run-id, run-directory and config snapshots.

These dataclasses ARE the schema. draccus reads them for the YAML file (through
its own built-in ``--config_path``) and for dotted CLI overrides, which is why
this package contains no argparse and never touches ``RecordConfig``'s
``--policy.path`` / ``__get_path_fields__`` machinery.
"""

import logging
import shutil
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import draccus
import draccus.cfgparsing

from lerobot.robots.config import RobotConfig
from lerobot.teleoperators.config import TeleoperatorConfig

logger = logging.getLogger(__name__)

# Bumped when a change to these dataclasses makes an older YAML mean something
# different. preflight.check_config_version compares the file's `version:` key
# against this, so a run started from a stale YAML dies before the robot moves.
#
# NOT bumped for StageConfig.start_pose (2026-10-19), by this rule: the field is
# optional and defaults to None, which is exactly the old behaviour, so every
# version-1 YAML written before it still means the same thing. The other
# direction needs no bump either -- an older stage_runner handed a YAML that HAS
# start_pose dies in draccus on the unknown field at parse time, before preflight
# and long before the robot moves, which is the failure a bump would have bought.
# The same rule, and the same answer, for the `leader_reset` executor and the
# top-level `teleop:` block (2026-10-19): both are opt-in, and a YAML without
# them means what it meant.
# The same again for the `base_replay` and `base_align` executors and their
# StageConfig.replay / StageConfig.align blocks (2026-10-24): both blocks default
# to None and only those two executors read them, so a YAML without them is
# unchanged. The one visible difference is cosmetic -- resolved_config_text does
# not pop them, so a new snapshot shows `replay: null` and `align: null` on every
# stage (popping them would break the snapshot tests that compare against
# draccus.dump with only start_pose and teleop filtered, and nothing reads
# snapshots back).
LATEST_CONFIG_VERSION: int = 1

TERMINATOR_TIMEOUT: str = "timeout"
TERMINATOR_MANUAL: str = "manual"
# A policy stage that a camera classifier may end before its timer -- see
# completion.py. timeout_s stays required: it is the ceiling.
TERMINATOR_CLASSIFIER: str = "classifier"
# A policy stage that ends a fixed time after a gripper closes on something --
# see completion.GripTracker. For a stage whose length the grasp sets (stage 6:
# shake for a while once the beaker is held) while the cameras cannot tell its
# end. timeout_s stays required: it is the ceiling.
TERMINATOR_GRASP: str = "grasp"
TERMINATOR_TYPES: tuple[str, ...] = (
    TERMINATOR_TIMEOUT,
    TERMINATOR_MANUAL,
    TERMINATOR_CLASSIFIER,
    TERMINATOR_GRASP,
)
# Terminators run by a completion.CompletionWatcher.
WATCHED_TERMINATORS: tuple[str, ...] = (TERMINATOR_CLASSIFIER, TERMINATOR_GRASP)
COMPLETION_MODE_FIRE: str = "fire"
COMPLETION_MODE_SHADOW: str = "shadow"
COMPLETION_MODES: tuple[str, ...] = (COMPLETION_MODE_FIRE, COMPLETION_MODE_SHADOW)

EXECUTOR_LEROBOT_POLICY: str = "lerobot_policy"
# A stage that runs NO policy: the operator drives the arms with the leader arm
# (the `teleop:` block) to hand the robot an object and set its pose, the way
# lerobot-record's reset section does between episodes. Records no frames, and
# ends on the right arrow or its ceiling -- see executors.run_leader_reset_stage.
# The same string is pinned in aggregate.py, which keeps such stages off the
# success-metric axis.
EXECUTOR_LEADER_RESET: str = "leader_reset"
# Two stages that drive the BASE and nothing else, the 10/24 field loop
# (heading_align.py / stage1_align.py) as chain stages:
#
# * base_replay -- follows a demo's base path (x.pos / theta.pos targets read
#   from a local pose-labelled dataset) with odometry feedback, arms held;
# * base_align -- turns and drives the base until the cam_high view matches the
#   demo's arrival frame (heading by SIFT rotation, distance by landmark scale).
#
# Both run their own tick loop and RECORD every tick into the trial's episode,
# tagged with the stage name, so the replay, the measurement holds and the
# corrections are frames like any policy stage's. Neither runs a checkpoint, so
# neither needs a policy_path or a bundle. A chain may hold any number of them,
# anywhere, each with its own dataset / reference (StageConfig.replay / .align).
# The same strings are pinned in aggregate.py, which keeps such stages off the
# success-metric axis.
EXECUTOR_BASE_REPLAY: str = "base_replay"
EXECUTOR_BASE_ALIGN: str = "base_align"
BASE_EXECUTORS: frozenset[str] = frozenset({EXECUTOR_BASE_REPLAY, EXECUTOR_BASE_ALIGN})
# Executors that run a checkpoint, i.e. need a policy_path and a bundle. The base
# executors are deliberately not here: they run no policy.
POLICY_EXECUTORS: frozenset[str] = frozenset({EXECUTOR_LEROBOT_POLICY})

# draccus owns this flag; we only scan argv for it so the verbatim source copy
# in snapshot_configs knows which file to copy (draccus does not expose it).
CONFIG_PATH_FLAG: str = "--config_path"

SOURCE_CONFIG_FILENAME: str = "config.source.yaml"
RESOLVED_CONFIG_FILENAME: str = "config.resolved.yaml"

# Prepended to the resolved dump. It is a record of what ran, not a re-runnable
# input: draccus.dump writes camera enums as `!!python/object/apply:` tags that
# draccus's own loader refuses, so feeding this file back to --config_path ends
# in a ConstructorError. The warning lives in the file because that is where the
# operator is standing when they reach for it.
RESOLVED_CONFIG_HEADER: str = (
    "# RECORD ONLY -- this snapshot is NOT re-parseable (draccus.dump writes\n"
    "# python/object tags its own loader rejects). Re-run from\n"
    f"# {SOURCE_CONFIG_FILENAME} beside it, which is a verbatim copy of the\n"
    "# input, and repeat any CLI override you see below.\n"
)


@dataclass
class GripConfig:
    """What a classifier or grasp terminator reads from the grippers.

    Widths are observation.state values in metres. On the 10/25 demos
    (all11_merged, stages 2 and 6) a gripper reads about 0 closed on nothing,
    0.013-0.019 open and 0.0054-0.0082 around a tube; around the beaker it
    closes to about 0 as well, so in stage 6 only the closing can be seen.
    """

    # observation.state names of the gripper joints to read, e.g.
    # left_left_carriage_joint.pos. preflight checks the robot has them.
    state_names: list[str] = field(default_factory=list)
    # classifier: the classifier may end the stage only while EVERY named
    # gripper has stayed inside [hold_min_m, hold_max_m] for settle_s -- both
    # tubes held. Both required when a classifier terminator has `grip`.
    hold_min_m: float | None = None
    hold_max_m: float | None = None
    # grasp: a gripper that opened past open_above_m and then stayed below
    # close_below_m for settle_s has grasped, at the start of that stay.
    # Opening past open_above_m again cancels it, so a re-grasp restarts the wait.
    open_above_m: float = 0.010
    close_below_m: float = 0.002
    settle_s: float = 0.5


@dataclass
class TerminatorConfig:
    """How a stage ends. A MAPPING, never a scalar.

    P2's keyframe terminator and P3's classifier both need parameters
    (velocity threshold, hold frames), and adding them to a mapping keeps every
    already-written YAML parsing; turning a scalar into a mapping does not.
    """

    type: str = TERMINATOR_TIMEOUT
    # timeout: the required cutoff. manual: an optional ceiling; when it is None
    # plan_stage substitutes defaults.manual_ceiling_s, because record_loop's
    # `while timestamp < control_time_s` raises TypeError against None on the
    # first iteration. classifier, grasp: the required ceiling, as for timeout.
    timeout_s: float | None = None
    # classifier: a local checkpoint directory or a Hub repo id holding the
    # classifier and its completion.json. grasp: optional -- the classifier is
    # then scored for the record only and never ends the stage.
    model_path: str | None = None
    # classifier only. None -> the value in the checkpoint's completion.json,
    # which the offline evaluation chose. Score at or above `threshold` for
    # `hold_s` seconds, not before `min_s` into the stage, ends the stage.
    threshold: float | None = None
    hold_s: float | None = None
    min_s: float | None = None
    # classifier, grasp. "fire" ends the stage; "shadow" only logs when it
    # would have, and the timer ends the stage.
    mode: str = COMPLETION_MODE_FIRE
    # classifier: an optional gripper gate (hold_min_m / hold_max_m). grasp:
    # required (open_above_m / close_below_m).
    grip: GripConfig | None = None
    # grasp only. Seconds after the latest grasp at which the stage ends.
    wait_s: float | None = None


@dataclass
class StartPoseSourceConfig:
    """Where a start pose came from. Written by ``python -m stage_runner.start_pose``.

    Carried into the stage_reposition event verbatim, because a pose is a
    number with no meaning on its own: which dataset, which episode out of how
    many, chosen how. Without it a pose pasted months ago cannot be checked
    against the checkpoint now in ``policy_path``.
    """

    dataset_repo_id: str = ""
    # The revision asked for (the training config's, else lerobot's CODEBASE_VERSION
    # tag) and the commit it resolved to; null for a local --dataset-root.
    dataset_revision: str | None = None
    dataset_commit: str | None = None
    # The checkpoint whose train_config.json named the dataset. preflight warns
    # when it differs from the stage's policy_path (a swapped checkpoint may have
    # been trained on other demonstrations).
    policy_path: str = ""
    episode_index: int | None = None
    # Episodes whose first frame entered the pick (non-finite rows are dropped).
    n_episodes: int | None = None
    method: str = ""
    # Standardised L2 distance of the chosen episode to the per-joint median.
    distance: float | None = None


@dataclass
class StartPoseConfig:
    """Arm joint targets to move to before a stage starts. ARM JOINTS ONLY.

    Keys are the robot's ACTION feature names for the arm joints of both arms
    (``left_joint_0.pos`` ... ``right_joint_5.pos``), never a gripper and never
    the base -- preflight.check_stage_start_poses rejects either before connect().
    The gripper is not commanded by the move: the boundary hold before it keeps
    the previous stage's last gripper command, so a held object is not released
    (unverified on hardware); the base is not moved (see reposition.py).
    """

    positions: dict[str, float] = field(default_factory=dict)
    # Floor of the paced goal time. The actual goal time is stretched beyond it
    # whenever a joint would otherwise exceed velocity_max x
    # velocity_safety_factor -- the same pacing rule as WidowXAIFollower's
    # send_action. 3.0 s rather than connect()'s fixed 2.0 s because the arm may
    # be carrying an object.
    min_goal_time_s: float = 3.0
    # Max |measured - target| per arm joint, in rad, after the move. A larger
    # error aborts the trial instead of starting the stage from a wrong pose.
    # UNMEASURED default: no settled tracking error of this arm has been recorded.
    position_tolerance_rad: float = 0.05
    source: StartPoseSourceConfig = field(default_factory=StartPoseSourceConfig)


@dataclass
class BaseReplayConfig:
    """What a ``base_replay`` stage drives the base along. Only that executor reads it.

    The targets are a demo episode's base path as POSES -- cumulative forward
    distance ``x.pos`` (m) and unwrapped heading ``theta.pos`` (rad) per row --
    read before connect from a dataset on local disk (base_assets). Never from
    the velocity dataset the demo was recorded as: its ``fps: 30`` timestamps
    are not the ticks the base actually ran at, and integrating them gave 59.5
    deg of an 85.8 deg turn on 10/24. ``derive_pose.py`` (vault
    .claude/scripts/heading-align) writes the pose copy from the acquisition's
    measured tick times.
    """

    # Repo id of a POSE-labelled LeRobot dataset (action names include x.pos and
    # theta.pos, as derive_pose.py writes). Read from local disk only: the robot
    # never downloads at run time.
    dataset: str = ""
    # Local dataset directory. None -> HF_LEROBOT_HOME / dataset, where
    # derive_pose.py wrote test_replay_timing_1024_basepose on the robot PC.
    root: str | None = None
    episode: int = 0
    # Inclusive row range of that episode. until_row 297 -> 298 targets, as
    # stage1_align ran stage 1 of episode 0. -1 (the default) is refused by
    # preflight on purpose: the last row is where the demo's base STOPPED, which
    # only the operator's segmentation knows. from_row > 0 replays a later piece
    # of the episode; its targets are then taken relative to row from_row - 1,
    # the pose the base is at when the piece starts.
    from_row: int = 0
    until_row: int = -1
    # Wall time per target row. 1/21 is the field's replay rate (stage1_align
    # REPLAY_HZ) -- the rate the demo's base commands actually ran at, so the
    # replay takes as long as the demo did.
    row_period_s: float = 1.0 / 21.0


@dataclass
class BaseAlignConfig:
    """How a ``base_align`` stage matches the live cam_high view to a demo frame.

    The reference is EITHER a frame of a local LeRobot dataset (reference_dataset,
    episode, frame) OR an image file captured at the demo spot (reference_image);
    preflight refuses both or neither. Loaded before connect (base_assets), so a
    missing file stops the run before the robot is energised, and no dataset is
    opened between stages.
    """

    reference_dataset: str = ""
    # None -> HF_LEROBOT_HOME / reference_dataset. Local disk only.
    reference_root: str | None = None
    episode: int = 0
    # The demo's ARRIVAL frame (heading_align ARRIVAL_TICK: the demo's base
    # stopped plus 5 ticks, e.g. 302 for episode 0). -1 is refused by preflight.
    frame: int = -1
    # ... or an image file (PNG/JPG, read as RGB). Exactly one of the two.
    reference_image: str | None = None
    # Observation key of the camera that sees the landmark. The field used
    # cam_high, the head camera that turns with the base.
    camera: str = "cam_high"
    # [x, y, w, h] of the landmark (the stage-2 rack) in the REFERENCE image, the
    # region the distance estimate scales. The field recomputed it with
    # dark_blob_box(reference) every run; the stage logs that automatic box next
    # to this one and warns when they differ.
    landmark_roi: list[int] = field(default_factory=list)
    # Distance from the camera to the landmark at the demo arrival (m). Required:
    # it turns the landmark's apparent scale into a forward move. 0.60 for the
    # 10/24 stage-2 rack (two samples, about +-10 %).
    landmark_distance_m: float = 0.0
    # Tolerances that end the loop as "aligned": the field's --yaw-tol and
    # --dist-tol-cm defaults.
    yaw_tol_deg: float = 2.0
    dist_tol_cm: float = 1.0
    # Turns and drives together, as the field's --max-moves; at most
    # max_moves + 1 measurements. Coarse turns count toward it too.
    max_moves: int = 6
    # Robot vertical axis in camera coordinates (10/24 estimate from in-place
    # turns). Yaw is measured about this axis, so a tilted head camera does not
    # read pitch as heading.
    camera_up: list[float] = field(
        default_factory=lambda: [-0.06288349, -0.84574326, -0.52987169]
    )
    # Override for the intrinsics; None (default) = read them from the RealSense
    # stream profile at stage start. Give both or neither; the mock robot, which
    # has no RealSense, needs them.
    camera_matrix: list[list[float]] | None = None
    dist_coeffs: list[float] | None = None
    # P gain of the heading tracker during an align turn (the field's 2.0).
    turn_kp: float = 2.0
    # Small-rotation shortfall option, OFF by default (0 = the 10/24 behaviour).
    # Turns below 3 deg fell 50-70 % short on 10/24 (cause not measured); > 0
    # floors the commanded turn rate at this many rad/s while the turn is still
    # short of its target. Its effect is unmeasured, hence off.
    min_turn_rate_rad_s: float = 0.0
    # Field fallback for distance when SIFT scale has < 8 inliers: live rack-box
    # width / ROI width. OFF by default; on 10/24 it was live (no ROI was passed)
    # and it fired once (run f2).
    distance_width_fallback: bool = False
    # Field fallback for HEADING, ON by default as on 10/24 (heading_for_step):
    # when the fine SIFT estimate is unusable (< 15 inliers or > 20 deg) the
    # horizontal position of the dark rack blob gives a coarse heading; the stage
    # turns on it and measures again, and a coarse estimate never counts as
    # aligned. Without it, two of the ten 10/24 arrivals (c1 +20.9 deg / 8
    # inliers, f1 +15.3 deg / 7 inliers) would have ended the stage. false = abort
    # whenever the fine estimate is unusable.
    coarse_heading: bool = True


@dataclass
class StageConfig:
    # `id` is the join key to the hand-filled labels file and to every event
    # line, so it has to stay stable across trials even when `name` changes.
    id: str = ""
    name: str = ""
    executor: str = EXECUTOR_LEROBOT_POLICY
    # Hub repo id, local directory, or mock://<name> for the hardware-free path.
    # Empty for a `leader_reset` stage, which runs no checkpoint.
    policy_path: str = ""
    # Becomes frame["task"] for every frame this stage records. ACT ignores it;
    # it is recorded so a VLA executor and the per-stage segmentation of one
    # episode both have it.
    instruction: str = ""
    terminator: TerminatorConfig = field(default_factory=TerminatorConfig)
    # Parsed and snapshotted, unused in P1. It exists now because once trials
    # start running, this YAML is tied to experiment logs and a schema change
    # costs more than a code change.
    precondition: str | None = None
    # Optional. None (the default, and every YAML written before 2026-10-19) =
    # the stage starts wherever the previous stage left the arms, exactly as
    # before. Set = the arms are moved there first; generate the block with
    # `python -m stage_runner.start_pose` rather than by hand.
    start_pose: StartPoseConfig | None = None
    # The block of a `base_replay` stage (what path to drive) and of a
    # `base_align` stage (what view to match). None on every other stage, and
    # preflight refuses either block on an executor that would ignore it.
    replay: BaseReplayConfig | None = None
    align: BaseAlignConfig | None = None
    # A `lerobot_policy` stage only. True = base torque is switched off for the
    # stage's loop and back on after it, however it ended, so the policy's base
    # command moves nothing (stage 6: pi0 trained on "shake, then drive to the
    # fridge" starts driving before the stage ends). The run-wide
    # `robot.enable_base_motor_torque: false` does the same for a chain with no
    # base stage; this is for a chain whose base stages need the torque.
    release_base_torque: bool = False


@dataclass
class DatasetConfig:
    # The one field that changes every trial: LeRobotDataset.create mkdirs with
    # exist_ok=False, so a reused repo_id kills the run. Set it HERE, in the
    # YAML. draccus takes `--dataset.repo_id=` as a dotted override like it does
    # for any other field, and that still works, but an overridden value is
    # absent from the verbatim config.source.yaml snapshot and present only in
    # the non-reparseable config.resolved.yaml -- so the run can no longer be
    # re-created from what the run directory holds.
    repo_id: str = ""
    fps: int = 30
    root: str | None = None
    video: bool = True
    # false: the chain's episode is never saved -- no video encode at the end of
    # the trial, and the dataset stays at 0 episodes. Frames are still buffered
    # during the run, because a base stage holds the grippers at the last
    # RECORDED goal (_recorded_gripper_goals) and falls back to the measured
    # width without one, which loosens a grip on a held beaker. For intervention
    # runs, whose intervention dataset is saved per take regardless.
    save: bool = True
    num_image_writer_processes: int = 0
    num_image_writer_threads_per_camera: int = 4


@dataclass
class DefaultsConfig:
    executor: str = EXECUTOR_LEROBOT_POLICY
    # The numeric control_time_s a `manual` stage still needs, and the runaway
    # guard for a missed right-arrow press.
    manual_ceiling_s: float = 300.0


@dataclass
class OutputConfig:
    root: str = "outputs/stage_runner"
    # None -> resolve_run_id derives one from the dataset name and local time.
    run_id: str | None = None


@dataclass
class StageRunnerConfig:
    # RobotConfig is a draccus ChoiceRegistry (lerobot/robots/config.py:23), so
    # the nested `robot:` block is resolved by its `type:` key against the
    # registry that record_adapter.register_plugins() populates. Registering our
    # own subclass here instead would fork the robot schema.
    robot: RobotConfig
    # The leader arm a `leader_reset` stage drives the robot with -- the same
    # block lerobot-record takes as `--teleop.*`, resolved by its `type:` key
    # against TeleoperatorConfig's registry (register_plugins() populates it).
    # None (the default, and every YAML written before 2026-10-19) = no leader
    # arm is constructed or connected; preflight refuses a leader_reset stage
    # without one.
    teleop: TeleoperatorConfig | None = None
    version: int = LATEST_CONFIG_VERSION
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    defaults: DefaultsConfig = field(default_factory=DefaultsConfig)
    # Order IS execution order.
    stages: list[StageConfig] = field(default_factory=list)
    output: OutputConfig = field(default_factory=OutputConfig)
    display_data: bool = False
    # Same name and default as lerobot-record's --play_sounds: each stage start
    # is spoken ("Stage s4, pour liquid ...") through spd-say on Linux.
    play_sounds: bool = True


def parse_config(argv: Sequence[str] | None = None) -> StageRunnerConfig:
    """Parse YAML + CLI overrides into StageRunnerConfig.

    Must run AFTER record_adapter.register_plugins(): an unregistered
    `type: mobileai_robot` fails inside draccus type resolution, not with a
    message about a missing plugin.
    """
    # Our own device registration -- the counterpart to register_plugins() for
    # the one robot that ships inside this package. draccus resolves
    # `type: stage_runner_mock_robot` HERE, at parse time, out of RobotConfig's
    # choice registry, and that entry exists only once mock_robot.py has run its
    # @RobotConfig.register_subclass. It used to ride on the package __init__
    # importing mock_robot eagerly; __init__ is now lazy so that
    # stage_runner.aggregate stays importable without lerobot or torch, which
    # moved the import here. Function scope, not module scope: nothing else in
    # this module needs it, and the smoke path must not depend on some other
    # module having been imported first. mock_teleop is the same for the leader
    # arm: `type: stage_runner_mock_teleop` resolves here, out of
    # TeleoperatorConfig's registry.
    from stage_runner import mock_robot, mock_teleop  # noqa: F401

    return draccus.parse(config_class=StageRunnerConfig, args=argv)


def source_config_path(argv: Sequence[str] | None = None) -> Path | None:
    """Return the ``--config_path`` value from argv, or None.

    draccus consumes the flag without exposing the path it read, and the
    verbatim snapshot needs the file itself -- the comments in it are what carry
    the rad-unit assumption and the p95 timeout provenance forward.
    """
    # argv is None means draccus read sys.argv[1:], so scan the same tokens.
    tokens = list(sys.argv[1:] if argv is None else argv)
    for index, token in enumerate(tokens):
        if token == CONFIG_PATH_FLAG:
            if index + 1 < len(tokens):
                return Path(tokens[index + 1])
            return None
        if token.startswith(CONFIG_PATH_FLAG + "="):
            return Path(token.split("=", 1)[1])
    return None


def resolve_run_id(config: StageRunnerConfig, now: datetime | None = None) -> str:
    """Explicit output.run_id, else <dataset name>_<local timestamp>.

    LOCAL time, not UTC: the run id and every wall_clock_iso in the event log are
    the join keys to the video capture times and to the operator's handwritten
    sheet, both of which are in local time.
    """
    if config.output.run_id:
        return config.output.run_id
    stamp = (now or datetime.now()).strftime("%Y%m%dT%H%M%S")
    dataset_name = config.dataset.repo_id.rsplit("/", 1)[-1]
    return f"{dataset_name}_{stamp}"


def resolve_run_directory(config: StageRunnerConfig, run_id: str) -> Path:
    """Return output.root / run_id, refusing a directory that already exists.

    The refusal is the point: events.jsonl and the two config snapshots are the
    only record a trial leaves besides the dataset, and a reused run_id would
    append events from two trials into one unsplittable stream. Creation itself
    is the caller's (call sequence step 9, mkdir with exist_ok=False); this
    function does not create, so calling it does not commit to a run.
    """
    run_directory = Path(config.output.root).expanduser() / run_id
    if run_directory.exists():
        raise FileExistsError(
            f"run directory already exists: {run_directory} -- "
            "pick a new dataset.repo_id or set output.run_id in the YAML"
        )
    return run_directory


def snapshot_configs(
    config: StageRunnerConfig, run_directory: Path, source_path: Path | None
) -> None:
    """Write config.source.yaml (verbatim) and config.resolved.yaml (parsed).

    Neither file alone answers both questions. The verbatim copy keeps the
    comments -- the rad-unit assumption and the p95 provenance live only there --
    and the resolved dump keeps any CLI override and every default
    that was never written down.

    The resolved dump is a RECORD, not an input. draccus.dump serialises a
    camera enum as a ``!!python/object/apply:`` tag that draccus's own loader
    then refuses (measured against configs/chain_task45.yaml: ConstructorError on
    cam_high.color_mode), so the obvious move -- re-running a past trial with
    ``--config_path <run>/config.resolved.yaml`` -- ends in a YAML constructor
    traceback. The header written below says so inside the file, which is where
    the operator is when they try it.
    """
    # exist_ok=True, not False: step 9's own mkdir(exist_ok=False) is what
    # guarantees a fresh directory (with resolve_run_directory's existence check
    # in front of it), so repeating exist_ok=False here would turn the normal
    # path into a FileExistsError.
    run_directory.mkdir(parents=True, exist_ok=True)

    if source_path is None:
        logger.warning(
            f"no {CONFIG_PATH_FLAG} in argv: {SOURCE_CONFIG_FILENAME} not "
            "written, only the resolved snapshot"
        )
    else:
        shutil.copyfile(source_path, run_directory / SOURCE_CONFIG_FILENAME)

    (run_directory / RESOLVED_CONFIG_FILENAME).write_text(
        RESOLVED_CONFIG_HEADER + resolved_config_text(config), encoding="utf-8"
    )


def resolved_config_text(config: StageRunnerConfig) -> str:
    """``draccus.dump(config)`` minus every ``start_pose: null`` and ``teleop: null``.

    draccus.dump is ``save_config(encode(config))`` (draccus/cfgparsing.py), so
    dropping the key from the encoded stage dicts before saving yields exactly
    the dump a config without the field produced. A config with no start pose
    and no leader arm therefore snapshots byte-for-byte as it did before
    StageConfig.start_pose and StageRunnerConfig.teleop existed, and an absent
    key still reads as its default (None).
    """
    encoded = draccus.encode(config)
    if encoded.get("teleop") is None:
        encoded.pop("teleop", None)
    for stage in encoded.get("stages") or []:
        if isinstance(stage, dict) and stage.get("start_pose") is None:
            stage.pop("start_pose", None)
    return draccus.cfgparsing.save_config(encoded)
