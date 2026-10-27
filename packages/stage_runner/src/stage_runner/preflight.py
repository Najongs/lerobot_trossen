"""Startup gates: everything that must kill the process before the robot moves.

Every check here runs before ``robot.connect()`` and before a single checkpoint
weight is downloaded, so a misconfigured run costs seconds instead of a powered
arm driving to a wrong pose or a several-hundred-MB download that ends in a
safetensors size-mismatch deep inside ``from_pretrained``.

Every failure raises ``PreflightError``, which ``cli.main`` maps to exit code 2
and prints as one line without a traceback: these are operator errors in the
YAML, not bugs in the package. Each message names the key to change.
"""

import logging
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

import lerobot
from lerobot.cameras.configs import ColorMode, Cv2Rotation
from lerobot.configs.policies import PreTrainedConfig
from lerobot.robots import Robot
from lerobot.utils.constants import ACTION, HF_LEROBOT_HOME, OBS_STATE
from lerobot.utils.control_utils import is_headless, sanity_check_dataset_name

# Module scope, unlike the executors import in check_stage_definitions: reposition
# reaches only transitions -> lerobot.robots, which this module imports already,
# so it adds neither torch nor a cycle to the gate.
from stage_runner import reposition, transitions
from stage_runner.config import (
    BASE_EXECUTORS,
    EXECUTOR_BASE_ALIGN,
    EXECUTOR_BASE_REPLAY,
    EXECUTOR_LEADER_RESET,
    LATEST_CONFIG_VERSION,
    POLICY_EXECUTORS,
    TERMINATOR_MANUAL,
    TERMINATOR_TIMEOUT,
    TERMINATOR_TYPES,
    StageConfig,
    StageRunnerConfig,
)

logger = logging.getLogger(__name__)

EVAL_DATASET_PREFIX: str = "eval_"


class PreflightError(RuntimeError):
    """A startup check refused the run. cli.main turns this into exit code 2."""


# The package pins ``lerobot[intelrealsense]>=0.4.4,<0.5``, but an editable
# checkout, a stale venv or a `uv sync` against a moved lock can all leave a
# different version importable. 0.6.0 reorganised the seam this package attaches
# to (``--policy.path`` gone from lerobot-record, the rollout loop split into
# lerobot-rollout), and the failure would surface as an AttributeError inside
# record_adapter with the robot already energised.
#
# THE OTHER HALF OF THIS PIN IS packages/stage_runner/pyproject.toml's
# `dependencies` line, and the two must move together: bump the dependency
# without these tuples and the first run after `uv sync` dies here quoting a pin
# that no longer exists anywhere in the repo. (The pointer in the other
# direction is already in that file's comment.)
SUPPORTED_LEROBOT_MINIMUM: tuple[int, int, int] = (0, 4, 4)
SUPPORTED_LEROBOT_BELOW: tuple[int, int, int] = (0, 5, 0)


def _format_version(version: tuple[int, int, int]) -> str:
    return ".".join(str(part) for part in version)


# Derived, never hand-written: a display string that is a third copy of the pin
# drifts from the values actually compared, and the operator then reads a range
# the code is not enforcing.
_LEROBOT_PIN: str = (
    f">={_format_version(SUPPORTED_LEROBOT_MINIMUM)},"
    f"<{_format_version(SUPPORTED_LEROBOT_BELOW)}"
)


def _parse_version(text: str) -> tuple[int, int, int] | None:
    """Parse a leading ``major.minor.patch`` out of a version string.

    Hand-rolled rather than pulled from ``packaging`` because that would be a
    declared dependency existing only to compare three integers. Trailing
    non-numeric suffixes are dropped ("0.5.0rc1" -> (0, 5, 0)), which is the
    conservative direction: a release candidate for an unsupported release is
    itself unsupported. Returns None when the string carries no numbers at all,
    e.g. a source checkout reporting "unknown".
    """
    parts: list[int] = []
    for chunk in text.split(".")[:3]:
        digits = ""
        for char in chunk:
            if not char.isdigit():
                break
            digits += char
        if not digits:
            break
        parts.append(int(digits))
    if not parts:
        return None
    while len(parts) < 3:
        parts.append(0)
    return (parts[0], parts[1], parts[2])


def check_lerobot_version() -> str:
    """Return the installed lerobot version string, or raise if it is off the pin.

    Returned rather than only checked because the trial_start event records it:
    a run's JSONL has to say which lerobot produced it, since the aggregator is
    re-run over old runs long after the venv has moved on.
    """
    version = str(lerobot.__version__)
    parsed = _parse_version(version)
    if parsed is None:
        raise PreflightError(
            f"cannot read the installed lerobot version ({version!r}); "
            f"stage_runner requires lerobot {_LEROBOT_PIN}. "
            "Reinstall with `uv sync` from the repo root."
        )
    if not SUPPORTED_LEROBOT_MINIMUM <= parsed < SUPPORTED_LEROBOT_BELOW:
        raise PreflightError(
            f"lerobot {version} is installed, but stage_runner requires "
            f"{_LEROBOT_PIN} (record_loop's signature is reorganised outside that "
            f"range). Reinstall with `uv sync` from the repo root, or pin "
            f"lerobot to {_format_version(SUPPORTED_LEROBOT_MINIMUM)}."
        )
    return version


def check_config_version(config: StageRunnerConfig) -> None:
    """Refuse a YAML written against a different schema generation.

    The YAML outlives the code: once trials are running it is tied to experiment
    logs and repro scripts, so an unrecognised generation must stop the run
    rather than be read with this generation's field meanings.
    """
    if config.version != LATEST_CONFIG_VERSION:
        raise PreflightError(
            f"config `version: {config.version}` is not supported; this "
            f"stage_runner reads version {LATEST_CONFIG_VERSION}. Set "
            f"`version: {LATEST_CONFIG_VERSION}` as the first key of the YAML, or "
            "run the stage_runner that matches the config."
        )


def check_stage_definitions(config: StageRunnerConfig) -> None:
    """Validate the stage list itself: ids, executors, policy paths, terminators.

    Also ``dataset.fps``, which belongs to the same class -- a config value that
    makes the control loop unrunnable -- and is checked in the one gate that runs
    before anything at all is constructed.

    Collects every problem before raising so one pass over the message fixes the
    whole YAML, instead of one failed run per typo.
    """
    # Imported inside the function so preflight stays importable without pulling
    # in record_adapter -> lerobot's record loop -> torch through the executors
    # module, and so a future executors module that wants a preflight helper
    # cannot create an import cycle.
    from stage_runner.executors import EXECUTORS

    if not config.stages:
        raise PreflightError(
            "`stages:` is empty -- there is nothing to run. Add at least one "
            "stage with an `id`, a `policy_path` and a `terminator`."
        )

    problems: list[str] = []

    if config.dataset.fps <= 0:
        # record_loop computes `sleep_time_s = 1 / fps` on its first iteration,
        # so a zero or negative fps is a ZeroDivisionError inside the control
        # loop WITH THE ROBOT ENERGISED -- the one place an operator typo must
        # never be allowed to land. manual_detection_margin_s(0) divides by the
        # same zero.
        problems.append(
            f"`dataset.fps: {config.dataset.fps}` must be positive -- record_loop "
            "computes 1 / fps inside the control loop, with the robot connected."
        )

    seen: set[str] = set()
    for index, stage in enumerate(config.stages):
        where = f"stages[{index}]"
        if not stage.id:
            problems.append(
                f"{where}: `id` is empty -- it is the join key between "
                "events.jsonl and the hand-filled labels file, so it cannot be "
                "blank."
            )
        elif stage.id in seen:
            problems.append(
                f"{where}: duplicate `id: {stage.id}` -- bundles and policy "
                "configs are keyed by stage id, so the second stage would run "
                "the first one's policy."
            )
        seen.add(stage.id)

        if stage.executor not in EXECUTORS:
            known = ", ".join(sorted(EXECUTORS))
            problems.append(
                f"{where} ({stage.id!r}): unknown `executor: {stage.executor}`. "
                f"Known executors: {known}."
            )
        elif stage.executor in BASE_EXECUTORS:
            problems.extend(_base_stage_problems(where, stage))
        elif stage.executor == EXECUTOR_LEADER_RESET:
            # A reset runs no checkpoint and moves the arms by hand, so both of
            # these would be silently ignored -- refused instead, with the fix.
            if stage.policy_path:
                problems.append(
                    f"{where} ({stage.id!r}): a `{EXECUTOR_LEADER_RESET}` stage "
                    f"runs no policy, but `policy_path: {stage.policy_path}` is "
                    "set. Delete it, or make this a `lerobot_policy` stage."
                )
            if stage.start_pose is not None:
                problems.append(
                    f"{where} ({stage.id!r}): a `{EXECUTOR_LEADER_RESET}` stage "
                    "is posed by hand, so `start_pose` would be overwritten the "
                    "moment the leader takes over. Put the `start_pose` on the "
                    "policy stage that follows the reset."
                )
            if config.teleop is None:
                problems.append(
                    f"{where} ({stage.id!r}): a `{EXECUTOR_LEADER_RESET}` stage "
                    "needs the leader arm: add a top-level `teleop:` block (the "
                    "values lerobot-record takes as --teleop.*)."
                )
        elif not stage.policy_path:
            # (A base stage never gets here: it has its own branch above, and
            # this fall-through would refuse it for the policy_path it must not
            # have.) Without this the blank path reaches
            # PreTrainedConfig.from_pretrained(""), which raises
            # `FileNotFoundError: [Errno 2] No such file or directory: ''` --
            # outside the PreflightError branch in cli.main, so the operator gets
            # a bare traceback instead of the line naming the key to fix.
            problems.append(
                f"{where} ({stage.id!r}): `policy_path` is empty. A "
                f"`{stage.executor}` stage runs a checkpoint: give it a hub repo "
                "id, a local directory, or `mock://hold` for a dry run."
            )

        if stage.executor in EXECUTORS and stage.executor not in BASE_EXECUTORS:
            # The blocks only the base executors read. Anywhere else they would
            # be parsed and silently ignored -- most likely a base stage whose
            # `executor:` line was forgotten, which would then run as a policy.
            for block in ("replay", "align"):
                if getattr(stage, block) is not None:
                    problems.append(
                        f"{where} ({stage.id!r}): a `{stage.executor}` stage ignores "
                        f"`{block}:`, which only a `{EXECUTOR_BASE_REPLAY}` / "
                        f"`{EXECUTOR_BASE_ALIGN}` stage reads. Set the stage's "
                        f"`executor:`, or delete the `{block}:` block."
                    )

        terminator = stage.terminator
        if terminator.type not in TERMINATOR_TYPES:
            known = ", ".join(TERMINATOR_TYPES)
            problems.append(
                f"{where} ({stage.id!r}): unknown `terminator.type: "
                f"{terminator.type}`. Known types: {known}."
            )
        elif terminator.type == TERMINATOR_TIMEOUT and terminator.timeout_s is None:
            problems.append(
                f"{where} ({stage.id!r}): `terminator.type: timeout` needs "
                "`timeout_s`. record_loop compares `timestamp < control_time_s`, "
                "and None raises TypeError on the first iteration."
            )

        if terminator.timeout_s is not None and terminator.timeout_s <= 0:
            # Not a crash: record_loop's `while timestamp < control_time_s` is
            # simply false on entry, so the stage records zero frames and the
            # trial looks like it ran.
            problems.append(
                f"{where} ({stage.id!r}): `terminator.timeout_s: "
                f"{terminator.timeout_s}` must be positive -- a non-positive "
                "value records zero frames without raising."
            )

        # The robot-independent half of the start pose gate; which KEYS the pose
        # may carry needs the constructed robot and is check_stage_start_poses.
        start_pose = stage.start_pose
        if start_pose is not None:
            if not start_pose.positions:
                problems.append(
                    f"{where} ({stage.id!r}): `start_pose.positions` is empty. "
                    "Generate the block with `python -m stage_runner.start_pose`, "
                    "or delete `start_pose:` to start the stage where the previous "
                    "one left the arms."
                )
            if not (
                math.isfinite(start_pose.min_goal_time_s)
                and start_pose.min_goal_time_s > 0
            ):
                # 0 would hand the driver a zero-length trajectory: the pacing
                # stretches it only for joints that actually move, so a pose a
                # few mrad away would be a step command.
                problems.append(
                    f"{where} ({stage.id!r}): `start_pose.min_goal_time_s: "
                    f"{start_pose.min_goal_time_s}` must be a positive number of "
                    "seconds."
                )
            if not (
                math.isfinite(start_pose.position_tolerance_rad)
                and start_pose.position_tolerance_rad > 0
            ):
                problems.append(
                    f"{where} ({stage.id!r}): `start_pose.position_tolerance_rad: "
                    f"{start_pose.position_tolerance_rad}` must be a positive "
                    "number of radians -- 0 can never be met, so every trial "
                    "would abort at this stage."
                )

    problems.extend(_task_string_collisions(config.stages))

    if config.stages and not any(
        stage.executor in POLICY_EXECUTORS for stage in config.stages
    ):
        # A chain of resets and base stages only would record no policy output
        # and trip the dataset-name rule the wrong way round (check_dataset_name's
        # no-policy branch); it is not an eval at all.
        executors_here = ", ".join(
            sorted({f"`{stage.executor}`" for stage in config.stages})
        )
        problems.append(
            f"no stage runs a policy (executors here: {executors_here}): add at "
            "least one `lerobot_policy` stage -- reset and base stages alone are "
            "not an eval."
        )

    if problems:
        raise PreflightError(
            "invalid stage definitions:\n  - " + "\n  - ".join(problems)
        )


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_positive(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def _base_stage_problems(where: str, stage: StageConfig) -> list[str]:
    """The robot-independent checks of one base_replay / base_align stage.

    Each refusal names the key to change. The ceiling floors come from
    base_stages, the module whose loops they bound, so the gate and the stage
    cannot disagree about how long a stage may need.
    """
    # Lazy for the same reason as the executors import in check_stage_definitions.
    from stage_runner import base_stages

    here = f"{where} ({stage.id!r})"
    executor = stage.executor
    problems: list[str] = []
    if stage.policy_path:
        problems.append(
            f"{here}: a `{executor}` stage drives the base and runs no policy, but "
            f"`policy_path: {stage.policy_path}` is set. Delete it."
        )
    # A base_replay stage may carry the NEXT policy stage's start_pose, so the
    # arms are posed before the base moves (user request 2026-10-25). runner._reposition_before_stage runs for any stage
    # with a start_pose, and the base loop then holds the arms where they are.
    if stage.start_pose is not None and executor != "base_replay":
        problems.append(
            f"{here}: a `{executor}` stage holds the arms where they are, so "
            "`start_pose` would not be applied. Put it on the policy stage that "
            "follows."
        )
    if stage.instruction:
        problems.append(
            f"{here}: a `{executor}` stage records its frames under its `name`; "
            "`instruction` would be ignored. Delete it and set `name`."
        )
    if not stage.name:
        problems.append(
            f"{here}: `name` is empty. A `{executor}` stage records every frame "
            "with its name as the task string, which is how the episode is split "
            "into stages afterwards."
        )
    timeout_s = stage.terminator.timeout_s
    if stage.terminator.type != TERMINATOR_TIMEOUT:
        problems.append(
            f"{here}: a `{executor}` stage needs `terminator.type: timeout` -- the "
            "timeout is the stage's runaway ceiling; it ends on its own (or on the "
            "right arrow) well before it."
        )

    if executor == EXECUTOR_BASE_REPLAY:
        replay = stage.replay
        if replay is None:
            problems.append(
                f"{here}: a `{executor}` stage needs a `replay:` block (dataset, "
                "episode, until_row)."
            )
        else:
            if not replay.dataset:
                problems.append(
                    f"{here}: `replay.dataset` is empty: give the repo id of the "
                    "pose-labelled dataset (derive_pose.py's copy)."
                )
            if not (_is_int(replay.episode) and replay.episode >= 0):
                problems.append(
                    f"{here}: `replay.episode: {replay.episode}` must be >= 0."
                )
            if not (
                _is_int(replay.from_row)
                and _is_int(replay.until_row)
                and 0 <= replay.from_row < replay.until_row
            ):
                problems.append(
                    f"{here}: `replay.from_row: {replay.from_row}` / `until_row: "
                    f"{replay.until_row}` must satisfy 0 <= from_row < until_row; "
                    "until_row is the last row to replay (where the demo's base "
                    "stopped), and has no default."
                )
            if not _is_positive(replay.row_period_s):
                problems.append(
                    f"{here}: `replay.row_period_s: {replay.row_period_s}` must be a "
                    "positive number of seconds (1/21 is the field's rate)."
                )
        if stage.align is not None:
            problems.append(
                f"{here}: a `{executor}` stage ignores `align:`; put it on a "
                f"`{EXECUTOR_BASE_ALIGN}` stage after this one."
            )
        return problems

    align = stage.align
    if stage.replay is not None:
        problems.append(
            f"{here}: a `{executor}` stage ignores `replay:`; put it on a "
            f"`{EXECUTOR_BASE_REPLAY}` stage before this one."
        )
    if align is None:
        problems.append(
            f"{here}: a `{executor}` stage needs an `align:` block (reference, "
            "landmark_roi, landmark_distance_m)."
        )
        return problems
    if bool(align.reference_dataset) == bool(align.reference_image):
        problems.append(
            f"{here}: give exactly one of `align.reference_dataset` (with episode "
            "and frame) and `align.reference_image`."
        )
    if align.reference_dataset and not (
        _is_int(align.episode)
        and _is_int(align.frame)
        and align.episode >= 0
        and align.frame >= 0
    ):
        problems.append(
            f"{here}: `align.episode: {align.episode}` / `align.frame: "
            f"{align.frame}` must both be >= 0; frame is the demo's arrival frame "
            "and has no default."
        )
    roi = align.landmark_roi
    if not (
        len(roi) == 4
        and all(_is_int(value) for value in roi)
        and roi[0] >= 0
        and roi[1] >= 0
        and roi[2] > 0
        and roi[3] > 0
    ):
        problems.append(
            f"{here}: `align.landmark_roi: {list(roi)}` must be [x, y, w, h] "
            "integers with x, y >= 0 and w, h > 0 -- the landmark's box in the "
            "reference (dark_blob_box(reference) is what the field used)."
        )
    if not _is_positive(align.landmark_distance_m):
        problems.append(
            f"{here}: `align.landmark_distance_m: {align.landmark_distance_m}` must "
            "be the positive camera-to-landmark distance at the demo arrival, in m "
            "(0.60 for the 10/24 rack)."
        )
    for name in ("yaw_tol_deg", "dist_tol_cm"):
        value = getattr(align, name)
        if not _is_positive(value):
            problems.append(f"{here}: `align.{name}: {value}` must be positive.")
    if not (_is_int(align.max_moves) and align.max_moves >= 1):
        problems.append(f"{here}: `align.max_moves: {align.max_moves}` must be >= 1.")
    up = align.camera_up
    if not (len(up) == 3 and all(math.isfinite(value) for value in up)) or not (
        0.9 <= math.hypot(*up) <= 1.1
    ):
        problems.append(
            f"{here}: `align.camera_up: {list(up)}` must be a 3-vector of norm "
            "about 1 (the robot's vertical axis in camera coordinates)."
        )
    matrix, dist = align.camera_matrix, align.dist_coeffs
    if (matrix is None) != (dist is None):
        problems.append(
            f"{here}: give both `align.camera_matrix` and `align.dist_coeffs`, or "
            "neither (then both are read from the RealSense)."
        )
    elif matrix is not None and dist is not None:
        if not (
            len(matrix) == 3
            and all(len(row) == 3 for row in matrix)
            and all(math.isfinite(value) for row in matrix for value in row)
        ):
            problems.append(
                f"{here}: `align.camera_matrix` must be a finite 3x3 matrix."
            )
        if len(dist) not in (4, 5, 8, 12, 14) or not all(
            math.isfinite(value) for value in dist
        ):
            problems.append(
                f"{here}: `align.dist_coeffs` must be 4, 5, 8, 12 or 14 finite "
                f"values (OpenCV's distortion models), got {len(dist)}."
            )
    if not _is_positive(align.turn_kp):
        problems.append(f"{here}: `align.turn_kp: {align.turn_kp}` must be positive.")
    if not (
        isinstance(align.min_turn_rate_rad_s, (int, float))
        and math.isfinite(align.min_turn_rate_rad_s)
        and align.min_turn_rate_rad_s >= 0
    ):
        problems.append(
            f"{here}: `align.min_turn_rate_rad_s: {align.min_turn_rate_rad_s}` must "
            "be >= 0 (0 = off, the 10/24 behaviour)."
        )
    if (
        stage.terminator.type == TERMINATOR_TIMEOUT
        and timeout_s is not None
        and _is_int(align.max_moves)
        and align.max_moves >= 1
    ):
        needed = base_stages.align_ceiling_floor_s(align.max_moves)
        if timeout_s < needed:
            problems.append(
                f"{here}: `terminator.timeout_s: {timeout_s}` is shorter than an "
                f"align of up to {align.max_moves} moves may need: each of the "
                f"{align.max_moves + 1} measurements may wait "
                f"{base_stages.CAPTURE_STILL_MAX_S:g} s for a still base and take "
                f"{base_stages.MEASURE_BUDGET_S:g} s. Raise it to at least "
                f"{math.ceil(needed)} s, or lower `align.max_moves`."
            )
    return problems


def _task_string(stage: StageConfig) -> str:
    """The string a stage's frames are recorded under (frame["task"])."""
    if stage.executor in POLICY_EXECUTORS:
        return stage.instruction or stage.name
    return stage.name


def _task_string_collisions(stages: Sequence[StageConfig]) -> list[str]:
    """A base stage whose name is another stage's task string.

    A base stage's frames carry its name as the task string, and that string is
    how one episode is split back into stages (tasks.parquet). Sharing it with
    another stage merges the two in every per-stage reading of the dataset.
    Scoped to base stages on purpose: two policy stages that share an
    instruction were legal before base stages existed and stay legal -- a VLA
    chain may well run the same instruction twice.
    """
    problems: list[str] = []
    for index, stage in enumerate(stages):
        if stage.executor not in BASE_EXECUTORS or not stage.name:
            continue
        for other_index, other in enumerate(stages):
            if other_index == index or _task_string(other) != stage.name:
                continue
            problems.append(
                f"stages[{index}] ({stage.id!r}): `name: {stage.name}` is also the "
                f"task string of stages[{other_index}] ({other.id!r}), so their "
                "frames could not be told apart in the dataset. Give the base "
                "stage a name of its own."
            )
            break
    return problems


def check_leader_reset_stages(
    config: StageRunnerConfig, robot: Robot, teleop: object | None
) -> None:
    """The robot-dependent half of the leader_reset gate, before anything connects.

    The leader's action goes to the follower UNCHANGED (record_adapter.call_reset_loop
    uses identity pipelines), so every leader key must be a follower action key
    and every follower arm key must come from the leader -- a missing one would
    hold that joint at a stale goal for the whole reset, an extra one raises
    inside send_action with the arms energised. And the leader must be able to
    align itself to the follower first (leader_reset.align_leader_to_follower).
    """
    if not any(stage.executor == EXECUTOR_LEADER_RESET for stage in config.stages):
        return
    if teleop is None:
        raise PreflightError(
            "a `leader_reset` stage needs the leader arm, and none was constructed "
            "-- add a top-level `teleop:` block."
        )
    problems: list[str] = []
    base_keys = set(transitions.BASE_VELOCITY_KEYS)
    leader_keys = set(getattr(teleop, "action_features", {}) or {}) - base_keys
    follower_keys = set(robot.action_features) - base_keys
    if leader_keys - follower_keys:
        problems.append(
            f"leader key(s) {sorted(leader_keys - follower_keys)} are not action "
            f"features of {robot.name}"
        )
    if follower_keys - leader_keys:
        problems.append(
            f"{robot.name} action key(s) {sorted(follower_keys - leader_keys)} "
            "get no command from the leader"
        )
    if not callable(getattr(teleop, "align_to", None)):
        problems.append(
            f"{type(teleop).__name__} has no align_to(), so the leader cannot be "
            "moved to the follower before the hand-over (the follower's first "
            "copied command would be a jump)"
        )
    if problems:
        raise PreflightError(
            f"the `teleop:` leader arm ({type(teleop).__name__}) cannot drive this "
            "robot in a leader_reset stage:\n  - " + "\n  - ".join(problems)
        )


# Methods the base stages call on robot.base. On MobileAIRobot it is the
# TrossenSlate built in __init__ (mobileai.py), so it exists before connect.
_BASE_METHODS: tuple[str, ...] = ("get_pose", "get_vel", "update_state", "set_cmd_vel")


def check_base_stages(config: StageRunnerConfig, robot: Robot) -> None:
    """The robot-dependent half of the base stage gate, before anything connects.

    The base must take velocity commands and report odometry, and must be
    powered; an align stage's camera must exist, must have an intrinsics source,
    and must stream the unrotated RGB image the estimator assumes. The reference
    image against the camera's size is base_assets' check, which has the image.
    """
    staged = [stage for stage in config.stages if stage.executor in BASE_EXECUTORS]
    if not staged:
        return
    problems: list[str] = []
    missing_keys = sorted(
        set(transitions.BASE_VELOCITY_KEYS) - set(robot.action_features)
    )
    if missing_keys:
        problems.append(
            f"robot {robot.name!r} takes no base velocity command {missing_keys}"
        )
    base = getattr(robot, "base", None)
    missing_methods = [
        name for name in _BASE_METHODS if not callable(getattr(base, name, None))
    ]
    if missing_methods:
        problems.append(
            f"robot {robot.name!r} has no base with {missing_methods} (robot.base is "
            f"{type(base).__name__}), so the stage can neither drive the base nor "
            "read its odometry"
        )
    if getattr(robot.config, "enable_base_motor_torque", True) is False:
        problems.append(
            "`robot.enable_base_motor_torque` is false: the base would ignore every "
            "command silently; set robot.enable_base_motor_torque: true."
        )
    # Both executors drive the base with PR 69's pose tracker, which lives in
    # lerobot_robot_trossen -- a package stage_runner does not declare. Found
    # missing here rather than at the first base stage, mid-chain.
    from stage_runner import base_stages

    try:
        base_stages.import_tracker()
    except ImportError as error:
        problems.append(str(error))

    cameras = getattr(robot, "cameras", None) or {}
    camera_configs = getattr(robot.config, "cameras", None) or {}
    for stage in staged:
        align = stage.align
        if stage.executor != EXECUTOR_BASE_ALIGN or align is None:
            continue
        here = f"stage {stage.id!r}"
        if align.camera not in cameras:
            problems.append(
                f"{here}: `align.camera: {align.camera}` is not a camera of robot "
                f"{robot.name!r} (cameras: {sorted(cameras)})"
            )
        entry = camera_configs.get(align.camera)
        if (
            align.camera_matrix is None
            and getattr(entry, "type", None) != "intelrealsense"
        ):
            problems.append(
                f"{here}: no intrinsics source for `{align.camera}` (it is not a "
                "RealSense): give align.camera_matrix and dist_coeffs."
            )
        if entry is None:
            continue
        # Device intrinsics hold for the unrotated stream only
        # (camera_realsense.py), and the estimator's gray thresholds are tuned
        # on RGB; either setting would skew every estimate without an error.
        rotation = getattr(entry, "rotation", Cv2Rotation.NO_ROTATION)
        if rotation != Cv2Rotation.NO_ROTATION:
            problems.append(
                f"{here}: camera `{align.camera}` is configured with rotation "
                f"{rotation!r}; the intrinsics and the reference are for the "
                "unrotated stream. Set its rotation to 0."
            )
        color_mode = getattr(entry, "color_mode", ColorMode.RGB)
        if color_mode != ColorMode.RGB:
            problems.append(
                f"{here}: camera `{align.camera}` streams {color_mode!r}; the "
                "estimator's thresholds are tuned on RGB. Set its color_mode to rgb."
            )
    if problems:
        raise PreflightError(
            f"robot {robot.name!r} cannot run the base stages:\n  - "
            + "\n  - ".join(problems)
        )


def robot_state_dimension(robot: Robot) -> int:
    """Count the scalar entries of ``robot.observation_features``.

    Camera entries are tuple-typed ``(height, width, channels)`` and everything
    else is a scalar ``float``/``int`` type, so excluding tuples leaves exactly
    the vector that becomes ``observation.state``.
    """
    return sum(
        1
        for value in robot.observation_features.values()
        if not isinstance(value, tuple)
    )


def robot_action_dimension(robot: Robot) -> int:
    """Count ``robot.action_features``; no camera entries exist on that side."""
    return len(robot.action_features)


# A near-twin of policies._feature_dimension, kept separate ON PURPOSE: importing
# policies here would pull lerobot.policies.factory -- and torch -- into the gate
# that has to run before anything heavy loads. If the shape-reading rule ever
# changes (a checkpoint declaring `shape = (1, 16)`, say), BOTH copies move
# together, or trial_start.policies[].state_dim starts reporting a different
# number than the gate that let the run start.
def _feature_dimension(features: Mapping[str, object] | None, key: str) -> int | None:
    feature = (features or {}).get(key)
    shape = getattr(feature, "shape", None)
    if not shape:
        return None
    return int(shape[0])


def checkpoint_state_dimension(policy_config: PreTrainedConfig) -> int | None:
    """Read ``input_features['observation.state']`` out of the checkpoint config.

    MUST be called before ``make_policy``: it fills ``input_features`` from the
    recording dataset whenever the field is empty, after which the checkpoint's
    own dimension is no longer recoverable from the object. None means the
    checkpoint's config.json does not declare a state feature, and there is
    nothing to compare.
    """
    return _feature_dimension(policy_config.input_features, OBS_STATE)


def checkpoint_action_dimension(policy_config: PreTrainedConfig) -> int | None:
    """Read ``output_features['action']`` out of the checkpoint config.

    Also strictly before ``make_policy``, which overwrites ``output_features``
    from ``ds_meta`` unconditionally (factory.py:470) -- so after that call the
    checkpoint's own action dimension is gone and the mismatch only reappears as
    a safetensors size error inside ``from_pretrained``, hundreds of MB later.
    """
    return _feature_dimension(policy_config.output_features, ACTION)


# ACCEPTED RISK -- joint units are never asserted, only dimensions.
# meta/info.json carries no unit annotation, so there is nothing to compare
# against: our own P0 could only write "rad assumed". That is the dataset
# format not recording units rather than a hole in this check, and the best
# available action is leaving a record that we assumed rad -- which
# config.source.yaml does. Re-examine when an asset that DOES carry a unit
# annotation (a different robot, a different acquisition) gets mixed in.
def check_stage_dimensions(
    robot: Robot,
    stages: Sequence[StageConfig],
    policy_configs: Mapping[str, PreTrainedConfig],
) -> None:
    """Compare every checkpoint's state and action width against this robot.

    Both directions matter and they fail differently. A state mismatch is the
    ``include_base_in_state`` question (16-dim assets vs 14-dim); an action
    mismatch means the checkpoint was trained on a different robot entirely,
    because ``action_features`` always carries the base keys regardless of that
    flag.
    """
    state_dim = robot_state_dimension(robot)
    action_dim = robot_action_dimension(robot)
    problems: list[str] = []

    for stage in stages:
        policy_config = policy_configs.get(stage.id)
        if policy_config is None:
            # A mock:// stage: policies.load_policy_configs skips it, so there
            # is no checkpoint config.json to compare against.
            continue

        checkpoint_state = checkpoint_state_dimension(policy_config)
        if checkpoint_state is not None and checkpoint_state != state_dim:
            problems.append(
                f"stage {stage.id!r} ({stage.policy_path}): checkpoint expects a "
                f"{checkpoint_state}-dim observation.state, robot "
                f"{robot.name!r} produces {state_dim}. Flip "
                "`include_base_in_state` in the robot block (it adds x.vel and "
                "theta.vel, i.e. 2 dims) or use the checkpoint trained on this "
                "state layout."
            )

        checkpoint_action = checkpoint_action_dimension(policy_config)
        if checkpoint_action is not None and checkpoint_action != action_dim:
            problems.append(
                f"stage {stage.id!r} ({stage.policy_path}): checkpoint outputs a "
                f"{checkpoint_action}-dim action, robot {robot.name!r} takes "
                f"{action_dim}. `include_base_in_state` does not change this -- "
                "action_features always carries the base keys -- so the "
                "checkpoint was trained on a different robot."
            )

    if problems:
        raise PreflightError(
            "policy/robot dimension mismatch:\n  - " + "\n  - ".join(problems)
        )


def check_stage_start_poses(config: StageRunnerConfig, robot: Robot) -> None:
    """Refuse a start pose this robot cannot be moved to, before it is energised.

    Rejects a pose whose keys are not exactly this robot's arm joints, one that
    names a gripper or a base key, and one with a non-finite value. The keys are
    matched against what reposition.arm_handles reads off the CONSTRUCTED robot
    (``robot.arms.<side>.config.joint_names``), and the same
    reposition.start_pose_problems runs again right before the move, so this gate
    and the motion cannot disagree about what a valid pose is.

    A pose whose recorded provenance names a different checkpoint than the stage
    runs only WARNS: a newer checkpoint of the same stage is usually trained on
    the same demonstrations, and the operator is the one who knows.
    """
    posed = [stage for stage in config.stages if stage.start_pose is not None]
    if not posed:
        return
    try:
        handles = reposition.arm_handles(robot)
    except reposition.StageRepositionError as error:
        named = ", ".join(repr(stage.id) for stage in posed)
        raise PreflightError(
            f"stage(s) {named} set `start_pose`, but robot {robot.name!r} cannot be "
            f"moved to one: {error}. Delete `start_pose:` from those stages."
        ) from error

    action_keys = set(robot.action_features)
    absent = [
        key for key in reposition.expected_arm_keys(handles) if key not in action_keys
    ]
    if absent:
        # Would mean the per-arm drivers and the action features disagree about
        # the arm joint names, which no shipped robot does; refused rather than
        # assumed, because the pose is keyed by action feature names.
        raise PreflightError(
            f"robot {robot.name!r}: arm joint keys {absent} derived from the arm "
            "drivers are not action features, so a start pose cannot be keyed by "
            "them."
        )

    problems: list[str] = []
    for stage in posed:
        for problem in reposition.start_pose_problems(
            stage.start_pose.positions, handles
        ):
            problems.append(f"stage {stage.id!r} `start_pose.positions`: {problem}")
        source_policy = stage.start_pose.source.policy_path
        if source_policy and source_policy != stage.policy_path:
            logger.warning(
                f"stage {stage.id!r}: start_pose was computed from the training "
                f"dataset of {source_policy!r}, but the stage runs "
                f"{stage.policy_path!r}. Regenerate it if that checkpoint was "
                "trained on other demonstrations."
            )
        if not stage.start_pose.source.dataset_repo_id:
            logger.warning(
                f"stage {stage.id!r}: start_pose carries no `source.dataset_repo_id`, "
                "so the stage_reposition events cannot say which demonstration "
                "the pose came from."
            )
    if problems:
        raise PreflightError("invalid start pose:\n  - " + "\n  - ".join(problems))


def check_manual_terminator_is_reachable(config: StageRunnerConfig) -> None:
    """Refuse a ``manual`` stage in a session where the right arrow cannot fire.

    Two different ways it cannot fire, and only the first is one lerobot can see:

    1. ``is_headless()`` is True when pynput will not import. In that case
       ``init_keyboard_listener`` still returns a usable events dict with
       ``listener=None``, so nothing ever sets ``exit_early``.
    2. The session is Wayland. ``is_headless()`` (control_utils.py) does nothing
       but ``import pynput`` and reports nothing about whether key events can be
       DELIVERED: under GNOME on Wayland, XWayland sets DISPLAY, pynput imports
       fine, ``is_headless()`` returns False -- and the listener then receives
       nothing, because Wayland does not hand global key events to it. This is
       the failure the repo README documents as live on these machines
       (`## eval 함정`: →·←·ESC 가 무반응인데 에러도 없으면 세션이 Wayland다).

    Either way the outcome is the same and it is worse than a crash: the stage
    runs to ``defaults.manual_ceiling_s`` -- five unattended minutes of policy
    rollout on a real robot -- and is then logged as a clean ``timeout``, a
    silently corrupted measurement. Hence a hard gate, not a warning. Esc does
    not rescue it either: it is the same listener.
    """
    manual_ids = [
        stage.id
        for stage in config.stages
        if stage.terminator.type == TERMINATOR_MANUAL
    ]
    if not manual_ids:
        return

    named = ", ".join(repr(stage_id) for stage_id in manual_ids)
    remedy = (
        "Run from a session where the arrow keys reach the listener, or give the "
        "stage `terminator: {type: timeout, timeout_s: <seconds>}`."
    )
    if is_headless():
        raise PreflightError(
            f"stage(s) {named} use `terminator.type: manual`, but this session is "
            "headless (pynput did not import), so the right arrow never fires and "
            "the stage would run to defaults.manual_ceiling_s and be logged as a "
            f"timeout. {remedy}"
        )
    session_type = os.environ.get("XDG_SESSION_TYPE", "")
    if session_type.lower() == "wayland":
        raise PreflightError(
            f"stage(s) {named} use `terminator.type: manual`, but XDG_SESSION_TYPE "
            "is 'wayland': pynput imports and the listener starts, yet no key "
            "event is ever delivered to it, so →, ← and ESC all do nothing while "
            "the stage runs to defaults.manual_ceiling_s and is logged as a clean "
            "timeout. Log out and back in choosing 'GNOME on Xorg' (see the "
            f"README's `## eval 함정`). {remedy}"
        )


def check_dataset_name(
    config: StageRunnerConfig, policy_configs: Mapping[str, PreTrainedConfig]
) -> None:
    """Enforce lerobot's ``eval_`` naming rule before the dataset is created.

    ``sanity_check_dataset_name`` (control_utils.py:186-198) enforces both
    directions: with a policy the repo_id's name part MUST start with ``eval_``,
    without one it must NOT. It also does ``_, dataset_name = repo_id.split("/")``,
    so a repo_id with any other number of slashes raises an unpacking error there
    rather than a readable one.
    """
    repo_id = config.dataset.repo_id
    if not repo_id:
        raise PreflightError(
            "`dataset.repo_id` is empty. Set it in the YAML to "
            "`<owner>/eval_<name>` -- a fresh name every trial, because "
            "LeRobotDataset.create mkdirs with exist_ok=False. The equivalent "
            "CLI override `--dataset.repo_id=` also works, but it is not "
            "captured in config.source.yaml, so the run stops being "
            "reproducible from its own snapshot."
        )
    if repo_id.count("/") != 1:
        raise PreflightError(
            f"`dataset.repo_id: {repo_id}` must be exactly `<owner>/<name>`: "
            "lerobot's sanity_check_dataset_name unpacks `repo_id.split('/')` "
            "into two parts and anything else raises there."
        )

    policy_config = next(iter(policy_configs.values()), None)
    if policy_config is not None:
        try:
            sanity_check_dataset_name(repo_id, policy_config)
        except ValueError as error:
            raise PreflightError(
                f"{error} Rename to `<owner>/{EVAL_DATASET_PREFIX}<name>`; the "
                "prefix is what marks the dataset as policy output rather than a "
                "teleop recording."
            ) from error
        return

    # Every stage is a mock:// stage, so there is no PreTrainedConfig to hand
    # over -- and sanity_check_dataset_name keys its branch off `policy_cfg is
    # None`, which would invert the rule and demand a name WITHOUT the prefix.
    # A mock policy is still a policy, so the eval_ requirement is applied here
    # directly. This is the branch the hardware-free smoke config takes.
    dataset_name = repo_id.split("/")[1]
    if any(stage.policy_path for stage in config.stages):
        if not dataset_name.startswith(EVAL_DATASET_PREFIX):
            raise PreflightError(
                f"`dataset.repo_id: {repo_id}` runs policies, so its name part "
                f"must start with `{EVAL_DATASET_PREFIX}` "
                f"(lerobot's convention for policy output). Rename to "
                f"`<owner>/{EVAL_DATASET_PREFIX}{dataset_name}`."
            )
        return

    # Currently unreachable, and kept: check_stage_definitions rejects an empty
    # `policy_path` for every policy executor and a chain with no policy stage at
    # all, so "no stage declares a policy_path" cannot get this far -- the
    # leader_reset and base executors, which run without a checkpoint, only ever
    # sit beside a policy stage. It stays as the explicit no-policy branch for an
    # executor that may one day legitimately make up a whole run (a start-state
    # randomizer), which would otherwise re-invert this rule by accident.
    try:
        sanity_check_dataset_name(repo_id, None)
    except ValueError as error:
        raise PreflightError(
            f"{error} No stage declares a `policy_path`, so this run records no "
            f"policy output and the name must NOT start with "
            f"`{EVAL_DATASET_PREFIX}`."
        ) from error


def dataset_root_path(config: StageRunnerConfig) -> Path:
    """The directory ``LeRobotDataset.create`` will mkdir for this run.

    Reproduces LeRobotDatasetMetadata.create's own line -- ``Path(root) if root
    is not None else HF_LEROBOT_HOME / repo_id`` -- because there is no API that
    answers "where would this land" without creating it.
    """
    if config.dataset.root is not None:
        return Path(config.dataset.root)
    return HF_LEROBOT_HOME / config.dataset.repo_id


def check_dataset_root_is_free(config: StageRunnerConfig) -> None:
    """Refuse a repo_id whose directory already exists.

    Forgetting to bump ``dataset.repo_id`` is the most likely operator error of
    the whole run -- the robot PC's cache already holds dozens of ``eval_*`` and
    an aborted start leaves an empty one behind (README `## eval 함정`) -- and
    without this gate it surfaces as a raw ``FileExistsError`` traceback out of
    ``LeRobotDataset.create``, AFTER the run directory and both config snapshots
    have been written. Checking here turns it into the same one-line exit 2 as
    every other config mistake, and leaves no orphan run directory behind.
    """
    root = dataset_root_path(config)
    if root.exists():
        raise PreflightError(
            f"dataset directory {root} already exists; LeRobotDataset.create "
            "mkdirs it with exist_ok=False and would die there with a "
            "FileExistsError. Every trial needs a fresh name: set a new "
            "`dataset.repo_id: <owner>/eval_<name>` in the YAML (or delete that "
            "directory if it is an aborted run)."
        )


def run_preflight(
    config: StageRunnerConfig,
    robot: Robot,
    policy_configs: Mapping[str, PreTrainedConfig],
    teleop: object | None = None,
) -> None:
    """Run every gate that needs the constructed robot and the checkpoint configs.

    Called after ``make_robot`` and ``load_policy_configs`` but before
    ``robot.connect()`` and before ``load_bundles`` downloads any weights.

    ``check_config_version`` and ``check_stage_definitions`` also run earlier in
    ``cli.main`` (they need neither the robot nor the network, so they gate the
    run before anything is constructed). Repeating them here costs microseconds
    and makes this function a complete gate on its own, so a caller that only
    knows about ``run_preflight`` cannot skip them. ``check_lerobot_version`` is
    not repeated: it returns the version string cli.main needs for the
    trial_start event, so it has its own call site.
    """
    check_config_version(config)
    check_stage_definitions(config)
    check_stage_dimensions(robot, config.stages, policy_configs)
    check_stage_start_poses(config, robot)
    check_leader_reset_stages(config, robot, teleop)
    check_base_stages(config, robot)
    check_manual_terminator_is_reachable(config)
    check_dataset_name(config, policy_configs)
    check_dataset_root_is_free(config)
    logger.info(
        f"Preflight passed: {len(config.stages)} stage(s), robot {robot.name!r} "
        f"({robot_state_dimension(robot)}-dim state, "
        f"{robot_action_dimension(robot)}-dim action), dataset "
        f"{config.dataset.repo_id}."
    )
