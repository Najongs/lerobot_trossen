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
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

import lerobot
from lerobot.configs.policies import PreTrainedConfig
from lerobot.robots import Robot
from lerobot.utils.constants import ACTION, HF_LEROBOT_HOME, OBS_STATE
from lerobot.utils.control_utils import is_headless, sanity_check_dataset_name

from stage_runner.config import (
    LATEST_CONFIG_VERSION,
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
        elif not stage.policy_path:
            # Without this the blank path reaches
            # PreTrainedConfig.from_pretrained(""), which raises
            # `FileNotFoundError: [Errno 2] No such file or directory: ''` --
            # outside the PreflightError branch in cli.main, so the operator gets
            # a bare traceback instead of the line naming the key to fix.
            problems.append(
                f"{where} ({stage.id!r}): `policy_path` is empty. A "
                f"`{stage.executor}` stage runs a checkpoint: give it a hub repo "
                "id, a local directory, or `mock://hold` for a dry run."
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

    if problems:
        raise PreflightError(
            "invalid stage definitions:\n  - " + "\n  - ".join(problems)
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

    # Currently unreachable, and kept: check_stage_definitions now rejects an
    # empty `policy_path` for every known executor, so "no stage declares a
    # policy_path" cannot get this far. It stays as the explicit no-policy branch
    # for the first executor that legitimately runs without a checkpoint (a
    # teleop reset stage, a start-state randomizer), which is the P2 case that
    # would otherwise re-invert this rule by accident.
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
    check_manual_terminator_is_reachable(config)
    check_dataset_name(config, policy_configs)
    check_dataset_root_is_free(config)
    logger.info(
        f"Preflight passed: {len(config.stages)} stage(s), robot {robot.name!r} "
        f"({robot_state_dimension(robot)}-dim state, "
        f"{robot_action_dimension(robot)}-dim action), dataset "
        f"{config.dataset.repo_id}."
    )
