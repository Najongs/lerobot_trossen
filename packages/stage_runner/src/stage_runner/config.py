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

from lerobot.robots.config import RobotConfig

logger = logging.getLogger(__name__)

# Bumped when a change to these dataclasses makes an older YAML mean something
# different. preflight.check_config_version compares the file's `version:` key
# against this, so a run started from a stale YAML dies before the robot moves.
LATEST_CONFIG_VERSION: int = 1

TERMINATOR_TIMEOUT: str = "timeout"
TERMINATOR_MANUAL: str = "manual"
TERMINATOR_TYPES: tuple[str, ...] = (TERMINATOR_TIMEOUT, TERMINATOR_MANUAL)

EXECUTOR_LEROBOT_POLICY: str = "lerobot_policy"

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
    # first iteration.
    timeout_s: float | None = None


@dataclass
class StageConfig:
    # `id` is the join key to the hand-filled labels file and to every event
    # line, so it has to stay stable across trials even when `name` changes.
    id: str = ""
    name: str = ""
    executor: str = EXECUTOR_LEROBOT_POLICY
    # Hub repo id, local directory, or mock://<name> for the hardware-free path.
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
    version: int = LATEST_CONFIG_VERSION
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    defaults: DefaultsConfig = field(default_factory=DefaultsConfig)
    # Order IS execution order.
    stages: list[StageConfig] = field(default_factory=list)
    output: OutputConfig = field(default_factory=OutputConfig)
    display_data: bool = False


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
    # module having been imported first.
    from stage_runner import mock_robot  # noqa: F401

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
        RESOLVED_CONFIG_HEADER + draccus.dump(config), encoding="utf-8"
    )
