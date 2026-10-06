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

# results is stdlib-only, so the import direction stays config -> results and
# never the reverse: aggregate.py imports results on a machine with no draccus
# and no lerobot, which is the guarantee the package __init__'s laziness keeps.
from stage_runner.results import (
    TERMINATED_BY_COMPLETE,
    TERMINATED_BY_MANUAL,
    TERMINATED_BY_REACHED,
)

logger = logging.getLogger(__name__)

# Bumped when a change to these dataclasses makes an older YAML mean something
# different. preflight.check_config_version compares the file's `version:` key
# against SUPPORTED_CONFIG_VERSIONS, so a run started from a stale YAML dies
# before the robot moves.
#
# Version 2 adds the `chain:` block, which EXPANDS into `stages` instead of the
# operator writing 22 of them by hand. Version 1 -- a hand-written stage list
# with no chain block -- is still read, unchanged: the hardware-free smoke
# config is a version 1 file and it is the regression test for everything the
# chain did not touch.
LATEST_CONFIG_VERSION: int = 2
SUPPORTED_CONFIG_VERSIONS: tuple[int, ...] = (1, 2)
CHAIN_CONFIG_VERSION: int = 2

TERMINATOR_TIMEOUT: str = "timeout"
TERMINATOR_MANUAL: str = "manual"
# A chain POLICY stage: ends when completion.CompletionMonitorStep fires, or on
# the right arrow, or at p90 x timeout_factor. The TYPE is "completion"; the
# OUTCOME is results.TERMINATED_BY_{COMPLETE,MANUAL,TIMEOUT}.
TERMINATOR_COMPLETION: str = "completion"
# A chain RESET stage: ends when the ramp arrives at the designated pose and
# holds it for settle_s. Outcome `reached` or `not_reached`.
TERMINATOR_REACHED: str = "reached"
TERMINATOR_TYPES: tuple[str, ...] = (
    TERMINATOR_TIMEOUT,
    TERMINATOR_MANUAL,
    TERMINATOR_COMPLETION,
    TERMINATOR_REACHED,
)

EXECUTOR_LEROBOT_POLICY: str = "lerobot_policy"
# The two chain executors. Separate from lerobot_policy because each does
# something that one must NOT do: chain_policy arms and disarms the completion
# monitor and re-points the one-hot, chain_reset enters record_loop with a
# trajectory policy and no checkpoint at all.
EXECUTOR_CHAIN_POLICY: str = "chain_policy"
EXECUTOR_CHAIN_RESET: str = "chain_reset"

# StageConfig.kind. A reset stage records frames into the same episode but is
# not a measurement of the policy, and every consumer (the report, the base
# integral, the success table) has to be able to drop it.
STAGE_KIND_POLICY: str = "policy"
STAGE_KIND_RESET: str = "reset"

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

    # ---- chain fields (version 2). Filled by expand_chain, not by hand. ----
    # A hand-written version 1 stage list leaves every one of these at its
    # default, which is exactly a single policy stage with no requirement -- so
    # nothing about version 1 changes.
    kind: str = STAGE_KIND_POLICY
    # 1..11. The chain's stage number, which is NOT the index in `stages`: the
    # expansion interleaves resets, so stage 11 sits at index 21.
    stage_number: int | None = None
    # 1-based one-hot index handed to TaskOneHotStep.set_stage before this
    # stage's record_loop. None for a model with no one-hot, and for resets.
    onehot_index: int | None = None
    # The terminators this stage MAY end with. Anything else breaks the chain
    # (runner.run_trial). Empty means "any", which is version 1's behaviour.
    required_terminator: list[str] = field(default_factory=list)
    # Reset stages only: the first reset of a chain has no preceding policy
    # stage, so its anchor is wherever the operator left the arms. Counted
    # separately in the report -- an 11-stage chain has 11 resets, of which 10
    # are boundaries between two policy stages.
    initial_reset: bool = False


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
class CompletionConfig:
    """The YAML's ``chain.completion:`` block. Mirrors completion.CompletionSettings.

    A second dataclass rather than reusing that one directly, because this is
    the draccus-facing SCHEMA (it is dumped into config.resolved.yaml and parsed
    out of the operator's file) while the other is what the monitor runs on.
    ``to_settings()`` is the one conversion, so the two cannot drift silently.
    """

    p_done: float = 0.95
    p_hold_s: float = 1.0
    stall_s: float = 3.0
    # TWO knobs, not one: `stall_track_rad` is how far the arm may lag the
    # command it is being given (a steady-state following error under load is a
    # stopped arm), `stall_arm_rad` is how far the MEASURED arm may travel
    # across the window (it is not). See CompletionSettings.
    stall_track_rad: float = 0.08
    stall_arm_rad: float = 0.05
    stall_base: float = 0.05
    # The departure latch: completion is refused for the whole stage until the
    # arm leaves `departure_arm_rad` of its designated start pose, or the
    # commanded base integrates past one of the two base thresholds.
    departure_arm_rad: float = 0.10
    departure_base_rot_rad: float = 0.17
    departure_base_fwd_m: float = 0.10
    # control_time_s = stage p90_s * this. Reaching it is a chain failure.
    timeout_factor: float = 1.3
    # The right arrow means "this stage is done, go on". ON for BOTH models by
    # user decision (2026-10-06): the automatic signal may be late or absent --
    # M1 does not stop at the end scene of t04 at all -- and seeing the whole
    # 1->11 chain once is worth more than an unattended measurement that stops
    # at stage 4. Automatic and manual completions are counted SEPARATELY
    # (terminated_by "complete" vs "manual"), so the distinction survives into
    # the report. ESC still aborts.
    allow_manual_complete: bool = True

    def to_settings(self):
        # Imported here, not at module scope: completion imports lerobot's
        # ProcessorStep (and so torch), and this module is imported by preflight
        # and by the config parse that must run before anything heavy loads.
        from stage_runner.completion import CompletionSettings

        return CompletionSettings(
            p_done=self.p_done,
            p_hold_s=self.p_hold_s,
            stall_s=self.stall_s,
            stall_track_rad=self.stall_track_rad,
            stall_arm_rad=self.stall_arm_rad,
            stall_base=self.stall_base,
            departure_arm_rad=self.departure_arm_rad,
            departure_base_rot_rad=self.departure_base_rot_rad,
            departure_base_fwd_m=self.departure_base_fwd_m,
        )


@dataclass
class ResetConfig:
    """The YAML's ``chain.reset:`` block. Mirrors reset_policy.ResetSettings."""

    t_min_s: float = 1.5
    v_des_rad_s: float = 0.524
    max_jump_rad: float = 1.5
    # The INITIAL reset only (the one from wherever a human left the arms to
    # `from_stage`'s pose). Tighter than the boundary limit on purpose: a
    # boundary gap is a measured 0.25-1.28 rad, while the initial gap is
    # whatever a person happened to leave, and a 1.5 rad sweep across the
    # workspace from an unknown pose is the one ramp nothing has validated.
    initial_max_jump_rad: float = 0.6
    tol_rad: float = 0.05
    settle_s: float = 1.0
    ceiling_factor: float = 3.0
    settle_check_tries: int = 3
    settle_check_gap_s: float = 0.1
    settle_check_tol_rad: float = 0.02
    # Whether the chain starts with a reset to stage `from_stage`'s pose. On by
    # default: P2 says the model does not start at all when the start
    # observation is slightly off, and the whole point of the reset is to put
    # the arms back inside the demonstration distribution.
    initial: bool = True
    # RESET-ONLY: drop every policy stage from the expansion and run the ramps
    # back to back. This is bring-up step ② and ③ (empty-handed, then holding an
    # object), and it is a MODE rather than an operator procedure on purpose.
    #
    # The procedure it replaces was "press ESC right after the reset reports
    # reached", and that is not reset-only: by the time a human reacts, the
    # policy has already sent several ticks at 21 Hz. On the first run of a new
    # runner, with the arms possibly holding glassware, those ticks are the whole
    # risk the step exists to retire.
    #
    # The ramps are the SAME ones the full chain runs -- same executor, same
    # plan, same ids -- because nothing between them moves the arm: the reset to
    # stage k+1 anchors exactly where the reset to stage k left it, which is the
    # boundary gap this step measures.
    only: bool = False

    def to_settings(self):
        from stage_runner.reset_policy import ResetSettings

        return ResetSettings(
            t_min_s=self.t_min_s,
            v_des_rad_s=self.v_des_rad_s,
            max_jump_rad=self.max_jump_rad,
            initial_max_jump_rad=self.initial_max_jump_rad,
            tol_rad=self.tol_rad,
            settle_s=self.settle_s,
            ceiling_factor=self.ceiling_factor,
            settle_check_tries=self.settle_check_tries,
            settle_check_gap_s=self.settle_check_gap_s,
            settle_check_tol_rad=self.settle_check_tol_rad,
        )

    def worst_case_ceiling_s(self) -> float:
        """The ceiling for the LARGEST ramp the settings allow.

        ``plan_stage`` has to put a number on the stage_start event before the
        arm has been read, and the real ceiling depends on the measured anchor.
        This is the bound: a ramp at ``max_jump_rad``. The ACTUAL ceiling and
        duration are in the stage_end event's ``reason_detail`` (``reset_T``).
        """
        duration = max(self.t_min_s, self.max_jump_rad / self.v_des_rad_s)
        return (duration + self.settle_s) * self.ceiling_factor


@dataclass
class ChainModelConfig:
    """Which checkpoint the whole chain runs, and how it is conditioned."""

    # One path for all 11 stages. ONE checkpoint, loaded ONCE: the design is a
    # single multi-stage ACT plus a stage one-hot, not eleven experts.
    policy_path: str = ""
    # K for the stage one-hot. 11 for M1/M2 and for the new `tph` models; null
    # for a checkpoint with no one-hot (then the state is 14 or 16 wide and the
    # stage is whatever the scene says it is).
    onehot_k: int | None = None
    # Applied to the checkpoint's own config after loading, the same way
    # `lerobot-record --policy.n_action_steps=` does. 30 is the exec confirmed
    # on 2026-10-02. null leaves the checkpoint's value.
    n_action_steps: int | None = None
    # REQUIRED for a real checkpoint, and required for a mock one too (there is
    # no config.json to derive it from). The checkpoint's
    # output_features.action.shape is still THE FACT -- 16 means no progress, 17
    # means progress -- and preflight refuses a run where the key and the width
    # disagree. What it no longer does is accept `null` and derive it quietly:
    # this is the single value that decides whether the recording dataset
    # declares a 17th action feature, and therefore at what width the normalizer
    # loads, so a config that does not state it cannot be read later to find out
    # which of the two models ran (2026-10-06).
    #
    # The default stays None so draccus can tell "absent" from "false"; absent is
    # what preflight rejects.
    has_progress: bool | None = None


@dataclass
class ChainConfig:
    """``chain:`` -- present in a version 2 config, absent in a version 1 one."""

    enabled: bool = False
    # configs/chain/stage_params.json. Generated in the training repo; see
    # configs/chain/README.md for the schema and the refusal list.
    params_path: str = ""
    from_stage: int = 1
    to_stage: int = 11
    model: ChainModelConfig = field(default_factory=ChainModelConfig)
    completion: CompletionConfig = field(default_factory=CompletionConfig)
    reset: ResetConfig = field(default_factory=ResetConfig)


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
    # Order IS execution order. In a version 2 chain config this is left EMPTY
    # in the YAML and filled by expand_chain before preflight runs -- writing 22
    # stages by hand is 22 chances to put the wrong one-hot on the wrong stage.
    stages: list[StageConfig] = field(default_factory=list)
    output: OutputConfig = field(default_factory=OutputConfig)
    display_data: bool = False
    chain: ChainConfig = field(default_factory=ChainConfig)


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


def policy_stage_id(stage_number: int) -> str:
    """``t04`` for stage 4. The join key to the hand-filled label sheet."""
    return f"t{stage_number:02d}"


def reset_stage_id(stage_number: int, *, initial: bool) -> str:
    """``reset_pre_01`` for the initial reset, ``reset_to_05`` for a boundary one.

    Named by the stage it resets TO, not by the boundary it sits in: the target
    pose belongs to the NEXT stage, and a reader of the log asking "which pose
    was it driving to" has the answer in the id.
    """
    return (
        f"reset_pre_{stage_number:02d}"
        if initial
        else f"reset_to_{stage_number:02d}"
    )


def expand_chain(config: StageRunnerConfig, params) -> list[StageConfig]:
    """Turn ``chain:`` into the stage list the runner walks. The ONE place that order lives.

    With ``chain.reset.only`` it is the RAMPS ALONE -- 11 reset stages, no
    policy stage at all. See :class:`ResetConfig`.

    For ``from_stage=1, to_stage=11`` the result is 22 stages::

        reset_pre_01, t01, reset_to_02, t02, ..., reset_to_11, t11

    i.e. **11 policy stages and 11 resets**, of which 10 are boundaries between
    two policy stages and one is the initial reset
    (``StageConfig.initial_reset``). The design brief says "10 resets" and means
    the ten boundaries; both numbers are reported, because an operator counting
    ramps on the robot sees eleven.

    ``params`` is a :class:`stage_runner.chain_params.ChainParams`. Taken as an
    argument rather than loaded here so that the loader's refusals (and its
    ``dataset.fps`` cross-check) happen in ``cli.main``'s preflight block, where
    a bad file is one line and exit code 2 instead of a traceback.

    Every field a chain stage needs is set HERE and nowhere else. The failure
    this prevents is the one a hand-written 22-stage YAML makes: a one-hot index
    that does not match the stage, which is silent -- the model runs, it just
    runs the wrong stage's conditioning, and offline §89 already showed the
    one-hot is weak enough that nothing would look obviously wrong.
    """
    chain = config.chain
    model = chain.model
    allowed_policy = [TERMINATED_BY_COMPLETE]
    if chain.completion.allow_manual_complete:
        allowed_policy.append(TERMINATED_BY_MANUAL)
    reset_ceiling = chain.reset.worst_case_ceiling_s()

    stages: list[StageConfig] = []
    for number in range(chain.from_stage, chain.to_stage + 1):
        initial = number == chain.from_stage
        # reset.only forces the first ramp in: without it the range would start
        # at the SECOND stage's pose with the arm wherever the operator left it,
        # so the first measured gap would not be a stage boundary at all.
        if not initial or chain.reset.initial or chain.reset.only:
            stages.append(
                StageConfig(
                    id=reset_stage_id(number, initial=initial),
                    name=f"reset to task{number:02d}",
                    executor=EXECUTOR_CHAIN_RESET,
                    # NO checkpoint. The ramp is arithmetic; there is nothing to
                    # load, and preflight's "a known executor needs a
                    # policy_path" rule is executor-aware for exactly this.
                    policy_path="",
                    # Recorded as frame["task"] for every reset frame, which is
                    # how a boundary is found in the dataset afterwards.
                    instruction=f"reset:{number:02d}",
                    terminator=TerminatorConfig(
                        type=TERMINATOR_REACHED, timeout_s=reset_ceiling
                    ),
                    kind=STAGE_KIND_RESET,
                    stage_number=number,
                    onehot_index=None,
                    required_terminator=[TERMINATED_BY_REACHED],
                    initial_reset=initial,
                )
            )
        if chain.reset.only:
            continue
        stage_params = params.stage(number)
        stages.append(
            StageConfig(
                id=policy_stage_id(number),
                name=stage_params.name,
                executor=EXECUTOR_CHAIN_POLICY,
                policy_path=model.policy_path,
                instruction=stage_params.instruction,
                terminator=TerminatorConfig(
                    type=TERMINATOR_COMPLETION,
                    timeout_s=stage_params.timeout_s(chain.completion.timeout_factor),
                ),
                kind=STAGE_KIND_POLICY,
                stage_number=number,
                onehot_index=number if model.onehot_k else None,
                required_terminator=list(allowed_policy),
                initial_reset=False,
            )
        )
    return stages


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
