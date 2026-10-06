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
from lerobot.utils.constants import (
    ACTION,
    HF_LEROBOT_HOME,
    OBS_ENV_STATE,
    OBS_STATE,
)
from lerobot.utils.control_utils import is_headless, sanity_check_dataset_name

from stage_runner.config import (
    CHAIN_CONFIG_VERSION,
    EXECUTOR_CHAIN_RESET,
    LATEST_CONFIG_VERSION,
    STAGE_KIND_POLICY,
    STAGE_KIND_RESET,
    SUPPORTED_CONFIG_VERSIONS,
    TERMINATOR_COMPLETION,
    TERMINATOR_MANUAL,
    TERMINATOR_REACHED,
    TERMINATOR_TIMEOUT,
    TERMINATOR_TYPES,
    StageConfig,
    StageRunnerConfig,
)

logger = logging.getLogger(__name__)

EVAL_DATASET_PREFIX: str = "eval_"

# Executors that deliberately run WITHOUT a checkpoint. Named by their config
# constant so a rename cannot leave this set pointing at a string nothing uses.
EXECUTORS_WITHOUT_CHECKPOINT: frozenset[str] = frozenset({EXECUTOR_CHAIN_RESET})

# The 17th action feature a `tph` checkpoint produces. The NAME is the mechanism
# (record_adapter.build_dataset_features declares it so make_policy's
# unconditional output_features overwrite lands on 17), so it lives next to the
# check that decides whether to declare it.
PROGRESS_ACTION_NAME: str = "progress"

# Matches policies.MOCK_POLICY_SCHEME. Duplicated rather than imported for the
# reason the whole module is: `policies` pulls lerobot.policies.factory and so
# torch into the gate that has to run before anything heavy loads.
_MOCK_POLICY_SCHEME: str = "mock://"


def runs_only_mock_policies(config: StageRunnerConfig) -> bool:
    """True when no stage loads a real checkpoint.

    Several gates mean something different on this path -- there is no
    checkpoint to compare dimensions against, nothing is energised, and nothing
    can run away -- so it is named once here instead of being re-derived at each
    one. `check_dataset_name` already had its own copy of this question and
    keeps it, because it needs the policy_configs mapping rather than the config.
    """
    paths = [stage.policy_path for stage in config.stages if stage.policy_path]
    if config.chain.enabled and config.chain.model.policy_path:
        paths.append(config.chain.model.policy_path)
    return bool(paths) and all(
        path.startswith(_MOCK_POLICY_SCHEME) for path in paths
    )


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
    """Refuse a YAML written against a schema generation this build cannot read.

    The YAML outlives the code: once trials are running it is tied to experiment
    logs and repro scripts, so an unrecognised generation must stop the run
    rather than be read with this generation's field meanings.

    TWO generations are accepted. Version 1 is a hand-written stage list and is
    read EXACTLY as before -- the hardware-free smoke config is one, and it is
    the regression test for everything the chain did not touch. Version 2 adds
    the `chain:` block.

    The two cross-checks below are the ones that matter: a `chain:` block in a
    version 1 file would be parsed and then IGNORED (a chain run that silently
    executed zero stages), and `version: 2` with no chain block and no stages is
    an operator who wrote half a config.
    """
    if config.version not in SUPPORTED_CONFIG_VERSIONS:
        supported = ", ".join(str(v) for v in SUPPORTED_CONFIG_VERSIONS)
        raise PreflightError(
            f"config `version: {config.version}` is not supported; this "
            f"stage_runner reads version(s) {supported} (latest "
            f"{LATEST_CONFIG_VERSION}). Set `version: {LATEST_CONFIG_VERSION}` as "
            "the first key of the YAML, or run the stage_runner that matches the "
            "config."
        )
    if config.chain.enabled and config.version < CHAIN_CONFIG_VERSION:
        raise PreflightError(
            f"`chain.enabled: true` needs `version: {CHAIN_CONFIG_VERSION}`, but "
            f"the file says `version: {config.version}`. A version "
            f"{config.version} config has no chain semantics, so the block would "
            "be parsed, snapshotted and then ignored -- a run with zero stages."
        )
    if config.version >= CHAIN_CONFIG_VERSION and not (
        config.chain.enabled or config.stages
    ):
        raise PreflightError(
            f"`version: {config.version}` with neither `chain.enabled: true` nor "
            "a `stages:` list: there is nothing to run. Set `chain.enabled: true` "
            "and `chain.params_path`, or write the stages out."
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
        elif stage.executor in EXECUTORS_WITHOUT_CHECKPOINT:
            # A reset stage runs arithmetic, not a checkpoint. Demanding a
            # policy_path here would force every chain YAML to invent a fake one,
            # and the branch below exists to stop a BLANK path reaching
            # PreTrainedConfig.from_pretrained("").
            if stage.policy_path:
                problems.append(
                    f"{where} ({stage.id!r}): a `{stage.executor}` stage loads no "
                    f"checkpoint, but `policy_path: {stage.policy_path}` is set. "
                    "Leave it empty -- a path here would be snapshotted into "
                    "config.resolved.yaml as if it had run."
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


def checkpoint_env_state_dimension(policy_config: PreTrainedConfig) -> int | None:
    """Read ``input_features['observation.environment_state']``, or None.

    None is the ROUND-1 answer (M1/M2/M3, ``tph``): no env token. A number is a
    ROUND-2 checkpoint, which feeds the stage one-hot to ACT twice -- the tail of
    the 27-D state AND a separate encoder token
    (``modeling_act.py:344-346, 465-466``). ``task_onehot_patch`` writes both
    from one vector, so this width MUST equal K; anything else means the
    checkpoint is not the one the chain thinks it is.

    Same ordering rule as :func:`checkpoint_state_dimension`: read BEFORE
    ``make_policy``. The input side survives that call (``factory.py:471`` guards
    the assignment with ``if not cfg.input_features``) but only because the
    checkpoint declared it -- a dataset-filled config would carry no ENV key at
    all, and then nothing would be comparable.
    """
    return _feature_dimension(policy_config.input_features, OBS_ENV_STATE)


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
    *,
    onehot_k: int | None = None,
    allow_extra_action: bool = False,
) -> None:
    """Compare every checkpoint's state and action width against this robot.

    Both directions matter and they fail differently. A state mismatch is the
    ``include_base_in_state`` question (16-dim assets vs 14-dim); an action
    mismatch means the checkpoint was trained on a different robot entirely,
    because ``action_features`` always carries the base keys regardless of that
    flag.

    TWO WIDENINGS ARE LEGITIMATE and both are declared by the caller, never
    guessed here:

    * ``onehot_k`` -- a multi-stage ACT reads ``16 + K`` state values, because
      ``task_onehot_patch`` builds ``[arms14, 0, 0, one_hot(K)]`` in the
      preprocessor. The robot still emits 14 (or 16), so the raw comparison
      would reject every one-hot checkpoint there is.
    * ``allow_extra_action`` -- a 17-D `tph` checkpoint, whose extra slot the
      dataset declares (see :func:`extra_action_names`).

    Passing neither reproduces the pre-chain behaviour exactly.

    A ROUND-2 checkpoint additionally declares
    ``observation.environment_state`` (``FeatureType.ENV``, shape ``[K]``) --
    the same stage one-hot handed to ACT as its own encoder token. That is
    ACCEPTED here, not required: it has to agree with ``onehot_k`` and with the
    state's one-hot tail, and declaring it with ``onehot_k`` null is refused
    because nothing would then write the key.
    """
    state_dim = robot_state_dimension(robot)
    action_dim = robot_action_dimension(robot)
    # 14 arm values + 2 zeroed base slots + K, which is what the one-hot step
    # builds and asserts against the checkpoint itself
    # (task_onehot_patch.insert_task_onehot). Stated here so the REFUSAL happens
    # before any weight downloads rather than inside make_pre_post_processors.
    expected_state = 16 + int(onehot_k) if onehot_k else state_dim
    problems: list[str] = []

    for stage in stages:
        policy_config = policy_configs.get(stage.id)
        if policy_config is None:
            # A mock:// stage, or a reset stage: neither has a checkpoint
            # config.json to compare against.
            continue

        checkpoint_state = checkpoint_state_dimension(policy_config)
        if onehot_k and checkpoint_state is not None and checkpoint_state != expected_state:
            problems.append(
                f"stage {stage.id!r} ({stage.policy_path}): checkpoint expects a "
                f"{checkpoint_state}-dim observation.state, but a K={onehot_k} "
                f"stage one-hot builds 16+{onehot_k}={expected_state} "
                "([arms14, 0, 0, one_hot]). Either `chain.model.onehot_k` is "
                "wrong or this is not a one-hot checkpoint."
            )
        elif not onehot_k and checkpoint_state is not None and checkpoint_state != state_dim:
            problems.append(
                f"stage {stage.id!r} ({stage.policy_path}): checkpoint expects a "
                f"{checkpoint_state}-dim observation.state, robot "
                f"{robot.name!r} produces {state_dim}. Flip "
                "`include_base_in_state` in the robot block (it adds x.vel and "
                "theta.vel, i.e. 2 dims) or use the checkpoint trained on this "
                "state layout."
            )

        # The round-2 env token. Accepted, never REQUIRED -- a round-1
        # checkpoint declares no ENV feature and must keep passing unchanged --
        # but when it is declared it has to agree with K, in BOTH directions:
        #
        # * declared and onehot_k is null: nothing would write the key, and ACT
        #   dies on `batch[OBS_ENV_STATE]` (modeling_act.py:466) on the FIRST
        #   FRAME, i.e. after robot.connect(). Refusing here moves that to
        #   before any weights download.
        # * declared K wide but not the chain's K: the env token IS the state
        #   tail, so the two widths cannot legitimately differ.
        checkpoint_env = checkpoint_env_state_dimension(policy_config)
        if checkpoint_env is not None and not onehot_k:
            problems.append(
                f"stage {stage.id!r} ({stage.policy_path}): checkpoint declares "
                f"observation.environment_state ({checkpoint_env}-dim, the "
                "round-2 stage token), but `chain.model.onehot_k` is null. "
                "Nothing would write that key and ACT reads it on the first "
                "frame -- set onehot_k to the stage count this checkpoint was "
                "trained with."
            )
        elif checkpoint_env is not None and checkpoint_env != int(onehot_k):
            problems.append(
                f"stage {stage.id!r} ({stage.policy_path}): checkpoint declares a "
                f"{checkpoint_env}-dim observation.environment_state but "
                f"`chain.model.onehot_k` is {onehot_k}. The env token is the same "
                "one-hot as the state tail, so the two widths must be equal."
            )
        # Independent of the 16+K check above ON PURPOSE: that one compares the
        # state against the chain's DECLARED K, this one compares the two things
        # the CHECKPOINT itself declares. An internally inconsistent config.json
        # earns both messages, and each is separately actionable.
        if (
            checkpoint_env is not None
            and checkpoint_state is not None
            and checkpoint_env != checkpoint_state - 16
        ):
            problems.append(
                f"stage {stage.id!r} ({stage.policy_path}): checkpoint declares a "
                f"{checkpoint_env}-dim observation.environment_state and a "
                f"{checkpoint_state}-dim observation.state, whose one-hot tail is "
                f"{checkpoint_state - 16} wide. Those are the same vector in the "
                "trained model; this checkpoint's config.json is inconsistent."
            )

        checkpoint_action = checkpoint_action_dimension(policy_config)
        if allow_extra_action and checkpoint_action is not None:
            # extra_action_names raises PreflightError itself on any width other
            # than equal or one wider, which is the whole check for this branch.
            extra_action_names(checkpoint_action, action_dim)
        elif checkpoint_action is not None and checkpoint_action != action_dim:
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


def extra_action_names(
    checkpoint_action_dim: int | None, robot_action_dim: int
) -> tuple[str, ...]:
    """Which action features the dataset must DECLARE on top of the robot's own.

    ``()`` when the checkpoint's action is exactly the robot's width (16 for
    this rig: ``[left7, right7, x.vel, theta.vel]``), ``("progress",)`` when it
    is one wider -- the 17th slot the `tph` recipe trains.

    Anything else raises, and raising is the point: ``make_policy`` overwrites
    ``cfg.output_features`` from the recording dataset UNCONDITIONALLY
    (factory.py:470), so a checkpoint whose head is some other width would build
    a head matching the DATASET and then die in ``load_state_dict`` with a
    safetensors size mismatch several hundred MB into ``from_pretrained``, on a
    machine that has already powered the arms. The width is readable from
    config.json alone, before a single weight byte moves.

    Called with the checkpoint's OWN dimension, which means strictly before
    ``make_policy`` -- after it, ``output_features`` is the dataset's and the
    checkpoint's width is unrecoverable from the object.
    """
    if checkpoint_action_dim is None:
        # The checkpoint's config.json declares no action feature. Nothing to
        # widen and nothing to compare; make_policy will fill it from the
        # dataset, which is upstream's own behaviour for a fresh policy.
        return ()
    extra = checkpoint_action_dim - robot_action_dim
    if extra == 0:
        return ()
    if extra == 1:
        return (PROGRESS_ACTION_NAME,)
    raise PreflightError(
        f"the checkpoint outputs a {checkpoint_action_dim}-dim action and the "
        f"robot takes {robot_action_dim}. The chain knows two widths: equal "
        f"(a 16-D model such as M1) and one wider (a 17-D `tph` model, whose "
        f"extra slot is the progress scalar `{PROGRESS_ACTION_NAME}`). "
        f"{checkpoint_action_dim} is neither, so either the checkpoint was "
        "trained on a different robot or `include_base_in_state` is not the "
        "flag that explains the difference -- action_features always carries "
        "the base keys, so that flag does not change this number."
    )


def _checkpoint_document(policy_path: str) -> Mapping[str, object] | None:
    """A local checkpoint's ``config.json`` as a plain dict, or None.

    None for a hub id, a mock path, or a directory with no readable
    ``config.json``. Deliberately NOT a ``PreTrainedConfig``: this runs in the
    gate that must not import ``lerobot.policies.factory``, and the one key it
    reads (``temporal_ensemble_coeff``) is a scalar in the file.
    """
    if not policy_path or policy_path.startswith(_MOCK_POLICY_SCHEME):
        return None
    candidate = Path(policy_path).expanduser() / "config.json"
    if not candidate.is_file():
        return None
    import json

    try:
        document = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return document if isinstance(document, dict) else None


def temporal_ensemble_problem(
    config: StageRunnerConfig, policy_config=None
) -> str | None:
    """The message refusing ``temporal_ensemble_coeff`` with ``n_action_steps>1``.

    ACT enforces this itself, in ``ACTConfig.__post_init__``, and the chain
    WALKS AROUND that enforcement without meaning to: ``policies.load_chain_bundles``
    assigns ``policy_config.n_action_steps = 30`` AFTER ``from_pretrained`` has
    run, which is exactly how ``lerobot-record --policy.n_action_steps=`` works
    and is why the assignment is there -- but a plain attribute write re-runs no
    validator. A checkpoint saved with ``temporal_ensemble_coeff: 0.01`` and
    ``n_action_steps: 1`` therefore loads cleanly, gets widened to 30, and runs a
    combination upstream declares NotImplementedError on.

    What it costs on this rig is measured, not theoretical: the ensemble runs the
    policy on every tick instead of once per chunk, the loop went 20% slower and
    the base over-rotated by 1.29x (aa30006), and the base holds its last
    velocity command until the next ``send_action`` -- so a slow loop drives
    every rotation further.

    ``scripts/_chain_preflight.py`` already refuses this, but only for runs
    started through ``scripts/eval_chain.sh``. ``python -m stage_runner`` is a
    supported entry point and had no gate at all.

    ``policy_config`` is the ``PreTrainedConfig`` ``cli.main`` has already
    fetched, when there is one; otherwise the checkpoint's ``config.json`` is
    read from disk. Either way this happens BEFORE the assignment.
    """
    chain = config.chain
    coefficient = None
    checkpoint_steps = None
    if policy_config is not None:
        coefficient = getattr(policy_config, "temporal_ensemble_coeff", None)
        checkpoint_steps = getattr(policy_config, "n_action_steps", None)
    else:
        document = _checkpoint_document(chain.model.policy_path)
        if document is None:
            return None
        coefficient = document.get("temporal_ensemble_coeff")
        checkpoint_steps = document.get("n_action_steps")
    if coefficient is None:
        return None
    # The EFFECTIVE value: the YAML's override when it sets one (that is the
    # number that will be assigned), else whatever the checkpoint carries.
    effective = (
        chain.model.n_action_steps
        if chain.model.n_action_steps is not None
        else checkpoint_steps
    )
    try:
        steps = int(effective or 1)
    except (TypeError, ValueError):
        steps = 1
    if steps <= 1:
        return None
    return (
        f"the checkpoint `{chain.model.policy_path}` was saved with "
        f"`temporal_ensemble_coeff: {coefficient}`, and this run would execute "
        f"`n_action_steps: {steps}`. ACT refuses that combination in its own "
        "__post_init__, but the chain assigns n_action_steps AFTER "
        "from_pretrained (as `--policy.n_action_steps=` does), which re-runs no "
        "validator -- so it would have run anyway. Measured cost on this rig: "
        "the loop 20% slower and base over-rotation 1.29x (aa30006). Either set "
        "`chain.model.n_action_steps: 1`, or point at a checkpoint whose "
        "config.json has `temporal_ensemble_coeff: null`."
    )


def threshold_problems(chain) -> list[str]:
    """Range-check every numeric knob of ``chain.completion`` and ``chain.reset``.

    EVERY ONE OF THEM, not the four that happened to be checked before. These are
    the numbers that decide when a stage is finished and how far the arms are
    driven, they come out of a hand-edited YAML, and draccus coerces whatever it
    finds to a float without an opinion about its sign. The two failure shapes
    they produce are both SILENT:

    * a threshold at or below 0 makes a test that can never pass (``stall_s: 0``
      leaves ``_covered_window`` asking for a zero-length window -- every stage
      then runs to its timeout and the chain "fails" on eleven stages that were
      finished) or one that always passes (``departure_arm_rad: 0`` latches the
      departure on tick 1, which turns the whole latch off);
    * ``stall_track_rad < stall_arm_rad`` inverts the pair the 2026-10-06 split
      exists to separate: the tracking allowance has to be the LOOSER of the two,
      because a loaded arm holding still lags its command by more than it travels.

    WRITTEN AS ``not (x > 0)``, NEVER AS ``x <= 0``. Every comparison with NaN is
    False, so ``x <= 0`` PASSES for a NaN and the value goes on to be a joint
    threshold -- the same NaN-shaped hole this session closed in the completion
    monitor and in lerobot's clamp. The ``not (...)`` form rejects it.
    """
    completion = chain.completion
    reset = chain.reset
    problems: list[str] = []

    def positive(path: str, value: float, why: str) -> None:
        if not (float(value) > 0.0):
            problems.append(f"`chain.{path}: {value}` must be > 0 -- {why}")

    def above_one(path: str, value: float, why: str) -> None:
        if not (float(value) > 1.0):
            problems.append(f"`chain.{path}: {value}` must be > 1.0 -- {why}")

    if not (0.0 < float(completion.p_done) <= 1.0):
        problems.append(
            f"`chain.completion.p_done: {completion.p_done}` must be in (0, 1]: "
            "the progress output is a fraction of the stage."
        )
    positive(
        "completion.p_hold_s",
        completion.p_hold_s,
        "it is the window the progress output must hold p_done over, and a "
        "zero-length window is satisfied by the first tick that reads high",
    )
    positive(
        "completion.stall_s",
        completion.stall_s,
        "it is the window that has to be free of motion; at 0 the window is "
        "never COVERED (_covered_window needs a sample at or before the cutoff), "
        "so no stage can ever complete and every one runs to its timeout",
    )
    positive(
        "completion.stall_arm_rad",
        completion.stall_arm_rad,
        "it is the peak-to-peak travel allowed across the stall window, and at 0 "
        "a real arm's encoder noise is read as motion forever",
    )
    positive(
        "completion.stall_track_rad",
        completion.stall_track_rad,
        "it is how far the arm may lag its command and still count as stopped; "
        "at 0 a loaded arm holding against gravity never counts as stopped",
    )
    positive(
        "completion.stall_base",
        completion.stall_base,
        "it is the commanded base velocity below which the base counts as "
        "stopped; at 0 no tick ever qualifies (the test is `>=`)",
    )
    if float(completion.stall_track_rad) < float(completion.stall_arm_rad):
        problems.append(
            f"`chain.completion.stall_track_rad: {completion.stall_track_rad}` is "
            f"TIGHTER than `stall_arm_rad: {completion.stall_arm_rad}`. The two "
            "were split on 2026-10-06 precisely because the tracking allowance "
            "must be the LOOSER one: a steady-state following error of 0.06 rad "
            "under load is a perfectly stopped arm, while 0.06 rad of travel is "
            "not. Inverted, a stationary loaded arm is declared 'still moving' "
            "and the stage runs to its timeout -- which is the regression the "
            "split fixed."
        )
    positive(
        "completion.departure_arm_rad",
        completion.departure_arm_rad,
        "it is the distance the arm must leave its starting pose by for the "
        "departure latch to set; at 0 the latch sets on tick 1, i.e. it is OFF, "
        "and a stage that never moves can be declared complete (the module "
        "docstring of completion.py says what that cost)",
    )
    positive(
        "completion.departure_base_rot_rad",
        completion.departure_base_rot_rad,
        "it is the integrated commanded rotation that latches the departure; at "
        "0 the latch sets on tick 1 and is therefore OFF",
    )
    positive(
        "completion.departure_base_fwd_m",
        completion.departure_base_fwd_m,
        "it is the integrated commanded travel that latches the departure; at 0 "
        "the latch sets on tick 1 and is therefore OFF",
    )
    above_one(
        "completion.timeout_factor",
        completion.timeout_factor,
        "it multiplies the stage's p90 length, so a factor at or below 1 times "
        "out the slowest tenth of the demonstrations by construction",
    )

    positive(
        "reset.t_min_s",
        reset.t_min_s,
        "it is the floor on the ramp duration; at 0 a tiny correction becomes a "
        "single 21 Hz step instead of a gentle second and a half",
    )
    positive(
        "reset.v_des_rad_s",
        reset.v_des_rad_s,
        "the ramp's duration is max_jump / v_des, so a non-positive value is a "
        "division by zero or a negative duration",
    )
    positive(
        "reset.max_jump_rad",
        reset.max_jump_rad,
        "it is the gap a BOUNDARY ramp is refused above, and it also sizes the "
        "ramp duration",
    )
    positive(
        "reset.initial_max_jump_rad",
        reset.initial_max_jump_rad,
        "it is the gap the INITIAL reset (from wherever a human left the arms) "
        "is refused above, and a non-positive value refuses every start",
    )
    if float(reset.initial_max_jump_rad) > float(reset.max_jump_rad):
        problems.append(
            f"`chain.reset.initial_max_jump_rad: {reset.initial_max_jump_rad}` is "
            f"larger than `max_jump_rad: {reset.max_jump_rad}`. The initial ramp "
            "starts from a pose nothing has measured, so it must be the TIGHTER "
            "of the two, never the looser."
        )
    positive(
        "reset.tol_rad",
        reset.tol_rad,
        "it is the arrival radius; at 0 arrival is never declared and the reset "
        "ends `not_reached`, which is a chain failure with no retry",
    )
    positive(
        "reset.settle_s",
        reset.settle_s,
        "it is the window arrival is judged over; at 0 arrival is declared on "
        "the first tick that happens to be inside tol",
    )
    above_one(
        "reset.ceiling_factor",
        reset.ceiling_factor,
        "the ramp is tick-based, so arrival is only possible while the achieved "
        "loop rate is at least dataset.fps / ceiling_factor -- a factor at or "
        "below 1 makes the ceiling shorter than the ramp itself",
    )
    return problems


def check_chain_definitions(
    config: StageRunnerConfig, params, policy_config=None
) -> None:
    """Gate the expanded chain: ids, kinds, one-hot indices, required terminators.

    Everything here is produced by ``config.expand_chain``, so a failure is a
    defect in this package rather than an operator error. It still runs: the
    expansion is the one place that pairs a stage number with a one-hot index,
    and a wrong pairing is SILENT -- the model runs, it just runs the wrong
    stage's conditioning, and offline §89 already showed the one-hot is weak
    enough that nothing downstream would look obviously wrong.
    """
    chain = config.chain
    problems: list[str] = []

    if not chain.params_path:
        problems.append(
            "`chain.params_path` is empty. It points at "
            "configs/chain/stage_params.json, which carries the designated start "
            "poses, the p10/p90 lengths and the end poses -- none of which the "
            "runner can derive."
        )
    if not chain.model.policy_path:
        problems.append(
            "`chain.model.policy_path` is empty. One checkpoint runs all 11 "
            "stages; there is no per-stage path."
        )
    elif chain.model.policy_path.startswith(_MOCK_POLICY_SCHEME) and chain.model.onehot_k:
        problems.append(
            f"`chain.model.policy_path: {chain.model.policy_path}` is a mock and "
            f"`chain.model.onehot_k: {chain.model.onehot_k}`. The one-hot step "
            "asserts that the CHECKPOINT declares a 16+K observation.state, and "
            "a mock declares whatever the dataset does -- so installing it would "
            "assert nothing while looking like it had. Set `onehot_k: null` for a "
            "mock chain; the one-hot wiring is covered by the unit test against "
            "task_onehot_patch.py."
        )
    if not 1 <= chain.from_stage <= chain.to_stage:
        problems.append(
            f"`chain.from_stage: {chain.from_stage}` / `chain.to_stage: "
            f"{chain.to_stage}`: need 1 <= from_stage <= to_stage."
        )
    problems.extend(threshold_problems(chain))
    coefficient_problem = temporal_ensemble_problem(config, policy_config)
    if coefficient_problem:
        problems.append(coefficient_problem)

    numbers = list(range(chain.from_stage, chain.to_stage + 1))
    absent = [n for n in numbers if n not in params.stages]
    if absent:
        problems.append(
            f"{params.path} has no parameters for stage(s) {absent}, which "
            f"chain.from_stage/to_stage asks to run."
        )

    policy_stages = [s for s in config.stages if s.kind == STAGE_KIND_POLICY]
    reset_stages = [s for s in config.stages if s.kind == STAGE_KIND_RESET]
    if chain.reset.only:
        # The ramps alone (bring-up (2) and (3)). A policy stage reaching this
        # expansion would move the arms under a checkpoint in the one mode whose
        # entire purpose is that nothing does.
        if policy_stages:
            problems.append(
                f"`chain.reset.only: true` but the expansion carries "
                f"{len(policy_stages)} policy stage(s) "
                f"{[s.id for s in policy_stages]}. Reset-only means the ramps "
                "alone; a policy stage here would drive the arms in the one mode "
                "that exists so that nothing does."
            )
        if [s.stage_number for s in reset_stages] != numbers:
            problems.append(
                f"reset-only expanded to ramps for "
                f"{[s.stage_number for s in reset_stages]}, expected {numbers}."
            )
    elif [s.stage_number for s in policy_stages] != numbers:
        problems.append(
            f"the expanded chain runs policy stages "
            f"{[s.stage_number for s in policy_stages]}, expected {numbers}."
        )
    for stage in policy_stages:
        if stage.terminator.type != TERMINATOR_COMPLETION:
            problems.append(
                f"policy stage {stage.id!r} has `terminator.type: "
                f"{stage.terminator.type}`, expected {TERMINATOR_COMPLETION!r}."
            )
        if chain.model.onehot_k and stage.onehot_index != stage.stage_number:
            problems.append(
                f"policy stage {stage.id!r} (stage {stage.stage_number}) carries "
                f"one-hot index {stage.onehot_index}. The one-hot is 1-BASED and "
                "must equal the stage number -- a mismatch runs the wrong stage's "
                "conditioning without any error."
            )
        if chain.model.onehot_k and not 1 <= (stage.onehot_index or 0) <= int(
            chain.model.onehot_k
        ):
            problems.append(
                f"policy stage {stage.id!r}: one-hot index {stage.onehot_index} is "
                f"outside 1..{chain.model.onehot_k}."
            )
    for stage in reset_stages:
        if stage.terminator.type != TERMINATOR_REACHED:
            problems.append(
                f"reset stage {stage.id!r} has `terminator.type: "
                f"{stage.terminator.type}`, expected {TERMINATOR_REACHED!r}."
            )
        if stage.stage_number not in params.stages:
            problems.append(
                f"reset stage {stage.id!r} targets stage {stage.stage_number}, "
                f"which {params.path} has no start pose for."
            )
    if not config.stages:
        problems.append(
            "the chain expanded to zero stages. Check chain.from_stage / "
            "chain.to_stage."
        )

    if problems:
        raise PreflightError("invalid chain definition:\n  - " + "\n  - ".join(problems))

    logger.info(
        f"chain: {len(policy_stages)} policy stage(s) "
        f"{numbers[0]}..{numbers[-1]}, {len(reset_stages)} reset(s) of which "
        f"{sum(1 for s in reset_stages if s.initial_reset)} initial and "
        f"{sum(1 for s in reset_stages if not s.initial_reset)} at a boundary; "
        f"params {params.path} (fps {params.fps}, generated from "
        f"{params.source.get('sim_commit', 'unknown')})"
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
    remedy = (
        "Run from a session where the arrow keys reach the listener, or give the "
        "stage `terminator: {type: timeout, timeout_s: <seconds>}`."
    )
    if config.chain.enabled and runs_only_mock_policies(config):
        # A mock chain has no robot: nothing is energised, nothing can run away,
        # and an unreachable keyboard costs a test run rather than an
        # unsupervised rollout. Skipping the gate here is what lets the
        # end-to-end chain test run over ssh on a machine with no display --
        # which is the only environment this package is developed in.
        logger.info(
            "every stage runs a mock policy, so the keyboard-reachability gate "
            "is skipped: there is no robot to stop"
        )
        return
    if config.chain.enabled:
        # UNCONDITIONAL for a chain, whatever the terminators say. Esc is the
        # ONLY human input a chain has: there is no inter-episode teleop reset
        # to stop in, the whole 1->11 run is one episode, and 22 stages of
        # policy rollout and automatic pose ramps run back to back. A session
        # where the listener never receives a key is a session where the
        # 'operator with the e-stop' safety net has no software half at all.
        # `allow_manual_complete` makes the right arrow load-bearing on top of
        # that: without it a stage whose automatic signal never comes runs to
        # its timeout and fails the chain.
        named = (
            ", ".join(repr(i) for i in manual_ids)
            if manual_ids
            else "the whole chain"
        )
        remedy = (
            "Run from a session where the arrow keys reach the listener (see the "
            "README's `## eval 함정`). A chain cannot be run without them: Esc is "
            "the only stop and the right arrow is the only way past a stage whose "
            "automatic completion does not come."
        )
        if is_headless():
            raise PreflightError(
                "this session is headless (pynput did not import), so neither "
                "Esc nor the right arrow can ever fire. A chain drives 22 stages "
                f"back to back with no other human input. {remedy}"
            )
        session_type = os.environ.get("XDG_SESSION_TYPE", "")
        if session_type.lower() == "wayland":
            raise PreflightError(
                "XDG_SESSION_TYPE is 'wayland': pynput imports and the listener "
                "starts, yet no key event is ever delivered to it, so Esc and the "
                "arrows all do nothing silently. A chain drives 22 stages back to "
                f"back with no other human input. {remedy}"
            )
        return
    if not manual_ids:
        return

    named = ", ".join(repr(stage_id) for stage_id in manual_ids)
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

    if config.chain.enabled:
        # A chain run is policy output whatever its stages look like, including
        # reset-only: it is an eval of the runner on the robot and it lands in
        # the same `~/eval_logs` series. Without this branch a reset-only chain
        # (whose stages carry NO policy_path, because a ramp loads no
        # checkpoint) would fall through to sanity_check_dataset_name(repo_id,
        # None) below, which INVERTS the rule and demands a name WITHOUT the
        # prefix -- the exact re-inversion the comment at the end of this
        # function warns about.
        if not dataset_name_part(repo_id).startswith(EVAL_DATASET_PREFIX):
            raise PreflightError(
                f"`dataset.repo_id: {repo_id}` is a chain run, so its name part "
                f"must start with `{EVAL_DATASET_PREFIX}` (lerobot's convention "
                f"for policy output). Rename to "
                f"`<owner>/{EVAL_DATASET_PREFIX}{dataset_name_part(repo_id)}`."
            )
        return

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


def dataset_name_part(repo_id: str) -> str:
    """The part after the single slash. Callers check the slash count first."""
    return repo_id.split("/")[-1]


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
    check_stage_dimensions(
        robot,
        config.stages,
        policy_configs,
        # Both widenings are DECLARED by the config, never guessed from the
        # checkpoint: a run that silently accepted a 27-D state because the
        # checkpoint happened to want one would also silently accept the wrong K.
        onehot_k=config.chain.model.onehot_k if config.chain.enabled else None,
        allow_extra_action=config.chain.enabled,
    )
    check_manual_terminator_is_reachable(config)
    check_dataset_name(config, policy_configs)
    check_dataset_root_is_free(config)
    logger.info(
        f"Preflight passed: {len(config.stages)} stage(s), robot {robot.name!r} "
        f"({robot_state_dimension(robot)}-dim state, "
        f"{robot_action_dimension(robot)}-dim action), dataset "
        f"{config.dataset.repo_id}."
    )
