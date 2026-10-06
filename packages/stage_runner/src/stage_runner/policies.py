"""Seam 2 of 2 to installed lerobot: a checkpoint's config, its policy and its two
processors, loaded and kept together as one inseparable triple.

Every ``lerobot.policies`` import in this package lives here. Loading happens
before ``robot.connect()``, so a cold HF cache never downloads inside a control
loop.
"""

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import PolicyAction, PolicyProcessorPipeline
from lerobot.utils.constants import ACTION, OBS_STATE

from stage_runner.config import StageConfig

logger = logging.getLogger(__name__)

# A policy_path carrying this scheme never touches the hub, the disk or torch
# weights; it resolves to stage_runner.mock_policy instead. lerobot's own mock
# branch is unusable from an installed package -- it does
# `from tests.mocks.mock_robot import MockRobot`, which is a ModuleNotFoundError
# outside a source checkout (measured 2026-09-08).
MOCK_POLICY_SCHEME: str = "mock://"


@dataclass(frozen=True)
class PolicyBundle:
    """One checkpoint's config, policy, preprocessor and postprocessor as a unit.

    Splitting them is the failure this type exists to prevent. Mixing a
    preprocessor from one checkpoint with a policy from another raises nothing
    at all: it unnormalizes with the wrong mean and std and drives the arms to
    wrong joint positions, silently. ``stage_id`` and ``policy_path`` travel
    with the triple so the pairing that actually ran is auditable in the JSONL.

    The annotations describe the production types. ``mock_policy`` puts
    duck-typed stand-ins in these fields on purpose -- ``record_loop`` does no
    isinstance check on any of the three, only ``is not None``.

    ``warmed_up`` records whether :func:`warm_up_bundle` actually completed its
    pass for this bundle, so :func:`bundle_descriptor` can put it in trial_start.
    It defaults to False and load_bundles sets it, which means a bundle built
    anywhere else (mock_policy) reports the truth without being edited: it was
    not warmed.
    """

    stage_id: str
    policy_path: str
    config: PreTrainedConfig
    policy: PreTrainedPolicy
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]]
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction]
    warmed_up: bool = False


def load_policy_config(policy_path: str) -> PreTrainedConfig:
    """Fetch one checkpoint's config.json and mark where it came from.

    ``PreTrainedConfig.from_pretrained`` does NOT set ``pretrained_path``;
    upstream ``record()`` assigns it by hand (lerobot_record.py:237-238). Skip
    that assignment and ``make_policy`` builds RANDOM weights while
    ``make_pre_post_processors`` builds stats-free processors that normalize to
    identity -- both without an error, and the first symptom is wrong joint
    commands on hardware.
    """
    config = PreTrainedConfig.from_pretrained(policy_path)
    config.pretrained_path = policy_path
    return config


def load_policy_configs(stages: Sequence[StageConfig]) -> dict[str, PreTrainedConfig]:
    """One config object per STAGE, keyed by stage id, fetched before connect.

    Two stages sharing a policy_path still get two objects: ``make_policy``
    mutates the config in place (it overwrites ``output_features`` from the
    recording dataset), so a shared object would carry the first stage's
    mutations into the second stage's dimension check.

    Only config.json is fetched here, which is what lets preflight compare the
    checkpoint's own state and action dims against the robot before any
    safetensors download starts.
    """
    configs: dict[str, PreTrainedConfig] = {}
    for stage in stages:
        if stage.policy_path.startswith(MOCK_POLICY_SCHEME):
            # No config.json exists anywhere for a mock path; preflight skips
            # the dimension check for these stages by their absence here.
            continue
        logger.info(f"Loading policy config for stage {stage.id}: {stage.policy_path}")
        configs[stage.id] = load_policy_config(stage.policy_path)
    return configs


def load_bundle(
    stage_id: str,
    policy_path: str,
    policy_config: PreTrainedConfig,
    dataset_meta: LeRobotDatasetMetadata,
) -> PolicyBundle:
    """Build the policy and both processors for one stage. Mock paths never reach here.

    Two deliberate omissions from upstream ``record()``'s call:

    1. ``rename_observations_processor`` is NOT passed. An override key that
       matches no saved step raises KeyError at load, and that step was
       verified present for act_task04 only -- a second checkpoint without it
       would kill the run at load time for no gain, since we do no renaming.
    2. ``dataset_stats`` is NOT passed. For ACT it is inert (only the Groot
       branch of make_pre_post_processors reads it); the normalizer stats come
       from the checkpoint's own safetensors, which is precisely what makes
       swapping N policies against one recording dataset correct.
    """
    policy = make_policy(policy_config, ds_meta=dataset_meta)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_config,
        pretrained_path=policy_config.pretrained_path,
        preprocessor_overrides={"device_processor": {"device": policy_config.device}},
    )
    return PolicyBundle(
        stage_id=stage_id,
        policy_path=policy_path,
        config=policy_config,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
    )


def load_bundles(
    stages: Sequence[StageConfig],
    policy_configs: Mapping[str, PreTrainedConfig],
    dataset_meta: LeRobotDatasetMetadata,
) -> dict[str, PolicyBundle]:
    """Every stage's bundle, keyed by stage id, loaded before robot.connect().

    Called after the dataset exists because ``make_policy`` needs ``ds_meta``.
    Identical policy_paths are loaded twice on purpose: deduplicating into one
    cached bundle is an unconfirmed proposal, and this is the only function it
    would ever go in.

    Each real bundle also gets one warm-up forward here (:func:`warm_up_bundle`),
    because this is the last point before ``robot.connect()`` where a stall is
    free and the alternative is paying it on the first recorded frame of a
    stage, inside the hertz the operator reads to set the timers.
    """
    bundles: dict[str, PolicyBundle] = {}
    for stage in stages:
        if stage.policy_path.startswith(MOCK_POLICY_SCHEME):
            # Imported here, not at module top: mock_policy imports PolicyBundle
            # from this module, so a top-level import is a cycle.
            from stage_runner.mock_policy import make_mock_bundle

            bundles[stage.id] = make_mock_bundle(
                stage.id, stage.policy_path, dataset_meta
            )
            continue
        logger.info(f"Loading policy for stage {stage.id}: {stage.policy_path}")
        bundle = load_bundle(
            stage.id, stage.policy_path, policy_configs[stage.id], dataset_meta
        )
        # Mock bundles are deliberately NOT warmed: they run no kernels, so
        # there is no first-call cost to move, and their duck-typed policy is
        # not obliged to accept a tensor batch.
        #
        # The RESULT is kept, not discarded. warm_up_bundle returns False for the
        # cases it swallows -- no input_features to shape a batch from, or a
        # policy that refuses the shaped batch -- and those are exactly the runs
        # where the first recorded frame of the stage still pays first-inference
        # CUDA/cuDNN init, which depresses that stage's hertz. hertz is the number
        # the MANDATORY FIRST-RUN PROCEDURE reads to set the timeouts, so whether
        # the warm-up ran has to be readable next to it in events.jsonl. Until
        # this line it reached the console only, and init_logging runs with
        # log_file=None, so it was gone the moment the terminal scrolled.
        bundles[stage.id] = replace(bundle, warmed_up=warm_up_bundle(bundle))
    return bundles


def load_chain_bundles(
    stages: Sequence[StageConfig],
    policy_config: PreTrainedConfig | None,
    dataset_meta: LeRobotDatasetMetadata,
    *,
    onehot_k: int | None,
    initial_stage: int,
    n_action_steps: int | None = None,
) -> tuple[dict[str, PolicyBundle], Any | None]:
    """ONE checkpoint for every policy stage of a chain, plus its one-hot step.

    Returns ``(bundles keyed by stage id, the TaskOneHotStep or None)``.

    ONE, not eleven. The design is a single multi-stage ACT conditioned on a
    stage one-hot, so eleven copies would be eleven times the weights and eleven
    CUDA contexts' worth of memory for the same network -- and ``load_bundles``'s
    rule that two stages sharing a ``policy_path`` still get two objects exists
    for the OPPOSITE case (``make_policy`` mutates the config in place, so two
    DIFFERENT checkpoints must not share one). Here there is one config, mutated
    once, and every stage points at the same triple.

    The one-hot step is installed into that one preprocessor at
    ``initial_stage`` and re-pointed per stage by the executor
    (``TaskOneHotStep.set_stage``). That re-pointing IS the per-stage
    conditioning; there is nothing else to switch.

    ``n_action_steps`` is applied to the checkpoint's config before
    ``make_policy``, which is what ``lerobot-record --policy.n_action_steps=``
    does. 30 is the exec confirmed on 2026-10-02.

    The one-hot import is LAZY AND CONDITIONAL on ``onehot_k``:
    ``lerobot_robot_trossen.task_onehot_patch`` cannot be imported without
    importing that package's ``__init__``, which imports ``mobileai`` and
    through it the ``trossen_slate`` SDK. A mock chain (``onehot_k: null``) must
    never reach that import -- it is the rule that keeps this package testable
    off the robot (CLAUDE.md 실기 안전).
    """
    if n_action_steps is not None and policy_config is not None:
        previous = getattr(policy_config, "n_action_steps", None)
        policy_config.n_action_steps = int(n_action_steps)
        logger.info(
            f"chain: n_action_steps {previous} -> {n_action_steps} "
            "(checkpoint config override, as --policy.n_action_steps does)"
        )

    policy_paths = {stage.policy_path for stage in stages if stage.policy_path}
    if len(policy_paths) > 1:
        raise ValueError(
            f"a chain runs ONE checkpoint, but the stages name {sorted(policy_paths)}. "
            "Set chain.model.policy_path and let config.expand_chain fill the stages."
        )
    policy_path = next(iter(policy_paths), "")

    if policy_path.startswith(MOCK_POLICY_SCHEME):
        # The hardware-free chain. No checkpoint exists, so there is nothing to
        # load, nothing to warm and -- crucially -- no config.json whose
        # observation.state width the one-hot step could be validated against,
        # which is why onehot_k is refused for a mock path (preflight says the
        # same thing with an actionable message).
        from stage_runner.mock_policy import make_mock_bundle

        if onehot_k:
            raise ValueError(
                f"chain.model.policy_path {policy_path!r} is a mock and "
                f"chain.model.onehot_k is {onehot_k}. The one-hot step asserts "
                "the checkpoint declares a 16+K state, and a mock declares "
                "whatever the dataset does -- so installing it would assert "
                "nothing. Set onehot_k: null for a mock chain."
            )
        bundle = make_mock_bundle("chain", policy_path, dataset_meta)
    else:
        bundle = load_bundle(
            stage_id="chain",
            policy_path=policy_path,
            policy_config=policy_config,
            dataset_meta=dataset_meta,
        )
        bundle = replace(bundle, warmed_up=warm_up_bundle(bundle))

    onehot = None
    if onehot_k:
        from lerobot_robot_trossen.task_onehot_patch import insert_task_onehot

        onehot = insert_task_onehot(
            bundle.preprocessor, policy_config, initial_stage, int(onehot_k)
        )
        logger.info(
            f"chain: stage one-hot installed before the normalizer, K={onehot_k}, "
            f"initial stage {initial_stage}. Every stage re-points it; the "
            "'stage i/K active' line is re-emitted each time and its ABSENCE "
            "means the one-hot did not switch -- stop the run."
        )

    bundles = {
        stage.id: replace(bundle, stage_id=stage.id)
        for stage in stages
        if stage.policy_path
    }
    return bundles, onehot


def warm_up_bundle(bundle: PolicyBundle) -> bool:
    """One throwaway forward so the first RECORDED frame does not pay for it.

    ``make_policy`` constructs the network and moves it to the device but never
    forwards, so without this each bundle's first ``select_action`` -- which
    happens inside ``record_loop``, on frame 1 of that bundle's stage -- pays
    first-call CUDA kernel load and cuDNN algorithm selection for the ACT
    backbone at its first input shape. That time lands inside the stage's
    ``elapsed_s`` (executors.py) while producing one frame, so it DEPRESSES that
    stage's frames/elapsed_s -- and the per-stage ``hertz`` in events.jsonl is
    the single number the MANDATORY FIRST-RUN PROCEDURE has the operator read
    before setting the 12.9 / 20.8 s timeouts. The bias is largest on the
    SHORTEST stage, which is exactly the one the timer question is sharpest for.
    Every bundle is warmed, not only the first: only the process-wide CUDA
    context is shared, and each policy's own first forward is its own cost.

    Called from load_bundles, which runs BEFORE ``robot.connect()``, so a stall
    here happens while nothing is energised -- the same window that already
    accepts a cold HF cache download.

    NEVER RAISES, and returns whether the pass actually ran. A warm-up is an
    optimisation: a shape this construction gets wrong, a policy that wants a
    key ``input_features`` does not declare (a language-conditioned checkpoint
    wanting ``task``), or a device that rejects the allocation must all cost a
    log line and nothing else. The real ``select_action`` in record_loop is
    where such a problem legitimately surfaces.
    """
    features = getattr(bundle.config, "input_features", None)
    if not features:
        logger.info(
            f"stage {bundle.stage_id}: no input_features to shape a warm-up batch "
            f"from; the first inference cost stays inside the stage"
        )
        return False
    device = str(getattr(bundle.config, "device", "") or "cpu")
    try:
        batch: dict[str, Any] = {}
        for key, feature in features.items():
            shape = tuple(getattr(feature, "shape", ()) or ())
            if not shape:
                raise ValueError(f"input feature {key!r} declares no shape")
            # Zeros in NORMALIZED space: normalization lives in the
            # preprocessor pipeline (make_pre_post_processors), not in the
            # policy, so select_action's input is already normalized and zeros
            # are a valid point in it. The VALUES are irrelevant here -- only
            # the shapes and dtypes decide which kernels get compiled.
            batch[key] = torch.zeros((1, *shape), dtype=torch.float32, device=device)
        with torch.inference_mode():
            bundle.policy.select_action(batch)
    except Exception as error:  # optimisation only: see the docstring
        logger.warning(
            f"stage {bundle.stage_id}: warm-up forward skipped ({error!r}). The "
            f"first frame of this stage pays the first-inference cost, which "
            f"depresses that stage's hertz in events.jsonl"
        )
        return False
    finally:
        # ALWAYS, including after a failure part-way through: ACT's
        # select_action fills an action-chunk queue, and a queue left holding
        # actions inferred from ZEROS would be popped by the first real frames
        # of the stage. record_loop's entry also resets (lerobot_record.py:331-335),
        # so this is belt and braces -- but the belt is what keeps a warm-up
        # from ever commanding the arms.
        reset = getattr(bundle.policy, "reset", None)
        if callable(reset):
            reset()
    logger.info(f"stage {bundle.stage_id}: warm-up forward done")
    return True


def bundle_descriptor(bundle: PolicyBundle) -> dict[str, Any]:
    """The JSON-safe record of what a stage actually ran, for the trial_start event.

    These dims are read AFTER make_policy, so they are the effective run dims,
    not the checkpoint's own -- preflight already compared those before any
    weights downloaded. ``temporal_ensemble_coeff`` is RECORDED, not enforced:
    P1 runs with temporal ensembling off and the 60k ACT checkpoints carry null
    already, so this field is where a run that silently turned it on becomes
    visible afterwards.

    ``warmed_up`` is the one field here that is about the RUN rather than the
    checkpoint, and it belongs with them because it qualifies a measurement: when
    it is false, this stage's first recorded frame paid first-inference CUDA
    kernel load and cuDNN algorithm selection, so the stage's ``hertz`` in
    events.jsonl is biased LOW -- worst on the shortest stage, which is the one
    the timer question is sharpest for. Reading that hertz to set the 12.9 / 20.8
    s timeouts without knowing this is how a timer gets set from a number that
    measured the warm-up. Mock bundles report false and mean it: they are not
    warmed, and having no kernels to load they need no warming either.

    Every value goes through getattr because mock bundles carry a duck-typed
    config, and a missing attribute here must not kill a real run's logging.
    """
    config = bundle.config
    return {
        "stage_id": bundle.stage_id,
        "policy_path": bundle.policy_path,
        # Not getattr'd: warmed_up is PolicyBundle's own field with a default, so
        # every bundle has it, mock ones included.
        "warmed_up": bundle.warmed_up,
        "device": str(getattr(config, "device", "") or ""),
        "state_dim": _feature_dimension(
            getattr(config, "input_features", None), OBS_STATE
        ),
        "action_dim": _feature_dimension(
            getattr(config, "output_features", None), ACTION
        ),
        "temporal_ensemble_coeff": getattr(config, "temporal_ensemble_coeff", None),
        "n_action_steps": getattr(config, "n_action_steps", None),
    }


def _feature_dimension(features: Mapping[str, Any] | None, key: str) -> int | None:
    """First axis of one PolicyFeature's shape, or None when it is not declared.

    A near-twin of ``preflight._feature_dimension``, and deliberately a second
    copy: preflight has to stay importable without torch, which importing this
    module would drag in. Both read the same field for the same purpose -- one
    GATES the run, one RECORDS it in trial_start -- so a change to the
    shape-reading rule belongs in both, or the log reports a dimension the gate
    never checked.
    """
    if not features:
        return None
    shape = getattr(features.get(key), "shape", None)
    if not shape:
        return None
    return int(shape[0])
