"""Feed a multi-stage ACT its stage number (off by default, opt in per run).

Why this exists
---------------
``trossen-ai-simulation`` trains one ACT over all 11 stages (task01..task11).
ACT has no language input, so the stage is given to it as a one-hot appended
after the 16-D state (``scripts/train_multi.py:apply_task_onehot``), and the two
base slots of the state (``x.vel``, ``theta.vel``, indices 14 and 15) are
trained as zeros (``base_state: zero``). The checkpoint therefore declares
``observation.state`` as ``16 + K`` wide (27 for K=11), while the robot emits
14 values (``include_base_in_state=false``, the default) or 16 (``true``).
Nothing on ``main`` builds the remaining columns, so such a checkpoint dies in
the normalizer on the first frame.

This module inserts one step into the policy preprocessor, immediately before
the normalizer:

    robot state (14 or 16)  ->  [arms 0..13, 0, 0, one_hot(i, K)]  (16 + K)

Round 2: the same one-hot, also as its own encoder token
-------------------------------------------------------
The second training round feeds the stage to ACT *twice*: still as the tail of
the 27-D state, and additionally as a separate encoder token. Such a checkpoint
declares one extra input feature::

    "observation.environment_state": {"type": "ENV", "shape": [11]}

and ``ACT`` tokenizes ``batch["observation.environment_state"]`` through
``encoder_env_state_input_proj`` (``modeling_act.py:344-346, 465-466``). The
tensor is the SAME one-hot vector as the state tail -- not a second signal -- so
this step writes both from one ``hot`` tensor and ``set_stage`` moves both at
once. ``FeatureType.ENV`` is absent from ACT's ``normalization_mapping``
(``configuration_act.py:89-95``), so the normalizer leaves that key untouched
(``normalize_processor.py:305`` falls back to ``IDENTITY``).

A round-1 checkpoint (M1/M2/M3, ``tph``) declares no ENV feature and the key is
then NOT added -- behaviour is bit-for-bit what it was. Nothing else changes:
``observation.state`` is still 16+K wide in both rounds.

The base slots are always written as exact zeros -- measured base velocity is
never fed to the policy, because the training data had ``action == state`` on
those slots (a teleop echo) and a policy that sees them learns to copy them.

The eval dataset keeps the robot's own 14/16-D state; only the policy input is
widened. Use ``LEROBOT_BASE_VEL_LOG`` if you need the measured base velocity.

Usage
-----
One stage per ``lerobot-record`` run. The stage number is 1-based as in the
dataset names (task01 = 1)::

    LEROBOT_TASK_ONEHOT=2/11 uv run lerobot-record ... --policy.path=<all11 ckpt> \\
        --policy.n_action_steps=5

When it fails, and how far the robot has got by then:

- malformed value, a checkpoint whose ``observation.state`` is not ``16+K`` wide, or
  one whose ``observation.environment_state`` is not ENV-typed or not K wide:
  raised from ``make_pre_post_processors``, which ``lerobot-record`` calls *before*
  ``robot.connect()`` -- nothing has moved.
- robot state neither 14 nor 16 wide (e.g. ``include_velocity``/``include_effort`` on):
  raised on the first frame, i.e. *after* ``robot.connect()`` (the arms have done their
  connect motion) but before any policy action is sent.
- the patch could not be installed: a warning only (an exception here would unregister
  the whole plugin), and the run then dies in the normalizer on the first frame. **If the
  ``installed`` line below is missing from the log, stop the run.**

Only the ``lerobot-record`` entry point is covered. Callers that import
``make_pre_post_processors`` under their own name (``stage_runner``) or run
``python -m lerobot.scripts.lerobot_record`` are not; call ``insert_task_onehot`` there.

Layout: training data recorded with ``mobileai_robot`` lists the state as the 14 arm
values (left 6 + carriage, right 6 + carriage) followed by ``x.vel``, ``theta.vel`` --
the same order ``MobileAIRobot.observation_features`` emits -- so columns 0..13 are
taken as-is. A checkpoint trained on base-first data (the MuJoCo teleop layout) would
pass the width check and be fed wrong columns; do not use this with such checkpoints.

To confirm a run used it::

    grep LEROBOT_TASK_ONEHOT <run log>
"""

import logging
import os
import re

import torch
from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor.pipeline import ProcessorStep

logger = logging.getLogger(__name__)

_ENV_VAR = "LEROBOT_TASK_ONEHOT"
_STATE_KEY = "observation.state"
# ``lerobot.utils.constants.OBS_ENV_STATE``, spelled out like _STATE_KEY above.
# This exact literal is what ACT reads off the batch (modeling_act.py:466), so
# the key is matched BY NAME and not by "the first ENV-typed input feature":
# a checkpoint that declared an ENV feature under some other key would get a
# token the model never looks up.
_ENV_STATE_KEY = "observation.environment_state"
_ARM_DIMS = 14
_BASE_DIMS = 2


def _parse(value: str) -> tuple[int, int]:
    m = re.fullmatch(r"\s*(\d+)\s*/\s*(\d+)\s*", value)
    if not m:
        raise RuntimeError(
            f"{_ENV_VAR}={value!r}: expected '<stage>/<K>', e.g. '2/11' (1-based stage)"
        )
    stage, k = int(m.group(1)), int(m.group(2))
    if not 1 <= stage <= k:
        raise RuntimeError(f"{_ENV_VAR}={value!r}: stage must be in 1..{k}")
    return stage, k


class TaskOneHotStep(ProcessorStep):
    """Widen ``observation.state`` to ``[arms, 0, 0, one_hot]``. Placed before the normalizer."""

    def __init__(
        self, stage: int, k: int, expected_dim: int, env_k: int | None = None
    ):
        if env_k is not None and env_k != k:
            # Defensive: _build_step already refuses this, but the step is also
            # constructed directly (tests, future callers) and a mismatch here
            # would feed the encoder a token of the wrong width.
            raise RuntimeError(
                f"{_ENV_VAR}: {_ENV_STATE_KEY} is {env_k} wide but the one-hot is "
                f"K={k}; the env token IS the one-hot, so the widths must match"
            )
        self.index = stage - 1
        self.k = k
        self.expected_dim = expected_dim
        # None == a round-1 checkpoint: the key is then never added, so such a
        # run is byte-for-byte what it was before round 2 existed.
        self.env_k = env_k
        self._announced = False

    def set_stage(self, stage: int) -> None:
        """Re-point the one-hot at another stage, for a chain inside ONE process.

        ``lerobot-record`` runs one stage per process and reads the stage from
        the environment once, so the index was fixed at construction. The chain
        runner (``stage_runner``) re-enters ``record_loop`` for every stage of a
        single episode against ONE loaded checkpoint, and the one-hot is the only
        thing that differs between those entries.

        ``_announced`` is cleared on purpose: the "stage i/K active" line is the
        only evidence in the log that the policy was fed the one-hot the operator
        asked for, and a chain that announced stage 1 and then silently ran
        stages 2..11 would be indistinguishable from one that never switched. The
        line is emitted once per stage, not once per run.

        Both places the one-hot appears move together: ``__call__`` builds the
        state tail and the ``observation.environment_state`` token from the SAME
        ``hot`` tensor, so there is no second index to re-point and no way for
        the two to disagree.

        Deliberately NOT done in ``reset()``: ``record_loop`` resets the
        preprocessor on entry to every stage (lerobot_record.py:332-335), so a
        reset that re-pointed the index would undo the caller's choice right
        after it was made. ``reset()`` stays a no-op.
        """
        if not 1 <= stage <= self.k:
            raise RuntimeError(
                f"{_ENV_VAR}: stage must be in 1..{self.k}, got {stage}"
            )
        self.index = stage - 1
        self._announced = False

    def __call__(self, transition):
        from lerobot.processor.core import TransitionKey

        obs = transition.get(TransitionKey.OBSERVATION)
        if not isinstance(obs, dict) or _STATE_KEY not in obs:
            return transition
        state = obs[_STATE_KEY]
        width = state.shape[-1]
        if width not in (_ARM_DIMS, _ARM_DIMS + _BASE_DIMS):
            raise RuntimeError(
                f"{_ENV_VAR}: robot state is {width} wide; expected 14 (include_base_in_state=false) "
                f"or 16 (true)"
            )
        arms = state[..., :_ARM_DIMS]
        base = torch.zeros(
            *state.shape[:-1], _BASE_DIMS, dtype=state.dtype, device=state.device
        )
        hot = torch.zeros(
            *state.shape[:-1], self.k, dtype=state.dtype, device=state.device
        )
        hot[..., self.index] = 1.0
        new_state = torch.cat([arms, base, hot], dim=-1)
        if new_state.shape[-1] != self.expected_dim:
            raise RuntimeError(
                f"{_ENV_VAR}: built {new_state.shape[-1]}-D state, checkpoint expects "
                f"{self.expected_dim}"
            )
        if not self._announced:
            logger.info(
                f"{_ENV_VAR}: stage {self.index + 1}/{self.k} active -- policy state "
                f"{width} -> {new_state.shape[-1]} (base slots zeroed)"
                + (" (+env token)" if self.env_k is not None else "")
            )
            self._announced = True
        obs = dict(obs)
        obs[_STATE_KEY] = new_state
        if self.env_k is not None:
            # The SAME vector as the state tail, as its own encoder token
            # (modeling_act.py:466). Cloned rather than aliased to the slice that
            # went into `cat`: an in-place step downstream must not be able to
            # move one copy without the other.
            obs[_ENV_STATE_KEY] = hot.clone()
        transition = transition.copy()
        transition[TransitionKey.OBSERVATION] = obs
        return transition

    # ProcessorStep protocol -- this step is never serialized with the checkpoint.
    def get_config(self):
        return {"stage": self.index + 1, "k": self.k, "env_k": self.env_k}

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        pass

    def reset(self):
        pass

    def transform_features(self, features):
        """Declare the env token; say nothing about ``observation.state``.

        ``observation.state`` is deliberately left alone: the checkpoint already
        declares it 16+K wide, and this step only fills the columns that
        declaration promised. The env token is different -- it is a key that did
        not exist upstream of this step -- so it is declared here, the way
        ``tokenizer_processor.py:296-320`` declares the keys it adds.

        NOTHING IN THE 0.4.4 RECORD PATH READS THIS. The only caller of
        ``transform_features`` is ``pipeline_features.aggregate_pipeline_dataset_features``
        (``pipeline_features.py:91``), and both ``record()``
        (``lerobot_record.py:449-462``) and this repo's chain
        (``record_adapter.py:229-238``) call it on the ROBOT pipelines
        (``teleop_action`` / ``robot_observation``), never on the policy
        preprocessor this step lives in. So the env token never reaches
        ``dataset.features`` and no column is added to the recording -- which is
        the wanted outcome: the observation features are the robot's, and this
        step puts the token in the BATCH only.

        Returned as fresh dicts (the pipeline hands the same object to every
        step, ``pipeline.py:1332-1336``), and a missing OBSERVATION bucket is
        tolerated rather than created.
        """
        if self.env_k is None:
            return features
        observation = (features or {}).get(PipelineFeatureType.OBSERVATION)
        if observation is None or _ENV_STATE_KEY in observation:
            return features
        features = dict(features)
        observation = dict(observation)
        observation[_ENV_STATE_KEY] = PolicyFeature(
            type=FeatureType.ENV, shape=(self.env_k,)
        )
        features[PipelineFeatureType.OBSERVATION] = observation
        return features


def _expected_state_dim(policy_cfg) -> int | None:
    ft = (getattr(policy_cfg, "input_features", None) or {}).get(_STATE_KEY)
    shape = getattr(ft, "shape", None)
    return int(shape[0]) if shape else None


def _env_state_dim(policy_cfg) -> int | None:
    """K declared by ``input_features['observation.environment_state']``, or None.

    None means a ROUND-1 checkpoint (M1/M2/M3, ``tph``): no env token, nothing
    added to the batch, behaviour unchanged.

    Looked up BY KEY, not via ``PreTrainedConfig.env_state_feature``
    (``configs/policies.py:139-145``), which returns the first ENV-typed feature
    under any key -- while ACT only ever reads this one literal
    (``modeling_act.py:466``). The type is still asserted, because an ENV-shaped
    feature declared as STATE would be MEAN_STD-normalized by the normalizer and
    the token would silently stop being a one-hot. ``FeatureType`` subclasses
    ``str``, so this also works on a config.json read back as plain strings.
    """
    ft = (getattr(policy_cfg, "input_features", None) or {}).get(_ENV_STATE_KEY)
    if ft is None:
        return None
    ftype = getattr(ft, "type", None)
    name = getattr(ftype, "value", ftype)
    if name != FeatureType.ENV.value:
        raise RuntimeError(
            f"{_ENV_VAR}: checkpoint declares {_ENV_STATE_KEY} with type {name!r}, "
            f"expected {FeatureType.ENV.value!r}. ACT only tokenizes it when the "
            "feature is ENV-typed, and any other type gets normalized"
        )
    shape = getattr(ft, "shape", None)
    if not shape:
        raise RuntimeError(
            f"{_ENV_VAR}: checkpoint declares {_ENV_STATE_KEY} with no shape, so "
            "the env token width is unknown"
        )
    return int(shape[0])


def _build_step(policy_cfg, stage: int, k: int) -> "TaskOneHotStep":
    """The one place the checkpoint is validated and the step is constructed.

    Both entry points go through it -- ``insert_task_onehot`` (chain runner) and
    the ``make_pre_post_processors`` wrapper (``lerobot-record``) -- so there is
    one set of refusals rather than two that can drift apart.
    """
    expected = _expected_state_dim(policy_cfg)
    if expected != 16 + k:
        raise RuntimeError(
            f"{_ENV_VAR}: checkpoint declares observation.state={expected}, "
            f"but stage one-hot needs 16+{k}={16 + k}. Wrong checkpoint or wrong K."
        )
    if not 1 <= stage <= k:
        raise RuntimeError(f"{_ENV_VAR}: stage must be in 1..{k}")
    env_k = _env_state_dim(policy_cfg)
    if env_k is not None and env_k != k:
        raise RuntimeError(
            f"{_ENV_VAR}: checkpoint declares {_ENV_STATE_KEY}={env_k} but the "
            f"one-hot is K={k}. The env token IS the state tail, so a different "
            "width means this is not the checkpoint you think it is."
        )
    return TaskOneHotStep(stage, k, expected, env_k=env_k)


def _insert(preprocessor, step):
    from lerobot.processor.normalize_processor import NormalizerProcessorStep

    steps = list(preprocessor.steps)
    at = next(
        (i for i, s in enumerate(steps) if isinstance(s, NormalizerProcessorStep)), None
    )
    if at is None:
        raise RuntimeError(
            f"{_ENV_VAR}: preprocessor has no normalizer step; refusing to guess a position"
        )
    steps.insert(at, step)
    preprocessor.steps = steps
    return step


def insert_task_onehot(preprocessor, policy_cfg, stage: int, k: int):
    """Insert the step into an already-built preprocessor (for callers other than lerobot-record).

    RETURNS THE STEP. ``lerobot-record`` never needs the handle -- one stage per
    process -- but the chain runner holds it and calls
    :meth:`TaskOneHotStep.set_stage` before each stage's ``record_loop``, which
    is the only way one loaded checkpoint can serve all 11 stages in a single
    episode. Returning it also means the caller can assert the step is installed
    instead of inferring it from a log line that may not have been flushed yet.

    A round-2 checkpoint (one that declares ``observation.environment_state``)
    needs no different call: the step reads the ENV feature off ``policy_cfg``
    and starts writing that key too. ``step.env_k`` is the handle for a caller
    that wants to log which round it got.
    """
    return _insert(preprocessor, _build_step(policy_cfg, stage, k))


def apply_task_onehot_patch() -> bool:
    """Install the preprocessor wrapper if the variable is set; return True if installed.

    Never raises at import: lerobot imports this package by name prefix and treats any
    exception as a failed plugin, which would take the ``mobileai_robot`` registration
    down with it. A malformed value or a wrong checkpoint is reported from
    ``make_pre_post_processors`` instead -- ``lerobot_record`` builds the processors
    *before* ``robot.connect()``, so the error lands before anything moves.
    """
    value = os.environ.get(_ENV_VAR)
    if not value:
        return False
    try:
        from lerobot.scripts import lerobot_record

        original = lerobot_record.make_pre_post_processors
        if getattr(original, "_task_onehot_wrapped", False):
            return True
        try:
            parsed, parse_error = _parse(value), None
        except RuntimeError as exc:
            parsed, parse_error = None, exc

        def wrapped(policy_cfg, *args, **kwargs):
            if parse_error is not None:
                raise parse_error
            stage, k = parsed
            pre, post = original(policy_cfg, *args, **kwargs)
            step = _insert(pre, _build_step(policy_cfg, stage, k))
            logger.info(
                f"{_ENV_VAR}={value}: one-hot step inserted before the normalizer"
                + (
                    f" (+env token, {_ENV_STATE_KEY}={step.env_k})"
                    if step.env_k is not None
                    else ""
                )
            )
            return pre, post

        wrapped._task_onehot_wrapped = True
        wrapped.__wrapped__ = original
        lerobot_record.make_pre_post_processors = wrapped
        print(f"[lerobot_robot_trossen] {_ENV_VAR}={value} installed", flush=True)
        return True
    except Exception as exc:  # never break plugin registration
        logger.warning(f"{_ENV_VAR} set but the patch could not be installed: {exc!r}")
        return False
