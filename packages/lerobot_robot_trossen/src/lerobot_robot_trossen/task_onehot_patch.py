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

- malformed value, or a checkpoint whose ``observation.state`` is not ``16+K`` wide:
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
from lerobot.processor.pipeline import ProcessorStep

logger = logging.getLogger(__name__)

_ENV_VAR = "LEROBOT_TASK_ONEHOT"
_STATE_KEY = "observation.state"
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

    def __init__(self, stage: int, k: int, expected_dim: int):
        self.index = stage - 1
        self.k = k
        self.expected_dim = expected_dim
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
            )
            self._announced = True
        obs = dict(obs)
        obs[_STATE_KEY] = new_state
        transition = transition.copy()
        transition[TransitionKey.OBSERVATION] = obs
        return transition

    # ProcessorStep protocol -- this step is never serialized with the checkpoint.
    def get_config(self):
        return {"stage": self.index + 1, "k": self.k}

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        pass

    def reset(self):
        pass

    def transform_features(self, features):
        # Feature bookkeeping is the checkpoint's: it already declares the widened state.
        return features


def _expected_state_dim(policy_cfg) -> int | None:
    ft = (getattr(policy_cfg, "input_features", None) or {}).get(_STATE_KEY)
    shape = getattr(ft, "shape", None)
    return int(shape[0]) if shape else None


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
    """
    expected = _expected_state_dim(policy_cfg)
    if expected != 16 + k:
        raise RuntimeError(
            f"{_ENV_VAR}: checkpoint declares observation.state={expected}, "
            f"but stage one-hot needs 16+{k}={16 + k}. Wrong checkpoint or wrong K."
        )
    if not 1 <= stage <= k:
        raise RuntimeError(f"{_ENV_VAR}: stage must be in 1..{k}")
    return _insert(preprocessor, TaskOneHotStep(stage, k, expected))


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
            expected = _expected_state_dim(policy_cfg)
            if expected != 16 + k:
                raise RuntimeError(
                    f"{_ENV_VAR}={value}: checkpoint declares observation.state={expected}, "
                    f"but stage one-hot needs 16+{k}={16 + k}. Wrong checkpoint or wrong K."
                )
            _insert(pre, TaskOneHotStep(stage, k, expected))
            logger.info(
                f"{_ENV_VAR}={value}: one-hot step inserted before the normalizer"
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
