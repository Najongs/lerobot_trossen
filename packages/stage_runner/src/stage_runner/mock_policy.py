"""Hardware-free, download-free policy triple for the smoke path.

``record_loop`` runs no isinstance check on the policy or its two processors --
it only tests ``is not None`` on all three (lerobot_record.py:332, :357) and
then reaches for ``policy.config.device``, ``policy.config.use_amp``,
``policy.reset()`` and ``policy.select_action(batch)`` through
``predict_action`` (control_utils.py:100-115). Duck typing therefore suffices:
the ``PolicyBundle`` field annotations are documentation, not enforcement, and
nothing here loads a checkpoint, touches the hub or builds a torch.nn.Module.

The returned action tensor is a REAL torch tensor because
``make_robot_action`` does ``.squeeze(0)`` and ``.to("cpu")`` on it
(policies/utils.py:194-195).
"""

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_STATE

if TYPE_CHECKING:
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    from stage_runner.policies import PolicyBundle

logger = logging.getLogger(__name__)

MOCK_POLICY_HOLD: str = "hold"
MOCK_POLICY_SINE: str = "sine"
MOCK_POLICY_NAMES: tuple[str, ...] = (MOCK_POLICY_HOLD, MOCK_POLICY_SINE)

# Small and slow on purpose: the smoke stages are one second long, and a sweep
# large enough to be visible per frame would look like a runaway command in the
# recorded observation.state rather than a mock.
_SINE_AMPLITUDE_RAD: float = 0.05
_SINE_FREQUENCY_HZ: float = 0.2


@dataclass
class MockPolicyConfig:
    """The subset of ``PreTrainedConfig`` that the runner and record_loop read.

    ``device`` is pinned to "cpu": ``get_safe_torch_device`` answers anything
    starting with "cuda" through a bare ``assert torch.cuda.is_available()``
    (utils/utils.py:60-61), which surfaces as an AssertionError with no message
    from inside predict_action on a machine without a GPU.
    """

    device: str = "cpu"
    use_amp: bool = False
    # Recorded into the trial_start event by policies.bundle_descriptor. A mock
    # keeps the shape of a 60k ACT checkpoint: temporal ensembling off, one
    # action consumed per step.
    n_action_steps: int | None = 1
    temporal_ensemble_coeff: float | None = None
    pretrained_path: str | None = None
    input_features: dict[str, PolicyFeature] = field(default_factory=dict)
    output_features: dict[str, PolicyFeature] = field(default_factory=dict)


class MockPolicy:
    """Emits an action built from the observation instead of from weights.

    Both modes hold every arm joint at the position the observation reports and
    force the base velocity keys to 0.0, so the mock never commands the base and
    the only motion in a smoke run is the one joint ``sine`` sweeps. The mapping
    is by NAME, not by index, because observation.state drops the base keys when
    ``include_base_in_state`` is false while the action never does.

    ``hold`` TRACKS THE PRESENT POSITION rather than anchoring to the first
    observation after reset, which is a wording divergence from the spec
    ("repeats the first observation's arm positions"). It is the same trajectory
    here: the mock arm is position-controlled, so commanding the present position
    every frame and commanding frame 0's position every frame land on the same
    values. Only ``sine`` anchors, because it must not integrate its own offset.
    If the mock ever gains drift, this is the line that has to change for
    ``hold`` to mean "pull back to the frame-0 pose".
    """

    def __init__(
        self,
        config: MockPolicyConfig,
        *,
        action_names: list[str],
        state_names: list[str],
        mode: str,
        fps: int,
    ) -> None:
        self.config = config
        self._mode = mode
        self._fps = fps
        # None for an action key with no observation counterpart -- in practice
        # x.vel / theta.vel, which are commanded but not always observed.
        self._state_index: list[int | None] = [
            state_names.index(name) if name in state_names else None
            for name in action_names
        ]
        self._sweep_index: int | None = next(
            (index for index, name in enumerate(action_names) if name.endswith(".pos")),
            None,
        )
        self._step = 0
        self._sweep_anchor: float | None = None

    def reset(self) -> None:
        # record_loop calls this on entry to EVERY stage, which is what flushes a
        # real ACT policy's leftover action chunk. Restarting the sweep here is
        # what makes that per-stage reset observable in the recorded data.
        self._step = 0
        self._sweep_anchor = None

    def select_action(self, batch: dict[str, Any]) -> torch.Tensor:
        state = batch[OBS_STATE].squeeze(0).tolist()
        values = [
            0.0 if index is None else float(state[index]) for index in self._state_index
        ]
        if self._mode == MOCK_POLICY_SINE and self._sweep_index is not None:
            # Anchored to the position seen on the FIRST frame after reset, not
            # added to the current one: the arm follows every command, so an
            # offset applied to the observation each frame integrates into an
            # unbounded ramp (0.05 rad amplitude reached 0.135 rad in 12 frames
            # before this anchor existed, and a 300 s manual ceiling would keep
            # it going). The sweep must stay inside +/-amplitude of one point.
            if self._sweep_anchor is None:
                self._sweep_anchor = values[self._sweep_index]
            phase = 2.0 * math.pi * _SINE_FREQUENCY_HZ * self._step / self._fps
            values[self._sweep_index] = (
                self._sweep_anchor + _SINE_AMPLITUDE_RAD * math.sin(phase)
            )
        self._step += 1
        # [1, action_dim]: make_robot_action squeezes the batch dimension back
        # off and zips what is left against ds_features[ACTION]["names"].
        return torch.tensor([values], dtype=torch.float32)


class MockProcessor:
    """Identity stand-in for a ``PolicyProcessorPipeline``.

    record_loop calls ``preprocessor(observation)`` / ``postprocessor(action)``
    and ``.reset()`` on both, and nothing else.
    """

    def __call__(self, value: Any) -> Any:
        return value

    def reset(self) -> None:
        return None


def make_mock_bundle(
    stage_id: str, policy_path: str, dataset_meta: "LeRobotDatasetMetadata"
) -> "PolicyBundle":
    """Build the policy triple for a ``mock://<name>`` stage.

    Feature names come from the dataset metadata, so the mock always matches
    whatever robot created the dataset and can never disagree with it about
    dimensions.
    """
    # Imported here rather than at module scope: policies.py is the module that
    # dispatches mock:// paths into this one, so a top-level import back into it
    # is a cycle whenever policies.py is the first of the two to be imported.
    from stage_runner.policies import PolicyBundle

    mode = policy_path.split("://", 1)[-1]
    if mode not in MOCK_POLICY_NAMES:
        raise ValueError(
            f"Stage '{stage_id}' asks for unknown mock policy '{policy_path}'. "
            f"Known names: {', '.join(MOCK_POLICY_NAMES)}."
        )

    action_names = list(dataset_meta.features[ACTION]["names"])
    state_names = list(dataset_meta.features[OBS_STATE]["names"])
    config = MockPolicyConfig(
        pretrained_path=policy_path,
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(len(state_names),))
        },
        output_features={
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(len(action_names),))
        },
    )
    policy = MockPolicy(
        config,
        action_names=action_names,
        state_names=state_names,
        mode=mode,
        fps=dataset_meta.fps,
    )
    logger.info(
        f"Stage '{stage_id}' uses mock policy '{mode}' "
        f"({len(state_names)}-dim state -> {len(action_names)}-dim action)"
    )
    return PolicyBundle(
        stage_id=stage_id,
        policy_path=policy_path,
        config=config,
        policy=policy,
        preprocessor=MockProcessor(),
        postprocessor=MockProcessor(),
    )
