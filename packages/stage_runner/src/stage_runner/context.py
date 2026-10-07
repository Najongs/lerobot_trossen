"""The frozen object every stage executor receives.

Its own module so executors, runner and cli can all import it without a cycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Annotation-only. Nothing here is constructed or isinstance-checked at
    # runtime, so lerobot and (through policies) torch stay out of the one
    # module that executors, runner and cli all import.
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.robots import Robot

    from stage_runner.chain_params import ChainParams
    from stage_runner.completion import CompletionMonitorStep, CompletionSettings
    from stage_runner.config import StageRunnerConfig
    from stage_runner.events import EventLog
    from stage_runner.policies import PolicyBundle
    from stage_runner.record_adapter import RecordProcessors
    from stage_runner.reset_policy import ResetSettings


@dataclass(frozen=True)
class ChainRuntime:
    """Everything the two chain executors share, assembled once in cli.main.

    Its own type rather than six more fields on :class:`StageContext`, because
    it is all or nothing: a chain run has every one of these and a version 1
    run has none, so ``context.chain is None`` is the single question an
    executor asks.

    ``monitor`` and ``onehot`` are the two objects that CARRY STATE ACROSS
    STAGES and must therefore be created exactly once:

    * the monitor lives in the robot action pipeline, which ``record_loop``
      never resets (it resets the policy and the two POLICY processors only,
      lerobot_record.py:332-335), so its window is cleared by
      ``begin_stage`` and by nothing else;
    * ``onehot`` is the step inside the ONE loaded checkpoint's preprocessor.
      One checkpoint serves all 11 stages, and re-pointing this step is the only
      difference between stage k's inference and stage k+1's.

    ``action_names`` / ``state_names`` come from the recording dataset's
    metadata and are what :class:`~stage_runner.reset_policy.ResetPolicy` zips
    its ramp against -- by NAME, because observation.state drops the base keys
    when ``include_base_in_state`` is false while the action never does.
    """

    params: ChainParams
    monitor: CompletionMonitorStep
    completion: CompletionSettings
    reset: ResetSettings
    # True when the checkpoint's action is 17 wide, i.e. the dataset declares a
    # `progress` feature and the completion monitor may read it. Derived from
    # the checkpoint, asserted against the YAML if the YAML states it.
    has_progress: bool
    action_names: tuple[str, ...]
    state_names: tuple[str, ...]
    # None for a checkpoint with no stage one-hot (a 14/16-D model).
    onehot: Any | None = None
    # K of the one-hot, for the log. None when onehot is None.
    onehot_k: int | None = None
    allow_manual_complete: bool = True


@dataclass(frozen=True)
class StageContext:
    """Everything a stage executor needs, assembled once in cli.main.

    An executor is ``(StageContext, StageConfig) -> StageResult``, and that
    signature is frozen. An executor added later that needs a handle of its own
    reads it from its own lazy loader or gets a new OPTIONAL defaulted field
    appended here -- a frozen dataclass takes additive fields without touching
    any existing call.

    Frozen blocks rebinding, not mutation: `events` is the dict core's keyboard
    listener writes into and executors read, and `dataset` accumulates the
    episode buffer. Both are meant to change; what must not change is which
    robot, dataset and log the stages share.
    """

    config: StageRunnerConfig
    robot: Robot
    dataset: LeRobotDataset
    processors: RecordProcessors
    events: dict[str, bool]
    bundles: dict[str, PolicyBundle]
    log: EventLog
    run_directory: Path
    # OPTIONAL and defaulted, which is what a frozen dataclass takes additively
    # without touching any existing construction -- the docstring above promised
    # that and this is the first field to use it. None in a version 1 run.
    chain: ChainRuntime | None = None
    # 리더암 텔레옵 장치 (lerobot Teleoperator). None 이면 텔레옵 구간이 없다.
    teleop: Any | None = None

    def buffered_frame_count(self) -> int:
        """Frames written into the current episode buffer so far.

        lerobot maintains this counter itself: add_frame reads it as the frame
        index (lerobot_dataset.py:1188) and increments it per frame (:1223), and
        save_episode pops it as the episode length (:1249). Reading it is what
        gives the JSONL its frame_idx and StageResult its frames with zero
        instrumentation inside the control loop.

        LeRobotDataset.__init__ leaves episode_buffer None until the first
        add_frame (:724, :1184), so a read before the first frame -- exactly
        what the stage_start event does -- returns 0 rather than raising. Our
        own path never sees that: create() installs a buffer (:1687) and
        save_episode reinstalls a fresh one (:1355 -> :1604).
        """
        buffer = self.dataset.episode_buffer
        if buffer is None:
            return 0
        return int(buffer["size"])
