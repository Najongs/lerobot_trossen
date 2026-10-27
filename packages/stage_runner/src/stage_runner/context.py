"""The frozen object every stage executor receives.

Its own module so executors, runner and cli can all import it without a cycle.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Annotation-only. Nothing here is constructed or isinstance-checked at
    # runtime, so lerobot and (through policies) torch stay out of the one
    # module that executors, runner and cli all import.
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.robots import Robot
    from lerobot.teleoperators import Teleoperator

    from stage_runner.config import StageRunnerConfig
    from stage_runner.events import EventLog
    from stage_runner.policies import PolicyBundle
    from stage_runner.record_adapter import RecordProcessors


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
    # The connected leader arm, the first appended field the docstring above
    # provides for. None unless a stage uses the leader_reset executor: cli.main
    # constructs and connects it only then.
    teleop: Teleoperator | None = None
    # Replay targets and align references, keyed by stage.id, loaded from local
    # files before connect by base_assets.load_base_assets -- so a base stage
    # anywhere in the chain, and any number of them, opens no dataset and reads
    # no file between stages. Empty unless a stage uses a base executor. Typed
    # Any so this module does not import numpy for the values
    # (base_assets.ReplayTargets / base_assets.AlignReference).
    base_assets: Mapping[str, Any] = field(default_factory=dict)
    # The gripper goals a leader_reset stage handed over on
    # StageResult.gripper_goals, kept here ONLY while they are newer than the
    # episode buffer's last action row; empty otherwise. A reset records no
    # frames, so after one the buffer's last row is the previous POLICY
    # stage's command, not the grip the operator just set with the leader arm.
    # A base stage reads this before the buffer (base_loop._hold_from) so a
    # base_replay or base_align right after a reset keeps that grip instead of
    # re-opening the grippers to an older command. run_trial writes it at every
    # boundary: set after a reset, cleared after any stage that recorded frames
    # (the buffer is the newer source again). Mutable on purpose, like
    # `events`: the dict is fixed, its contents follow the chain.
    reset_gripper_goals: dict[str, float] = field(default_factory=dict)

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
