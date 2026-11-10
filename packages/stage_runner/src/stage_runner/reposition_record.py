"""Record frames while the arms move to a stage's start pose (``dataset.record_reposition``).

WHY. A chain with start poses spends about 3 s before every stage driving the
arms to that stage's start pose, and nothing was recorded during it: the
episode video jumped from the previous stage's last frame to the next stage's
first, twelve times in an 11-stage chain (37 s of a 313 s run on the robot,
2026-11-01). An operator watching the recording could not see the move itself
-- the arms swinging, a gripper loosening on a beaker -- which is exactly the
part a dropped object or a bad pose asks about.

HOW. reposition.move_arms_to_start_pose spends every wait of the move -- the
expiry wait before it, the move's goal_time, the settle polls after it --
through ``sleep``. This module's :meth:`RepositionFrameRecorder.sleep` is that
sleep: while it waits it records one frame per 1/fps, built the way the base
stages build theirs (base_loop.StageFrameWriter: processed observation, action
through build_dataset_frame, the task string). It SENDS NOTHING: the arms are
on the driver's own trajectory, and a send_action would replace it.

WHAT THE FRAME'S ACTION SAYS. The goal in force, not a policy output:
* the arm joints -- the boundary hold until the move is commanded, then the
  start pose target (``commanded``);
* the grippers -- the boundary hold's gripper goal (the previous stage's last
  command, runner._recorded_gripper_goals), else the measured width;
* the base -- 0, which is what stop_base left it at (the runner refuses the
  move when the base was not confirmed stopped).
The frames carry their own task string, ``start_pose_before_<stage id>``, so a
reader -- or a training script -- can tell them from a policy's frames and drop
them.

A RECORDING FAILURE DOES NOT FAIL THE MOVE. An exception from
get_observation or add_frame is logged once and recording stops for the rest of
this move; the wait itself goes on with a plain sleep, so the arms' move and
its reached / not-reached verdict are exactly what they would have been without
recording. A frame half-added when the error came is the dataset's problem to
reject at save time, the same as for any other stage.

FRAME ACCOUNTING. These frames land BETWEEN the previous stage_end and this
stage_start, so every stage's own frame_idx delta still equals its frames
(aggregate's integrity check) and frame_idx still indexes the episode. The
stage_reposition event carries ``frames`` and ``frame_idx_start`` for them.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

from lerobot.utils.robot_utils import precise_sleep

from stage_runner import transitions
from stage_runner.base_loop import StageFrameWriter

if TYPE_CHECKING:
    from stage_runner.context import StageContext

logger = logging.getLogger(__name__)

TASK_PREFIX: str = "start_pose_before_"


def reposition_task(stage_id: str) -> str:
    """The task string of the frames recorded before ``stage_id``."""
    return f"{TASK_PREFIX}{stage_id}"


class RepositionFrameRecorder:
    """A ``sleep`` for move_arms_to_start_pose that records frames while it waits.

    ``hold_action`` is the boundary hold the runner sent before the move
    (StopBaseOutcome.action), or None at the chain's first stage. Only its
    ``.pos`` keys are used; the base is always written as 0.
    """

    def __init__(
        self,
        context: StageContext,
        stage_id: str,
        hold_action: Mapping[str, float] | None,
        *,
        clock: Callable[[], float] = time.perf_counter,
        pause: Callable[[float], None] = precise_sleep,
    ) -> None:
        self._context = context
        self._robot = context.robot
        self._writer = StageFrameWriter(context.dataset, reposition_task(stage_id))
        self.stage_id = stage_id
        self.period_s = 1.0 / int(context.config.dataset.fps)
        self._clock = clock
        self._pause = pause
        self._goals: dict[str, float] = {
            key: float(value)
            for key, value in (hold_action or {}).items()
            if key not in transitions.BASE_VELOCITY_KEYS
        }
        self._next_tick: float | None = None
        self.frames = 0
        self.error: str | None = None

    @property
    def recording(self) -> bool:
        return self.error is None

    def commanded(self, target: Mapping[str, float]) -> None:
        """The move was commanded: the arm joints' goal is now the start pose."""
        self._goals.update({key: float(value) for key, value in target.items()})

    def sleep(self, duration_s: float) -> None:
        """Wait ``duration_s``, recording a frame whenever one is due."""
        deadline = self._clock() + max(0.0, float(duration_s))
        while True:
            now = self._clock()
            if self.recording and (self._next_tick is None or now >= self._next_tick):
                self._tick()
                # Paced from the previous due time, not from now, so a slow
                # get_observation does not drift the rate; a tick later than a
                # whole period starts the schedule over instead of bursting.
                base = now if self._next_tick is None else self._next_tick
                self._next_tick = base + self.period_s
                now = self._clock()
                if self._next_tick < now:
                    self._next_tick = now + self.period_s
            if now >= deadline:
                return
            until = deadline if not self.recording else min(deadline, self._next_tick)
            self._pause(max(0.0, until - now))

    def _tick(self) -> None:
        try:
            raw = self._robot.get_observation()
            observation = self._context.processors.robot_observation(raw)
            action: dict[str, float] = {}
            for key in self._robot.action_features:
                if key in transitions.BASE_VELOCITY_KEYS:
                    action[key] = 0.0
                elif key in self._goals:
                    action[key] = self._goals[key]
                else:
                    # Stage 0 (no boundary hold) or a key the hold did not
                    # carry: the measured position, i.e. "hold where it is".
                    action[key] = float(raw[key])
            self._writer.write(observation, action)
            self.frames += 1
        except Exception as error:  # noqa: BLE001 -- recording must not fail the move
            self.error = repr(error)
            logger.warning(
                f"stage {self.stage_id!r}: stopped recording the start pose move "
                f"after {self.frames} frames; the move itself goes on: {error!r}",
                exc_info=True,
            )


__all__ = [
    "TASK_PREFIX",
    "RepositionFrameRecorder",
    "reposition_task",
]
