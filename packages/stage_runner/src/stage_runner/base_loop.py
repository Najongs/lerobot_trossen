"""The tick loop the base stages run in place of record_loop, and the frame it records.

WHY NOT record_loop. record_loop takes its action from a policy or a leader arm
and has no seam for "the base command comes from a pose tracker, the arms hold
still". The base stages therefore run their own loop -- and it has to keep
everything record_loop guarantees, because the episode the trial saves is ONE
episode across every stage:

* one ``get_observation``, one full-width ``send_action`` and one ``add_frame``
  per tick, in that order, paced to ``dataset.fps`` with lerobot's own
  ``precise_sleep`` (lerobot_record.py:348-427). A base stage's frames are
  indistinguishable in shape from a policy stage's: same features, the stage's
  task string, ``frame_index`` consecutive across the boundary;
* the observation goes through ``processors.robot_observation`` and the action
  through ``processors.robot_action`` exactly as record_loop sends them, and
  the frame is built with ``build_dataset_frame`` from the same feature dict;
* ``save_episode`` is never called (runner.run_trial saves once).

SINGLE WRITER. While a base stage runs, this loop on the main thread is the only
caller of ``robot.get_observation``, ``robot.send_action`` and
``dataset.add_frame``. The align stage's measurement worker gets a COPY of an
image this loop already read and touches nothing else.

THE PROBE. ``MobileAIRobot`` hides two facts a pose tracker needs: whether this
tick's ``update_state()`` refreshed the odometry cache (a failure only logs a
throttled warning, mobileai.py get_observation) and whether ``set_cmd_vel``
went through (same, send_action). For the stage's duration :class:`BaseProbe`
stands in for ``robot.base`` and records both, delegating every call -- so
neither ``mobileai.py`` nor the robot API changes, and the odometry read after
``get_observation`` is ``robot.base.get_pose()`` on the cache that call just
refreshed, at no extra serial cost. ``close()`` puts the original back on every
exit path, before the runner's boundary ``stop_base`` talks to the driver.

ARMS HOLD. Every action carries the arm joints at their position on the stage's
first observation -- and the GRIPPERS at the goal of the last recorded action
when the episode already has frames, so a gripper closed on an object by the
previous stage keeps its squeeze through the base motion (see
:meth:`BaseTickLoop._hold_from`).

NOTHING HERE TALKS TO A PHASE. lerobot_robot_trossen's loop_rate_log tags
record_loop's ticks "policy"/"teleop"; this loop's ticks carry phase None, and
the serial re-arm (which only acts in a tagged policy loop) stays off -- each
base transaction pays the driver's full wait. Faking a phase would mix these
ticks into the policy rate windows. The tick periods
this loop measures are logged per stage instead (:meth:`BaseTickLoop.tick_stats`).
"""

from __future__ import annotations

import logging
import math
import statistics
import time
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from lerobot.datasets.utils import build_dataset_frame
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.robot_utils import precise_sleep

from stage_runner import reposition, transitions

if TYPE_CHECKING:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from stage_runner.config import StageConfig
    from stage_runner.context import StageContext

logger = logging.getLogger(__name__)

# What poll_keys returns. Closed: the executors branch on exactly these.
KEY_STOP: str = "stop"
KEY_RERECORD: str = "rerecord"
KEY_RIGHT: str = "right"

# Where the held gripper goals came from, written into the stage event.
# "reset": the last gripper command of a leader_reset stage with no recorded
# stage after it (StageContext.reset_gripper_goals) -- the grip the operator
# set. "recorded": the last action row of the episode buffer (the previous
# stage's last command). "measured": the stage's first observation -- the
# chain's first stage, a flat robot with no gripper, or a buffer that could
# not be read.
GRIPPER_HOLD_RESET: str = "reset"
GRIPPER_HOLD_RECORDED: str = "recorded"
GRIPPER_HOLD_MEASURED: str = "measured"

# Base system states that make base_state_fault() report a fault, compared by
# NAME so this module imports nothing from lerobot_robot_trossen
# (SlateBaseSystemState in mobileai.py): the e-stop, and every SYS_ERR* code.
_FAULT_STATE: str = "SYS_ESTOP"
_FAULT_PREFIX: str = "SYS_ERR"


class BaseProbe:
    """``robot.base`` for the duration of a base stage: the real driver, observed.

    Delegates everything to the wrapped object; ``update_state`` and
    ``set_cmd_vel`` additionally record what they returned. Return values are
    passed through unchanged and exceptions propagate unchanged -- the probe
    changes no behaviour of the robot, it only remembers.

    Per tick (reset by :meth:`begin_tick`): ``last_update_ok`` is None until
    this tick's ``update_state`` ran, then its result; ``last_cmd_ok`` likewise
    for ``set_cmd_vel``. Per stage: the failure counts and the current and
    longest run of consecutive refused commands -- CONSECUTIVE because one
    dropped serial write is not a fault and eleven in a row is an e-stop
    (stage1_align.py's rule).
    """

    def __init__(self, base: Any) -> None:
        # __getattr__ (not __getattribute__) runs only for names the probe does
        # not have itself, so these attributes are the probe's own.
        self.wrapped = base
        self.last_update_ok: bool | None = None
        self.last_cmd_ok: bool | None = None
        self.update_calls = 0
        self.update_failed_ticks = 0
        self.cmd_calls = 0
        self.cmd_refused_ticks = 0
        self.cmd_refused_run = 0
        self.max_cmd_refused_run = 0

    def __getattr__(self, name: str) -> Any:
        # Everything the stage does not observe -- get_pose, get_vel, read(),
        # enable_motor_torque, a mock's true_pose -- goes straight through.
        return getattr(self.wrapped, name)

    def begin_tick(self) -> None:
        self.last_update_ok = None
        self.last_cmd_ok = None

    def update_state(self) -> Any:
        self.update_calls += 1
        # Recorded as a failure BEFORE the call, so an update_state that raises
        # leaves "not refreshed" behind rather than the previous tick's True.
        self.last_update_ok = False
        result = self.wrapped.update_state()
        self.last_update_ok = bool(result)
        if not result:
            self.update_failed_ticks += 1
        return result

    def set_cmd_vel(self, linear: float, angular: float) -> Any:
        self.cmd_calls += 1
        self.last_cmd_ok = False
        result = self.wrapped.set_cmd_vel(linear, angular)
        # TrossenSlate.set_cmd_vel returns False when the Modbus transaction
        # failed (mobileai.py tests it that way); anything else is "accepted".
        accepted = result is not False
        self.last_cmd_ok = accepted
        if accepted:
            self.cmd_refused_run = 0
        else:
            self.cmd_refused_ticks += 1
            self.cmd_refused_run += 1
            self.max_cmd_refused_run = max(
                self.max_cmd_refused_run, self.cmd_refused_run
            )
        return result

    def counters(self) -> dict[str, int]:
        """The per-stage counts, for the stage event."""
        return {
            "update_calls": self.update_calls,
            "update_failed_ticks": self.update_failed_ticks,
            "cmd_calls": self.cmd_calls,
            "cmd_refused_ticks": self.cmd_refused_ticks,
            "max_cmd_refused_run": self.max_cmd_refused_run,
        }


class StageFrameWriter:
    """One ``add_frame`` per tick, built exactly as record_loop builds it.

    lerobot_record.py:353-354 and :409-412: the observation frame from the
    PROCESSED observation, the action frame from the action values (not the
    send_action echo), both through ``build_dataset_frame`` on the dataset's own
    features, plus the task string. ``validate_frame`` then accepts exactly the
    features plus ``task``, so any drift from that recipe fails loudly on the
    first frame rather than writing a wrong one.
    """

    def __init__(self, dataset: LeRobotDataset, task: str) -> None:
        self._dataset = dataset
        self._task = task

    def write(
        self, observation: Mapping[str, Any], action: Mapping[str, float]
    ) -> None:
        features = self._dataset.features
        frame = {
            **build_dataset_frame(features, observation, prefix=OBS_STR),
            **build_dataset_frame(features, action, prefix=ACTION),
            "task": self._task,
        }
        self._dataset.add_frame(frame)


class BaseTickLoop:
    """observe -> act, paced to dataset.fps, recording every tick; see the module doc.

    The contract the executors keep: every tick is ``observe()`` then exactly
    one ``act()``. ``act`` sends and records; a stage that decides to stop
    after ``observe`` simply returns without ``act`` -- that tick then sends
    nothing and records nothing, and the executor zeroes the base directly.

    Constructing the loop installs the probe as ``robot.base``; ``close()``
    (idempotent) restores the original. Construct it inside the executor's
    try, or call close() on every path -- a probe left installed would outlive
    the stage.
    """

    def __init__(
        self, context: StageContext, stage: StageConfig, *, ceiling_s: float
    ) -> None:
        robot = context.robot
        base = getattr(robot, "base", None)
        if base is None:
            # preflight.check_base_stages makes this unreachable from cli.
            raise RuntimeError(
                f"stage {stage.id!r}: robot {robot.name!r} has no .base to drive"
            )
        # connect() no longer waits for the arms' 2 s move to the staged pose
        # (lerobot_robot_trossen.parking): the first get_observation does. A
        # base stage first in a chain would take that wait inside its first
        # tick, against its ceiling and its tick periods, so wait it out here
        # before the clock starts. Robots without the call (the mock, an older
        # robot package) have nothing to wait for.
        wait_for_arrival = getattr(robot, "wait_for_arrival", None)
        self.arrival_wait_s = (
            float(wait_for_arrival()) if callable(wait_for_arrival) else 0.0
        )
        self._context = context
        self._robot = robot
        self.stage_id = stage.id
        self.task = stage.name
        self.ceiling_s = float(ceiling_s)
        self.fps = int(context.config.dataset.fps)
        self.period_s = 1.0 / self.fps
        self._writer = StageFrameWriter(context.dataset, stage.name)
        self.frames_before = context.buffered_frame_count()
        self._original_base = base
        self.probe = BaseProbe(base)
        robot.base = self.probe
        self._closed = False
        self.started = time.perf_counter()

        self.held_arm_positions: dict[str, float] | None = None
        self.gripper_hold_source: str | None = None
        self.ticks = 0
        self._tick_started: float | None = None
        self._last_tick_started: float | None = None
        self._raw_observation: dict[str, Any] | None = None
        self.periods: list[float] = []
        self.last_period_s: float | None = None
        self.work_s: list[float] = []

    # ---------------------------------------------------------------- frames

    @property
    def frames(self) -> int:
        """Frames this stage has recorded so far (episode-buffer delta)."""
        return self._context.buffered_frame_count() - self.frames_before

    # ------------------------------------------------------------------ tick

    def observe(
        self,
    ) -> tuple[dict[str, Any], tuple[float, float, float] | None, float]:
        """Start a tick: one observation, plus this tick's odometry and start time.

        Returns ``(processed observation, odometry or None, tick start)``. The
        odometry is None unless THIS tick's ``update_state`` succeeded -- the
        field's Base.read rule (pr69 base_pose_step_test.py: get_pose only if
        fresh) -- and also None when get_pose raises, has the wrong length or
        carries a non-finite value. The tracker turns None into a zero command
        for that tick.
        """
        if self._closed:
            raise RuntimeError(f"stage {self.stage_id!r}: the tick loop is closed")
        if self._tick_started is not None:
            raise RuntimeError(
                f"stage {self.stage_id!r}: observe() twice without act(); every "
                "tick is one observation and one action"
            )
        started = time.perf_counter()
        if self._last_tick_started is not None:
            self.last_period_s = started - self._last_tick_started
            self.periods.append(self.last_period_s)
        self._last_tick_started = started
        self._tick_started = started
        self.ticks += 1
        self.probe.begin_tick()
        raw = self._robot.get_observation()
        if self.held_arm_positions is None:
            self.held_arm_positions = self._hold_from(raw)
        self._raw_observation = raw
        observation = self._context.processors.robot_observation(raw)
        return observation, self.read_odometry(), started

    def act(
        self, observation: Mapping[str, Any], x_vel: float, theta_vel: float
    ) -> dict[str, float]:
        """End a tick: send the held arms and this base command, record, pace.

        Sanitizing and the +-1 clamp of the base command stay in
        ``send_action`` (MobileAIRobot._sanitize_base_command), as for a policy.
        The frame records the action as built -- the tracker's command -- which
        is what record_loop records for a policy too (its ``action_values``, not
        the robot's echo).
        """
        if self._tick_started is None or self._raw_observation is None:
            raise RuntimeError(
                f"stage {self.stage_id!r}: act() without observe() first"
            )
        held = self.held_arm_positions or {}
        action: dict[str, float] = {}
        for key in self._robot.action_features:
            if key == "x.vel":
                action[key] = float(x_vel)
            elif key == "theta.vel":
                action[key] = float(theta_vel)
            else:
                action[key] = held[key]
        processors = self._context.processors
        self._robot.send_action(
            processors.robot_action((action, self._raw_observation))
        )
        self._writer.write(observation, action)
        tick_started = self._tick_started
        self._tick_started = None
        self._raw_observation = None
        work_s = time.perf_counter() - tick_started
        self.work_s.append(work_s)
        precise_sleep(max(self.period_s - work_s, 0.0))
        return action

    # ------------------------------------------------------------ base reads

    def read_odometry(self) -> tuple[float, float, float] | None:
        """``robot.base.get_pose()`` if this tick refreshed it, else None. Never raises."""
        if self.probe.last_update_ok is not True:
            return None
        try:
            pose = [float(value) for value in self._original_base.get_pose()]
        except Exception:
            logger.debug(f"stage {self.stage_id!r}: get_pose raised", exc_info=True)
            return None
        if len(pose) != 3 or not all(math.isfinite(value) for value in pose):
            return None
        return pose[0], pose[1], pose[2]

    def read_base_velocity(self) -> tuple[float, float] | None:
        """Measured (forward m/s, turn rad/s) if this tick refreshed it, else None.

        None counts as "not still" wherever stillness is tested: a base whose
        state could not be read is not known to have stopped.
        """
        if self.probe.last_update_ok is not True:
            return None
        try:
            values = [float(value) for value in self._original_base.get_vel()[:2]]
        except Exception:
            logger.debug(f"stage {self.stage_id!r}: get_vel raised", exc_info=True)
            return None
        if len(values) != 2 or not all(math.isfinite(value) for value in values):
            return None
        return values[0], values[1]

    def base_state_name(self) -> str | None:
        """The name of the base's last reported system state, or None.

        MobileAIRobot.get_observation keeps ``_last_base_state`` current on
        every successful refresh that carries a known code (mobileai.py
        _log_base_state_transition), at no serial cost. A PRIVATE attribute:
        a rename there disables the e-stop guard silently, which is why the
        tracking-error and refused-command guards stay independent of it. The
        mock has no such attribute, so this is None there.
        """
        state = getattr(self._robot, "_last_base_state", None)
        if state is None:
            return None
        name = getattr(state, "name", None)
        return str(name) if name is not None else str(state)

    def base_state_fault(self) -> str | None:
        """``SYS_ESTOP`` or a ``SYS_ERR*`` state name when the base reports one, else None."""
        name = self.base_state_name()
        if name is None:
            return None
        if name == _FAULT_STATE or name.startswith(_FAULT_PREFIX):
            return name
        return None

    # ------------------------------------------------------------------ keys

    def poll_keys(self) -> str | None:
        """The operator's key for this tick: KEY_STOP, KEY_RERECORD, KEY_RIGHT or None.

        Esc and the left arrow are checked first: lerobot's listener sets
        exit_early together with each of them, so a right-arrow test first
        would read Esc as "accept and go on". exit_early is CONSUMED here, as
        record_loop consumes it when it breaks on it (lerobot_record.py:343-345),
        so a press that ended this stage does not survive into the next one.
        The two abort flags are never cleared (runner.run_trial's rule).
        """
        events = self._context.events
        right = bool(events.get("exit_early"))
        if right:
            events["exit_early"] = False
        if events.get("stop_recording"):
            return KEY_STOP
        if events.get("rerecord_episode"):
            return KEY_RERECORD
        if right:
            return KEY_RIGHT
        return None

    # ----------------------------------------------------------------- stats

    def elapsed_s(self, now: float | None = None) -> float:
        return (time.perf_counter() if now is None else now) - self.started

    def ceiling_reached(self, now: float) -> bool:
        return now - self.started >= self.ceiling_s

    def median_period_s(self, since: int = 0) -> float:
        """Median tick period from tick ``since`` on, or 1/fps with fewer than 2 periods.

        The tracker's ``nominal_dt`` seed (pr69 used 1/target_fps). The MEASURED
        period is used when there is one, because this loop ticks at the
        hardware rate (about 20 Hz measured), not at the dataset's 30 fps.
        """
        periods = self.periods[since:]
        if len(periods) < 2:
            return self.period_s
        return float(statistics.median(periods))

    def tick_stats(self, periods: list[float] | None = None) -> dict[str, float | int]:
        """Mean, p95 and max tick period, and the count slower than 1/fps.

        ``periods`` defaults to every tick of the stage; an executor passes a
        subset (the ticks during an align measurement) to see what the worker
        thread cost the loop.
        """
        values = list(self.periods if periods is None else periods)
        if not values:
            return {
                "tick_count": self.ticks,
                "tick_mean_s": None,
                "tick_p95_s": None,
                "tick_max_s": None,
                "slow_ticks": 0,
            }
        ordered = sorted(values)
        p95 = ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]
        # One period of slack on 1/fps would hide nothing useful; "slow" here
        # is a tick that took more than 1.5 nominal periods, i.e. a frame the
        # dataset's frame_index/fps timeline stretched.
        return {
            "tick_count": self.ticks,
            "tick_mean_s": float(statistics.fmean(values)),
            "tick_p95_s": float(p95),
            "tick_max_s": float(ordered[-1]),
            "slow_ticks": sum(1 for value in values if value > 1.5 * self.period_s),
        }

    # -------------------------------------------------------------- teardown

    def close(self) -> None:
        """Restore the original ``robot.base``. Idempotent, never raises."""
        if self._closed:
            return
        self._closed = True
        current = getattr(self._robot, "base", None)
        if current is not self.probe:
            logger.warning(
                f"stage {self.stage_id!r}: robot.base was replaced while the stage "
                f"ran ({type(current).__name__}); restoring the stage's original "
                "base anyway"
            )
        try:
            self._robot.base = self._original_base
        except Exception:
            logger.exception(f"stage {self.stage_id!r}: could not restore robot.base")

    # ------------------------------------------------------------------- arms

    def _hold_from(self, raw_observation: Mapping[str, Any]) -> dict[str, float]:
        """The arm half of every action this stage sends.

        Each arm joint at its position on this first observation
        (transitions.build_hold_action, base keys dropped), so the arms do not
        move. The GRIPPERS are the exception when the episode already has
        frames: they take the goal of the last recorded action
        (runner._recorded_gripper_goals, the same read the runner uses before a
        start pose). A gripper closed on an object has a measured position equal
        to the object's width while the last command is narrower; "hold at the
        measured position" would remove that squeeze for the whole replay. The
        boundary hold the runner sent just before this stage already used the
        measured position for the few milliseconds of the boundary; re-sending
        the previous command restores the goal in force.

        A leader_reset stage records no frames, so after one the buffer's last
        row is the last POLICY stage's command, not the grip the operator just
        set with the leader arm; re-sending it could open the grippers on a
        handed-over tube for the whole base motion. The runner therefore keeps a
        reset's goals on StageContext.reset_gripper_goals until a stage records
        frames, and they come FIRST here -- also for a reset that is the chain's
        first stage, where the buffer is still empty.

        A chain's first stage has no frames and holds the measured values
        without a warning.
        """
        hold = {
            key: value
            for key, value in transitions.build_hold_action(
                self._robot, raw_observation
            ).items()
            if key not in transitions.BASE_VELOCITY_KEYS
        }
        self.gripper_hold_source = GRIPPER_HOLD_MEASURED
        grippers = [key for key in hold if reposition.is_gripper_key(key)]
        reset_goals = self._context.reset_gripper_goals
        if grippers and reset_goals:
            hold.update(
                {key: reset_goals[key] for key in grippers if key in reset_goals}
            )
            self.gripper_hold_source = GRIPPER_HOLD_RESET
        elif grippers and self._context.buffered_frame_count() > 0:
            # Imported here: runner imports executors, which imports
            # base_stages, which imports this module.
            from stage_runner.runner import _recorded_gripper_goals

            goals = _recorded_gripper_goals(self._context)
            if goals is not None:
                hold.update({key: goals[key] for key in grippers if key in goals})
                self.gripper_hold_source = GRIPPER_HOLD_RECORDED
        return hold


def stop_image_writer_after(error: BaseException, dataset: Any) -> None:
    """What ``@safe_stop_image_writer`` does for record_loop, for our own loop.

    image_writer.py:26-36: on an ``Exception`` (not a KeyboardInterrupt), stop
    the dataset's image writer before re-raising, so its threads do not hang
    the process. Never raises: it runs on the way out of an exception that is
    the one the operator has to see.
    """
    if not isinstance(error, Exception):
        return
    image_writer = getattr(dataset, "image_writer", None)
    if image_writer is None:
        return
    try:
        image_writer.stop()
    except Exception:
        logger.exception("could not stop the image writer after the stage raised")
