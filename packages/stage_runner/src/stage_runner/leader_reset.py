"""The hardware half of a ``leader_reset`` stage: hand over to the leader arm and back.

A ``leader_reset`` stage (executors.run_leader_reset_stage) is lerobot-record's
reset section placed anywhere in the chain: the operator moves the leader arm,
the follower copies it, no frame is recorded, and the right arrow ends it. Its
use is the one the runner could not do before -- putting a tube or a beaker in
the gripper and setting the pose a stage was trained to start from.

THREE THINGS HAPPEN AROUND THAT LOOP, each one here because the plain upstream
loop gets it wrong on this rig:

* THE LEADER IS ALIGNED TO THE FOLLOWER FIRST. record_loop's first iteration
  sends the leader's joint positions to the follower as they are. After
  connect() both sit in the same staged pose, but in the middle of a chain the
  follower is wherever the previous policy left it, and the first command is a
  jump of up to a radian per joint -- the policy -> leader arm hand-over at which
  three of the five 9/27 crashes happened (`Joint 3 velocity limit exceeded`;
  that the jump CAUSED them is unverified). So the leader is first driven, under
  position control, to the follower's measured joints -- gripper included, so a
  held object is not let go -- and only then handed back to gravity
  compensation. The follower then starts copying a leader that is already where
  it is.
* THE BASE TORQUE IS RELEASED, so the operator can push the base by hand, and
  restored before the next stage (decided 2026-10-19). Only when the robot was
  connected WITH base torque: a run that keeps the base immobilised
  (``enable_base_motor_torque: false``) is left exactly as it is. A zero
  velocity is commanded before each switch, so the base never wakes up holding
  a velocity.
* THE BASE COMMAND IS ZERO THROUGHOUT. MobileAILeaderTeleop.get_action returns
  the base's own MEASURED velocity as its x.vel / theta.vel (mobileai_leader.py),
  which is harmless while recording -- the base is unpowered then -- and a
  feedback loop on a powered base: a nudge measured as velocity is commanded
  back as velocity. ResetActionFilter overwrites both with 0.0.

Reached through getattr -- ``robot.base``, ``teleop.align_to`` -- the same
convention as reposition.py and transitions.zero_base_directly, so the mock
robot and the mock leader arm stand in with the same attribute shape.
"""

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from stage_runner.reposition import is_gripper_key
from stage_runner.transitions import BASE_VELOCITY_KEYS, zero_base_directly

logger = logging.getLogger(__name__)

POSITION_SUFFIX: str = ".pos"

# The leader is driven toward the follower with the operator's hands near it, so
# its alignment move is paced slower than the follower's own moves: this is the
# AVERAGE joint speed, and the controller's trajectory peaks at ~2x the average
# (the measurement quoted in config_widowxai_follower.py's velocity_safety_factor
# comment). UNMEASURED as a comfort figure -- a conservative round number.
LEADER_ALIGN_AVERAGE_SPEED_RAD_S: float = 0.5
# Floor of the alignment goal time: the leader's own connect() moves to its
# staged pose over a fixed 2.0 s (widowxai_leader.py configure()).
LEADER_ALIGN_MIN_GOAL_TIME_S: float = 2.0

# base_torque values of the stage_reset event. Closed, because a reader buckets
# on them.
#   released     -- torque was switched off for the loop and switched back on
#   not_released -- the robot was connected without base torque (or has no base
#                   that can switch it); nothing was switched
#   restore_failed -- switching it back on FAILED: the next stage does not run
#   release_failed -- a policy stage's `release_base_torque` could not switch it
#                     off: the policy does not run (stage_base_torque only)
BASE_TORQUE_RELEASED: str = "released"
BASE_TORQUE_NOT_RELEASED: str = "not_released"
BASE_TORQUE_RESTORE_FAILED: str = "restore_failed"
BASE_TORQUE_RELEASE_FAILED: str = "release_failed"


class LeaderResetError(RuntimeError):
    """The hand-over to or from the leader arm cannot be done safely."""


def _arm_keys(teleop: Any) -> list[str]:
    """The leader's joint position keys -- arms and grippers, never the base."""
    return [
        key
        for key in teleop.action_features
        if key.endswith(POSITION_SUFFIX) and key not in BASE_VELOCITY_KEYS
    ]


def follower_targets(robot: Any, teleop: Any) -> dict[str, float]:
    """The follower's MEASURED position for every joint the leader commands.

    The leader's keys and the follower's are the same names (left_joint_0.pos ...
    right_left_carriage_joint.pos): record_loop sends the leader's action to the
    follower unchanged, which only works because they are. A key the follower
    does not report is refused rather than guessed.
    """
    observation = robot.get_observation()
    keys = _arm_keys(teleop)
    missing = [key for key in keys if key not in observation]
    if missing:
        raise LeaderResetError(
            f"the follower reports no measured position for leader key(s) {missing}"
        )
    targets = {key: float(observation[key]) for key in keys}
    non_finite = [key for key, value in targets.items() if not math.isfinite(value)]
    if non_finite:
        raise LeaderResetError(
            f"the follower's measured position is non-finite for {non_finite}"
        )
    return targets


def align_goal_time(start: Mapping[str, float], target: Mapping[str, float]) -> float:
    """Goal time for the leader's alignment move, from its largest ARM joint delta.

    Gripper carriages are in metres, a different unit, and move a few
    centimetres at most, so they do not pace the move.
    """
    deltas = [
        abs(target[key] - start[key])
        for key in target
        if key in start and not _is_gripper(key)
    ]
    largest = max(deltas, default=0.0)
    return max(LEADER_ALIGN_MIN_GOAL_TIME_S, largest / LEADER_ALIGN_AVERAGE_SPEED_RAD_S)


def _is_gripper(key: str) -> bool:
    # reposition owns the one gripper naming rule; a second copy would drift.
    return is_gripper_key(key)


@dataclass
class LeaderAlignResult:
    """What align_leader_to_follower did, for the stage_reset event."""

    goal_time_s: float
    target: dict[str, float] = field(default_factory=dict)
    leader_before: dict[str, float] = field(default_factory=dict)
    leader_after: dict[str, float] = field(default_factory=dict)
    # Max |leader - follower| over the ARM joints after the move. Logged, not
    # gated: once the leader is back in gravity compensation it sags a little,
    # and the operator is about to move it anyway.
    max_arm_error_rad: float = 0.0


def align_leader_to_follower(robot: Any, teleop: Any) -> LeaderAlignResult:
    """Drive the leader to the follower's measured joints, then free it again.

    Blocking. ``teleop.align_to(positions, goal_time)`` is the leader arm's own
    method (BiWidowXAILeaderRobot.align_to): position mode, one paced move,
    back to gravity compensation -- the sequence its configure() already runs to
    reach the staged pose. preflight refuses a leader without it.
    """
    align_to = getattr(teleop, "align_to", None)
    if not callable(align_to):
        raise LeaderResetError(
            f"{type(teleop).__name__} has no align_to(); the leader cannot be "
            "moved to the follower before the hand-over"
        )
    target = follower_targets(robot, teleop)
    before = {key: float(value) for key, value in teleop.get_action().items()}
    goal_time = align_goal_time(before, target)
    logger.info(
        f"aligning the leader arm to the follower over {goal_time:.2f}s "
        "(hands off the leader until it stops)"
    )
    align_to(target, goal_time)
    after = {key: float(value) for key, value in teleop.get_action().items()}
    arm_errors = [
        abs(after[key] - target[key])
        for key in target
        if key in after and not _is_gripper(key)
    ]
    return LeaderAlignResult(
        goal_time_s=goal_time,
        target=target,
        leader_before={key: before[key] for key in target if key in before},
        leader_after={key: after[key] for key in target if key in after},
        max_arm_error_rad=max(arm_errors, default=0.0),
    )


def _base_switch(robot: Any) -> Any | None:
    """``robot.base.enable_motor_torque`` when this run may switch it, else None.

    Switched only when the robot was CONNECTED with base torque on
    (``enable_base_motor_torque: true``): restoring "on" to a base that was
    deliberately left unpowered would power it up behind the operator's back.
    """
    if not getattr(getattr(robot, "config", None), "enable_base_motor_torque", False):
        return None
    switch = getattr(getattr(robot, "base", None), "enable_motor_torque", None)
    return switch if callable(switch) else None


def base_torque_switchable(robot: Any) -> bool:
    """True when release/restore_base_torque would act, False when they no-op."""
    return _base_switch(robot) is not None


def _switch_succeeded(returned: Any) -> tuple[bool, str]:
    """TrossenSlate.enable_motor_torque returns (success, message)."""
    if isinstance(returned, tuple) and returned:
        return bool(returned[0]), str(returned[1]) if len(returned) > 1 else ""
    return returned is not False, ""


def release_base_torque(robot: Any) -> bool:
    """Zero the base command, then switch base torque off. True = released.

    A failure to release is NOT fatal: the base simply stays powered at zero
    velocity, and the operator cannot push it -- logged, and the reset goes on.
    """
    switch = _base_switch(robot)
    if switch is None:
        return False
    problem = zero_base_directly(robot)
    if problem:
        logger.warning(f"base not zeroed before releasing its torque: {problem}")
    try:
        ok, message = _switch_succeeded(switch(False))
    except Exception:
        logger.exception("could not release the base torque; it stays on")
        return False
    if not ok:
        logger.warning(f"base torque release refused ({message}); it stays on")
        return False
    logger.info("base torque released: the base can be pushed by hand")
    return True


def restore_base_torque(robot: Any) -> str:
    """Zero the base command, then switch base torque back on. "" = restored.

    Returns what went wrong otherwise. The zero comes FIRST so the base does not
    re-energise holding whatever velocity was last written while it was free.
    """
    switch = _base_switch(robot)
    if switch is None:
        return "this robot has no base torque switch"
    problem = zero_base_directly(robot)
    if problem:
        return f"base not zeroed before restoring its torque: {problem}"
    try:
        ok, message = _switch_succeeded(switch(True))
    except Exception as error:
        return f"base.enable_motor_torque(True) raised {error!r}"
    if not ok:
        return f"base.enable_motor_torque(True) failed: {message}"
    logger.info("base torque restored")
    return ""


class ResetActionFilter:
    """The reset loop's teleop_action_processor: zero the base, remember the grippers.

    record_loop calls ``teleop_action_processor((action, observation))`` on every
    leader action and sends the result (record_adapter.call_reset_loop), so this
    sees every command the reset sends. It returns a copy with ``x.vel`` and
    ``theta.vel`` forced to 0.0 -- see the module docstring for why the leader's
    own values must never reach a powered base -- and keeps the gripper goals of
    the last one, which the runner holds through a following start pose move.
    """

    def __init__(self) -> None:
        self.last_action: dict[str, float] | None = None
        self.sent_count = 0

    def __call__(self, transition: tuple[Mapping[str, Any], Any]) -> dict[str, Any]:
        action, _observation = transition
        filtered = dict(action)
        for key in BASE_VELOCITY_KEYS:
            filtered[key] = 0.0
        self.last_action = {key: float(value) for key, value in filtered.items()}
        self.sent_count += 1
        return filtered

    def gripper_goals(self) -> dict[str, float] | None:
        if self.last_action is None:
            return None
        goals = {
            key: value for key, value in self.last_action.items() if _is_gripper(key)
        }
        return goals or None
