"""Stop the mobile base at every stage boundary.

The arms are position-controlled, so "the last command is still standing" already
means stopped. The base is not: it takes a VELOCITY command and keeps executing it
until the next ``send_action``, so the gap between two stages (and the gap between
the last stage and ``save_episode``, which flushes video encoders for seconds) is
time the base spends driving on the previous stage's command.
"""

# ACCEPTED RISK (2026-09-08, deliberate): if the process is SIGKILLed or
# loses power, nothing here runs and the base keeps its last command. The
# correct prescription is a command timeout, and that layer is base firmware,
# not this runner. P1's mechanism is a person standing beside the robot with
# the e-stop, and the P1 operator is the author. Re-examine when the setup
# runs unattended (repeated trials with nobody present, remote demo).

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from lerobot.robots import Robot

logger = logging.getLogger(__name__)

BASE_VELOCITY_KEYS: frozenset[str] = frozenset({"x.vel", "theta.vel"})

# Which mechanism actually zeroed the base. The caller writes this into the
# transition event, because "the base was commanded to stop" and "the full hold
# action went through" are different facts and a run where only the second
# failed must not read like a clean one.
STOP_PATH_PRIMARY: str = "primary"
STOP_PATH_FALLBACK: str = "fallback"
STOP_PATH_FAILED: str = "failed"

# The `reason` string that goes on the transition event, derived from the path
# in ONE place. Two independently written strings for the same fact drift, and
# the drift is invisible until someone greps the logs for the failure case.
REASON_BY_STOP_PATH: dict[str, str] = {
    STOP_PATH_PRIMARY: "stop_base",
    STOP_PATH_FALLBACK: "stop_base_fallback",
    STOP_PATH_FAILED: "stop_base_failed",
}


@dataclass(frozen=True)
class StopBaseOutcome:
    """What stop_base did, so the caller can log WHICH path fired.

    ``action`` is the dict handed to ``send_action`` on the primary path and
    None otherwise -- on the fallback path no action dict exists, because the
    fallback deliberately does not build one (that is the whole point of it).
    """

    path: str
    action: dict[str, float] | None = None
    error: str = ""

    @property
    def reason(self) -> str:
        return REASON_BY_STOP_PATH[self.path]

    @property
    def base_is_stopped(self) -> bool:
        """True when SOMETHING zeroed the base. False means it may still drive."""
        return self.path in (STOP_PATH_PRIMARY, STOP_PATH_FALLBACK)


def build_hold_action(
    robot: Robot, observation: Mapping[str, Any] | None = None
) -> dict[str, float]:
    """Build a full-width action that holds the arms and zeroes the base.

    A base-only action such as ``{"x.vel": 0.0, "theta.vel": 0.0}`` RAISES, which
    is why this fills every action feature: MobileAIRobot.send_action forwards the
    arm-matching keys to the arms (mobileai.py:542-544), WidowXAIFollower.send_action
    builds ``goal_pos`` by filtering ``.pos`` keys (widowxai_follower.py:233-237),
    and widowxai_follower.py:272 then does
    ``delta = abs(goal_pos[joint_name] - present_pos[joint_name])`` for every name in
    ``config.joint_names``. An arm-empty action is a KeyError there -- before
    anything reaches the motors, so the base is not stopped either.

    Commanding each arm joint to its present position makes every delta 0, so the
    velocity pacing in that same loop does nothing and the arms do not move.
    """
    if observation is None:
        observation = robot.get_observation()

    action: dict[str, float] = {}
    for key in robot.action_features:
        if key in BASE_VELOCITY_KEYS:
            # Set unconditionally rather than copied from the observation: with
            # include_base_in_state=False the base velocities are absent from
            # observation_features but are ALWAYS present in action_features
            # (mobileai.py:331-339), so there is nothing to copy and the whole
            # point of this call would be silently dropped.
            action[key] = 0.0
            continue
        if key not in observation:
            raise KeyError(
                f"{robot.name}: action feature {key!r} has no matching key in the "
                f"observation, so it cannot be commanded to hold its present value"
            )
        value = float(observation[key])
        if not math.isfinite(value):
            # Raising here is safe ONLY because stop_base's fallback zeroes the
            # base without this action: refusing to command the arms no longer
            # costs the base stop. A NaN that reached the arms would not fail
            # loudly -- WidowXAIFollower's pacing does
            # `abs(goal_pos[j] - present_pos[j]) > limit`, every comparison with
            # NaN is False, so the pacing silently does not fire and the NaN
            # goal reaches set_all_positions. The fork already recorded what NaN
            # does on the command side once: "max(-MAX, NaN) returns -MAX -- a
            # NaN command reaches the base as full-speed reverse"
            # (mobileai.py:54-55).
            # HARDENING, not an observed defect: no arm present position has
            # ever been seen non-finite on this rig. The fork's documented NaN
            # incident is on the base velocity READ path (mobileai.py:38,
            # 54-55, 159), which is sanitized there.
            raise ValueError(
                f"{robot.name}: observation value for action feature {key!r} is "
                f"{value!r}, which cannot be commanded as a hold position"
            )
        action[key] = value
    return action


def zero_base_directly(robot: Robot) -> str:
    """Command the base to zero WITHOUT an observation or the action API.

    Returns "" on success and a description of what stopped it otherwise. Never
    raises: it is the last line of defence and its own failure must not replace
    the failure that brought us here.

    This bypasses ``send_action`` on purpose. The full hold action needs a
    complete ``get_observation``, which on MobileAIRobot reads all three
    RealSense cameras through an unguarded ``async_read`` (mobileai.py:526-528)
    -- so the most likely mid-stage fault on this rig is exactly the fault that
    would stop the base from being stopped. The arms need no command at all
    (position-controlled, they hold), so the arm half of the hold action must
    never be able to cost the base stop.

    ``robot.base.set_cmd_vel(0.0, 0.0)`` is the fork's own precedent: it is the
    FIRST statement of MobileAIRobot.disconnect() (mobileai.py:568-569).
    Everything is reached through getattr so this module keeps no hard
    dependency on the Trossen package and a robot with no base (our mock) is a
    no-op rather than a crash.
    """
    base = getattr(robot, "base", None)
    if base is None:
        return "robot exposes no .base, so there is no base command to fall back on"
    set_cmd_vel = getattr(base, "set_cmd_vel", None)
    if not callable(set_cmd_vel):
        return f"{type(base).__name__}.set_cmd_vel is not callable"
    try:
        accepted = set_cmd_vel(0.0, 0.0)
    except Exception as error:  # last line of defence: see the docstring
        return f"base.set_cmd_vel(0.0, 0.0) raised {error!r}"
    # TrossenSlate.set_cmd_vel returns False when the Modbus transaction failed;
    # mobileai.py:554 and :569 both test it that way. Unlike send_action's echo
    # this IS a delivery signal, so a False here is worth propagating.
    if accepted is False:
        return "base.set_cmd_vel(0.0, 0.0) returned False (base transaction failed)"
    return ""


def stop_base(robot: Robot, *, reraise_interrupt: bool = True) -> StopBaseOutcome:
    """Zero the base at a stage boundary and report which path did it.

    PRIMARY path: one ``get_observation``, one full hold action, one
    ``send_action``. It stays primary because a base-only action dict raises a
    KeyError at widowxai_follower.py:272 before anything reaches the motors --
    see :func:`build_hold_action`.

    FALLBACK path: if the observation or the send raises, :func:`zero_base_directly`
    commands the base straight through ``robot.base.set_cmd_vel(0.0, 0.0)``.
    Zeroing the base is the safety-critical half and it must not be hostage to
    three RealSense reads.

    NEVER raises for an ordinary failure -- the caller gets an outcome to log.
    A KeyboardInterrupt or SystemExit IS re-raised, but only AFTER the fallback
    has run, so the operator's second Ctrl+C still gets through and the base is
    stopped before it does.

    ``reraise_interrupt=False`` suppresses that one re-raise and returns the
    outcome instead. It exists for the caller that is ALREADY unwinding on an
    exception it will re-raise itself (runner._stop_base_never_raises): there the
    interrupt has nothing left to protect -- the fallback above has just stopped
    the base and the process is on its way out -- while letting it escape costs
    the truth. The escaping interrupt was caught one frame up and filed as
    stop_base_path="failed", so an ordinary Ctrl+C during a stage was logged as
    "the base may still be moving" even though the fallback had zeroed it, and an
    indicator that cries wolf on a routine operator abort is one an operator
    learns to ignore before the run where it is real.

    ``stop_action`` IS INTENT, NOT CONFIRMATION on the primary path, and nothing
    available there can make it confirmation. ``MobileAIRobot.send_action`` does
    not raise when the base write fails: a failed Modbus transaction only emits a
    throttled warning (mobileai.py:551-558), and the echo it returns carries the
    *sanitized commanded* velocities either way (mobileai.py:562-566), so the echo
    cannot prove delivery either. What the echo DOES show is the arm side, where
    the follower clamps the goal positions it accepted -- a disagreement there is
    real information. Both are logged, named, so a post-hoc reader of a run where
    the base kept drifting has the commanded dict, the accepted echo and the
    base-write warning on the same timeline instead of only the first. The durable
    fix is a delivery signal out of MobileAIRobot.send_action, which is the sibling
    package's call to make. The FALLBACK path does have one (set_cmd_vel's bool).
    """
    try:
        action = build_hold_action(robot)
        echo = robot.send_action(action)
    except BaseException as primary_error:
        fallback_error = zero_base_directly(robot)
        if fallback_error:
            path = STOP_PATH_FAILED
            detail = (
                f"hold action failed ({primary_error!r}) and the direct base "
                f"command did not go through either: {fallback_error}"
            )
            logger.error(f"stop_base: THE BASE MAY STILL BE MOVING -- {detail}")
        else:
            path = STOP_PATH_FALLBACK
            detail = (
                f"hold action failed ({primary_error!r}); the base was zeroed "
                f"directly through base.set_cmd_vel(0.0, 0.0) instead. The arms "
                f"were NOT commanded, which is safe: they are position-controlled "
                f"and hold their last position."
            )
            logger.warning(f"stop_base fell back: {detail}")
        # A Ctrl+C landing inside get_observation must still reach the operator,
        # but only after the base is stopped -- which the fallback above just
        # did. `except Exception` here would swallow the emergency case; letting
        # it through before the fallback would skip the base stop entirely.
        # reraise_interrupt=False is the failure-path caller saying "I am already
        # re-raising; give me the true path instead" -- see the docstring.
        if reraise_interrupt and not isinstance(primary_error, Exception):
            raise
        return StopBaseOutcome(path=path, action=None, error=detail)

    logger.info(f"stop_base commanded: {action}")
    logger.info(f"stop_base echo (accepted by the robot, not a delivery ack): {echo}")
    return StopBaseOutcome(path=STOP_PATH_PRIMARY, action=action)
