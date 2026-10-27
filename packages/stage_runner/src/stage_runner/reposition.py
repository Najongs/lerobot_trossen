"""Move both arms to a stage's start pose before the stage starts.

A stage policy was trained on demonstrations that all began from roughly one arm
pose. In a chain, the stage instead begins wherever the previous stage left the
arms, which is a pose that stage's training data may never contain. An optional
``start_pose`` per stage (config.StartPoseConfig, computed offline by
``python -m stage_runner.start_pose``) closes that gap: before the stage starts,
the arm joints are driven to the pose of one real demonstration's first frame.

THREE THINGS THIS MODULE NEVER COMMANDS, each a decision rather than an omission:

* THE GRIPPERS. ``TrossenArmDriver.set_arm_positions`` sets the arm joints and
  leaves the gripper joint out of the vector entirely (help() on the installed
  trossen_arm 1.9.0: "The size of the vectors should be equal to the number of arm
  joints"), so the gripper keeps the goal that stands when the move begins.
  Commanding the gripper to its demo-frame value would release the object
  whenever the demo's grip width differs from today's.
  WHICH goal stands is decided one step earlier, at the boundary: stop_base's
  hold sets every ``.pos`` key, the gripper included, to its MEASURED position,
  and on a gripper closed on a beaker that is the beaker's width -- no position
  error, so little or no squeeze, for the whole multi-second move. Before a stage
  with a start pose the runner therefore builds that hold with the previous
  stage's last gripper COMMAND instead (runner._recorded_gripper_goals ->
  transitions.build_hold_action(held_positions=...)), and the stage_reposition
  event records which one was in force (gripper_goal_source). That a held
  object stays held through the move is UNVERIFIED ON HARDWARE: the mock gripper
  has no grip force to lose.
* THE BASE. It is velocity-controlled and its position is the previous stage's
  outcome; there is no base "pose" to restore. The runner only moves the arms
  after the boundary's stop_base has stopped it (runner._reposition_before_stage).
* ANYTHING THROUGH ``robot.send_action``. That path is the policy's: it needs a
  full-width action including the gripper (widowxai_follower.py:316 indexes every
  configured joint name), and its pacing uses a ~0.1 s minimum window with
  blocking=False. A start pose is a multi-second, blocking move of the arm joints
  only, which is what the driver call above is for.

THE MOVE WAITS FOR THE PREVIOUS TRAJECTORY TO EXPIRE before it is commanded.
trossen_arm 1.9.0 builds every new set_*_positions trajectory -- a quintic per
joint -- from the COMMANDED position, velocity and acceleration of the
trajectory still running, not from the measured state, and a non-zero start
acceleration's effect grows with the new horizon squared. At a boundary the last
arm command is the stop_base hold, a ~0.1 s trajectory still mid-course when the
3 s move was sent ~0.045 s later; the move inherited its acceleration ~900x
amplified and swung both arms up to 160 deg the wrong way before arriving
(robot logs 2026-10-20; a 1 s pause before the move removed the swing). So
move_arms_to_start_pose first sleeps until the last command's send time plus
its goal_time plus TRAJECTORY_EXPIRY_MARGIN_S, read off the follower's own
record (WidowXAIFollower.last_command_sent_s / last_command_goal_time_s), and
only then builds the move from a commanded state of (hold target, 0, 0). No
extra arm command is sent and the grippers are not touched by the wait.

Everything is reached through getattr -- ``robot.arms.left_arm.driver`` and
``.config.joint_names`` -- the same convention as transitions.zero_base_directly,
so this module has no import-time dependency on the Trossen package and the mock
robot can stand in with the same attribute shape.
"""

import logging
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from stage_runner.transitions import BASE_VELOCITY_KEYS

logger = logging.getLogger(__name__)

# How the gripper is told apart from the arm joints: BY NAME. The WidowX AI's
# joint list is joint_0..joint_5 plus "left_carriage_joint" -- the gripper
# carriage, in metres rather than radians (config_widowxai_follower.py
# `joint_names`) -- and MobileAIRobot prefixes each with left_/right_, so the
# dataset and action names are e.g. "right_left_carriage_joint.pos". "gripper" is
# there for a future end effector that names itself plainly; neither substring
# occurs in an arm joint name of this robot.
GRIPPER_NAME_MARKERS: tuple[str, ...] = ("carriage", "gripper")

POSITION_SUFFIX: str = ".pos"

# The (arms-attribute, feature-prefix) pairs of BiWidowXAIFollowerRobot, which
# prefixes every arm feature with "left_" / "right_" (bi_widowxai_follower.py:80-84)
# and holds the two arms as .left_arm / .right_arm (:56-57).
ARM_ATTRIBUTES: tuple[tuple[str, str], ...] = (
    ("left_arm", "left_"),
    ("right_arm", "right_"),
)

# After the blocking call returns, how long the arms get to come within
# position_tolerance_rad before the move is declared NOT reached. The left arm is
# commanded non-blocking first and the right arm blocking with the same
# goal_time, so by the time the right arm's call returns the left arm's
# trajectory has had at least as long; this window covers controller settling,
# not a second trajectory. UNMEASURED on hardware -- a guard against starting a
# stage from a wrong pose, not a characterised settling time.
SETTLE_TIMEOUT_S: float = 1.0
SETTLE_POLL_S: float = 0.05

# How far past the previous trajectory's computed end the move waits. The
# recorded send time is the HOST's: the driver starts the trajectory once the
# command has crossed the link and its control loop has picked it up, so the
# real end can trail sent + goal_time by the link latency plus a loop period or
# two (milliseconds on this wired arm). 50 ms covers that with room, and costs
# 50 ms on a multi-second move. UNMEASURED on hardware -- the 2026-10-20 runs
# measured the swing, not the driver's start latency.
TRAJECTORY_EXPIRY_MARGIN_S: float = 0.05

# The wait used when an arm object carries no last-command record at all (a
# lerobot_robot_trossen older than the record, or a robot that is not a
# WidowXAIFollower), so the remaining time cannot be computed. 1.0 s is the
# pause the robot-PC patch of 2026-10-20 used, which removed the swing on every
# boundary it ran on. It is NOT a bound: a paced policy frame can run longer
# (a 1.5 rad jump on joint 0 needs ~1.2 s at the 0.4 safety factor), so this
# case is logged as a warning, not treated as equivalent to a recorded wait.
UNRECORDED_EXPIRY_WAIT_S: float = 1.0

# expiry_wait_source values of the stage_reposition event. Closed.
#   recorded   -- every arm reported its last command; the wait is the time
#                 left on the later one plus the margin (0 when both expired)
#   no_command -- the record exists but no arm has been commanded through it
#                 yet, so there is no trajectory to inherit from: no wait
#   unrecorded -- an arm has no record; UNRECORDED_EXPIRY_WAIT_S was waited
EXPIRY_WAIT_RECORDED: str = "recorded"
EXPIRY_WAIT_NO_COMMAND: str = "no_command"
EXPIRY_WAIT_UNRECORDED: str = "unrecorded"
EXPIRY_WAIT_SOURCES: tuple[str, ...] = (
    EXPIRY_WAIT_RECORDED,
    EXPIRY_WAIT_NO_COMMAND,
    EXPIRY_WAIT_UNRECORDED,
)

# outcome values of the stage_reposition event. Closed, because a reader buckets
# on them: only "reached" lets the stage run its policy.
REPOSITION_REACHED: str = "reached"
REPOSITION_NOT_REACHED: str = "not_reached"
REPOSITION_ERROR: str = "error"
REPOSITION_REFUSED: str = "refused"
REPOSITION_SKIPPED: str = "skipped"
REPOSITION_INTERRUPTED: str = "interrupted"
REPOSITION_OUTCOMES: tuple[str, ...] = (
    REPOSITION_REACHED,
    REPOSITION_NOT_REACHED,
    REPOSITION_ERROR,
    REPOSITION_REFUSED,
    REPOSITION_SKIPPED,
    REPOSITION_INTERRUPTED,
)


# gripper_goal_source values of the stage_reposition event: the gripper goal in
# force during the move. Closed, because a reader buckets on them.
#   previous_command -- the boundary hold re-sent the previous stage's last
#                       gripper command (the intended case)
#   measured         -- the last command could not be read, so the hold used the
#                       measured position: the grip may be loose during the move
#   unchanged        -- nothing commanded the gripper at this boundary: stage 0
#                       (whatever connect() or the operator left), or a fallback
#                       stop_base that commanded no arm joint
GRIPPER_GOAL_PREVIOUS_COMMAND: str = "previous_command"
GRIPPER_GOAL_MEASURED: str = "measured"
GRIPPER_GOAL_UNCHANGED: str = "unchanged"
GRIPPER_GOAL_SOURCES: tuple[str, ...] = (
    GRIPPER_GOAL_PREVIOUS_COMMAND,
    GRIPPER_GOAL_MEASURED,
    GRIPPER_GOAL_UNCHANGED,
)


class StageRepositionError(RuntimeError):
    """The robot cannot be moved to this start pose; nothing was commanded."""


def is_gripper_key(key: str) -> bool:
    """True for a gripper feature name such as ``left_left_carriage_joint.pos``."""
    stem = key.removesuffix(POSITION_SUFFIX).lower()
    return any(marker in stem for marker in GRIPPER_NAME_MARKERS)


def is_base_key(key: str) -> bool:
    return key in BASE_VELOCITY_KEYS


def is_arm_joint_key(key: str) -> bool:
    """An arm joint POSITION: ``.pos``, not a gripper, not the base.

    The ``.pos`` test also drops the per-joint ``.vel`` / ``.eff`` / ``.ext_eff``
    features include_velocity / include_effort add to observation.state
    (widowxai_follower.py:75-87), which are not positions to command.
    """
    return (
        key.endswith(POSITION_SUFFIX)
        and not is_gripper_key(key)
        and not is_base_key(key)
    )


@dataclass(frozen=True)
class ArmHandle:
    """One arm as this module drives it: its driver and its arm joints, by name.

    ``arm_joint_names`` are the driver's own names (``joint_0``...), in DRIVER
    ORDER -- the order of ``config.joint_names``, which is the order WidowXAIFollower
    already relies on when it zips ``driver.get_all_positions()`` against it
    (widowxai_follower.py:285-291). ``keys`` are the matching robot action feature
    names (``left_joint_0.pos``...), which is what a start pose is keyed by.
    ``arm`` is the arm object itself, kept so its last-command record is read at
    MOVE time (wait_for_previous_trajectory), not when the handle was built.
    """

    name: str
    driver: Any
    arm_joint_names: tuple[str, ...]
    keys: tuple[str, ...]
    velocity_safety_factor: float
    arm: Any = None


def arm_handles(robot: Any) -> list[ArmHandle]:
    """Find both arms' drivers and arm-joint names on ``robot``, or raise.

    Config-derived only, so preflight can call it on a constructed robot BEFORE
    connect(): WidowXAIFollower builds its TrossenArmDriver in __init__
    (widowxai_follower.py:41) and nothing here talks to it.

    The gripper is identified by name and must be the LAST joint, exactly one per
    arm: ``set_arm_positions`` takes a vector of the arm joints only, i.e. the
    all-joint vector minus the end effector, so mapping a pose onto that vector
    by name is only correct if the non-gripper names are a prefix of
    ``joint_names``. A joint list that breaks that assumption is refused here
    rather than mapped by a guess.
    """
    arms = getattr(robot, "arms", None)
    if arms is None:
        raise StageRepositionError(
            f"{getattr(robot, 'name', type(robot).__name__)} exposes no .arms, so "
            "there is no per-arm driver to move to a start pose (the Mobile AI "
            "robot and the bimanual mock layout have one)"
        )
    handles: list[ArmHandle] = []
    for attribute, prefix in ARM_ATTRIBUTES:
        arm = getattr(arms, attribute, None)
        driver = getattr(arm, "driver", None)
        arm_config = getattr(arm, "config", None)
        joint_names = list(getattr(arm_config, "joint_names", None) or [])
        if arm is None or driver is None or not joint_names:
            raise StageRepositionError(
                f"robot.arms.{attribute} has no driver or no config.joint_names"
            )
        for method in ("get_arm_positions", "set_arm_positions", "get_joint_limits"):
            if not callable(getattr(driver, method, None)):
                raise StageRepositionError(
                    f"robot.arms.{attribute}.driver has no callable {method}()"
                )
        arm_names = [
            name for name in joint_names if not is_gripper_key(name + POSITION_SUFFIX)
        ]
        grippers = [name for name in joint_names if name not in arm_names]
        if len(grippers) != 1 or joint_names[: len(arm_names)] != arm_names:
            raise StageRepositionError(
                f"robot.arms.{attribute}.config.joint_names={joint_names}: expected "
                "the arm joints followed by exactly one gripper joint last, which "
                "is the layout set_arm_positions' arm-only vector assumes"
            )
        handles.append(
            ArmHandle(
                name=attribute,
                driver=driver,
                arm_joint_names=tuple(arm_names),
                keys=tuple(f"{prefix}{name}{POSITION_SUFFIX}" for name in arm_names),
                velocity_safety_factor=float(
                    getattr(arm_config, "velocity_safety_factor", 0.0) or 0.0
                ),
                arm=arm,
            )
        )
    return handles


def expected_arm_keys(handles: Sequence[ArmHandle]) -> tuple[str, ...]:
    """Every key a start pose for this robot must carry, and nothing else."""
    return tuple(key for handle in handles for key in handle.keys)


def start_pose_problems(
    positions: Mapping[str, float], handles: Sequence[ArmHandle]
) -> list[str]:
    """Everything wrong with ``positions`` for these arms; empty means usable.

    The ONE validation, called by preflight before connect() and again by
    move_arms_to_start_pose before the first driver call, so the gate and the
    motion cannot disagree about what a valid pose is.
    """
    problems: list[str] = []
    keys = set(positions)
    expected = set(expected_arm_keys(handles))
    grippers = sorted(key for key in keys if is_gripper_key(key))
    bases = sorted(key for key in keys if is_base_key(key))
    if grippers:
        problems.append(
            f"gripper key(s) {grippers}: a start pose never commands the gripper, so "
            "a held object is not released -- delete these lines"
        )
    if bases:
        problems.append(
            f"base key(s) {bases}: the base is velocity-controlled and is never "
            "moved to a pose -- delete these lines"
        )
    missing = sorted(expected - keys)
    if missing:
        problems.append(f"missing arm joint key(s) {missing}")
    unknown = sorted(keys - expected - set(grippers) - set(bases))
    if unknown:
        problems.append(
            f"key(s) {unknown} are not arm joints of this robot (expected exactly "
            f"{sorted(expected)})"
        )
    non_finite = sorted(
        key
        for key, value in positions.items()
        if not isinstance(value, (int, float)) or not math.isfinite(float(value))
    )
    if non_finite:
        problems.append(
            f"non-finite value(s) for {non_finite}: a NaN goal does not fail loudly "
            "on this arm -- every pacing comparison against NaN is False -- so it is "
            "refused here"
        )
    return problems


@dataclass
class RepositionResult:
    """What move_arms_to_start_pose did, for the stage_reposition event."""

    reached: bool
    goal_time_s: float
    start_positions: dict[str, float] = field(default_factory=dict)
    end_positions: dict[str, float] = field(default_factory=dict)
    max_abs_error: float = 0.0
    detail: str = ""
    # The wait for the previous trajectory to expire, before the move.
    expiry_wait_s: float = 0.0
    expiry_wait_source: str = EXPIRY_WAIT_NO_COMMAND
    previous_goal_time_s: dict[str, float | None] = field(default_factory=dict)


@dataclass(frozen=True)
class ExpiryWait:
    """What wait_for_previous_trajectory did."""

    wait_s: float
    source: str
    # Per arm (handle.name): the goal_time of its last recorded command, None
    # when it has none -- the number that says how long the wait could have been.
    previous_goal_time_s: dict[str, float | None]


def wait_for_previous_trajectory(
    handles: Sequence[ArmHandle],
    *,
    sleep: Callable[[float], None],
    clock: Callable[[], float],
) -> ExpiryWait:
    """Sleep until every arm's last trajectory has run out, plus the margin.

    The module docstring says why: the driver seeds the next trajectory from the
    commanded state of the running one, so the start pose move must be built
    only once that state is (target, 0, 0). Each arm is timed on its own record
    -- the right arm is commanded a little after the left -- and the wait is for
    the LATER expiry. Only the last command per arm matters, because each
    set_*_positions replaces the trajectory that was running.

    ``clock`` must be the clock the record was stamped with:
    WidowXAIFollower uses time.perf_counter, which is move_arms_to_start_pose's
    default.
    """
    expiries: list[float] = []
    goal_times: dict[str, float | None] = {}
    unrecorded: list[str] = []
    for handle in handles:
        if not (
            hasattr(handle.arm, "last_command_sent_s")
            and hasattr(handle.arm, "last_command_goal_time_s")
        ):
            unrecorded.append(handle.name)
            goal_times[handle.name] = None
            continue
        sent = handle.arm.last_command_sent_s
        goal_time = handle.arm.last_command_goal_time_s
        if sent is None or goal_time is None:
            goal_times[handle.name] = None
            continue
        goal_times[handle.name] = float(goal_time)
        expiries.append(float(sent) + float(goal_time))

    if unrecorded:
        wait_s = UNRECORDED_EXPIRY_WAIT_S
        source = EXPIRY_WAIT_UNRECORDED
        logger.warning(
            f"{', '.join(unrecorded)} keep no record of their last arm command "
            f"(an older lerobot_robot_trossen?); waiting a fixed {wait_s:.2f}s "
            "for it to expire, which does not cover a long paced frame"
        )
    elif not expiries:
        wait_s = 0.0
        source = EXPIRY_WAIT_NO_COMMAND
    else:
        wait_s = max(0.0, max(expiries) + TRAJECTORY_EXPIRY_MARGIN_S - clock())
        source = EXPIRY_WAIT_RECORDED
    if wait_s > 0.0:
        sleep(wait_s)
    logger.info(
        f"waited {wait_s:.3f}s for the previous arm trajectory to expire "
        f"({source}; last goal_time "
        + ", ".join(
            f"{name}={'none' if value is None else f'{value:.3f}s'}"
            for name, value in goal_times.items()
        )
        + ")"
    )
    return ExpiryWait(wait_s=wait_s, source=source, previous_goal_time_s=goal_times)


def _read_positions(handles: Sequence[ArmHandle]) -> dict[str, float]:
    positions: dict[str, float] = {}
    for handle in handles:
        values = [float(value) for value in handle.driver.get_arm_positions()]
        if len(values) != len(handle.keys):
            # The arm-only vector's length is the driver's answer to "how many arm
            # joints", and the name mapping above is only right if it agrees.
            raise StageRepositionError(
                f"{handle.name}: driver.get_arm_positions() returned {len(values)} "
                f"values for {len(handle.keys)} arm joints {list(handle.keys)}"
            )
        positions.update(zip(handle.keys, values, strict=True))
    return positions


def paced_goal_time(
    handles: Sequence[ArmHandle],
    start: Mapping[str, float],
    target: Mapping[str, float],
    min_goal_time_s: float,
) -> float:
    """The goal time that keeps every joint under its safe velocity, floored.

    The same rule WidowXAIFollower.send_action applies per frame
    (widowxai_follower.py:311-320): ``goal_time >= delta / (velocity_max *
    velocity_safety_factor)`` for every joint, where velocity_max is the driver's
    hard limit and the factor (0.4 on this robot) leaves room for the trajectory's
    PEAK velocity, measured at ~2.07x the commanded average (the comment on
    velocity_safety_factor in config_widowxai_follower.py). ONE goal time for both
    arms -- the larger of the two -- so they arrive together and the settle check
    below waits on one deadline.

    ``start`` is read AFTER wait_for_previous_trajectory, so it is where the
    boundary hold left the arms, not a pose they were still moving through.

    The floor is ``min_goal_time_s`` rather than the per-frame 0.1 s minimum:
    connect() and disconnect() move to the staged pose over a fixed 2.0 s
    (widowxai_follower.py:149-153, :356-360), and a start-pose move -- possibly
    with an object in the gripper -- should be no faster than that.
    """
    goal_time = float(min_goal_time_s)
    for handle in handles:
        limits = list(handle.driver.get_joint_limits())
        for index, key in enumerate(handle.keys):
            v_max = float(getattr(limits[index], "velocity_max", 0.0))
            safe_v = v_max * handle.velocity_safety_factor
            if safe_v > 0:
                goal_time = max(goal_time, abs(target[key] - start[key]) / safe_v)
    return goal_time


def _check_position_limits(
    handles: Sequence[ArmHandle], target: Mapping[str, float]
) -> None:
    """Refuse a target outside a joint's position limits before anything moves.

    The demo's first frame was a reachable pose of SOME arm, but limits are a
    per-controller setting (set_joint_limits exists on the driver), and a target
    past one is better refused with a name than handed to the controller.
    """
    problems: list[str] = []
    for handle in handles:
        limits = list(handle.driver.get_joint_limits())
        for index, key in enumerate(handle.keys):
            low = getattr(limits[index], "position_min", None)
            high = getattr(limits[index], "position_max", None)
            value = target[key]
            if low is not None and value < float(low):
                problems.append(f"{key}={value:.4f} < position_min {float(low):.4f}")
            if high is not None and value > float(high):
                problems.append(f"{key}={value:.4f} > position_max {float(high):.4f}")
    if problems:
        raise StageRepositionError(
            "start pose outside the joint limits: " + "; ".join(problems)
        )


def move_arms_to_start_pose(
    robot: Any,
    positions: Mapping[str, float],
    *,
    min_goal_time_s: float,
    position_tolerance_rad: float,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.perf_counter,
) -> RepositionResult:
    """Drive both arms' joints to ``positions`` and wait until they are there.

    Raises StageRepositionError for a pose that cannot be commanded (structure,
    keys, limits) BEFORE the first driver call -- and before the expiry wait, so
    a refusal costs no time; any driver exception propagates.
    Returns ``reached=False`` when the arms were commanded but did not come within
    ``position_tolerance_rad`` by SETTLE_TIMEOUT_S after the blocking call -- the
    runner then refuses to start the stage from that pose.

    THE MOVE IS BLOCKING, in the sense that matters: this function does not return
    until the move is over. The left arm is commanded with ``blocking=False`` and
    the right arm with ``blocking=True`` and the same goal_time, so both
    trajectories run concurrently and the right arm's call is the wait; issuing
    both blocking would move the arms one after the other and double the time.
    The settle check then reads BOTH arms back, which is what covers the left arm.

    BEFORE THE MOVE, wait_for_previous_trajectory sleeps until the last command
    each arm received (the boundary hold, usually ~0.1 s) has expired plus
    TRAJECTORY_EXPIRY_MARGIN_S: otherwise the driver seeds the multi-second move
    with that command's mid-course acceleration and the arms swing. The start
    positions are read AFTER the wait, so the goal_time is paced from where the
    hold left the arms. The wait sends nothing.

    No frame is recorded here: this runs between record_loop calls, so the
    episode buffer does not grow and frame_idx is the same before and after.
    """
    handles = arm_handles(robot)
    problems = start_pose_problems(positions, handles)
    if problems:
        raise StageRepositionError("invalid start pose: " + "; ".join(problems))
    target = {key: float(value) for key, value in positions.items()}
    _check_position_limits(handles, target)

    expiry = wait_for_previous_trajectory(handles, sleep=sleep, clock=clock)
    start = _read_positions(handles)
    goal_time = paced_goal_time(handles, start, target, min_goal_time_s)
    logger.info(
        f"moving arms to the start pose over {goal_time:.2f}s "
        f"(max joint delta {max(abs(target[k] - start[k]) for k in target):.3f} rad); "
        "grippers and base are not commanded"
    )
    for index, handle in enumerate(handles):
        last = index == len(handles) - 1
        handle.driver.set_arm_positions(
            # BY NAME into driver order: handle.keys is built from
            # config.joint_names, which is the driver's joint order.
            [target[key] for key in handle.keys],
            goal_time=goal_time,
            blocking=last,
        )

    deadline = clock() + SETTLE_TIMEOUT_S
    while True:
        end = _read_positions(handles)
        max_abs_error = max(abs(end[key] - target[key]) for key in target)
        if max_abs_error <= position_tolerance_rad:
            return RepositionResult(
                reached=True,
                goal_time_s=goal_time,
                start_positions=start,
                end_positions=end,
                max_abs_error=max_abs_error,
                detail=(
                    f"reached within {position_tolerance_rad} rad "
                    f"(max error {max_abs_error:.4f} rad)"
                ),
                expiry_wait_s=expiry.wait_s,
                expiry_wait_source=expiry.source,
                previous_goal_time_s=dict(expiry.previous_goal_time_s),
            )
        if clock() >= deadline:
            worst = max(target, key=lambda key: abs(end[key] - target[key]))
            return RepositionResult(
                reached=False,
                goal_time_s=goal_time,
                start_positions=start,
                end_positions=end,
                max_abs_error=max_abs_error,
                detail=(
                    f"start pose NOT reached: {worst} is {end[worst]:.4f} rad, "
                    f"target {target[worst]:.4f} (error {max_abs_error:.4f} > "
                    f"tolerance {position_tolerance_rad}) {SETTLE_TIMEOUT_S}s after "
                    f"the {goal_time:.2f}s move"
                ),
                expiry_wait_s=expiry.wait_s,
                expiry_wait_source=expiry.source,
                previous_goal_time_s=dict(expiry.previous_goal_time_s),
            )
        sleep(SETTLE_POLL_S)
