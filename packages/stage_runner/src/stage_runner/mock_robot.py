"""Hardware-free robot for the smoke path.

lerobot's own ``mock_robot`` registry entry is unusable from an installed
package: ``make_robot_from_config`` handles it with
``from tests.mocks.mock_robot import MockRobot`` (robots/utils.py:81), and the
``tests`` package is not shipped inside the distribution -- the import raises
ModuleNotFoundError (measured 2026-09-08). Hence our own robot, registered
under a distinct type string so the two entries never collide.

Resolution goes through lerobot's real generic factory
(``make_robot_from_config`` -> ``make_device_from_device_class``), so the smoke
path exercises the production selection path with no test branch in production
code.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any

from lerobot.robots.config import RobotConfig
from lerobot.robots.robot import Robot
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

logger = logging.getLogger(__name__)

MOCK_ROBOT_TYPE: str = "stage_runner_mock_robot"

# Seeded so every joint starts at a DIFFERENT non-zero position: a hold action
# that wrongly zeroed the arms instead of reading the present positions would be
# indistinguishable from a correct one if every joint sat at 0.0. The seed is
# (index + 1) * step, not index * step, because joint_0 would otherwise start at
# exactly 0.0 -- the one value such a bug produces -- and the smoke test's
# "no joint is 0.0" assertion would then depend on the mock POLICY having moved
# it, which is a different module's business.
_JOINT_SEED_STEP_RAD: float = 0.1

# `arm_layout` values. "flat" is the original mock (joint_0..joint_N-1, no arm
# membership) and stays the default so every existing smoke config runs
# byte-for-byte as before. "bimanual" reproduces the Mobile AI kit's names and
# structure -- left_/right_ prefixes, a gripper joint last per arm, and
# robot.arms.<side>.driver -- which is what reposition.arm_handles reads, so the
# start pose path runs through the same code on the mock as on hardware.
ARM_LAYOUT_FLAT: str = "flat"
ARM_LAYOUT_BIMANUAL: str = "bimanual"

# One WidowX AI arm's joint names, copied from WidowXAIFollowerConfig.joint_names
# (config_widowxai_follower.py) rather than imported, so the mock keeps working
# on an interpreter without the Trossen package.
_WIDOWXAI_JOINT_NAMES: tuple[str, ...] = (
    "joint_0",
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "left_carriage_joint",
)
_BIMANUAL_PREFIXES: tuple[tuple[str, str], ...] = (
    ("left_arm", "left_"),
    ("right_arm", "right_"),
)
# PLACEHOLDERS, not measurements: the shape of TrossenArmDriver.get_joint_limits()
# so the pacing arithmetic has something to divide by. The 3*pi on joints 3-5 is
# the figure widowxai_follower.py's pacing comment quotes for joints 3 and 4;
# the rest are round numbers. Nothing about the real arm's limits is asserted by
# a test that uses these.
_MOCK_VELOCITY_MAX: tuple[float, ...] = (
    3.14159,
    3.14159,
    3.14159,
    9.42478,
    9.42478,
    9.42478,
    1.0,
)
_MOCK_POSITION_LIMIT_RAD: float = 6.28318
# WidowXAIFollowerConfig.velocity_safety_factor's default.
_MOCK_VELOCITY_SAFETY_FACTOR: float = 0.4
# The goal_time the mock records for every send_action: WidowXAIFollower's
# minimum window at its defaults, min_time_to_move_multiplier / loop_rate =
# 3.0 / 30. The mock does not reproduce the follower's pacing stretch, so a
# test that needs a longer previous trajectory sets the record itself.
_MOCK_MIN_TIME_TO_MOVE_S: float = 3.0 / 30


@RobotConfig.register_subclass(MOCK_ROBOT_TYPE)
@dataclass
class MockRobotConfig(RobotConfig):
    # 14 = the two 7-joint WidowXAI arms of the Mobile AI kit, so the smoke
    # path's state/action dimensions match the real asset without hardware.
    joint_count: int = 14
    # Same meaning as MobileAIRobotConfig.include_base_in_state: dropping the
    # base from the observation shortens observation.state but NOT the action.
    # Reproduced here because that asymmetry is what preflight's separate state
    # and action dimension checks exist for.
    include_base_in_state: bool = True
    # "flat" (default) or "bimanual" -- see ARM_LAYOUT_BIMANUAL. bimanual is
    # always 2 x 7 joints, so joint_count must stay 14 with it.
    arm_layout: str = ARM_LAYOUT_FLAT


@dataclass(frozen=True)
class _MockJointLimit:
    """The three attributes reposition reads off a trossen_arm.JointLimit."""

    velocity_max: float
    position_min: float
    position_max: float


@dataclass
class _MockArmConfig:
    joint_names: list[str]
    velocity_safety_factor: float = _MOCK_VELOCITY_SAFETY_FACTOR


class MockArmDriver:
    """The slice of TrossenArmDriver the start pose move uses, over the mock's joints.

    Reads and writes the robot's own joint dict, so a pose set here is what the
    next get_observation reports and what the next stage's policy starts from.
    The move completes instantly -- no goal_time is slept, which is what keeps
    the smoke run fast and also why nothing here is evidence about timing.

    Every call is RECORDED, and the gripper methods exist only to be recorded:
    the feature under test must never reach them, and a test can only assert
    "the gripper got nothing" if a call to it would leave a trace.
    """

    def __init__(self, robot: "MockRobot", prefix: str) -> None:
        self._robot = robot
        self._prefix = prefix
        self.arm_position_commands: list[dict[str, Any]] = []
        self.gripper_commands: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        # Test hooks. raise_on_set: the exception set_arm_positions raises (the
        # controller refusing a goal). stall_joint: an arm joint name that does
        # not move, to model a joint blocked by contact.
        self.raise_on_set: BaseException | None = None
        self.stall_joint: str | None = None

    @property
    def _arm_names(self) -> list[str]:
        return list(_WIDOWXAI_JOINT_NAMES[:-1])

    def get_arm_positions(self) -> list[float]:
        return [
            self._robot._joint_positions[f"{self._prefix}{name}"]
            for name in self._arm_names
        ]

    def get_all_positions(self) -> list[float]:
        return [
            self._robot._joint_positions[f"{self._prefix}{name}"]
            for name in _WIDOWXAI_JOINT_NAMES
        ]

    def get_joint_limits(self) -> list[_MockJointLimit]:
        return [
            _MockJointLimit(
                velocity_max=v_max,
                position_min=-_MOCK_POSITION_LIMIT_RAD,
                position_max=_MOCK_POSITION_LIMIT_RAD,
            )
            for v_max in _MOCK_VELOCITY_MAX
        ]

    def set_arm_positions(
        self, goal_positions: list[float], goal_time: float = 2.0, blocking: bool = True
    ) -> None:
        self.arm_position_commands.append(
            {
                "goal_positions": [float(value) for value in goal_positions],
                "goal_time": float(goal_time),
                "blocking": bool(blocking),
            }
        )
        if self.raise_on_set is not None:
            raise self.raise_on_set
        if len(goal_positions) != len(self._arm_names):
            # The real driver's contract: "The size of the vectors should be
            # equal to the number of arm joints."
            raise ValueError(
                f"set_arm_positions got {len(goal_positions)} values for "
                f"{len(self._arm_names)} arm joints"
            )
        for name, value in zip(self._arm_names, goal_positions, strict=True):
            if name == self.stall_joint:
                continue
            self._robot._joint_positions[f"{self._prefix}{name}"] = float(value)

    def set_all_positions(self, *args: Any, **kwargs: Any) -> None:
        self.gripper_commands.append(("set_all_positions", args, kwargs))

    def set_gripper_position(self, *args: Any, **kwargs: Any) -> None:
        self.gripper_commands.append(("set_gripper_position", args, kwargs))


class _MockArm:
    def __init__(self, robot: "MockRobot", prefix: str) -> None:
        self.config = _MockArmConfig(joint_names=list(_WIDOWXAI_JOINT_NAMES))
        self.driver = MockArmDriver(robot, prefix)
        # WidowXAIFollower's last-command record (widowxai_follower.py
        # __init__), which reposition waits on before a start pose move. None
        # until the first send_action: the mock's connect() moves nothing, so
        # unlike the real configure() it leaves no trajectory to wait for.
        self.last_command_sent_s: float | None = None
        self.last_command_goal_time_s: float | None = None


class _MockArms:
    """Stand-in for MobileAIRobot.arms (a BiWidowXAIFollowerRobot)."""

    def __init__(self, robot: "MockRobot") -> None:
        self.left_arm = _MockArm(robot, "left_")
        self.right_arm = _MockArm(robot, "right_")


class MockRobot(Robot):
    """A robot that answers from memory instead of from a serial bus.

    Position-controlled arms and a velocity-commanded base are both modeled,
    because the base holding its last command across a control gap is the whole
    reason ``transitions.stop_base`` exists.
    """

    config_class: type[MockRobotConfig] = MockRobotConfig
    name: str = MOCK_ROBOT_TYPE

    def __init__(self, config: MockRobotConfig) -> None:
        super().__init__(config)
        self.config = config
        # No cameras: image_writer_threads then computes to 0, so no image
        # writer and no video encoding run and the smoke path never touches
        # ffmpeg. THE RATE THIS ROBOT SUSTAINS IS THEREFORE AN ARTIFACT -- it
        # answers get_observation out of a dict and pays neither a RealSense
        # read nor a PNG encode, so a clean 30 Hz in a smoke run says nothing
        # about the loop rate on hardware (another session measured 20.4 Hz
        # there). The per-stage `hertz` of the first REAL run is the only
        # measurement of that.
        self.cameras: dict[str, Any] = {}
        # Every action that reached send_action, in order, so a test can assert
        # what the transition actually sent rather than that it did not raise.
        self.sent_actions: list[dict[str, float]] = []
        self._connected = False
        if config.arm_layout not in (ARM_LAYOUT_FLAT, ARM_LAYOUT_BIMANUAL):
            raise ValueError(
                f"unknown arm_layout {config.arm_layout!r}; expected "
                f"{ARM_LAYOUT_FLAT!r} or {ARM_LAYOUT_BIMANUAL!r}"
            )
        if config.arm_layout == ARM_LAYOUT_BIMANUAL and config.joint_count != len(
            _BIMANUAL_PREFIXES
        ) * len(_WIDOWXAI_JOINT_NAMES):
            raise ValueError(
                f"arm_layout bimanual is 2 x {len(_WIDOWXAI_JOINT_NAMES)} joints; "
                f"joint_count is {config.joint_count}"
            )
        self._joint_positions: dict[str, float] = {
            name: (index + 1) * _JOINT_SEED_STEP_RAD
            for index, name in enumerate(self._joint_names)
        }
        self._base_velocity: dict[str, float] = {"x.vel": 0.0, "theta.vel": 0.0}
        # Only the bimanual layout has per-arm drivers. The flat mock has no
        # .arms at all, which is also what makes preflight refuse a start pose on
        # it -- the same refusal a robot without per-arm drivers gets.
        if config.arm_layout == ARM_LAYOUT_BIMANUAL:
            self.arms = _MockArms(self)

    @property
    def _joint_names(self) -> list[str]:
        # Flat joint_0..joint_N-1 by default rather than the real left_/right_
        # prefixes: nothing in the runner's stage path parses arm membership,
        # only the ".pos" suffix and the dimension count, and a flat list keeps
        # joint_count honest. The start pose path DOES parse it, which is what
        # the bimanual layout is for.
        if self.config.arm_layout == ARM_LAYOUT_BIMANUAL:
            return [
                f"{prefix}{name}"
                for _, prefix in _BIMANUAL_PREFIXES
                for name in _WIDOWXAI_JOINT_NAMES
            ]
        return [f"joint_{index}" for index in range(self.config.joint_count)]

    @property
    def _joint_ft(self) -> dict[str, type]:
        return {f"{name}.pos": float for name in self._joint_names}

    @property
    def _base_ft(self) -> dict[str, type]:
        return {"x.vel": float, "theta.vel": float}

    @property
    def observation_features(self) -> dict[str, type | tuple]:
        base_ft = self._base_ft if self.config.include_base_in_state else {}
        return {**self._joint_ft, **base_ft}

    @property
    def action_features(self) -> dict[str, type]:
        # Base velocity is ALWAYS commandable, matching MobileAIRobot
        # (mobileai.py:337-339): include_base_in_state shortens the observation
        # only.
        return {**self._joint_ft, **self._base_ft}

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def connect(self, calibrate: bool = True) -> None:
        if self._connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")
        self._connected = True
        logger.info(f"{self} connected ({self.config.joint_count} joints, no cameras)")

    def calibrate(self) -> None:
        return None

    def configure(self) -> None:
        return None

    def get_observation(self) -> dict[str, Any]:
        if not self._connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        observation: dict[str, Any] = {
            f"{name}.pos": position for name, position in self._joint_positions.items()
        }
        if self.config.include_base_in_state:
            observation.update(self._base_velocity)
        return observation

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self._connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        goal_positions = {
            key.removesuffix(".pos"): float(value)
            for key, value in action.items()
            if key.endswith(".pos")
        }
        missing = [name for name in self._joint_names if name not in goal_positions]
        if missing:
            # WidowXAIFollower.send_action:316 evaluates
            # `goal_pos[joint_name] - present_pos[joint_name]` for EVERY name in
            # config.joint_names, so a base-only action such as
            # {"x.vel": 0.0, "theta.vel": 0.0} is a KeyError there rather than a
            # partial write. Reproduced so the smoke path fails the same way the
            # arm does instead of quietly accepting an action the robot rejects.
            raise KeyError(
                f"{self} received an action with no position for: {', '.join(missing)}"
            )

        # Position-controlled: the arm ends up where it was told, so "last
        # command held" means stopped. The base is the opposite -- the stored
        # velocity keeps being reported until the next command replaces it.
        self._joint_positions.update(goal_positions)
        if self.config.arm_layout == ARM_LAYOUT_BIMANUAL:
            # Left arm first, then right, each stamped after "its" call, as
            # BiWidowXAIFollowerRobot.send_action and the follower do.
            for arm in (self.arms.left_arm, self.arms.right_arm):
                arm.last_command_sent_s = time.perf_counter()
                arm.last_command_goal_time_s = _MOCK_MIN_TIME_TO_MOVE_S
        x_vel = float(action.get("x.vel", 0.0))
        theta_vel = float(action.get("theta.vel", 0.0))
        self._base_velocity = {"x.vel": x_vel, "theta.vel": theta_vel}

        sent: dict[str, float] = {
            **{f"{name}.pos": value for name, value in goal_positions.items()},
            "x.vel": x_vel,
            "theta.vel": theta_vel,
        }
        self.sent_actions.append(sent)
        return sent

    def disconnect(self) -> None:
        if not self._connected:
            # WARN AND RETURN, deliberately not DeviceNotConnectedError like
            # get_observation and send_action. cli.main's finally calls
            # disconnect() unconditionally, because a MobileAIRobot whose
            # cameras failed to enumerate is energised while is_connected is
            # False -- and the real MobileAIRobot.disconnect() has no
            # not-connected guard at all (mobileai.py:568-569, it goes straight
            # to base.set_cmd_vel(0.0, 0.0)). A mock that raises here would make
            # every run whose connect() failed print an ERROR traceback saying
            # "CHECK THE BASE IS STOPPED" about a robot that never moved.
            logger.warning(f"{self} was already disconnected; nothing to do")
            return
        # MobileAIRobot.disconnect() calls base.set_cmd_vel(0.0, 0.0) first
        # (mobileai.py:569), the backstop for a run that died before its own
        # stop_base. Deliberately NOT appended to sent_actions: that list is
        # what a test reads to find the transition action, and a teardown entry
        # after it would make "the last action zeroed the base" true for the
        # wrong reason.
        self._base_velocity = {"x.vel": 0.0, "theta.vel": 0.0}
        self._connected = False
        logger.info(f"{self} disconnected")
