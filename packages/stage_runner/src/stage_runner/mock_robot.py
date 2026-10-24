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
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
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

# MobileAIRobot's command clamp (_MAX_BASE_CMD_*_VEL in mobileai.py), applied
# on the simulated base path only, so a tracker that saturates sees the same
# ceiling as on the robot.
_MOCK_MAX_BASE_CMD: float = 1.0
# Integration step of the simulated base (s). A control tick is ~33 ms; a
# turn integrated in one step per tick would drift by the chord error.
_MOCK_BASE_SUBSTEP_S: float = 0.005
# The landmark's apparent scale is clamped to this range so a simulated base
# that drives through the landmark renders a huge sprite, not a division by 0.
_MOCK_SCENE_SCALE_RANGE: tuple[float, float] = (0.2, 3.0)


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
    # OPT-IN simulated base (MockSlate) with odometry, for the base_replay and
    # base_align stages. Off by default: every existing smoke config keeps a
    # robot with no .base at all, and tests that need one attach their own.
    # On, send_action drives robot.base.set_cmd_vel and get_observation calls
    # robot.base.update_state(), as MobileAIRobot does, and the observation's
    # x.vel / theta.vel become the MEASURED velocity instead of the command.
    simulate_base: bool = False
    # First-order lag of the simulated base's velocity behind its command (s).
    # 0 = the velocity follows instantly; the real base measured 0.10-0.16 s.
    base_lag_s: float = 0.0
    # Same name and meaning as MobileAIRobotConfig.enable_base_motor_torque:
    # whether connect() turns the base motors on. With it off the simulated
    # base accepts commands and does not move, as the real one does. True here
    # (the real default is False) so a simulate_base YAML needs no extra line;
    # it has no effect without a .base, which is the default.
    enable_base_motor_torque: bool = True
    # OPT-IN synthetic camera (mock_scene.render_scene) under this observation
    # key, e.g. "cam_high". None (default) = no cameras, as before. Only when
    # set is mock_scene -- and cv2 with it -- imported.
    scene_camera: str | None = None
    scene_width: int = 640
    scene_height: int = 480
    # The scene's pinhole matrix. The mock has no camera config entry to read
    # intrinsics from, so an align stage on it gives the same matrix in YAML.
    scene_camera_matrix: list[list[float]] = field(
        default_factory=lambda: [
            [392.6346, 0.0, 322.7227],
            [0.0, 392.0662, 236.2555],
            [0.0, 0.0, 1.0],
        ]
    )
    # Robot vertical axis in camera coordinates (the 10/24 estimate).
    scene_up: list[float] = field(
        default_factory=lambda: [-0.06288349, -0.84574326, -0.52987169]
    )
    scene_seed: int = 0
    # Odometry pose [x m, y m, theta rad] at which the camera sees the
    # reference view (yaw 0, landmark at its reference size).
    scene_reference_pose: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    # Distance from the reference pose to the landmark, straight ahead (m).
    # The landmark's apparent scale is this over the live distance to it.
    scene_landmark_distance_m: float = 0.60


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


def _wrap_angle(angle: float) -> float:
    """[-pi, pi), the range TrossenSlate.get_pose reports theta in."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class MockSlate:
    """The slice of trossen_slate.TrossenSlate the base stages use, simulated.

    A unicycle whose velocity follows the last ACCEPTED command through a
    first-order lag, integrated against the wall clock -- the stages tick in
    real time, so wall time is what makes a 0.5 rad/s command turn 0.5 rad per
    second here as on the robot. Like the real driver, ``get_pose`` and
    ``get_vel`` read a cache that only ``update_state`` refreshes, and
    ``update_state`` / ``set_cmd_vel`` return a bool instead of raising.

    TEST HOOKS, each modelling one field failure:

    * ``fail_updates`` -- that many next ``update_state`` calls return False
      and leave the cache untouched (a dropped serial read); the base keeps
      moving meanwhile.
    * ``refuse_cmds`` -- that many next ``set_cmd_vel`` calls return False and
      are ignored (the e-stop refusing commands, stage1_align.py:127-132).
    * ``ignore_cmds`` -- commands return True and do nothing (torque off, a
      stalled wheel).
    * ``command_gain`` -- the command is multiplied by this before it acts;
      -1 is a base that moves the wrong way.
    * ``pose_override`` / ``velocity_override`` -- returned by ``get_pose`` /
      ``get_vel`` instead of the cache (NaN, garbage, a jump, a base that never
      reads still).

    NOTHING HERE IS EVIDENCE ABOUT THE REAL BASE: no wheel slip, no deadband,
    no small-turn shortfall (10/24: 1.77 deg asked, 0.86 deg turned).
    """

    def __init__(
        self, *, lag_s: float = 0.0, clock: Callable[[], float] = time.perf_counter
    ) -> None:
        self.lag_s = float(lag_s)
        self._clock = clock
        self._x = 0.0
        self._y = 0.0
        # Unwrapped internally; get_pose wraps it like the driver.
        self._heading = 0.0
        self._v = 0.0
        self._w = 0.0
        self._command = (0.0, 0.0)
        self._last_t: float | None = None
        self._cached_pose = [0.0, 0.0, 0.0]
        self._cached_vel = [0.0, 0.0]
        self.torque_enabled = False
        self.fail_updates = 0
        self.refuse_cmds = 0
        self.ignore_cmds = False
        self.command_gain = 1.0
        self.pose_override: list[float] | None = None
        self.velocity_override: list[float] | None = None
        # Every set_cmd_vel call as (linear, angular, accepted), in order.
        self.commands: list[tuple[float, float, bool]] = []
        self.update_calls = 0
        self.update_failures = 0

    def _advance(self) -> None:
        """Integrate the motion from the last call up to now."""
        now = self._clock()
        if self._last_t is None:
            self._last_t = now
            return
        remaining = max(0.0, now - self._last_t)
        self._last_t = now
        if self.torque_enabled and not self.ignore_cmds:
            target_v, target_w = self._command
        else:
            target_v, target_w = 0.0, 0.0
        while remaining > 0.0:
            h = min(_MOCK_BASE_SUBSTEP_S, remaining)
            remaining -= h
            alpha = 1.0 if self.lag_s <= 0.0 else 1.0 - math.exp(-h / self.lag_s)
            self._v += alpha * (target_v - self._v)
            self._w += alpha * (target_w - self._w)
            mid = self._heading + self._w * h / 2.0
            self._x += self._v * math.cos(mid) * h
            self._y += self._v * math.sin(mid) * h
            self._heading += self._w * h

    def true_pose(self) -> tuple[float, float, float]:
        """Where the base really is now (x, y, unwrapped heading), cache or not.

        What a camera sees: the scene camera renders from this, so an odometry
        hook (pose_override, fail_updates) does not also blind the camera.
        """
        self._advance()
        return self._x, self._y, self._heading

    def place(self, x: float, y: float, heading: float) -> None:
        """Test helper: put the base somewhere, at rest, and refresh the cache."""
        self._advance()
        self._x, self._y, self._heading = float(x), float(y), float(heading)
        self._v = self._w = 0.0
        self._cached_pose = [self._x, self._y, _wrap_angle(self._heading)]
        self._cached_vel = [0.0, 0.0]

    def update_state(self) -> bool:
        self._advance()
        self.update_calls += 1
        if self.fail_updates > 0:
            self.fail_updates -= 1
            self.update_failures += 1
            return False
        self._cached_pose = [self._x, self._y, _wrap_angle(self._heading)]
        self._cached_vel = [self._v, self._w]
        return True

    def get_pose(self) -> list[float]:
        if self.pose_override is not None:
            return list(self.pose_override)
        return list(self._cached_pose)

    def get_vel(self) -> list[float]:
        if self.velocity_override is not None:
            return list(self.velocity_override)
        return list(self._cached_vel)

    def set_cmd_vel(self, linear: float, angular: float) -> bool:
        # The old command acts up to now; the new one from now on.
        self._advance()
        if self.refuse_cmds > 0:
            self.refuse_cmds -= 1
            self.commands.append((float(linear), float(angular), False))
            return False
        self.commands.append((float(linear), float(angular), True))
        self._command = (
            float(linear) * self.command_gain,
            float(angular) * self.command_gain,
        )
        return True

    def enable_motor_torque(self, enable: bool) -> tuple[bool, str]:
        self._advance()
        self.torque_enabled = bool(enable)
        return True, ""


class MockSceneCamera:
    """A camera that renders mock_scene from the simulated base's true pose.

    The view yaw is the base heading minus ``scene_reference_pose``'s, and the
    landmark sits ``scene_landmark_distance_m`` straight ahead of the reference
    pose, so turning shows up as yaw and driving toward it as scale. A sideways
    offset is NOT rendered (the scene is a pure rotation): the field's "a
    sideways offset folds into the yaw" (landmark_align trap 6) does not exist
    here.

    Every read returns a NEW array: frames go into the dataset by reference and
    the image writer saves them later, so a shared buffer would be overwritten
    under it.
    """

    def __init__(self, robot: "MockRobot", key: str) -> None:
        from stage_runner import mock_scene  # lazy: cv2 only when opted in

        self._render = mock_scene.render_scene
        self._robot = robot
        self.key = key
        config = robot.config
        self.width = int(config.scene_width)
        self.height = int(config.scene_height)
        self._camera_matrix = tuple(
            tuple(float(v) for v in row) for row in config.scene_camera_matrix
        )
        self._up = tuple(float(v) for v in config.scene_up)
        self._seed = int(config.scene_seed)
        self._reference = tuple(float(v) for v in config.scene_reference_pose)
        self._distance = float(config.scene_landmark_distance_m)
        self._connected = False
        self._cache_key: tuple[float, float] | None = None
        self._cache_image: Any = None
        # Test hook: (yaw_deg, landmark_scale) to render instead of the pose.
        self.view_override: tuple[float, float] | None = None
        self.reads = 0

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self, warmup: bool = True) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def view(self) -> tuple[float, float]:
        """(yaw_deg, landmark_scale) the camera sees from where the base is."""
        if self.view_override is not None:
            return self.view_override
        base = getattr(self._robot, "base", None)
        x_ref, y_ref, theta_ref = self._reference
        if base is None or not hasattr(base, "true_pose"):
            return 0.0, 1.0
        x, y, heading = base.true_pose()
        yaw_deg = math.degrees(_wrap_angle(heading - theta_ref))
        landmark_x = x_ref + self._distance * math.cos(theta_ref)
        landmark_y = y_ref + self._distance * math.sin(theta_ref)
        live_distance = math.hypot(landmark_x - x, landmark_y - y)
        low, high = _MOCK_SCENE_SCALE_RANGE
        scale = self._distance / max(live_distance, 1e-6)
        return yaw_deg, min(high, max(low, scale))

    def _frame(self) -> Any:
        if not self._connected:
            raise DeviceNotConnectedError(
                f"mock scene camera {self.key} is not connected."
            )
        self.reads += 1
        yaw_deg, scale = self.view()
        # Rendering costs a few ms; a still base asks for the same view every
        # tick, so the last image is reused (still copied, see the class doc).
        key = (round(yaw_deg, 4), round(scale, 5))
        if key != self._cache_key:
            self._cache_image = self._render(
                view_yaw_deg=yaw_deg,
                landmark_scale=scale,
                camera_matrix=self._camera_matrix,
                up=self._up,
                width=self.width,
                height=self.height,
                seed=self._seed,
            )
            self._cache_key = key
        return self._cache_image.copy()

    def async_read(self, timeout_ms: float = 200) -> Any:
        return self._frame()

    def read(self) -> Any:
        return self._frame()

    def read_latest(self, max_age_ms: int = 500) -> Any:
        return self._frame()


def _sanitize_base_command(x_vel: float, theta_vel: float) -> tuple[float, float]:
    """MobileAIRobot's command guard: non-finite -> 0.0, then clamp to +-1."""
    x_vel = x_vel if math.isfinite(x_vel) else 0.0
    theta_vel = theta_vel if math.isfinite(theta_vel) else 0.0
    return (
        max(-_MOCK_MAX_BASE_CMD, min(_MOCK_MAX_BASE_CMD, x_vel)),
        max(-_MOCK_MAX_BASE_CMD, min(_MOCK_MAX_BASE_CMD, theta_vel)),
    )


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
        # No cameras by default (scene_camera opts one in, below):
        # image_writer_threads then computes to 0, so no image writer and no
        # video encoding run and the smoke path never touches ffmpeg. THE RATE THIS ROBOT SUSTAINS IS THEREFORE AN ARTIFACT -- it
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
        self._check_simulation_config()
        # Built in __init__, not connect(), as MobileAIRobot builds its
        # TrossenSlate (mobileai.py:221): preflight inspects robot.base before
        # anything is powered.
        if config.simulate_base:
            self.base = MockSlate(lag_s=config.base_lag_s)
        if config.scene_camera is not None:
            self.cameras = {
                config.scene_camera: MockSceneCamera(self, config.scene_camera)
            }

    def _check_simulation_config(self) -> None:
        config = self.config
        if config.base_lag_s < 0.0 or not math.isfinite(config.base_lag_s):
            raise ValueError(
                f"base_lag_s must be a finite value >= 0, got {config.base_lag_s}"
            )
        if config.scene_camera is None:
            return
        if not config.scene_camera:
            raise ValueError("scene_camera must be a non-empty observation key or null")
        if config.scene_width <= 0 or config.scene_height <= 0:
            raise ValueError(
                f"scene size must be positive, got {config.scene_width}x{config.scene_height}"
            )
        matrix = config.scene_camera_matrix
        if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
            raise ValueError(f"scene_camera_matrix must be 3x3, got {matrix}")
        if len(config.scene_up) != 3 or math.hypot(*config.scene_up) == 0.0:
            raise ValueError(
                f"scene_up must be a non-zero 3-vector, got {config.scene_up}"
            )
        if len(config.scene_reference_pose) != 3:
            raise ValueError(
                f"scene_reference_pose must be [x, y, theta], got {config.scene_reference_pose}"
            )
        if not config.scene_landmark_distance_m > 0.0:
            raise ValueError(
                f"scene_landmark_distance_m must be > 0, got {config.scene_landmark_distance_m}"
            )

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
    def _camera_ft(self) -> dict[str, tuple]:
        # (height, width, 3), the shape MobileAIRobot declares per camera
        # (mobileai.py:250-254).
        return {
            key: (camera.height, camera.width, 3)
            for key, camera in self.cameras.items()
        }

    @property
    def observation_features(self) -> dict[str, type | tuple]:
        base_ft = self._base_ft if self.config.include_base_in_state else {}
        return {**self._joint_ft, **base_ft, **self._camera_ft}

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
        if self.config.simulate_base:
            # MobileAIRobot.connect switches base torque to the config value
            # (mobileai.py:286); the simulated base starts integrating here.
            self.base.enable_motor_torque(self.config.enable_base_motor_torque)
            self.base.update_state()
        for camera in self.cameras.values():
            camera.connect()
        self._connected = True
        cameras = ", ".join(self.cameras) or "no cameras"
        logger.info(f"{self} connected ({self.config.joint_count} joints, {cameras})")

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
        if self.config.simulate_base:
            # As MobileAIRobot.get_observation: refresh the base cache, then
            # report the MEASURED velocity (mobileai.py:423-452). A failed
            # refresh leaves the cache as it was, which is what the real robot
            # then reports too.
            self.base.update_state()
            if self.config.include_base_in_state:
                x_vel, theta_vel = (float(v) for v in self.base.get_vel()[:2])
                observation.update({"x.vel": x_vel, "theta.vel": theta_vel})
        elif self.config.include_base_in_state:
            observation.update(self._base_velocity)
        for key, camera in self.cameras.items():
            observation[key] = camera.async_read()
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
        if self.config.simulate_base:
            # The command reaches the base through robot.base, as on
            # MobileAIRobot.send_action (mobileai.py:475-490), so whatever wraps
            # robot.base sees it; a refusal is not an exception there either.
            x_vel, theta_vel = _sanitize_base_command(x_vel, theta_vel)
            if not self.base.set_cmd_vel(x_vel, theta_vel):
                logger.warning(f"{self} base refused set_cmd_vel({x_vel}, {theta_vel})")
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
        if self.config.simulate_base:
            self.base.set_cmd_vel(0.0, 0.0)
        for camera in self.cameras.values():
            camera.disconnect()
        self._connected = False
        logger.info(f"{self} disconnected")
