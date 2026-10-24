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
        self._joint_positions: dict[str, float] = {
            name: (index + 1) * _JOINT_SEED_STEP_RAD
            for index, name in enumerate(self._joint_names)
        }
        self._base_velocity: dict[str, float] = {"x.vel": 0.0, "theta.vel": 0.0}

    @property
    def _joint_names(self) -> list[str]:
        # Flat joint_0..joint_N-1 rather than the real left_/right_ prefixes:
        # nothing in the runner parses arm membership, only the ".pos" suffix
        # and the dimension count, and a flat list keeps joint_count honest.
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
            # WidowXAIFollower.send_action:272 evaluates
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
