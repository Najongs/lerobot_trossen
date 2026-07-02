import logging
import math
import threading
import time
from typing import Any

from trossen_slate import TrossenSlate

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots import Robot

from lerobot_robot_trossen import BiWidowXAIFollowerRobot, BiWidowXAIFollowerRobotConfig
from lerobot_robot_trossen.config_mobileai import MobileAIRobotConfig

logger = logging.getLogger(__name__)

# Shared state to allow teleoperator to access the latest base velocity from the robot
# This is used for passive recording of base movement
_base_velocity_lock = threading.Lock()
_latest_base_velocity = {"x.vel": 0.0, "theta.vel": 0.0}

# Sanity bounds for base velocity readings. The SLATE base maxes out at 1.0 m/s
# linear; these limits are generous but far below the garbage magnitudes (~1e28)
# produced when trossen_slate returns uninitialized/stale serial buffer bytes
# reinterpreted as float.
_MAX_BASE_LINEAR_VEL = 5.0  # m/s
_MAX_BASE_ANGULAR_VEL = 10.0  # rad/s


def _sanitize_base_velocity(x_vel: float, theta_vel: float) -> tuple[float, float]:
    """Replace non-finite or out-of-range base velocities with 0.0.

    Guards against corrupted readings from ``TrossenSlate.get_vel()`` (notably the
    first read of an episode), which would otherwise poison dataset stats and bake
    NaN into the policy normalizer.
    """
    if not math.isfinite(x_vel) or abs(x_vel) > _MAX_BASE_LINEAR_VEL:
        logger.warning(f"Discarding invalid base x.vel reading: {x_vel!r} -> 0.0")
        x_vel = 0.0
    if not math.isfinite(theta_vel) or abs(theta_vel) > _MAX_BASE_ANGULAR_VEL:
        logger.warning(
            f"Discarding invalid base theta.vel reading: {theta_vel!r} -> 0.0"
        )
        theta_vel = 0.0
    return x_vel, theta_vel


def get_latest_base_velocity() -> dict[str, float]:
    with _base_velocity_lock:
        return _latest_base_velocity.copy()


class MobileAIRobot(Robot):
    """
    [Mobile AI](https://www.trossenrobotics.com/mobile-ai) by Trossen Robotics
    """

    config_class = MobileAIRobotConfig
    name = "mobileai_robot"

    def __init__(self, config: MobileAIRobotConfig):
        super().__init__(config)
        self.config = config

        arms_config = BiWidowXAIFollowerRobotConfig(
            left_arm_ip_address=config.left_arm_ip_address,
            right_arm_ip_address=config.right_arm_ip_address,
            left_arm_max_relative_target=config.left_arm_max_relative_target,
            right_arm_max_relative_target=config.right_arm_max_relative_target,
            min_time_to_move_multiplier=config.min_time_to_move_multiplier,
            loop_rate=config.loop_rate,
            include_velocity=config.include_velocity,
            include_effort=config.include_effort,
            include_external_effort=config.include_external_effort,
            cameras={},
        )

        self.arms = BiWidowXAIFollowerRobot(arms_config)
        self.base = TrossenSlate()

        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _base_ft(self) -> dict[str, type]:
        return {"x.vel": float, "theta.vel": float}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.cameras
        }

    @property
    def observation_features(self) -> dict[str, type | tuple]:
        # Arm features (flag-aware: .pos plus optional .vel/.eff/.ext_eff) come from the bimanual
        # arms, plus the mobile base velocity (optional) and the shared cameras.
        base_ft = self._base_ft if self.config.include_base_in_state else {}
        return {**self.arms.observation_features, **base_ft, **self._cameras_ft}

    @property
    def action_features(self) -> dict[str, type]:
        return {**self.arms.action_features, **self._base_ft}

    @property
    def is_connected(self) -> bool:
        return self.arms.is_connected and all(
            cam.is_connected for cam in self.cameras.values()
        )

    def connect(self, calibrate: bool = True) -> None:
        self.arms.connect(calibrate=calibrate)
        base_init_success, message = self.base.init_base()
        if not base_init_success:
            raise ConnectionError(f"Failed to connect to Mobile AI base: {message}")

        self.base.enable_motor_torque(self.config.enable_base_motor_torque)

        for cam in self.cameras.values():
            cam.connect()

    @property
    def is_calibrated(self) -> bool:
        # Trossen Arm robots do not require calibration but we check both arms for consistency
        return self.arms.is_calibrated

    def calibrate(self) -> None:
        # Trossen Arm robots do not require calibration but we call calibrate on both arms for
        # consistency
        self.arms.calibrate()

    def configure(self) -> None:
        # Set the arm to position control mode
        self.arms.configure()

    def get_observation(self) -> dict[str, Any]:
        obs_dict = {}

        # Get arm observations
        arms_obs = self.arms.get_observation()
        obs_dict.update(arms_obs)

        # Get base observations. Refresh the cached chassis state first so get_vel()
        # does not return a stale/uninitialized buffer (the cause of garbage base
        # velocities on the first frame of an episode), then sanity-check the result.
        if not self.base.update_state():
            logger.warning(
                "Failed to refresh Mobile AI base state; using last cached velocity."
            )
        base_obs = self.base.get_vel()
        x_vel, theta_vel = _sanitize_base_velocity(base_obs[0], base_obs[1])

        # Update shared state (always, so the teleoperator can passively record base
        # movement even when base velocity is excluded from the observation).
        with _base_velocity_lock:
            _latest_base_velocity["x.vel"] = x_vel
            _latest_base_velocity["theta.vel"] = theta_vel

        # Expose base velocity as an observation feature only when configured. The
        # _nobasestate policies expect a 14-dim observation.state (arms only); adding
        # base here would make it 16-dim and break the policy normalizer.
        if self.config.include_base_in_state:
            obs_dict.update({"x.vel": x_vel, "theta.vel": theta_vel})

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        send_action_arms = self.arms.send_action(
            {k: v for k, v in action.items() if k in self.arms.action_features}
        )
        action_base_x_vel = action.get("x.vel", 0.0)
        action_base_theta_vel = action.get("theta.vel", 0.0)
        self.base.set_cmd_vel(action_base_x_vel, action_base_theta_vel)

        return {
            **send_action_arms,
            "x.vel": action_base_x_vel,
            "theta.vel": action_base_theta_vel,
        }

    def disconnect(self):
        if not self.base.set_cmd_vel(0.0, 0.0):
            # We log a warning but continue with disconnect
            logger.warning("Failed to stop Mobile AI base during disconnect.")

        try:
            self.arms.disconnect()
        except Exception as e:
            # We log a warning but continue with disconnect
            logger.warning(f"Error while disconnecting arms: {e}")

        for cam in self.cameras.values():
            cam.disconnect()
