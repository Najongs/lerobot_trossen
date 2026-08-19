import logging
import math
import os
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


# Hardware clamp bounds enforced inside TrossenSlate::set_cmd_vel. Verified in
# the installed trossen_slate 0.0.3 binary: the clamp is
# min(MAX, max(-MAX, v)) against the constants at .rodata 0x3e9f0 (-1.0f) and
# 0x3e9f4 (+1.0f). NaN compares false against everything, so max(-MAX, NaN)
# returns -MAX -- a NaN command reaches the base as full-speed reverse, and eval
# runs with enable_base_motor_torque=True, so that command is actually executed.
_MAX_BASE_CMD_LINEAR_VEL = 1.0  # m/s
_MAX_BASE_CMD_ANGULAR_VEL = 1.0  # rad/s

# A failing serial link would emit one warning per control-loop iteration
# (~20/s) and bury everything else, so base warnings are throttled per key.
_BASE_WARN_INTERVAL_S = 1.0
_last_base_warn: dict[str, float] = {}


def _warn_throttled(key: str, message: str) -> None:
    """Emit a warning at most once per ``_BASE_WARN_INTERVAL_S`` per key."""
    now = time.monotonic()
    last = _last_base_warn.get(key)
    if last is None or now - last >= _BASE_WARN_INTERVAL_S:
        _last_base_warn[key] = now
        logger.warning(message)


def _sanitize_base_command(x_vel: float, theta_vel: float) -> tuple[float, float]:
    """Clamp base velocity commands and replace non-finite values with 0.0.

    The read path has had this guard since the base velocity NaN incident
    (:func:`_sanitize_base_velocity`); the command path did not. The asymmetry
    was backwards: a corrupted reading poisons a dataset, a corrupted command
    drives the robot.
    """
    if not math.isfinite(x_vel):
        _warn_throttled("cmd_x", f"Non-finite base x.vel command {x_vel!r} -> 0.0")
        x_vel = 0.0
    if not math.isfinite(theta_vel):
        _warn_throttled(
            "cmd_theta", f"Non-finite base theta.vel command {theta_vel!r} -> 0.0"
        )
        theta_vel = 0.0
    x_vel = max(-_MAX_BASE_CMD_LINEAR_VEL, min(_MAX_BASE_CMD_LINEAR_VEL, x_vel))
    theta_vel = max(
        -_MAX_BASE_CMD_ANGULAR_VEL, min(_MAX_BASE_CMD_ANGULAR_VEL, theta_vel)
    )
    return x_vel, theta_vel


def get_latest_base_velocity() -> dict[str, float]:
    with _base_velocity_lock:
        return _latest_base_velocity.copy()


# Control-loop rate meter. send_action() runs exactly once per record/eval loop
# iteration, and the base velocity command set there is held until the *next*
# send_action(), so the wall-clock interval between consecutive calls is exactly
# the integration window that turns a base velocity command into rotation. A loop
# running at 15 Hz instead of the target 30 Hz therefore doubles every base
# rotation (the "~2x over-rotation" symptom). lerobot 0.4.0's record loop does
# NOT warn on slowdown -- busy_wait() with a negative argument just returns -- so
# we surface the real rate here instead.
#
# Opt-in diagnostic: set LEROBOT_LOOP_HZ_LOG=1 to enable. Off by default so normal
# operation has no extra logging and no accumulation overhead.
_LOOP_HZ_LOG_ENABLED = os.getenv("LEROBOT_LOOP_HZ_LOG", "").strip().lower() not in (
    "",
    "0",
    "false",
    "no",
)
_LOOP_HZ_WINDOW = 30  # frames per summary line (~1 s at 30 fps)
_LOOP_HZ_RESET_GAP_S = 1.0  # gaps longer than this (episode reset) are not counted
# "sec" accumulates per-loop-section wall time (seconds) over the window so the
# summary can pinpoint *where* a slow loop spends its time (arms read/write vs
# base I/O vs camera reads) in a single run, instead of black-box camera-drop trials.
_loop_hz_meter = {"prev_t": None, "count": 0, "sum_dt": 0.0, "max_dt": 0.0, "sec": {}}


def _add_loop_section(name: str, seconds: float) -> None:
    """Accumulate wall time for a named loop section (see _record_loop_tick)."""
    if not _LOOP_HZ_LOG_ENABLED:
        return
    sec = _loop_hz_meter["sec"]
    sec[name] = sec.get(name, 0.0) + seconds


def _reset_loop_window() -> None:
    _loop_hz_meter["count"] = 0
    _loop_hz_meter["sum_dt"] = 0.0
    _loop_hz_meter["max_dt"] = 0.0
    _loop_hz_meter["sec"] = {}


def _record_loop_tick() -> None:
    """Measure and periodically log the real control-loop rate.

    Called once per iteration from ``send_action``. Emits a summary every
    ``_LOOP_HZ_WINDOW`` frames with the mean and min instantaneous Hz over the
    window, plus the mean time spent in each instrumented section. ``target_fps /
    mean_hz`` is the base over-rotation multiplier: a mean near the target fps
    rules the loop-slowdown hypothesis out, a mean near half confirms it. The
    section breakdown (arms/base/cameras) says which I/O is the bottleneck.
    """
    if not _LOOP_HZ_LOG_ENABLED:
        return
    m = _loop_hz_meter
    now = time.perf_counter()
    prev = m["prev_t"]
    m["prev_t"] = now
    if prev is None:
        return
    dt = now - prev
    if dt > _LOOP_HZ_RESET_GAP_S:
        # Episode boundary / reset pause: drop the partial window so a long idle
        # gap does not masquerade as a slow loop.
        _reset_loop_window()
        return
    m["count"] += 1
    m["sum_dt"] += dt
    m["max_dt"] = max(m["max_dt"], dt)
    if m["count"] >= _LOOP_HZ_WINDOW:
        mean_hz = m["count"] / m["sum_dt"]
        min_hz = 1.0 / m["max_dt"]
        sections = "  ".join(
            f"{name}={m['sec'][name] / m['count'] * 1e3:.0f}ms"
            for name in sorted(m["sec"])
        )
        # "other" = loop time not inside any instrumented section (policy
        # select_action/preprocessing, dataset.add_frame, processors, busy_wait).
        other_s = m["sum_dt"] - sum(m["sec"].values())
        other = f"other={other_s / m['count'] * 1e3:.0f}ms"
        logger.info(
            f"Control loop rate over last {m['count']} frames: "
            f"mean={mean_hz:.1f} Hz, min={min_hz:.1f} Hz "
            f"(target_fps / mean_hz = base over-rotation multiplier)"
            + (f" | per-frame: {sections}  {other}" if sections else "")
        )
        _reset_loop_window()


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
            velocity_safety_factor=config.velocity_safety_factor,
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
        _t = time.perf_counter()
        arms_obs = self.arms.get_observation()
        _add_loop_section("arms_read", time.perf_counter() - _t)
        obs_dict.update(arms_obs)

        # Get base observations. Refresh the cached chassis state first so get_vel()
        # does not return a stale/uninitialized buffer (the cause of garbage base
        # velocities on the first frame of an episode), then sanity-check the result.
        _t = time.perf_counter()
        if not self.base.update_state():
            logger.warning(
                "Failed to refresh Mobile AI base state; using last cached velocity."
            )
        base_obs = self.base.get_vel()
        _add_loop_section("base_read", time.perf_counter() - _t)
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
            dt_s = time.perf_counter() - start
            _add_loop_section(f"cam:{cam_key}", dt_s)
            logger.debug(f"{self} read {cam_key}: {dt_s * 1e3:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        # Record the real control-loop period (see _record_loop_tick): the base
        # velocity command below is held until the next call, so a slow loop
        # over-rotates the base proportionally.
        _record_loop_tick()

        _t = time.perf_counter()
        send_action_arms = self.arms.send_action(
            {k: v for k, v in action.items() if k in self.arms.action_features}
        )
        _add_loop_section("arms_write", time.perf_counter() - _t)
        action_base_x_vel, action_base_theta_vel = _sanitize_base_command(
            action.get("x.vel", 0.0), action.get("theta.vel", 0.0)
        )
        _t = time.perf_counter()
        # set_cmd_vel carries the read half of the same Modbus transaction, so a
        # failure means both that the command may not have been applied and that
        # the cached state get_vel() returns is now stale -- the driver leaves the
        # cache untouched on failure and never recovers on its own.
        if not self.base.set_cmd_vel(action_base_x_vel, action_base_theta_vel):
            _warn_throttled(
                "base_write",
                "Mobile AI base transaction failed: the velocity command may not "
                "have been applied and the cached base velocity is now stale.",
            )
        _add_loop_section("base_write", time.perf_counter() - _t)

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
