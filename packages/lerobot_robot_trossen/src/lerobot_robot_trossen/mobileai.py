import logging
import math
import threading
import time
from enum import IntEnum
from typing import Any

from termcolor import colored
from trossen_slate import ChassisData, TrossenSlate

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots import Robot

from lerobot_robot_trossen import BiWidowXAIFollowerRobot, BiWidowXAIFollowerRobotConfig
from lerobot_robot_trossen.config_mobileai import MobileAIRobotConfig
from lerobot_robot_trossen.loop_rate_log import add_loop_section, record_loop_tick

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


class SlateBaseSystemState(IntEnum):
    """Values of ``ChassisData.system_state`` reported by the SLATE base.

    trossen_slate exposes the field as a plain int, and this table exists neither
    there nor in lerobot 0.4.4: it lived in the legacy
    ``lerobot/common/robot_devices/utils.py``, which the 0.4 restructuring deleted
    along with the emergency stop check this module restores. The installed
    firmware may therefore report codes that are missing here; see
    :func:`_known_base_state` for why those are ignored rather than reported.
    """

    SYS_INIT = 0x00
    SYS_NORMAL = 0x01
    SYS_REMOTE = 0x02
    SYS_ESTOP = 0x03
    SYS_CALIB = 0x04
    SYS_TEST = 0x05
    SYS_CHARGING = 0x06
    SYS_ERR = 0x10
    SYS_ERR_ID = 0x11
    SYS_ERR_COM = 0x12
    SYS_ERR_ENC = 0x13
    SYS_ERR_COLLISION = 0x14
    SYS_ERR_LOW_VOLTAGE = 0x15
    SYS_ERR_OVER_VOLTAGE = 0x16
    SYS_ERR_OVER_CURRENT = 0x17
    SYS_ERR_OVER_TEMP = 0x18


# lerobot's console format (lerobot/utils/utils.py, custom_format) carries no
# %(name)s and truncates the module path, so a warning from this fork is not
# distinguishable from a lerobot one. Hence an explicit prefix and bold red -- the
# one place where this fork's plain-sentence log convention is broken on purpose.
# termcolor drops the escapes on a non-tty and under NO_COLOR, so there is no
# hand-written isatty guard here.
_BASE_STATE_PREFIX = "[MOBILE AI BASE]"


def _base_state_banner(message: str, color: str = "red") -> str:
    """Prefix and highlight a base state message (see _BASE_STATE_PREFIX).

    The prefix stays on the recovery line too, so one grep finds both ends of an
    incident; only the colour differs, because a red line means "the base was not
    moving" and the closing line means the opposite.
    """
    return colored(f"{_BASE_STATE_PREFIX} {message}", color, attrs=["bold"])


def _known_base_state(state: int) -> SlateBaseSystemState | None:
    """Map a raw ``system_state`` to a known code, or None when it is unlisted.

    The driver's chassis buffer is uninitialized until the first successful
    ``update_state()``, and a partially failed transaction can leave a stale byte
    behind, so a value missing from the table above is far more likely to be
    garbage than a state the firmware means. Callers ignore it entirely rather
    than report it.
    """
    try:
        return SlateBaseSystemState(state)
    except ValueError:
        return None


def _base_state_warning(state: SlateBaseSystemState) -> str | None:
    """Describe an abnormal base system state, or None when it is fine.

    Every state below arrives in the same 26-register transaction that
    ``update_state()`` already performs, so watching more than the emergency stop
    costs nothing. Kept as one function so a state that turns out to be noise in
    practice (docked charging is the likely candidate) can be dropped by deleting
    its branch.
    """
    if state is SlateBaseSystemState.SYS_ESTOP:
        return "Emergency stop is engaged; the base will not move."
    if state is SlateBaseSystemState.SYS_CHARGING:
        return "The base is charging."
    if state >= SlateBaseSystemState.SYS_ERR:
        return f"The base reports a fault: {state.name}."
    return None


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


# Control-loop rate instrumentation lives in loop_rate_log.py. send_action()
# runs exactly once per record/eval loop iteration, and the base velocity
# command set there is held until the *next* send_action(), so the interval
# between calls is the integration window that turns a velocity command into
# rotation. The sections instrumented below are what the summary breaks down.


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
        # Destination struct for TrossenSlate.read(), which only copies the
        # driver's cache, so one buffer is reused for the whole session.
        self._chassis = ChassisData()
        # Last system_state seen on a successful read, so the warning fires on the
        # edge instead of on every frame. Deliberately not seeded in connect(): a
        # run that connects while the base is already in an abnormal state would
        # then have no transition left and would stay silent for its whole length.
        self._last_base_state: SlateBaseSystemState | None = None
        # Latched off after the first failed read, so a driver whose ChassisData
        # differs from the one this was written against costs one warning rather
        # than one exception per frame (see _read_base_state).
        self._base_state_monitor_ok = True
        # Whether a warning is currently open. Separate from _last_base_state on
        # purpose: the base can pass through SYS_INIT or SYS_REMOTE on its way out
        # of an emergency stop, and keying the closing line off the previous state
        # alone would swallow it whenever it does.
        self._base_warning_active = False

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

        # Refuse to start a run against a base that cannot move: an emergency stop
        # left engaged used to be noticed only once the recording was over. Runs
        # before enable_motor_torque so nothing is energized on the way out.
        if self.config.estop_check:
            self._raise_if_base_estopped()

        self.base.enable_motor_torque(self.config.enable_base_motor_torque)

        for cam in self.cameras.values():
            cam.connect()

    def _read_base_state(self, latch: bool = True) -> int | None:
        """Return the cached ``system_state``, or None when it is unavailable.

        ``read()`` copies the struct ``update_state()`` already filled, so this
        costs no serial traffic and is safe to call on every frame. Its return
        value is discarded on purpose: the C++ signature is ``void`` and only the
        docstring claims otherwise, so ``if not read(...)`` would be true forever.

        Never raises. ``record_loop()`` runs both the record and the reset phase
        and ``dataset.save_episode()`` is called after the reset phase, so an
        exception escaping ``get_observation()`` during a reset would discard the
        episode that was just recorded -- and pressing the emergency stop during a
        reset to push the base by hand is normal operation, not an error. A
        mismatch with the installed driver latches the monitor off for the rest of
        the process instead.
        """
        if not self._base_state_monitor_ok:
            return None
        try:
            self.base.read(self._chassis)
            return int(self._chassis.system_state)
        except Exception as e:
            if not latch:
                # connect() is a single call, not a loop, so one failure there is
                # no reason to give up on the per-frame monitor as well.
                logger.warning(f"Could not read Mobile AI base state: {e}.")
                return None
            self._base_state_monitor_ok = False
            logger.warning(
                f"Mobile AI base state monitoring disabled: {e}. The base "
                "emergency stop will not be reported for the rest of this run."
            )
            return None

    def _raise_if_base_estopped(self) -> None:
        """Fail the connection when the base is in emergency stop.

        Called from ``connect()`` only. The refresh below is a single ~21 ms
        Modbus transaction outside the control loop, where the same call would
        de-rate the loop and de-rating the loop multiplies base rotation. It
        cannot be skipped here: the driver's chassis buffer is uninitialized until
        the first successful refresh, so judging it without one judges garbage.

        Fails open on anything short of a confirmed ``SYS_ESTOP``. A missed check
        leaves the behavior we had before this patch, while a false positive
        stops someone from recording at all.
        """
        if not self.base.update_state():
            logger.warning(
                "Could not refresh Mobile AI base state; skipping the emergency "
                "stop check. Confirm the emergency stop is released before "
                "recording."
            )
            return
        if self._read_base_state(latch=False) != SlateBaseSystemState.SYS_ESTOP:
            return
        # Both record()'s finally and Robot.__del__ guard on is_connected, which is
        # False here because the cameras are connected further down, so neither
        # would release the arms: without this they stay torque-enabled and never
        # run their sleep-pose sequence.
        try:
            self.arms.disconnect()
        except Exception as e:
            logger.warning(f"Error while disconnecting arms after e-stop: {e}")
        raise RuntimeError(
            "Robot is in emergency stop state. Please release the emergency stop "
            "button and try again. Pass --robot.estop_check=false to start anyway."
        )

    def _log_base_state_transition(self, state: int) -> None:
        """Warn once when the base enters an abnormal state, once when it leaves.

        Warning only, never an exception -- :meth:`_read_base_state` explains what
        a raise on this path would cost.

        Not routed through :func:`_warn_throttled`, which rate-limits a condition
        that keeps re-firing (a failing serial link) and would print once a second
        for as long as the button stays down. An emergency stop is a state, not a
        stream of failures, so it is reported on its edges.
        """
        known = _known_base_state(state)
        if known is None:
            # An unlisted code is ignored and not remembered: remembering it would
            # make the next frame look like a transition back, so a register that
            # flickers between garbage and a real state would warn every frame.
            return
        if known == self._last_base_state:
            return
        self._last_base_state = known
        message = _base_state_warning(known)
        if message is not None:
            self._base_warning_active = True
            logger.warning(_base_state_banner(message))
            return
        if not self._base_warning_active:
            return
        if known is not SlateBaseSystemState.SYS_NORMAL:
            # Releasing the button can show SYS_INIT or SYS_REMOTE for a frame
            # first. That is not a recovery, so the warning stays open until the
            # base actually reports SYS_NORMAL -- otherwise the closing line goes
            # missing exactly when the release was not instantaneous.
            return
        self._base_warning_active = False
        logger.warning(_base_state_banner("The base state is back to normal.", "green"))

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
        add_loop_section("arms_read", time.perf_counter() - _t)
        obs_dict.update(arms_obs)

        # Get base observations. Refresh the cached chassis state first so get_vel()
        # does not return a stale/uninitialized buffer (the cause of garbage base
        # velocities on the first frame of an episode), then sanity-check the result.
        _t = time.perf_counter()
        if not self.base.update_state():
            _warn_throttled(
                "base_read",
                "Failed to refresh Mobile AI base state; using last cached velocity.",
            )
        else:
            # system_state arrived with the velocities update_state() just read and
            # read() only copies that cache, so watching the emergency stop adds no
            # serial traffic. Success path only: after a failed refresh the cached
            # state is stale and would report a transition that never happened.
            base_state = self._read_base_state()
            if base_state is not None:
                self._log_base_state_transition(base_state)
        base_obs = self.base.get_vel()
        add_loop_section("base_read", time.perf_counter() - _t)
        x_vel, theta_vel = _sanitize_base_velocity(base_obs[0], base_obs[1])

        # Update shared state (always, so the teleoperator can passively record base
        # movement even when base velocity is excluded from the observation).
        with _base_velocity_lock:
            _latest_base_velocity["x.vel"] = x_vel
            _latest_base_velocity["theta.vel"] = theta_vel

        # Expose base velocity as an observation feature only when configured. Off by
        # default: current policies expect a 14-dim observation.state (arms only), and
        # adding base here would make it 16-dim and break the policy normalizer.
        if self.config.include_base_in_state:
            obs_dict.update({"x.vel": x_vel, "theta.vel": theta_vel})

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_s = time.perf_counter() - start
            add_loop_section(f"cam:{cam_key}", dt_s)
            logger.debug(f"{self} read {cam_key}: {dt_s * 1e3:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        # Record the real control-loop period (see loop_rate_log.py): the base
        # velocity command below is held until the next call, so a slow loop
        # over-rotates the base proportionally.
        record_loop_tick()

        _t = time.perf_counter()
        send_action_arms = self.arms.send_action(
            {k: v for k, v in action.items() if k in self.arms.action_features}
        )
        add_loop_section("arms_write", time.perf_counter() - _t)
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
        add_loop_section("base_write", time.perf_counter() - _t)

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
