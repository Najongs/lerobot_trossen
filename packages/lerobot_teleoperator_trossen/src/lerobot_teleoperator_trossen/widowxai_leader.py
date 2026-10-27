import logging
import threading
import time

import trossen_arm
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot_robot_trossen.parking import (
    ARRIVAL_MARGIN_S,
    STAGED_MOVE_S,
    guard_sigint,
    park_at_exit,
    park_in_place,
    park_together,
    parking_in_place,
    queue_for_parking,
)

from lerobot_teleoperator_trossen.config_widowxai_leader import (
    WidowXAILeaderTeleopConfig,
)

logger = logging.getLogger(__name__)


class WidowXAILeaderTeleop(Teleoperator):
    """
    [WidowX AI](https://www.trossenrobotics.com/widowx-ai) by Trossen Robotics
    """

    config_class = WidowXAILeaderTeleopConfig
    name = "widowxai_leader_teleop"

    def __init__(self, config: WidowXAILeaderTeleopConfig):
        super().__init__(config)
        self.config = config
        self.driver = trossen_arm.TrossenArmDriver()
        # As in WidowXAIFollower: connect() and disconnect() do not wait for
        # their moves, so leader and follower arms move together. _arrival_s is
        # when the connect move to the staged pose ends (None once waited for);
        # _release_pending says gravity compensation, which must not start
        # mid-move, is still owed after it.
        self._arrival_s: float | None = None
        self._release_pending = False
        self._disconnecting = False
        self._park_thread: threading.Thread | None = None
        self._park_error: BaseException | None = None

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{joint_name}.pos": float for joint_name in self.config.joint_names}

    @property
    def feedback_features(self) -> dict[str, type]:
        # TODO(lukeschmitt-tr): Implement force feedback
        # Note that feedback is currently not implemented upstream huggingface/lerobot.
        return {}

    @property
    def is_connected(self) -> bool:
        # False from the moment disconnect() is called.
        return not self._disconnecting and self.driver.get_is_configured()

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.driver.configure(
            model=trossen_arm.Model.wxai_v0,
            end_effector=trossen_arm.StandardEndEffector.wxai_v0_leader,
            serv_ip=self.config.ip_address,
            clear_error=True,
        )
        if not self.is_calibrated and calibrate:
            self.calibrate()

        self.configure()
        park_at_exit(self, rank=2)
        logger.info(
            f"{self} connected; moving to the staged pose over {STAGED_MOVE_S:.2f}s "
            "without waiting"
        )

    @property
    def is_calibrated(self) -> bool:
        # Trossen Arm robots do not require calibration
        return True

    def calibrate(self) -> None:
        # Trossen Arm robots do not require calibration
        pass

    def configure(self) -> None:
        self.driver.set_all_modes(trossen_arm.Mode.position)
        # Move the arm to the staged positions. Gravity compensation follows in
        # wait_for_arrival(), once the move has ended.
        self.driver.set_all_positions(
            self.config.staged_positions,
            goal_time=STAGED_MOVE_S,
            blocking=False,
        )
        self._arrival_s = time.perf_counter() + STAGED_MOVE_S + ARRIVAL_MARGIN_S
        self._release_pending = True

    def _wait_for_staged_move(self) -> float:
        """Sleep until the connect move has ended; the seconds slept."""
        if self._arrival_s is None:
            return 0.0
        wait_s = max(0.0, self._arrival_s - time.perf_counter())
        if wait_s > 0.0:
            time.sleep(wait_s)
        self._arrival_s = None
        return wait_s

    def wait_for_arrival(self) -> float:
        """Block until the connect move has ended, then start gravity compensation.

        The state connect() used to return in. Returns the seconds waited; 0.0
        once done, so get_action can call it every frame.
        """
        wait_s = self._wait_for_staged_move()
        if self._release_pending:
            self.release()
        return wait_s

    def get_action(self) -> dict[str, float]:
        self.wait_for_arrival()
        start = time.perf_counter()
        action = self.driver.get_all_positions()
        action_dict = {
            f"{joint_name}.pos": val
            for joint_name, val in zip(
                self.config.joint_names,
                action,
                strict=True,
            )
        }
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return action_dict

    def align_to(
        self, positions: dict[str, float], goal_time: float, blocking: bool = True
    ) -> None:
        """Move to ``positions`` (keyed ``<joint>.pos``) under position control.

        Leaves the arm in position mode; :meth:`release` hands it back to gravity
        compensation. stage_runner's leader_reset stage uses the pair to bring the
        leader to the follower's pose before the follower starts copying it -- the
        same mode sequence configure() runs to reach the staged pose.
        """
        goal = [float(positions[f"{name}.pos"]) for name in self.config.joint_names]
        # Still in position mode if the connect move is running: let it end, and
        # skip the gravity compensation owed after it -- release() follows.
        self._wait_for_staged_move()
        self._release_pending = False
        self.driver.set_all_modes(trossen_arm.Mode.position)
        self.driver.set_all_positions(goal, goal_time=goal_time, blocking=blocking)

    def release(self) -> None:
        """Gravity compensation, the state wait_for_arrival() leaves the arm in."""
        self._release_pending = False
        self.driver.set_all_modes(trossen_arm.Mode.external_effort)
        self.driver.set_all_external_efforts(
            [0.0] * len(self.config.joint_names),
            goal_time=0.0,
            blocking=True,
        )

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # TODO(lukeschmitt-tr): Implement force feedback
        # Note that feedback is currently not implemented upstream huggingface/lerobot.
        raise NotImplementedError

    def disconnect(self) -> None:
        """Start parking the arm in the background and return.

        As in WidowXAIFollower.disconnect: a non-daemon thread runs the staged and
        sleep pose moves; wait_for_disconnect() joins it. Inside a finalizer it
        parks in place instead (lerobot_robot_trossen/parking.py).
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        self._disconnecting = True

        if parking_in_place():
            queue_for_parking(self)
            return
        # Before the thread starts: Ctrl+C can only be guarded from this thread.
        guard_sigint()
        self._park_thread = threading.Thread(
            target=self._park, name=f"park {self.config.ip_address}", daemon=False
        )
        self._park_thread.start()
        logger.info(f"{self} parking: staged then sleep pose, in the background")

    def _park(self) -> None:
        park_together([self])

    def _begin_park(self) -> None:
        self._wait_for_staged_move()
        self._release_pending = False
        self.driver.set_all_modes(trossen_arm.Mode.position)

    def wait_for_disconnect(self, timeout: float | None = None) -> BaseException | None:
        """Join the park thread; return what the park raised, or None.

        A park still running after ``timeout`` returns a TimeoutError. Returns
        None at once if disconnect() was never called.
        """
        thread = self._park_thread
        if thread is not None:
            thread.join(timeout)
            if thread.is_alive():
                return TimeoutError(f"{self} still parking after {timeout}s")
        return self._park_error

    def __del__(self) -> None:
        # Teleoperator.__del__ disconnects a leader dropped while connected; no
        # thread may start from a finalizer (lerobot_robot_trossen/parking.py).
        with park_in_place():
            super().__del__()
