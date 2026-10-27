"""A leader arm that answers from memory, for the hardware-free leader_reset path.

The counterpart of mock_robot.MockRobot for ``teleop:``. It reproduces the two
things of MobileAILeaderTeleop the leader_reset stage depends on:

* the KEYS -- left_/right_ WidowX AI joint names, a gripper joint last per arm,
  plus ``x.vel`` / ``theta.vel``, so preflight's key check and the identity
  pipeline from leader to follower see the real shape;
* the BASE ECHO -- the real leader returns the base's MEASURED velocity as its
  base action (mobileai_leader.py). ``echo_base_velocity`` stands in for it with
  a non-zero value, so a test can show that the value never reaches the robot.

``align_to`` moves the mock leader instantly and records the call; nothing here
is evidence about the real leader's motion, timing or gravity compensation.
"""

from dataclasses import dataclass
from typing import Any

from lerobot.teleoperators import Teleoperator, TeleoperatorConfig

MOCK_TELEOP_TYPE: str = "stage_runner_mock_teleop"

# Same names as mock_robot's bimanual layout and the real kit.
_JOINT_NAMES: tuple[str, ...] = (
    "joint_0",
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "left_carriage_joint",
)
_PREFIXES: tuple[str, ...] = ("left_", "right_")
# The real leader's staged pose (config_widowxai_leader.py staged_positions),
# which is also the follower's, so a mock leader starts where a freshly
# connected kit's leader does. Gripper 0.0 -- closed.
_STAGED: tuple[float, ...] = (0.0, 1.0472, 0.5236, 0.6283, 0.0, 0.0, 0.0)


@TeleoperatorConfig.register_subclass(MOCK_TELEOP_TYPE)
@dataclass
class MockTeleopConfig(TeleoperatorConfig):
    # What get_action reports for x.vel and theta.vel: the real leader's echo of
    # the measured base velocity. Non-zero by default so the echo is visible.
    echo_base_velocity: float = 0.25


class MockTeleop(Teleoperator):
    """Bimanual mock leader: positions in a dict, every align_to recorded."""

    config_class: type[MockTeleopConfig] = MockTeleopConfig
    name: str = MOCK_TELEOP_TYPE

    def __init__(self, config: MockTeleopConfig) -> None:
        super().__init__(config)
        self.config = config
        self._connected = False
        self.positions: dict[str, float] = {
            f"{prefix}{name}.pos": value
            for prefix in _PREFIXES
            for name, value in zip(_JOINT_NAMES, _STAGED, strict=True)
        }
        # Every align_to call as (positions, goal_time), in order.
        self.align_calls: list[tuple[dict[str, float], float]] = []
        # Test hooks. on_get_action(call_number, teleop) runs before each
        # get_action answers, so a test can move the "operator's hands" mid-loop.
        # raise_on_align: the exception align_to raises.
        self.get_action_count = 0
        self.on_get_action: Any = None
        self.raise_on_align: BaseException | None = None

    @property
    def action_features(self) -> dict[str, type]:
        return {
            **{key: float for key in self.positions},
            "x.vel": float,
            "theta.vel": float,
        }

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self, calibrate: bool = True) -> None:
        self._connected = True

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def get_action(self) -> dict[str, float]:
        self.get_action_count += 1
        if self.on_get_action is not None:
            self.on_get_action(self.get_action_count, self)
        echo = self.config.echo_base_velocity
        return {**self.positions, "x.vel": echo, "theta.vel": echo}

    def align_to(self, positions: dict[str, float], goal_time: float) -> None:
        if self.raise_on_align is not None:
            raise self.raise_on_align
        self.align_calls.append((dict(positions), float(goal_time)))
        self.positions.update({key: float(value) for key, value in positions.items()})

    def send_feedback(self, feedback: dict[str, float]) -> None:
        pass

    def disconnect(self) -> None:
        self._connected = False
