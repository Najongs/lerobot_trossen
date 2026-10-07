"""Hardware-free teleoperator for tests: answers one fixed arm pose (+ base 0).

The counterpart of :mod:`mock_robot` for the runner's teleop phase (the
leader-arm window before the first stage, see ``runner._run_teleop_phase``).
``record_loop`` only needs ``get_action()`` to return the robot's action keys;
everything else on the abstract base is a no-op here.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any

from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.teleoperators.teleoperator import Teleoperator

from stage_runner.mock_robot import _JOINT_SEED_STEP_RAD, REALISTIC_JOINT_NAMES

MOCK_TELEOP_TYPE: str = "stage_runner_mock_teleop"


@TeleoperatorConfig.register_subclass(MOCK_TELEOP_TYPE)
@dataclass
class MockTeleopConfig(TeleoperatorConfig):
    joint_count: int = 14
    realistic_joint_names: bool = True
    # None (default): every joint answers the mock robot's SEED pose
    # ((index + 1) * _JOINT_SEED_STEP_RAD, mock_robot.py) -- a leader nobody
    # touches, so the arm stays where it is and the initial reset's jump gate
    # sees the same pose as a run without a teleop phase. A float: every joint
    # answers that value, for a test that wants the window to move the arm.
    pose_rad: float | None = None
    include_base: bool = True


class MockTeleop(Teleoperator):
    config_class: type[MockTeleopConfig] = MockTeleopConfig
    name: str = MOCK_TELEOP_TYPE

    def __init__(self, config: MockTeleopConfig) -> None:
        super().__init__(config)
        self.config = config
        self._connected = False
        self.calls = 0

    @property
    def _joint_names(self) -> list[str]:
        if self.config.realistic_joint_names and self.config.joint_count == 14:
            return list(REALISTIC_JOINT_NAMES)
        return [f"joint_{index}" for index in range(self.config.joint_count)]

    @cached_property
    def action_features(self) -> dict[str, type]:
        features: dict[str, type] = {f"{name}.pos": float for name in self._joint_names}
        if self.config.include_base:
            features["x.vel"] = float
            features["theta.vel"] = float
        return features

    @cached_property
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
        return None

    def configure(self) -> None:
        return None

    def get_action(self) -> dict[str, Any]:
        self.calls += 1
        action: dict[str, Any] = {
            f"{name}.pos": (
                (index + 1) * _JOINT_SEED_STEP_RAD if self.config.pose_rad is None else self.config.pose_rad
            )
            for index, name in enumerate(self._joint_names)
        }
        if self.config.include_base:
            action["x.vel"] = 0.0
            action["theta.vel"] = 0.0
        return action

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        return None

    def disconnect(self) -> None:
        self._connected = False
