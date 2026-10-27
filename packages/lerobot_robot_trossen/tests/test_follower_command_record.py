"""WidowXAIFollower remembers the last trajectory it handed the driver.

stage_runner.reposition reads ``last_command_sent_s`` and
``last_command_goal_time_s`` to wait until that trajectory has expired before a
start pose move (trossen_arm 1.9.0 seeds a new trajectory from the commanded
state of the one still running). These tests pin what the two values mean for
each kind of call, with a fake driver and a fake clock: the real driver object
cannot be wrapped, but the follower's ``driver`` attribute can be replaced.
"""

import pytest
from lerobot_robot_trossen import widowxai_follower
from lerobot_robot_trossen.config_widowxai_follower import WidowXAIFollowerConfig
from lerobot_robot_trossen.widowxai_follower import WidowXAIFollower


class _Limit:
    def __init__(self, velocity_max: float) -> None:
        self.velocity_max = velocity_max


class _FakeDriver:
    """Records each set_all_positions with the fake clock reading at the call."""

    def __init__(self, clock: "_Clock", joint_count: int) -> None:
        self._clock = clock
        self._positions = [0.0] * joint_count
        self.calls: list[dict] = []

    def get_is_configured(self) -> bool:
        return True

    def set_all_modes(self, mode: object) -> None:
        pass

    def get_joint_limits(self) -> list[_Limit]:
        return [_Limit(1.0) for _ in self._positions]

    def get_all_positions(self) -> list[float]:
        return list(self._positions)

    def set_all_positions(self, goal_positions, goal_time=2.0, blocking=True) -> None:
        self.calls.append(
            {"at": self._clock.now, "goal_time": goal_time, "blocking": blocking}
        )
        # A blocking call returns once the goal is reached.
        self._clock.now += goal_time if blocking else 0.001
        self._positions = list(goal_positions)


class _Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def follower(monkeypatch: pytest.MonkeyPatch) -> tuple[WidowXAIFollower, _Clock]:
    clock = _Clock()
    monkeypatch.setattr(widowxai_follower.time, "perf_counter", clock)
    arm = WidowXAIFollower(WidowXAIFollowerConfig(ip_address="192.0.2.1"))
    arm.driver = _FakeDriver(clock, len(arm.config.joint_names))
    return arm, clock


def test_nothing_is_recorded_before_the_first_command(follower) -> None:
    arm, _ = follower
    assert arm.last_command_sent_s is None
    assert arm.last_command_goal_time_s is None


def test_a_blocking_move_is_recorded_from_before_the_call(follower) -> None:
    """Its return time would add a second goal_time; the start is the call."""
    arm, clock = follower
    arm.configure()
    (call,) = arm.driver.calls
    assert arm.last_command_sent_s == call["at"]
    assert arm.last_command_goal_time_s == 2.0
    # ...so it reads as expired the moment the blocking call has returned.
    assert arm.last_command_sent_s + arm.last_command_goal_time_s <= clock.now


def test_a_policy_frame_records_its_adaptive_goal_time_after_the_call(
    follower,
) -> None:
    arm, clock = follower
    arm.configure()
    names = arm.config.joint_names
    here = dict(zip(names, arm.driver.get_all_positions(), strict=True))
    # Inside the minimum window: goal_time = 3.0 / 30.
    arm.send_action({f"{name}.pos": value for name, value in here.items()})
    assert arm.last_command_goal_time_s == pytest.approx(0.1)
    assert arm.last_command_sent_s == clock.now  # after the call returned
    assert arm.last_command_sent_s > arm.driver.calls[-1]["at"]
    # A paced frame: 1.0 rad at 1.0 rad/s x 0.4 needs 2.5 s, and that is what
    # the record holds -- not the 0.1 s minimum.
    action = {f"{name}.pos": value for name, value in here.items()}
    action[f"{names[0]}.pos"] = here[names[0]] + 1.0
    arm.send_action(action)
    assert arm.driver.calls[-1]["goal_time"] == pytest.approx(2.5)
    assert arm.last_command_goal_time_s == pytest.approx(2.5)


def test_disconnect_records_its_last_blocking_move(follower) -> None:
    arm, _ = follower
    arm.configure()
    arm.driver.cleanup = lambda: None
    arm.disconnect()
    assert arm.last_command_sent_s == arm.driver.calls[-1]["at"]
    assert arm.last_command_goal_time_s == 2.0
