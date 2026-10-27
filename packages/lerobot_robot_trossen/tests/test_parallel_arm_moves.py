"""connect() and disconnect() hand the arm its staged / sleep moves and return.

Waiting for each 2 s move in turn made a leader + follower setup spend 8 s
connecting and 16 s disconnecting. These tests pin the replacement: connect
does not wait, the first read or command waits for the rest of the move, and
disconnect parks the arm on a background thread. A fake driver and a fake clock
stand in for trossen_arm, as in test_follower_command_record.py.
"""

import threading

import pytest
from lerobot_robot_trossen import widowxai_follower
from lerobot_robot_trossen.bi_widowxai_follower import BiWidowXAIFollowerRobot
from lerobot_robot_trossen.config_bi_widowxai_follower import (
    BiWidowXAIFollowerRobotConfig,
)
from lerobot_robot_trossen.config_widowxai_follower import WidowXAIFollowerConfig
from lerobot_robot_trossen.widowxai_follower import (
    ARRIVAL_MARGIN_S,
    STAGED_MOVE_S,
    WidowXAIFollower,
)


class _Clock:
    def __init__(self) -> None:
        self.now = 100.0
        self._lock = threading.Lock()

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        with self._lock:
            self.now += seconds


class _Limit:
    velocity_max = 1.0


class _Output:
    def __init__(self, positions: list[float]) -> None:
        self.positions = positions
        self.velocities = [0.0] * len(positions)
        self.efforts = [0.0] * len(positions)
        self.external_efforts = [0.0] * len(positions)


class _FakeDriver:
    """Logs every call with the fake clock reading; blocking moves take goal_time."""

    def __init__(self, clock: _Clock, joint_count: int, name: str) -> None:
        self._clock = clock
        self._positions = [0.0] * joint_count
        self.name = name
        self.calls: list[tuple] = []
        self.configured = False
        self.fail_moves = False

    def configure(self, **kwargs) -> None:
        self.configured = True

    def get_is_configured(self) -> bool:
        return self.configured

    def set_all_modes(self, mode: object) -> None:
        self.calls.append(("mode", self._clock.now, mode))

    def get_joint_limits(self) -> list[_Limit]:
        return [_Limit() for _ in self._positions]

    def get_all_positions(self) -> list[float]:
        return list(self._positions)

    def get_robot_output(self):
        class _Joint:
            all = _Output(list(self._positions))

        class _Robot:
            joint = _Joint()

        return _Robot()

    def set_all_positions(self, goal_positions, goal_time=2.0, blocking=True) -> None:
        if self.fail_moves:
            raise RuntimeError("Joint 3 velocity limit exceeded (injected)")
        self.calls.append(("move", self._clock.now, goal_time, blocking))
        self._clock.advance(goal_time if blocking else 0.001)
        self._positions = list(goal_positions)

    def cleanup(self) -> None:
        self.calls.append(("cleanup", self._clock.now))
        self.configured = False


_devices: list = []


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch):
    clock = _Clock()
    monkeypatch.setattr(widowxai_follower.time, "perf_counter", clock)
    monkeypatch.setattr(widowxai_follower.time, "sleep", clock.advance)
    yield clock
    # Park what a test left connected while the fake clock still runs; left to
    # __del__ at exit, each would wait out its moves in real seconds.
    while _devices:
        device = _devices.pop()
        if device.is_connected:
            device.disconnect()
        device.wait_for_disconnect(timeout=5.0)


def _follower(clock: _Clock, name: str = "192.0.2.1") -> WidowXAIFollower:
    arm = WidowXAIFollower(WidowXAIFollowerConfig(ip_address=name))
    arm.driver = _FakeDriver(clock, len(arm.config.joint_names), name)
    _devices.append(arm)
    return arm


def _moves(arm) -> list[tuple]:
    return [call for call in arm.driver.calls if call[0] == "move"]


def test_connect_sends_the_staged_move_and_returns(clock: _Clock) -> None:
    arm = _follower(clock)
    started = clock.now
    arm.connect()
    ((_, _, goal_time, blocking),) = _moves(arm)
    assert (goal_time, blocking) == (STAGED_MOVE_S, False)
    assert clock.now - started < 0.01, "connect() waited for the staged move"
    assert arm.is_connected


def test_the_first_read_waits_for_the_rest_of_the_move(clock: _Clock) -> None:
    arm = _follower(clock)
    arm.connect()
    sent = arm.last_command_sent_s
    clock.advance(0.5)
    arm.get_observation()
    assert clock.now == pytest.approx(sent + STAGED_MOVE_S + ARRIVAL_MARGIN_S)
    # Once there, a read costs no wait.
    before = clock.now
    arm.get_observation()
    assert clock.now == before


def test_a_command_is_not_sent_mid_move(clock: _Clock) -> None:
    """A frame sent mid-move would be seeded from the staged move's state."""
    arm = _follower(clock)
    arm.connect()
    arrival = arm.last_command_sent_s + STAGED_MOVE_S + ARRIVAL_MARGIN_S
    names = arm.config.joint_names
    arm.send_action({f"{name}.pos": 0.0 for name in names})
    _, frame_at, _, _ = _moves(arm)[-1]
    assert frame_at >= arrival


def test_disconnect_parks_in_the_background(clock: _Clock) -> None:
    arm = _follower(clock)
    arm.connect()
    arm.get_observation()
    before = clock.now
    gate = threading.Event()
    real_move = arm.driver.set_all_positions

    def held_move(*args, **kwargs):
        gate.wait(timeout=5.0)
        real_move(*args, **kwargs)

    arm.driver.set_all_positions = held_move
    arm.disconnect()
    # Returned while the park thread is still held at its first move.
    assert clock.now == before
    assert not arm.is_connected
    gate.set()
    assert arm.wait_for_disconnect(timeout=5.0) is None
    staged, sleep = _moves(arm)[1:]
    assert staged[2:] == (STAGED_MOVE_S, False)
    assert sleep[2:] == (STAGED_MOVE_S, False)
    # The sleep pose leg waits for the staged one to end.
    assert sleep[1] >= staged[1] + STAGED_MOVE_S
    assert arm.driver.calls[-1][0] == "cleanup"


def test_a_disconnect_before_arrival_lets_the_connect_move_end(clock: _Clock) -> None:
    arm = _follower(clock)
    arm.connect()
    arrival = arm.last_command_sent_s + STAGED_MOVE_S + ARRIVAL_MARGIN_S
    arm.driver.cleanup = lambda: None
    arm.disconnect()
    assert arm.wait_for_disconnect(timeout=5.0) is None
    _, staged_at, _, _ = _moves(arm)[1]
    assert staged_at >= arrival


def test_a_failed_park_is_reported_not_raised(clock: _Clock) -> None:
    arm = _follower(clock)
    arm.connect()
    arm.driver.fail_moves = True
    arm.disconnect()  # does not raise: the failure is on the park thread
    error = arm.wait_for_disconnect(timeout=5.0)
    assert isinstance(error, RuntimeError)
    assert "velocity limit" in str(error)


def test_a_dropped_robot_parks_in_place_not_on_a_thread(clock: _Clock) -> None:
    """Robot.__del__ disconnects it; a thread started there can deadlock."""
    robot = BiWidowXAIFollowerRobot(
        BiWidowXAIFollowerRobotConfig(
            left_arm_ip_address="192.0.2.1", right_arm_ip_address="192.0.2.2"
        )
    )
    drivers = []
    for arm, name in ((robot.left_arm, "left"), (robot.right_arm, "right")):
        arm.driver = _FakeDriver(clock, len(arm.config.joint_names), name)
        _devices.append(arm)
        drivers.append(arm.driver)
    robot.connect()
    started = []
    real_start = threading.Thread.start

    def recording_start(thread):
        started.append(thread.name)
        real_start(thread)

    threading.Thread.start = recording_start
    try:
        robot.__del__()
    finally:
        threading.Thread.start = real_start
    assert started == []
    for driver in drivers:
        assert driver.calls[-1][0] == "cleanup"
    assert not robot.left_arm.is_connected
    # In place, but both arms still move at once: each leg starts on both
    # arms before the wait for it.
    left = [call for call in drivers[0].calls if call[0] == "move"][1:]
    right = [call for call in drivers[1].calls if call[0] == "move"][1:]
    for left_leg, right_leg in zip(left, right, strict=True):
        assert abs(right_leg[1] - left_leg[1]) < 0.01


def test_one_arm_failing_to_park_does_not_stop_the_other(clock: _Clock) -> None:
    robot = BiWidowXAIFollowerRobot(
        BiWidowXAIFollowerRobotConfig(
            left_arm_ip_address="192.0.2.1", right_arm_ip_address="192.0.2.2"
        )
    )
    for arm, name in ((robot.left_arm, "left"), (robot.right_arm, "right")):
        arm.driver = _FakeDriver(clock, len(arm.config.joint_names), name)
        _devices.append(arm)
    robot.connect()
    robot.left_arm.driver.fail_moves = True
    robot.__del__()
    assert isinstance(robot.left_arm.wait_for_disconnect(), RuntimeError)
    assert robot.right_arm.wait_for_disconnect() is None
    assert robot.right_arm.driver.calls[-1][0] == "cleanup"


def test_wait_for_disconnect_without_disconnect_returns_none(clock: _Clock) -> None:
    arm = _follower(clock)
    assert arm.wait_for_disconnect(timeout=0.1) is None


def test_both_arms_start_their_staged_moves_together(clock: _Clock) -> None:
    robot = BiWidowXAIFollowerRobot(
        BiWidowXAIFollowerRobotConfig(
            left_arm_ip_address="192.0.2.1", right_arm_ip_address="192.0.2.2"
        )
    )
    for arm, name in ((robot.left_arm, "left"), (robot.right_arm, "right")):
        arm.driver = _FakeDriver(clock, len(arm.config.joint_names), name)
        _devices.append(arm)
    started = clock.now
    robot.connect()
    (left,) = _moves(robot.left_arm)
    (right,) = _moves(robot.right_arm)
    assert right[1] - left[1] < 0.01, "the right arm waited for the left one"
    assert clock.now - started < 0.01
    waited = robot.wait_for_arrival()
    assert waited == pytest.approx(STAGED_MOVE_S + ARRIVAL_MARGIN_S, abs=0.01)

    for arm in (robot.left_arm, robot.right_arm):
        arm.driver.cleanup = lambda: None
    robot.disconnect()
    assert robot.wait_for_disconnect(timeout=5.0) == []


@pytest.fixture
def sigint_guard(monkeypatch: pytest.MonkeyPatch):
    """A fresh Ctrl+C guard, with the process's own handler put back after."""
    import signal

    from lerobot_robot_trossen import parking

    original = signal.getsignal(signal.SIGINT)
    if original is parking._on_sigint:  # installed by an earlier disconnect()
        original = parking._previous_sigint
    signal.signal(signal.SIGINT, original)
    monkeypatch.setattr(parking, "_guard_installed", False)
    monkeypatch.setattr(parking, "_previous_sigint", None)
    parking.guard_sigint()
    assert parking._previous_sigint is original
    yield parking
    signal.signal(signal.SIGINT, original)


def _ctrl_c() -> None:
    import os
    import signal

    os.kill(os.getpid(), signal.SIGINT)


def test_ctrl_c_waits_for_a_running_park(sigint_guard) -> None:
    with sigint_guard._parking():
        try:
            _ctrl_c()
        except KeyboardInterrupt:
            pytest.fail("Ctrl+C interrupted a running park")
    with pytest.raises(KeyboardInterrupt):
        _ctrl_c()


def test_ctrl_c_works_again_once_the_guard_runs_out(
    sigint_guard, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A park that hangs must not make Ctrl+C useless."""
    with sigint_guard._parking():
        monkeypatch.setattr(sigint_guard, "_guard_until", 0.0)
        with pytest.raises(KeyboardInterrupt):
            _ctrl_c()


def test_the_exit_sweep_parks_what_is_left_together(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What a failed start leaves connected parks at exit, all arms at once."""
    from lerobot_robot_trossen import parking

    monkeypatch.setattr(parking, "_tracked", [])
    pairs = []
    for base in (1, 3):
        robot = BiWidowXAIFollowerRobot(
            BiWidowXAIFollowerRobotConfig(
                left_arm_ip_address=f"192.0.2.{base}",
                right_arm_ip_address=f"192.0.2.{base + 1}",
            )
        )
        for arm, name in ((robot.left_arm, "l"), (robot.right_arm, "r")):
            arm.driver = _FakeDriver(clock, len(arm.config.joint_names), name)
            _devices.append(arm)
        robot.connect()
        pairs.append(robot)
    pairs[1].disconnect()  # already handled by its caller: left alone
    assert pairs[1].wait_for_disconnect(timeout=5.0) == []
    calls_before = len(pairs[1].left_arm.driver.calls)

    parking._park_still_connected()

    arms = [pairs[0].left_arm, pairs[0].right_arm]
    legs = [[c for c in arm.driver.calls if c[0] == "move"][1:] for arm in arms]
    for leg in zip(*legs, strict=True):
        assert max(c[1] for c in leg) - min(c[1] for c in leg) < 0.01
    for arm in arms:
        assert arm.driver.calls[-1][0] == "cleanup"
        assert not arm.is_connected
    assert len(pairs[1].left_arm.driver.calls) == calls_before
