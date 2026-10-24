"""The leader arm's connect() and disconnect() do not wait for their moves either.

Same scheme as lerobot_robot_trossen/tests/test_parallel_arm_moves.py, plus the
leader's own step: gravity compensation may only start once the connect move
has ended, so it moves from connect() to the first use of the arm.
"""

import threading

import pytest
import trossen_arm
from lerobot_teleoperator_trossen import widowxai_leader
from lerobot_teleoperator_trossen.config_widowxai_leader import (
    WidowXAILeaderTeleopConfig,
)
from lerobot_teleoperator_trossen.widowxai_leader import (
    ARRIVAL_MARGIN_S,
    STAGED_MOVE_S,
    WidowXAILeaderTeleop,
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


class _FakeDriver:
    def __init__(self, clock: _Clock, joint_count: int) -> None:
        self._clock = clock
        self._positions = [0.0] * joint_count
        self.calls: list[tuple] = []
        self.configured = False

    def configure(self, **kwargs) -> None:
        self.configured = True

    def get_is_configured(self) -> bool:
        return self.configured

    def set_all_modes(self, mode: object) -> None:
        self.calls.append(("mode", self._clock.now, mode))

    def set_all_external_efforts(self, efforts, goal_time=0.0, blocking=True) -> None:
        self.calls.append(("efforts", self._clock.now))

    def get_all_positions(self) -> list[float]:
        return list(self._positions)

    def set_all_positions(self, goal_positions, goal_time=2.0, blocking=True) -> None:
        self.calls.append(("move", self._clock.now, goal_time, blocking))
        self._clock.advance(goal_time if blocking else 0.001)
        self._positions = list(goal_positions)

    def cleanup(self) -> None:
        self.calls.append(("cleanup", self._clock.now))
        self.configured = False


@pytest.fixture
def leader(monkeypatch: pytest.MonkeyPatch):
    clock = _Clock()
    monkeypatch.setattr(widowxai_leader.time, "perf_counter", clock)
    monkeypatch.setattr(widowxai_leader.time, "sleep", clock.advance)
    arm = WidowXAILeaderTeleop(WidowXAILeaderTeleopConfig(ip_address="192.0.2.3"))
    arm.driver = _FakeDriver(clock, len(arm.config.joint_names))
    yield arm, clock
    # Park it while the fake clock still runs; left to __del__ at exit, it
    # would wait out its moves in real seconds.
    if arm.is_connected:
        arm.disconnect()
    arm.wait_for_disconnect(timeout=5.0)


def _kinds(arm) -> list:
    return [call[2] if call[0] == "mode" else call[0] for call in arm.driver.calls]


def test_connect_returns_in_position_mode_mid_move(leader) -> None:
    arm, clock = leader
    started = clock.now
    arm.connect()
    assert clock.now - started < 0.01, "connect() waited for the staged move"
    assert _kinds(arm) == [trossen_arm.Mode.position, "move"]
    assert arm.driver.calls[-1][3] is False  # non-blocking


def test_gravity_compensation_starts_after_the_move_on_first_read(leader) -> None:
    arm, _ = leader
    arm.connect()
    _, sent, _, _ = arm.driver.calls[-1]
    arm.get_action()
    mode_call = arm.driver.calls[2]
    assert mode_call[2] == trossen_arm.Mode.external_effort
    assert mode_call[1] >= sent + STAGED_MOVE_S
    assert _kinds(arm)[2:] == [trossen_arm.Mode.external_effort, "efforts"]
    # Once, not per frame.
    arm.get_action()
    assert len(arm.driver.calls) == 4


def test_align_before_first_read_skips_the_owed_release(leader) -> None:
    """align_to then release() is the leader reset sequence; no extra mode hop."""
    arm, clock = leader
    arm.connect()
    arrival = clock.now + STAGED_MOVE_S + ARRIVAL_MARGIN_S - 0.001
    positions = {f"{name}.pos": 0.1 for name in arm.config.joint_names}
    arm.align_to(positions, goal_time=1.0)
    align_move = arm.driver.calls[-1]
    assert align_move[0] == "move" and align_move[1] >= arrival
    assert trossen_arm.Mode.external_effort not in _kinds(arm)
    arm.release()
    arm.get_action()
    assert _kinds(arm).count(trossen_arm.Mode.external_effort) == 1


def test_disconnect_parks_in_the_background(leader) -> None:
    arm, clock = leader
    arm.connect()
    arm.get_action()
    gate = threading.Event()
    real_move = arm.driver.set_all_positions

    def held_move(*args, **kwargs):
        gate.wait(timeout=5.0)
        real_move(*args, **kwargs)

    arm.driver.set_all_positions = held_move
    before = clock.now
    arm.disconnect()
    assert clock.now == before
    assert not arm.is_connected
    gate.set()
    assert arm.wait_for_disconnect(timeout=5.0) is None
    assert _kinds(arm)[-4:] == [trossen_arm.Mode.position, "move", "move", "cleanup"]


def test_a_dropped_leader_parks_in_place(leader) -> None:
    arm, _ = leader
    arm.connect()
    started = []
    real_start = threading.Thread.start

    def recording_start(thread):
        started.append(thread.name)
        real_start(thread)

    threading.Thread.start = recording_start
    try:
        arm.__del__()
    finally:
        threading.Thread.start = real_start
    assert started == []
    assert arm.driver.calls[-1][0] == "cleanup"


def test_dropped_leaders_park_in_place_together(leader) -> None:
    """The e-stop exit path: lerobot-teleoperate never disconnects the leader."""
    _, clock = leader
    from lerobot_teleoperator_trossen.bi_widowxai_leader import BiWidowXAILeaderRobot
    from lerobot_teleoperator_trossen.config_bi_widowxai_leader import (
        BiWidowXAILeaderRobotConfig,
    )

    pair = BiWidowXAILeaderRobot(
        BiWidowXAILeaderRobotConfig(
            left_arm_ip_address="192.0.2.3", right_arm_ip_address="192.0.2.2"
        )
    )
    for arm in (pair.left_arm, pair.right_arm):
        arm.driver = _FakeDriver(clock, len(arm.config.joint_names))
    pair.connect()
    started = clock.now
    pair.__del__()
    left, right = (
        [call for call in arm.driver.calls if call[0] == "move"][1:]
        for arm in (pair.left_arm, pair.right_arm)
    )
    for left_leg, right_leg in zip(left, right, strict=True):
        assert abs(right_leg[1] - left_leg[1]) < 0.01
    # About one connect move plus two legs, not twice the legs.
    assert clock.now - started < 3 * (STAGED_MOVE_S + ARRIVAL_MARGIN_S) + 0.1
    for arm in (pair.left_arm, pair.right_arm):
        assert arm.driver.calls[-1][0] == "cleanup"
