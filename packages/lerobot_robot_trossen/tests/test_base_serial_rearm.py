"""Base serial re-arm: the bit it sets, when it may run, its guards, the robot hook.

No hardware: the driver object is either a zeroed buffer laid out like
``base_driver::driver`` or the real ``trossen_slate`` 0.0.3 global with its fd
pointed at a pipe or pty for the duration of one test.
"""

import ctypes
import logging
import os
from types import SimpleNamespace

import pytest

from lerobot_robot_trossen import base_serial_rearm, loop_rate_log
from lerobot_robot_trossen.base_serial_rearm import (
    READ_SET_OFFSET,
    BaseSerialRearm,
    create_base_serial_rearm,
    loop_refusal_reason,
)


def fake_driver():
    """A zeroed buffer laid out like base_driver::driver (fd_set at +0x10)."""
    return (ctypes.c_ubyte * 0x98)()


@pytest.fixture(autouse=True)
def outside_any_loop():
    loop_rate_log._reset_phase(None, None)
    yield
    loop_rate_log._reset_phase(None, None)


def in_loop(phase, fps):
    loop_rate_log._reset_phase(phase, fps)


# ----- the bit ----------------------------------------------------------------------


def test_rearm_sets_the_bit_the_real_select_reads():
    read_end, write_end = os.pipe()
    try:
        os.write(write_end, b"x")
        driver = fake_driver()
        rearm = BaseSerialRearm(ctypes.addressof(driver), read_end, "/dev/ttyUSB0")
        in_loop("policy", 21.0)
        rearm.rearm_if_permitted()
        assert rearm.is_armed() and rearm.rearm_count == 1

        class Timeval(ctypes.Structure):
            _fields_ = [("tv_sec", ctypes.c_long), ("tv_usec", ctypes.c_long)]

        c_library = ctypes.CDLL(None, use_errno=True)
        read_set = ctypes.c_void_p(ctypes.addressof(driver) + READ_SET_OFFSET)
        ready = c_library.select(
            read_end + 1, read_set, None, None, ctypes.byref(Timeval(0, 0))
        )
        assert ready == 1
    finally:
        os.close(read_end)
        os.close(write_end)


@pytest.mark.parametrize("file_descriptor", [3, 8, 13, 1023])
def test_rearm_sets_exactly_the_glibc_fd_set_bit(file_descriptor):
    # select() above cannot tell a wrong bit apart when that bit happens to be
    # another readable fd (stdout under pytest), so pin the bytes as well.
    driver = fake_driver()
    rearm = BaseSerialRearm(ctypes.addressof(driver), file_descriptor, "/dev/ttyUSB0")
    in_loop("policy", 21.0)
    rearm.rearm_if_permitted()
    expected = bytearray(0x98)
    word = 0x10 + 8 * (file_descriptor // 64)
    expected[word : word + 8] = (1 << (file_descriptor % 64)).to_bytes(8, "little")
    assert bytes(driver) == bytes(expected)


# ----- when it may run --------------------------------------------------------------


@pytest.mark.parametrize(
    ("phase", "fps", "allowed"),
    [
        ("policy", 21.0, True),
        ("policy", 20.0, True),
        ("policy", 30.0, False),
        ("policy", None, False),
        ("teleop", 21.0, False),
        (None, 21.0, False),
    ],
)
def test_loop_refusal(phase, fps, allowed):
    assert (loop_refusal_reason(phase, fps) is None) == allowed


def test_nothing_happens_outside_a_policy_loop():
    driver = fake_driver()
    rearm = BaseSerialRearm(ctypes.addressof(driver), 13, "/dev/ttyUSB0")
    for phase, fps in ((None, None), ("teleop", 21.0), ("policy", 30.0)):
        in_loop(phase, fps)
        rearm.rearm_if_permitted()
    assert rearm.rearm_count == 0
    assert bytes(driver) == bytes(0x98)


def test_the_driver_is_put_back_after_a_permitted_loop():
    driver = fake_driver()
    rearm = BaseSerialRearm(ctypes.addressof(driver), 13, "/dev/ttyUSB0")
    in_loop("policy", 21.0)
    rearm.rearm_if_permitted()
    assert rearm.is_armed()
    in_loop("teleop", 21.0)  # the reset phase that follows an episode
    rearm.rearm_if_permitted()
    assert not rearm.is_armed()
    assert bytes(driver) == bytes(0x98)


def test_a_driver_armed_by_itself_is_left_alone():
    driver = fake_driver()
    driver[0x10 + 1] = 1 << 5  # fd 13 already set when the hook is built
    rearm = BaseSerialRearm(ctypes.addressof(driver), 13, "/dev/ttyUSB0")
    rearm.rearm_if_permitted()
    assert rearm.is_armed()


@pytest.mark.parametrize(
    ("fps", "armed_inside"), [(21, True), (30, False)], ids=["21fps", "30fps"]
)
def test_scope_follows_the_loop_rate_phase_tag(fps, armed_inside, caplog):
    """Through loop_rate_log's real record_loop wrapper, the way record() runs it."""
    driver = fake_driver()
    rearm = BaseSerialRearm(ctypes.addressof(driver), 13, "/dev/ttyUSB0")
    seen = []

    def one_tick(**kwargs):
        rearm.rearm_if_permitted()
        seen.append(rearm.is_armed())

    wrapped = loop_rate_log._wrap_record_loop(one_tick)
    with caplog.at_level(logging.INFO):
        wrapped(policy=object(), fps=fps)  # episode
        wrapped(policy=None, fps=fps)  # reset phase
    assert seen == [armed_inside, False]
    assert loop_rate_log.current_phase() is None
    if armed_inside:
        assert "active for this policy loop at 21 fps" in caplog.text
    else:
        assert "off for this loop: --dataset.fps=30 is above 21" in caplog.text


def test_permission_is_logged_once_per_change(caplog):
    driver = fake_driver()
    rearm = BaseSerialRearm(ctypes.addressof(driver), 13, "/dev/ttyUSB0")
    in_loop("policy", 21.0)
    with caplog.at_level(logging.INFO):
        for _ in range(50):
            rearm.rearm_if_permitted()
    assert caplog.text.count("active for this policy loop") == 1


# ----- guards -----------------------------------------------------------------------


def test_not_requested_means_nothing_is_touched(monkeypatch):
    monkeypatch.setattr(base_serial_rearm, "REQUESTED", False)
    assert create_base_serial_rearm() is None


def test_other_trossen_slate_versions_are_refused(monkeypatch, caplog):
    from importlib import metadata

    monkeypatch.setattr(base_serial_rearm, "REQUESTED", True)
    monkeypatch.setattr(metadata, "version", lambda name: "0.0.4")
    with caplog.at_level(logging.WARNING):
        assert create_base_serial_rearm() is None
    assert "0.0.4" in caplog.text


def _real_driver_address():
    trossen_slate = pytest.importorskip("trossen_slate")
    from importlib import metadata

    if metadata.version("trossen_slate") != "0.0.3":
        pytest.skip("layout only known for trossen_slate 0.0.3")
    library = ctypes.CDLL(trossen_slate.trossen_slate.__file__)
    return ctypes.addressof(
        ctypes.c_char.in_dll(library, base_serial_rearm.DRIVER_SYMBOL)
    )


def test_real_driver_object_is_found_and_rearmed(monkeypatch):
    """Point the real 0.0.3 driver's fd at a pty and check the bit lands in it."""
    driver_address = _real_driver_address()
    file_descriptor_field = ctypes.c_int.from_address(driver_address + 0x8)
    saved_file_descriptor = file_descriptor_field.value
    saved_set = bytes((ctypes.c_ubyte * 128).from_address(driver_address + 0x10))
    controller, device = os.openpty()
    try:
        file_descriptor_field.value = device
        monkeypatch.setattr(base_serial_rearm, "REQUESTED", True)
        # A pty is /dev/pts/N; the real port is /dev/ttyUSB*. Let the guard see
        # the name it will see on the robot.
        real_readlink = os.readlink
        monkeypatch.setattr(
            base_serial_rearm.os,
            "readlink",
            lambda path: "/dev/ttyUSB0"
            if path == f"/proc/self/fd/{device}"
            else real_readlink(path),
        )
        rearm = create_base_serial_rearm()
        assert rearm is not None
        assert rearm.file_descriptor == device
        in_loop("policy", 21.0)
        rearm.rearm_if_permitted()
        assert rearm.is_armed()
    finally:
        file_descriptor_field.value = saved_file_descriptor
        ctypes.memmove(driver_address + 0x10, saved_set, 128)
        os.close(controller)
        os.close(device)


@pytest.mark.parametrize(
    ("descriptor_value", "link_target", "message"),
    [
        (1024, None, "not an open descriptor"),
        (-1, None, "not an open descriptor"),
        (None, None, "not a serial port"),  # a pipe
        (None, "/dev/pts/3", "not a serial port"),
    ],
)
def test_guard_boundaries(monkeypatch, caplog, descriptor_value, link_target, message):
    driver_address = _real_driver_address()
    file_descriptor_field = ctypes.c_int.from_address(driver_address + 0x8)
    saved_file_descriptor = file_descriptor_field.value
    read_end, write_end = os.pipe()
    try:
        file_descriptor_field.value = (
            read_end if descriptor_value is None else descriptor_value
        )
        monkeypatch.setattr(base_serial_rearm, "REQUESTED", True)
        if link_target is not None:
            monkeypatch.setattr(
                base_serial_rearm.os, "readlink", lambda path: link_target
            )
        with caplog.at_level(logging.WARNING):
            assert create_base_serial_rearm() is None
        assert message in caplog.text
    finally:
        file_descriptor_field.value = saved_file_descriptor
        os.close(read_end)
        os.close(write_end)


# ----- the robot hook ---------------------------------------------------------------


def test_mobileai_rearms_right_before_each_base_transaction():
    from lerobot_robot_trossen.mobileai import MobileAIRobot

    calls = []
    robot = object.__new__(MobileAIRobot)
    robot.arms = SimpleNamespace(
        get_observation=lambda: {}, send_action=lambda action: {}, action_features={}
    )
    robot.base = SimpleNamespace(
        update_state=lambda: calls.append("update_state") or False,
        get_vel=lambda: (0.0, 0.0),
        set_cmd_vel=lambda x_velocity, theta_velocity: calls.append("set_cmd_vel")
        or True,
    )
    robot.config = SimpleNamespace(include_base_in_state=False)
    robot.cameras = {}
    robot._base_serial_rearm = SimpleNamespace(
        rearm_if_permitted=lambda: calls.append("rearm")
    )
    robot.get_observation()
    robot.send_action({"x.vel": 0.0, "theta.vel": 0.0})
    assert calls == ["rearm", "update_state", "rearm", "set_cmd_vel"]
