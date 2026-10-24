"""base_loop: the tick loop and frame writer the base stages record through.

Unit level, against a real LeRobotDataset (create(), video off) and the
bimanual mock robot with the simulated base -- no cli, no executor. What is
asserted, and why each matters on the robot:

* a frame is the dataset's features plus the stage's task, with the action as
  sent: arms held, the base command given -- a frame built any other way
  either KeyErrors in add_frame or records an action that was not sent;
* the loop paces to dataset.fps and frames grow by exactly one per act(), so
  stage_end.frames equals the frame_idx delta;
* the probe sees every update_state and set_cmd_vel, reports a failed refresh
  as "no odometry" for that tick, and is gone from robot.base after close() --
  on the exception path too, so the runner's stop_base talks to the driver;
* the grippers hold the measured value in a chain's first stage and the last
  recorded goal once the episode has frames;
* keys: Esc beats the right arrow it comes with, and exit_early is consumed.

NOTHING HERE IS EVIDENCE ABOUT THE REAL LOOP RATE: the mock answers from memory.
"""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from stage_runner import base_loop, record_adapter
from stage_runner.base_loop import BaseProbe, BaseTickLoop
from stage_runner.config import StageConfig
from stage_runner.context import StageContext
from stage_runner.events import EventLog
from stage_runner.mock_robot import MockRobot, MockRobotConfig

GRIPPERS = ("left_left_carriage_joint.pos", "right_left_carriage_joint.pos")
FPS = 30


def _context(tmp_path: Path, **robot_overrides: Any) -> StageContext:
    robot = MockRobot(
        MockRobotConfig(arm_layout="bimanual", simulate_base=True, **robot_overrides)
    )
    processors = record_adapter.make_processors()
    dataset = LeRobotDataset.create(
        "local/base_loop",
        FPS,
        root=tmp_path / "dataset",
        robot_type=robot.name,
        features=record_adapter.build_dataset_features(robot, processors, False),
        use_videos=False,
    )
    robot.connect()
    log = EventLog(tmp_path / "events.jsonl", "run")
    return StageContext(
        config=SimpleNamespace(dataset=SimpleNamespace(fps=FPS)),
        robot=robot,
        dataset=dataset,
        processors=processors,
        events={
            "exit_early": False,
            "stop_recording": False,
            "rerecord_episode": False,
        },
        bundles={},
        log=log,
        run_directory=tmp_path,
    )


def _loop(context: StageContext, name: str = "drive") -> BaseTickLoop:
    return BaseTickLoop(context, StageConfig(id="b", name=name), ceiling_s=10.0)


def _tick(loop: BaseTickLoop, x_vel: float = 0.0, theta_vel: float = 0.0) -> Any:
    observation, odometry, now = loop.observe()
    action = loop.act(observation, x_vel, theta_vel)
    return observation, odometry, now, action


# --------------------------------------------------------------------------
# Frames.


def test_a_tick_records_one_frame_with_the_held_arms_and_the_command(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    robot = context.robot
    first = robot.get_observation()
    loop = _loop(context)
    try:
        for _ in range(3):
            _tick(loop, 0.2, -0.1)
    finally:
        loop.close()
    assert loop.frames == 3 == context.buffered_frame_count()
    buffer = context.dataset.episode_buffer
    assert buffer["task"] == ["drive"] * 3
    names = list(context.dataset.features["action"]["names"])
    for row in buffer["action"]:
        values = dict(zip(names, (float(value) for value in row), strict=True))
        assert values["x.vel"] == pytest.approx(0.2)
        assert values["theta.vel"] == pytest.approx(-0.1)
        for key in robot.action_features:
            if key.endswith(".pos"):
                assert values[key] == pytest.approx(first[key]), key
    assert loop.gripper_hold_source == base_loop.GRIPPER_HOLD_MEASURED
    # The command reached the base through the probe, sanitized by send_action.
    assert robot.base.commands[-3:] == [(0.2, -0.1, True)] * 3


def test_the_loop_paces_to_the_dataset_fps(tmp_path: Path) -> None:
    context = _context(tmp_path)
    loop = _loop(context)
    started = time.perf_counter()
    try:
        for _ in range(15):
            _tick(loop)
    finally:
        loop.close()
    elapsed = time.perf_counter() - started
    assert elapsed == pytest.approx(15 / FPS, rel=0.25)
    stats = loop.tick_stats()
    assert stats["tick_count"] == 15
    assert stats["tick_mean_s"] == pytest.approx(1 / FPS, rel=0.25)
    assert loop.median_period_s() == pytest.approx(1 / FPS, rel=0.25)


def test_the_tick_contract_is_one_observe_then_one_act(tmp_path: Path) -> None:
    context = _context(tmp_path)
    loop = _loop(context)
    try:
        observation, _, _ = loop.observe()
        with pytest.raises(RuntimeError, match="twice"):
            loop.observe()
        loop.act(observation, 0.0, 0.0)
        with pytest.raises(RuntimeError, match="without observe"):
            loop.act(observation, 0.0, 0.0)
    finally:
        loop.close()
    with pytest.raises(RuntimeError, match="closed"):
        loop.observe()


def test_the_median_period_is_one_over_fps_before_two_ticks(tmp_path: Path) -> None:
    context = _context(tmp_path)
    loop = _loop(context)
    try:
        assert loop.median_period_s() == pytest.approx(1 / FPS)
        _tick(loop)
        assert loop.median_period_s() == pytest.approx(1 / FPS)
        assert loop.tick_stats()["tick_mean_s"] is None
    finally:
        loop.close()


# --------------------------------------------------------------------------
# Grippers.


def test_the_grippers_keep_the_last_recorded_goal_once_frames_exist(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    robot = context.robot
    # A previous stage's last command squeezed the grippers past where they
    # measure: record one frame with that goal, then let the arm "stop" wider.
    previous = _loop(context, name="previous")
    try:
        observation, _, _ = previous.observe()
        previous.held_arm_positions = {
            **previous.held_arm_positions,
            **{key: 0.01 for key in GRIPPERS},
        }
        previous.act(observation, 0.0, 0.0)
    finally:
        previous.close()
    for key in GRIPPERS:
        robot._joint_positions[key.removesuffix(".pos")] = 0.03

    loop = _loop(context)
    try:
        _tick(loop)
    finally:
        loop.close()
    assert loop.gripper_hold_source == base_loop.GRIPPER_HOLD_RECORDED
    for key in GRIPPERS:
        assert loop.held_arm_positions[key] == pytest.approx(0.01), key
    assert robot.sent_actions[-1][GRIPPERS[0]] == pytest.approx(0.01)


def test_an_unreadable_buffer_falls_back_to_the_measured_grippers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from stage_runner import runner

    context = _context(tmp_path)
    loop = _loop(context)
    try:
        _tick(loop)
    finally:
        loop.close()
    monkeypatch.setattr(runner, "_recorded_gripper_goals", lambda context: None)
    second = _loop(context)
    try:
        _tick(second)
    finally:
        second.close()
    assert second.gripper_hold_source == base_loop.GRIPPER_HOLD_MEASURED


# --------------------------------------------------------------------------
# Probe and odometry.


def test_the_probe_is_installed_for_the_stage_and_restored(tmp_path: Path) -> None:
    context = _context(tmp_path)
    slate = context.robot.base
    loop = _loop(context)
    assert isinstance(context.robot.base, BaseProbe)
    assert context.robot.base.wrapped is slate
    loop.close()
    assert context.robot.base is slate
    loop.close()  # idempotent
    assert context.robot.base is slate


def test_the_arms_connect_move_is_waited_out_before_the_clock_starts(
    tmp_path: Path,
) -> None:
    # The real robot's connect() returns while the arms are still moving to
    # the staged pose; the loop must wait before it stamps `started`.
    context = _context(tmp_path)
    calls: list[float] = []

    def wait_for_arrival() -> float:
        calls.append(time.perf_counter())
        return 1.5

    context.robot.wait_for_arrival = wait_for_arrival
    loop = _loop(context)
    try:
        assert len(calls) == 1
        assert calls[0] <= loop.started
        assert loop.arrival_wait_s == pytest.approx(1.5)
    finally:
        loop.close()


def test_a_robot_without_the_arrival_wait_starts_at_once(tmp_path: Path) -> None:
    context = _context(tmp_path)
    assert not hasattr(context.robot, "wait_for_arrival")
    loop = _loop(context)
    try:
        assert loop.arrival_wait_s == 0.0
    finally:
        loop.close()


def test_the_probe_is_restored_when_a_tick_raises(tmp_path: Path) -> None:
    context = _context(tmp_path)
    slate = context.robot.base

    def explode(action: dict[str, float]) -> dict[str, float]:
        raise OSError("serial gone")

    context.robot.send_action = explode
    loop = _loop(context)
    with pytest.raises(OSError, match="serial gone"):
        try:
            _tick(loop)
        finally:
            loop.close()
    assert context.robot.base is slate
    assert context.buffered_frame_count() == 0, "a failed send records nothing"


def test_a_failed_refresh_is_no_odometry_for_that_tick_only(tmp_path: Path) -> None:
    context = _context(tmp_path)
    slate = context.robot.base
    loop = _loop(context)
    try:
        _, odometry, _, _ = _tick(loop)
        assert odometry is not None
        slate.fail_updates = 1
        observation, odometry, _ = loop.observe()
        assert odometry is None
        assert loop.read_base_velocity() is None, "stale = not known to be still"
        assert loop.probe.last_update_ok is False
        loop.act(observation, 0.0, 0.0)
        _, odometry, _, _ = _tick(loop)
        assert odometry is not None
    finally:
        loop.close()
    assert loop.probe.update_failed_ticks == 1


@pytest.mark.parametrize(
    "pose",
    [
        [float("nan"), 0.0, 0.0],
        [0.0, 0.0],
        "garbage",
    ],
)
def test_garbage_odometry_reads_as_none(tmp_path: Path, pose: Any) -> None:
    context = _context(tmp_path)
    context.robot.base.pose_override = pose
    loop = _loop(context)
    try:
        _, odometry, _, _ = _tick(loop)
    finally:
        loop.close()
    assert odometry is None


def test_refused_commands_are_counted_in_runs(tmp_path: Path) -> None:
    context = _context(tmp_path)
    context.robot.base.refuse_cmds = 3
    loop = _loop(context)
    try:
        for _ in range(3):
            _tick(loop)
        assert loop.probe.cmd_refused_run == 3
        assert loop.probe.last_cmd_ok is False
        _tick(loop)
        assert loop.probe.cmd_refused_run == 0
        assert loop.probe.last_cmd_ok is True
    finally:
        loop.close()
    assert loop.probe.counters()["cmd_refused_ticks"] == 3
    assert loop.probe.counters()["max_cmd_refused_run"] == 3


def test_the_base_state_fault_reads_the_robots_last_state(tmp_path: Path) -> None:
    context = _context(tmp_path)
    loop = _loop(context)
    try:
        assert loop.base_state_fault() is None, "the mock reports no state"
        context.robot._last_base_state = SimpleNamespace(name="SYS_NORMAL")
        assert loop.base_state_fault() is None
        assert loop.base_state_name() == "SYS_NORMAL"
        context.robot._last_base_state = SimpleNamespace(name="SYS_ESTOP")
        assert loop.base_state_fault() == "SYS_ESTOP"
        context.robot._last_base_state = SimpleNamespace(name="SYS_ERR_COLLISION")
        assert loop.base_state_fault() == "SYS_ERR_COLLISION"
    finally:
        loop.close()


def test_odometry_follows_the_simulated_base(tmp_path: Path) -> None:
    context = _context(tmp_path)
    loop = _loop(context)
    try:
        _, start, _, _ = _tick(loop, 0.0, 0.5)
        for _ in range(9):
            _tick(loop, 0.0, 0.5)
        _, end, _, _ = _tick(loop, 0.0, 0.0)
        velocity = loop.read_base_velocity()
    finally:
        loop.close()
    turned = end[2] - start[2]
    assert turned == pytest.approx(0.5 * 10 / FPS, rel=0.3)
    assert velocity is not None and velocity[1] == pytest.approx(0.5)


# --------------------------------------------------------------------------
# Keys.


def test_esc_beats_the_right_arrow_and_exit_early_is_consumed(tmp_path: Path) -> None:
    context = _context(tmp_path)
    loop = _loop(context)
    try:
        assert loop.poll_keys() is None
        context.events["exit_early"] = True
        assert loop.poll_keys() == base_loop.KEY_RIGHT
        assert context.events["exit_early"] is False
        assert loop.poll_keys() is None
        context.events.update(exit_early=True, stop_recording=True)
        assert loop.poll_keys() == base_loop.KEY_STOP
        assert context.events["stop_recording"] is True, "abort flags are never cleared"
        context.events.update(stop_recording=False, rerecord_episode=True)
        assert loop.poll_keys() == base_loop.KEY_RERECORD
    finally:
        loop.close()


# --------------------------------------------------------------------------
# Image writer on the way out.


def test_the_image_writer_is_stopped_after_an_exception_only() -> None:
    stops: list[bool] = []
    dataset = SimpleNamespace(
        image_writer=SimpleNamespace(stop=lambda: stops.append(True))
    )
    base_loop.stop_image_writer_after(KeyboardInterrupt(), dataset)
    assert stops == []
    base_loop.stop_image_writer_after(RuntimeError("x"), dataset)
    assert stops == [True]
    base_loop.stop_image_writer_after(
        RuntimeError("x"), SimpleNamespace(image_writer=None)
    )


def test_the_camera_array_goes_into_the_frame_by_reference(tmp_path: Path) -> None:
    # record_loop's rule: images go into add_frame as read; only the align
    # worker's input is copied. With video off lerobot drops the camera from
    # the dataset (pipeline_features), so this checks the observation path.
    context = _context(tmp_path, scene_camera="cam_high")
    loop = _loop(context)
    try:
        observation, _, _ = loop.observe()
        assert isinstance(observation["cam_high"], np.ndarray)
        loop.act(observation, 0.0, 0.0)
    finally:
        loop.close()
    assert context.buffered_frame_count() == 1
