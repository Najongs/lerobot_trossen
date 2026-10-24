"""The mock robot's opt-in simulated base and scene camera, and its unchanged defaults.

The base_replay and base_align stages need a base with odometry and a camera
that sees the base turn; the mock grows both behind two config switches
(simulate_base, scene_camera). Every existing test and smoke config uses the
mock WITHOUT them, so the first thing pinned here is that the defaults are the
robot they were before: no .base, no cameras, the commanded velocity echoed in
the observation, and no cv2 scene module imported.

What is asserted, and why each matters for the stages that use them:

* MockSlate integrates the accepted command against the clock, and its pose
  and velocity come from a cache only update_state refreshes -- as the real
  driver -- so a stage that reads odometry without a fresh update sees it
  stale here too;
* each test hook models one field failure (dropped read, refused command,
  stalled or reversed base, garbage odometry), so a stage's guard for it can
  be tested;
* with simulate_base on, send_action reaches robot.base.set_cmd_vel (sanitized
  like MobileAIRobot) and get_observation calls update_state, so a wrapper
  installed as robot.base sees every command and refresh;
* the scene camera turns a base turn into a view yaw and a drive toward the
  landmark into a landmark scale, with the signs landmark_align expects;
* a chain with the camera still runs end to end through the real cli.

NOTHING HERE IS EVIDENCE ABOUT THE REAL BASE OR CAMERA.
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from stage_runner import cli, landmark_align, mock_scene, record_adapter
from stage_runner.mock_robot import MockRobot, MockRobotConfig, MockSlate

CONFIGS = Path(__file__).parent / "configs"
SMOKE_CONFIG = CONFIGS / "smoke_mock.yaml"


class _Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def _slate(lag_s: float = 0.0) -> tuple[MockSlate, _Clock]:
    clock = _Clock()
    slate = MockSlate(lag_s=lag_s, clock=clock)
    slate.enable_motor_torque(True)
    slate.update_state()
    return slate, clock


def _run_for(
    slate: MockSlate, clock: _Clock, seconds: float, tick: float = 0.033
) -> None:
    steps = round(seconds / tick)
    for _ in range(steps):
        clock.now += tick
        slate.update_state()


# ---------------------------------------------------------------------------
# defaults unchanged
# ---------------------------------------------------------------------------


def test_the_default_mock_has_no_base_and_no_cameras():
    robot = MockRobot(MockRobotConfig(joint_count=4))
    assert not hasattr(robot, "base"), (
        "기본 목업에 .base가 생기면 기존 시험의 전제가 바뀐다"
    )
    assert robot.cameras == {}
    assert set(robot.observation_features) == {
        "joint_0.pos",
        "joint_1.pos",
        "joint_2.pos",
        "joint_3.pos",
        "x.vel",
        "theta.vel",
    }
    robot.connect()
    action = {f"joint_{i}.pos": 0.5 for i in range(4)} | {
        "x.vel": 0.3,
        "theta.vel": -0.2,
    }
    robot.send_action(action)
    observation = robot.get_observation()
    # The default echoes the COMMAND, not a measured velocity.
    assert (observation["x.vel"], observation["theta.vel"]) == (0.3, -0.2)
    robot.disconnect()


def test_an_attached_test_base_is_not_driven_by_send_action():
    # test_smoke_runner and test_leader_reset attach their own fake .base and
    # count its calls; the default mock must keep NOT routing commands to it.
    class _CountingBase:
        calls = 0

        def set_cmd_vel(self, linear: float, angular: float) -> bool:
            _CountingBase.calls += 1
            return True

    robot = MockRobot(MockRobotConfig(joint_count=2))
    robot.base = _CountingBase()
    robot.connect()
    robot.send_action(
        {"joint_0.pos": 0.1, "joint_1.pos": 0.2, "x.vel": 0.4, "theta.vel": 0.0}
    )
    robot.get_observation()
    assert _CountingBase.calls == 0


def test_importing_the_mock_robot_does_not_import_the_scene():
    # mock_robot is imported by every parse_config; the scene (and its cv2
    # rendering) must load only when a config opts in.
    code = (
        "import sys\n"
        "from stage_runner.mock_robot import MockRobot, MockRobotConfig\n"
        "MockRobot(MockRobotConfig(simulate_base=True))\n"
        "assert 'stage_runner.mock_scene' not in sys.modules\n"
        "assert 'stage_runner.landmark_align' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_simulation_config_is_checked():
    with pytest.raises(ValueError, match="base_lag_s"):
        MockRobot(MockRobotConfig(base_lag_s=-1.0))
    with pytest.raises(ValueError, match="scene_camera_matrix"):
        MockRobot(MockRobotConfig(scene_camera="cam_high", scene_camera_matrix=[[1.0]]))
    with pytest.raises(ValueError, match="scene_reference_pose"):
        MockRobot(MockRobotConfig(scene_camera="cam_high", scene_reference_pose=[0.0]))


# ---------------------------------------------------------------------------
# MockSlate
# ---------------------------------------------------------------------------


def test_a_command_integrates_against_the_clock():
    slate, clock = _slate()
    slate.set_cmd_vel(0.5, 0.0)
    _run_for(slate, clock, 1.0)
    x, y, theta = slate.get_pose()
    assert x == pytest.approx(0.5, abs=0.02) and abs(y) < 1e-9 and theta == 0.0
    slate.set_cmd_vel(0.0, 0.5)
    _run_for(slate, clock, 2.0)
    assert slate.get_pose()[2] == pytest.approx(1.0, abs=0.02)
    assert slate.get_vel() == pytest.approx([0.0, 0.5])


def test_a_turn_with_forward_motion_curves():
    slate, clock = _slate()
    slate.set_cmd_vel(0.5, 0.5)
    _run_for(slate, clock, math.pi / 0.5)  # half a circle of radius 1 m
    x, y, theta = slate.get_pose()
    assert x == pytest.approx(0.0, abs=0.03) and y == pytest.approx(2.0, abs=0.03)
    assert abs(theta) == pytest.approx(math.pi, abs=0.03)


def test_heading_is_reported_wrapped():
    slate, clock = _slate()
    slate.set_cmd_vel(0.0, 1.0)
    _run_for(slate, clock, 4.0)
    assert slate.get_pose()[2] == pytest.approx(4.0 - 2 * math.pi, abs=0.05)
    assert slate.true_pose()[2] == pytest.approx(4.0, abs=0.05), "내부는 언랩 유지"


def test_pose_is_a_cache_only_update_state_refreshes():
    slate, clock = _slate()
    slate.set_cmd_vel(1.0, 0.0)
    clock.now += 0.5
    assert slate.get_pose()[0] == 0.0, "update_state 전에는 옛 캐시"
    slate.update_state()
    assert slate.get_pose()[0] == pytest.approx(0.5)


def test_a_failed_update_leaves_the_cache_while_the_base_moves():
    slate, clock = _slate()
    slate.set_cmd_vel(1.0, 0.0)
    slate.fail_updates = 2
    clock.now += 0.1
    assert slate.update_state() is False
    clock.now += 0.1
    assert slate.update_state() is False
    assert slate.get_pose()[0] == 0.0
    clock.now += 0.1
    assert slate.update_state() is True
    assert slate.get_pose()[0] == pytest.approx(0.3)
    assert slate.update_failures == 2


def test_refused_commands_return_false_and_do_not_act():
    slate, clock = _slate()
    slate.refuse_cmds = 2
    assert slate.set_cmd_vel(1.0, 0.0) is False
    assert slate.set_cmd_vel(1.0, 0.0) is False
    _run_for(slate, clock, 0.5)
    assert slate.get_pose()[0] == 0.0
    assert slate.set_cmd_vel(1.0, 0.0) is True
    assert [accepted for *_, accepted in slate.commands] == [False, False, True]


def test_ignored_commands_and_torque_off_do_not_move_the_base():
    slate, clock = _slate()
    slate.ignore_cmds = True
    assert slate.set_cmd_vel(1.0, 1.0) is True
    _run_for(slate, clock, 0.5)
    assert slate.get_pose() == [0.0, 0.0, 0.0]
    slate.ignore_cmds = False
    slate.enable_motor_torque(False)
    slate.set_cmd_vel(1.0, 1.0)
    _run_for(slate, clock, 0.5)
    assert slate.get_pose() == [0.0, 0.0, 0.0]


def test_a_reversed_base_moves_against_the_command():
    slate, clock = _slate()
    slate.command_gain = -1.0
    slate.set_cmd_vel(0.0, 0.4)
    _run_for(slate, clock, 1.0)
    assert slate.get_pose()[2] == pytest.approx(-0.4, abs=0.02)


def test_lag_delays_the_velocity():
    slate, clock = _slate(lag_s=0.13)
    slate.set_cmd_vel(0.0, 1.0)
    clock.now += 0.13
    slate.update_state()
    assert slate.get_vel()[1] == pytest.approx(1.0 - math.exp(-1.0), abs=0.01)


def test_overrides_replace_the_reads():
    slate, _ = _slate()
    slate.pose_override = [float("nan"), 0.0, 0.0]
    slate.velocity_override = [0.2, 0.0]
    assert math.isnan(slate.get_pose()[0])
    assert slate.get_vel() == [0.2, 0.0]
    assert slate.true_pose() == (0.0, 0.0, 0.0), "카메라가 보는 실제 위치는 그대로"


# ---------------------------------------------------------------------------
# MockRobot with the simulated base
# ---------------------------------------------------------------------------


def _bimanual(**overrides) -> MockRobot:
    return MockRobot(
        MockRobotConfig(arm_layout="bimanual", simulate_base=True, **overrides)
    )


def _hold_action(robot: MockRobot, x_vel: float, theta_vel: float) -> dict[str, float]:
    observation = robot.get_observation()
    action = {
        key: observation[key] for key in robot.action_features if key.endswith(".pos")
    }
    return {**action, "x.vel": x_vel, "theta.vel": theta_vel}


def test_send_action_drives_the_base_through_robot_base_sanitized():
    robot = _bimanual()
    robot.connect()
    assert robot.base.torque_enabled, "enable_base_motor_torque 기본 True로 연결"
    robot.send_action(_hold_action(robot, 0.2, -0.1))
    robot.send_action(_hold_action(robot, float("nan"), 7.0))
    assert [command[:2] for command in robot.base.commands] == [(0.2, -0.1), (0.0, 1.0)]
    assert robot.sent_actions[-1]["theta.vel"] == 1.0
    robot.disconnect()
    assert robot.base.commands[-1][:2] == (0.0, 0.0), "disconnect는 베이스를 0으로"


def test_torque_off_in_the_config_leaves_the_base_unpowered():
    robot = _bimanual(enable_base_motor_torque=False)
    robot.connect()
    assert not robot.base.torque_enabled


def test_the_observation_reports_the_measured_velocity():
    robot = _bimanual()
    robot.connect()
    robot.base.velocity_override = [0.05, -0.02]
    observation = robot.get_observation()
    assert (observation["x.vel"], observation["theta.vel"]) == (0.05, -0.02)
    before = robot.base.update_calls
    robot.get_observation()
    assert robot.base.update_calls == before + 1, "관측마다 update_state 한 번"


# ---------------------------------------------------------------------------
# scene camera
# ---------------------------------------------------------------------------


def _scene_robot(**overrides) -> MockRobot:
    robot = _bimanual(scene_camera="cam_high", **overrides)
    robot.connect()
    return robot


def test_the_scene_camera_is_an_observation_feature_and_key():
    robot = _scene_robot()
    assert robot.observation_features["cam_high"] == (480, 640, 3)
    first = robot.get_observation()["cam_high"]
    second = robot.get_observation()["cam_high"]
    assert first.shape == (480, 640, 3) and first.dtype == np.uint8
    assert first is not second, "프레임마다 새 배열(데이터셋이 참조로 담는다)"
    assert np.array_equal(first, mock_scene.render_scene())


def test_turning_the_base_turns_the_view_with_the_align_sign():
    robot = _scene_robot()
    reference = robot.get_observation()["cam_high"]
    robot.base.place(0.0, 0.0, math.radians(6.0))  # turned left
    live = robot.get_observation()["cam_high"]
    k = np.array(robot.config.scene_camera_matrix)
    up = np.array(robot.config.scene_up) / np.linalg.norm(robot.config.scene_up)
    yaw, _, _ = landmark_align.yaw_between(reference, live, k, np.zeros(5), up)
    assert yaw == pytest.approx(6.0, abs=0.3)


def test_driving_toward_the_landmark_makes_it_bigger():
    robot = _scene_robot(scene_reference_pose=[1.0, 2.0, math.pi / 2])
    robot.base.place(1.0, 2.0, math.pi / 2)
    assert robot.cameras["cam_high"].view() == pytest.approx((0.0, 1.0))
    robot.base.place(1.0, 2.03, math.pi / 2)  # 3 cm closer along +y
    yaw, scale = robot.cameras["cam_high"].view()
    assert yaw == pytest.approx(0.0) and scale == pytest.approx(0.60 / 0.57)


def test_the_view_override_renders_a_given_view():
    robot = _scene_robot()
    camera = robot.cameras["cam_high"]
    camera.view_override = (25.0, 1.0)
    assert np.array_equal(
        robot.get_observation()["cam_high"], mock_scene.render_scene(view_yaw_deg=25.0)
    )


def test_a_chain_with_the_scene_camera_runs_end_to_end(tmp_path, monkeypatch):
    # The camera adds an image feature to the dataset and an image writer to
    # the trial; the policy stages of the existing smoke chain must not care.
    argv = [
        "--config_path",
        str(SMOKE_CONFIG),
        "--dataset.repo_id=local/eval_mock_scene",
        f"--dataset.root={tmp_path / 'dataset'}",
        f"--output.root={tmp_path / 'outputs'}",
        "--robot.simulate_base=true",
        "--robot.scene_camera=cam_high",
    ]
    created: list[MockRobot] = []
    original = record_adapter.make_robot

    def capture(robot_config):
        robot = original(robot_config)
        created.append(robot)
        return robot

    monkeypatch.setattr(record_adapter, "make_robot", capture)
    assert cli.main(argv) == 0
    robot = created[0]
    assert robot.cameras["cam_high"].reads > 0
    assert robot.base.commands, "정책 단계의 베이스 명령이 robot.base로 갔다"
    # With video: false lerobot drops image features from the dataset
    # altogether (datasets/pipeline_features.py:120, "If False, image features
    # are excluded"), so the smoke chain records no frames of this camera: a
    # stage reads it from the observation, never from the dataset.
    info = (tmp_path / "dataset" / "meta" / "info.json").read_text()
    assert "observation.images.cam_high" not in info
