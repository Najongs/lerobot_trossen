"""The RUN-TIME half of the start pose: config, preflight, the move, its event.

Drives the real cli.main against tests/configs/smoke_mock_start_pose.yaml -- the
same mock chain as test_smoke_runner.py, with the bimanual mock layout and a
start_pose on both stages -- so lerobot's real record_loop runs and the move goes
through reposition.move_arms_to_start_pose against the mock's per-arm drivers,
the same attribute path (robot.arms.<side>.driver) it takes on hardware.

What is asserted, and why each matters on the robot:

* the ARMS get the pose, mapped BY NAME into each driver's joint order, and the
  first recorded frame of each stage is that pose -- so the policy really did
  start from it;
* the GRIPPERS get nothing: a stage that begins holding a beaker must not drop it;
* the BASE is stopped before the arms move at a boundary, and the first stage is
  moved too;
* a pose that is not reached ends the trial instead of starting the policy, and
  the log stays closed and paired;
* the stage_end -> stage_start residual means what it meant before start poses.

NOTHING HERE IS EVIDENCE ABOUT HARDWARE TIMING OR TRACKING. The mock driver
completes a move instantly and exactly; whether the real arms settle within
position_tolerance_rad in SETTLE_TIMEOUT_S, and how long a paced move takes, are
first-run measurements (reposition_s and max_abs_error_rad in the event).
"""

from __future__ import annotations

import contextlib
import json
import math
from pathlib import Path
from typing import Any
from unittest import mock

import pandas as pd
import pytest
from lerobot.robots import Robot
from lerobot.robots.config import RobotConfig
from stage_runner import (
    aggregate,
    cli,
    config,
    preflight,
    record_adapter,
    reposition,
    transitions,
)
from stage_runner.config import (
    StageConfig,
    StageRunnerConfig,
    StartPoseConfig,
    TerminatorConfig,
)
from stage_runner.mock_robot import MockRobot, MockRobotConfig

CONFIGS = Path(__file__).parent / "configs"
POSED_CONFIG = CONFIGS / "smoke_mock_start_pose.yaml"
REPO_ID = "local/eval_smoke_chain"
LEFT_ARM = [f"left_joint_{index}.pos" for index in range(6)]
RIGHT_ARM = [f"right_joint_{index}.pos" for index in range(6)]
GRIPPERS = ["left_left_carriage_joint.pos", "right_left_carriage_joint.pos"]
DRIVING_BASE = {"x.vel": 0.4, "theta.vel": 0.0}
STOPPED_BASE = {"x.vel": 0.0, "theta.vel": 0.0}


def _posed_targets() -> dict[str, dict[str, float]]:
    parsed = config.parse_config(["--config_path", str(POSED_CONFIG)])
    return {stage.id: dict(stage.start_pose.positions) for stage in parsed.stages}


def _bimanual_robot() -> MockRobot:
    return MockRobot(MockRobotConfig(id="unit", arm_layout="bimanual"))


def _pose(**overrides: float) -> dict[str, float]:
    positions = {key: 0.1 for key in LEFT_ARM + RIGHT_ARM}
    positions.update(overrides)
    return positions


# --------------------------------------------------------------------------
# Config: the field is optional and an old YAML is unchanged.


def test_an_old_yaml_parses_with_no_start_pose() -> None:
    """Every YAML written before the field existed means exactly what it meant."""
    smoke = config.parse_config(["--config_path", str(CONFIGS / "smoke_mock.yaml")])
    assert [stage.start_pose for stage in smoke.stages] == [None, None]
    assert smoke.version == config.LATEST_CONFIG_VERSION == 1

    # The production config too -- it needs the Trossen plugin for its robot type.
    record_adapter.register_plugins()
    chain = config.parse_config(
        [
            "--config_path",
            str(Path(__file__).parents[3] / "configs" / "chain_task45.yaml"),
        ]
    )
    assert [stage.start_pose for stage in chain.stages] == [None, None]


def test_the_resolved_snapshot_of_an_old_yaml_is_unchanged() -> None:
    """No start_pose -> no `start_pose: null` line in config.resolved.yaml.

    The dump equals draccus.dump(config) with exactly those lines removed, which
    is what the dump of a StageConfig without the field was. Checked once by hand
    against the pre-feature config.py on smoke_mock.yaml and chain_task45.yaml.
    """
    import draccus

    smoke = config.parse_config(["--config_path", str(CONFIGS / "smoke_mock.yaml")])
    text = config.resolved_config_text(smoke)
    assert "start_pose" not in text
    full = draccus.dump(smoke)
    assert text == "".join(
        line
        for line in full.splitlines(keepends=True)
        if "start_pose: null" not in line
    )

    posed = config.parse_config(["--config_path", str(POSED_CONFIG)])
    assert config.resolved_config_text(posed).count("start_pose:") == 2


def test_a_posed_yaml_parses_every_field() -> None:
    parsed = config.parse_config(["--config_path", str(POSED_CONFIG)])
    s4 = parsed.stages[0].start_pose
    assert s4 is not None
    assert list(s4.positions) == LEFT_ARM + RIGHT_ARM
    assert s4.positions["left_joint_0.pos"] == -0.25
    assert s4.min_goal_time_s == 3.0
    assert s4.position_tolerance_rad == 0.05
    assert s4.source.dataset_repo_id == "local/hand_written_smoke_pose"
    assert s4.source.episode_index == 0
    assert s4.source.n_episodes == 1
    assert s4.source.method == "hand_written_test_value"
    assert s4.source.distance == 0.0
    assert s4.source.dataset_revision is None


# --------------------------------------------------------------------------
# Preflight: refused before the robot moves.


def _config_with(
    start_pose: StartPoseConfig, arm_layout: str = "bimanual"
) -> StageRunnerConfig:
    return StageRunnerConfig(
        robot=MockRobotConfig(id="unit", arm_layout=arm_layout),
        stages=[
            StageConfig(
                id="s4",
                policy_path="mock://hold",
                terminator=TerminatorConfig(type="timeout", timeout_s=1.0),
                start_pose=start_pose,
            )
        ],
    )


@pytest.mark.parametrize(
    ("positions", "message"),
    [
        (_pose(**{"left_left_carriage_joint.pos": 0.02}), "gripper key"),
        (_pose(**{"x.vel": 0.0}), "base key"),
        ({key: 0.1 for key in LEFT_ARM + RIGHT_ARM[:-1]}, "missing arm joint key"),
        (_pose(**{"joint_0.pos": 0.1}), "not arm joints of this robot"),
        (_pose(left_joint_3_pos_typo=0.1), "not arm joints of this robot"),
        (_pose(**{"right_joint_2.pos": float("nan")}), "non-finite"),
        (_pose(**{"right_joint_2.pos": float("inf")}), "non-finite"),
    ],
)
def test_preflight_rejects_a_bad_start_pose(
    positions: dict[str, float], message: str
) -> None:
    robot = _bimanual_robot()
    with pytest.raises(preflight.PreflightError, match=message):
        preflight.check_stage_start_poses(
            _config_with(StartPoseConfig(positions=positions)), robot
        )
    assert robot.is_connected is False


def test_preflight_accepts_exactly_the_arm_joints() -> None:
    preflight.check_stage_start_poses(
        _config_with(StartPoseConfig(positions=_pose())), _bimanual_robot()
    )


def test_preflight_refuses_a_start_pose_on_a_robot_without_arm_drivers() -> None:
    """The flat mock has no robot.arms -- the same as any robot this cannot drive."""
    flat = MockRobot(MockRobotConfig(id="unit"))
    with pytest.raises(preflight.PreflightError, match="exposes no .arms"):
        preflight.check_stage_start_poses(
            _config_with(StartPoseConfig(positions={"joint_0.pos": 0.1}), "flat"), flat
        )


@pytest.mark.parametrize(
    ("start_pose", "message"),
    [
        (StartPoseConfig(positions={}), "is empty"),
        (StartPoseConfig(positions=_pose(), min_goal_time_s=0.0), "min_goal_time_s"),
        (
            StartPoseConfig(positions=_pose(), min_goal_time_s=float("nan")),
            "min_goal_time_s",
        ),
        (
            StartPoseConfig(positions=_pose(), position_tolerance_rad=0.0),
            "position_tolerance_rad",
        ),
    ],
)
def test_stage_definitions_reject_unusable_start_pose_parameters(
    start_pose: StartPoseConfig, message: str
) -> None:
    with pytest.raises(preflight.PreflightError, match=message):
        preflight.check_stage_definitions(_config_with(start_pose))


def test_a_bad_start_pose_stops_cli_before_connect(tmp_path: Path) -> None:
    """End to end: exit 2, and the robot was never connected or commanded."""
    bad = POSED_CONFIG.read_text().replace(
        "        right_joint_5.pos: -0.2\n",
        "        right_joint_5.pos: -0.2\n        right_left_carriage_joint.pos: 0.0\n",
        1,
    )
    assert bad != POSED_CONFIG.read_text()
    path = tmp_path / "bad.yaml"
    path.write_text(bad)
    robots: list[Any] = []
    real = record_adapter.make_robot

    def make_robot(robot_config: RobotConfig) -> Robot:
        robots.append(real(robot_config))
        return robots[-1]

    with mock.patch.object(record_adapter, "make_robot", make_robot):
        code = cli.main(
            [
                "--config_path",
                str(path),
                f"--dataset.root={tmp_path / 'dataset'}",
                f"--output.root={tmp_path / 'outputs'}",
            ]
        )
    assert code == cli.EXIT_PREFLIGHT
    (robot,) = robots
    assert robot.is_connected is False
    assert robot.sent_actions == []
    assert robot.arms.right_arm.driver.arm_position_commands == []


# --------------------------------------------------------------------------
# The move itself, on a constructed mock (no cli).


def test_the_move_maps_by_name_and_never_touches_the_gripper() -> None:
    robot = _bimanual_robot()
    robot.connect()
    grippers_before = {key: robot.get_observation()[key] for key in GRIPPERS}
    # A pose whose LEFT and RIGHT values are deliberately different per joint, so
    # a left/right swap or an off-by-one in driver order cannot pass.
    target = {key: 0.01 * (index + 1) for index, key in enumerate(LEFT_ARM + RIGHT_ARM)}

    result = reposition.move_arms_to_start_pose(
        robot, target, min_goal_time_s=3.0, position_tolerance_rad=0.05
    )

    assert result.reached is True
    left = robot.arms.left_arm.driver.arm_position_commands
    right = robot.arms.right_arm.driver.arm_position_commands
    assert [command["goal_positions"] for command in left] == [
        [target[k] for k in LEFT_ARM]
    ]
    assert [command["goal_positions"] for command in right] == [
        [target[k] for k in RIGHT_ARM]
    ]
    # Concurrent, then waited on: left non-blocking, right blocking, one goal time.
    assert [command["blocking"] for command in left + right] == [False, True]
    assert left[0]["goal_time"] == right[0]["goal_time"] == 3.0
    for arm in (robot.arms.left_arm, robot.arms.right_arm):
        assert arm.driver.gripper_commands == []
    observation = robot.get_observation()
    assert {key: observation[key] for key in GRIPPERS} == grippers_before
    assert {key: observation[key] for key in target} == target
    assert not any(key in result.start_positions for key in GRIPPERS)
    # Nothing went through the policy's path.
    assert robot.sent_actions == []


def test_the_goal_time_is_stretched_so_no_joint_exceeds_its_safe_velocity() -> None:
    """joint_0's placeholder limit is pi rad/s x 0.4 = 1.2566 rad/s; 5 rad needs ~3.98 s."""
    robot = _bimanual_robot()
    robot.connect()
    start = robot.get_observation()["left_joint_0.pos"]
    target = {key: robot.get_observation()[key] for key in LEFT_ARM + RIGHT_ARM}
    target["left_joint_0.pos"] = start + 5.0
    result = reposition.move_arms_to_start_pose(
        robot, target, min_goal_time_s=3.0, position_tolerance_rad=0.05
    )
    assert math.isclose(result.goal_time_s, 5.0 / (3.14159 * 0.4), rel_tol=1e-9)
    # A move inside the floor keeps the floor.
    small = dict(target, **{"left_joint_0.pos": start + 5.01})
    assert (
        reposition.move_arms_to_start_pose(
            robot, small, min_goal_time_s=3.0, position_tolerance_rad=0.05
        ).goal_time_s
        == 3.0
    )


def test_a_joint_that_does_not_arrive_is_not_reached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(reposition, "SETTLE_TIMEOUT_S", 0.05)
    robot = _bimanual_robot()
    robot.connect()
    robot.arms.right_arm.driver.stall_joint = "joint_3"
    result = reposition.move_arms_to_start_pose(
        robot, _pose(), min_goal_time_s=3.0, position_tolerance_rad=0.05
    )
    assert result.reached is False
    assert "right_joint_3.pos" in result.detail
    assert result.max_abs_error > 0.05


def test_a_target_past_a_joint_limit_is_refused_before_anything_moves() -> None:
    robot = _bimanual_robot()
    robot.connect()
    # A hold that has not expired yet: a refusal must not wait for it either.
    clock = _FakeClock()
    _set_last_command(robot, sent_s=clock(), left=0.1, right=0.1)
    slept: list[float] = []
    with pytest.raises(reposition.StageRepositionError, match="joint limits"):
        reposition.move_arms_to_start_pose(
            robot,
            _pose(**{"left_joint_1.pos": 7.0}),
            min_goal_time_s=3.0,
            position_tolerance_rad=0.05,
            sleep=slept.append,
            clock=clock,
        )
    assert robot.arms.left_arm.driver.arm_position_commands == []
    assert robot.arms.right_arm.driver.arm_position_commands == []
    assert slept == []


# --------------------------------------------------------------------------
# The wait for the previous trajectory to expire, before the move.
#
# trossen_arm 1.9.0 seeds a new trajectory from the COMMANDED state of the one
# still running; a 3 s move sent mid-way through the ~0.1 s boundary hold
# inherits its acceleration and swings the arms (reposition.py's docstring).
# The mock has no trajectories, so what is asserted is the arithmetic and the
# ORDER: the sleep is exactly the time left plus the margin, and it happens
# before the first arm command.


class _FakeClock:
    """A clock that only moves when the code under test sleeps."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now


def _set_last_command(
    robot: MockRobot, *, sent_s: float | None, left: float | None, right: float | None
) -> None:
    for arm, goal_time in ((robot.arms.left_arm, left), (robot.arms.right_arm, right)):
        arm.last_command_sent_s = sent_s
        arm.last_command_goal_time_s = goal_time


def _ordered_move(
    robot: MockRobot, clock: _FakeClock
) -> tuple[reposition.RepositionResult, list[tuple[str, Any]]]:
    """Move to the robot's own pose, logging sleeps and arm commands in order."""
    calls: list[tuple[str, Any]] = []

    def sleep(seconds: float) -> None:
        calls.append(("sleep", seconds))
        clock.now += seconds

    for side in ("left_arm", "right_arm"):
        driver = getattr(robot.arms, side).driver
        original = driver.set_arm_positions

        def set_arm_positions(
            *args: Any, _side: str = side, _original: Any = original, **kwargs: Any
        ) -> None:
            calls.append(("move", (_side, clock())))
            return _original(*args, **kwargs)

        driver.set_arm_positions = set_arm_positions
    here = robot.get_observation()
    result = reposition.move_arms_to_start_pose(
        robot,
        {key: here[key] for key in LEFT_ARM + RIGHT_ARM},
        min_goal_time_s=3.0,
        position_tolerance_rad=0.05,
        sleep=sleep,
        clock=clock,
    )
    return result, calls


def test_the_move_waits_out_the_previous_trajectory_before_its_first_command() -> None:
    robot = _bimanual_robot()
    robot.connect()
    clock = _FakeClock()
    # The hold was sent 45 ms ago with the 0.1 s minimum window (the 2026-10-20
    # sequence): 55 ms of it are left.
    _set_last_command(robot, sent_s=clock() - 0.045, left=0.1, right=0.1)

    result, calls = _ordered_move(robot, clock)

    expected = 0.1 - 0.045 + reposition.TRAJECTORY_EXPIRY_MARGIN_S
    assert calls[0][0] == "sleep"
    assert math.isclose(calls[0][1], expected, abs_tol=1e-9)
    # Both arm commands come after the wait, at a time the hold had expired.
    assert [name for name, _ in calls[1:3]] == ["move", "move"]
    for _, (_, at) in calls[1:3]:
        assert at >= 1000.0 - 0.045 + 0.1
    assert result.reached is True
    assert math.isclose(result.expiry_wait_s, expected, abs_tol=1e-9)
    assert result.expiry_wait_source == reposition.EXPIRY_WAIT_RECORDED
    assert result.previous_goal_time_s == {"left_arm": 0.1, "right_arm": 0.1}
    # The wait commands nothing: one move per arm, nothing to the gripper.
    for arm in (robot.arms.left_arm, robot.arms.right_arm):
        assert len(arm.driver.arm_position_commands) == 1
        assert arm.driver.gripper_commands == []
    assert robot.sent_actions == []


def test_the_later_of_the_two_arms_sets_the_wait() -> None:
    """A paced right-arm hold (0.25 s) outlasts the left's 0.1 s minimum."""
    robot = _bimanual_robot()
    robot.connect()
    clock = _FakeClock()
    _set_last_command(robot, sent_s=clock(), left=0.1, right=0.25)

    result, calls = _ordered_move(robot, clock)

    assert calls[0][0] == "sleep"
    assert math.isclose(
        calls[0][1], 0.25 + reposition.TRAJECTORY_EXPIRY_MARGIN_S, abs_tol=1e-9
    )
    assert result.previous_goal_time_s == {"left_arm": 0.1, "right_arm": 0.25}


def test_an_expired_previous_trajectory_costs_no_wait() -> None:
    robot = _bimanual_robot()
    robot.connect()
    clock = _FakeClock()
    # Expired, margin included, a moment ago.
    _set_last_command(
        robot,
        sent_s=clock() - 0.1 - reposition.TRAJECTORY_EXPIRY_MARGIN_S - 0.001,
        left=0.1,
        right=0.1,
    )

    result, calls = _ordered_move(robot, clock)

    assert [name for name, _ in calls] == ["move", "move"]
    assert result.expiry_wait_s == 0.0
    assert result.expiry_wait_source == reposition.EXPIRY_WAIT_RECORDED


def test_no_previous_command_means_no_wait() -> None:
    """The mock's connect() moves nothing, so there is nothing to inherit from."""
    robot = _bimanual_robot()
    robot.connect()
    assert robot.arms.left_arm.last_command_sent_s is None

    result, calls = _ordered_move(robot, _FakeClock())

    assert [name for name, _ in calls] == ["move", "move"]
    assert result.expiry_wait_s == 0.0
    assert result.expiry_wait_source == reposition.EXPIRY_WAIT_NO_COMMAND
    assert result.previous_goal_time_s == {"left_arm": None, "right_arm": None}


def test_a_policy_frame_through_send_action_is_what_the_wait_reads() -> None:
    """The record is written by send_action itself, not by a test helper."""
    robot = _bimanual_robot()
    robot.connect()
    robot.send_action(robot.get_observation())
    for arm in (robot.arms.left_arm, robot.arms.right_arm):
        assert arm.last_command_goal_time_s == pytest.approx(0.1)
        assert arm.last_command_sent_s is not None


def test_an_arm_without_a_record_waits_the_fixed_fallback() -> None:
    """An older robot package: no record, so a fixed wait and a warning."""
    robot = _bimanual_robot()
    robot.connect()
    del robot.arms.right_arm.last_command_sent_s
    del robot.arms.right_arm.last_command_goal_time_s

    # Patched rather than caplog: the suite runs with -p no:logging.
    with mock.patch.object(reposition.logger, "warning") as warning:
        result, calls = _ordered_move(robot, _FakeClock())

    assert calls[0] == ("sleep", reposition.UNRECORDED_EXPIRY_WAIT_S)
    assert [name for name, _ in calls[1:]] == ["move", "move"]
    assert result.expiry_wait_source == reposition.EXPIRY_WAIT_UNRECORDED
    (call,) = warning.call_args_list
    assert "right_arm" in call.args[0]


# --------------------------------------------------------------------------
# The chain, through cli.main.


def _run(
    tmp_path: Path,
    *,
    config_path: Path = POSED_CONFIG,
    after_stage: Any = None,
    instrument: Any = None,
    preset_events: dict[str, bool] | None = None,
) -> dict[str, Any]:
    """Drive cli.main once; patch only record_adapter's module-level seams."""
    captured: dict[str, Any] = {"robot": None, "base_at_encoder_flush": None}
    real_make_robot = record_adapter.make_robot

    def make_robot(robot_config: RobotConfig) -> Robot:
        robot = real_make_robot(robot_config)
        captured["robot"] = robot
        if instrument is not None:
            instrument(robot, captured)
        return robot

    real_manager = record_adapter.video_encoding_manager

    @contextlib.contextmanager
    def video_encoding_manager(dataset: Any) -> Any:
        with real_manager(dataset):
            try:
                yield
            finally:
                captured["base_at_encoder_flush"] = dict(
                    captured["robot"]._base_velocity
                )

    real_call = record_adapter.call_record_loop
    calls = {"n": 0}

    def call_record_loop(**kwargs: Any) -> Any:
        calls["n"] += 1
        result = real_call(**kwargs)
        if after_stage is not None:
            after_stage(calls["n"], captured)
        return result

    real_events = record_adapter.make_keyboard_events

    def make_keyboard_events() -> tuple[Any | None, dict[str, bool]]:
        listener, keyboard_events = real_events()
        keyboard_events.update(preset_events or {})
        return listener, keyboard_events

    argv = [
        "--config_path",
        str(config_path),
        f"--dataset.root={tmp_path / 'dataset'}",
        f"--output.root={tmp_path / 'outputs'}",
    ]
    raised: BaseException | None = None
    exit_code: int | None = None
    with contextlib.ExitStack() as stack:
        for patch in (
            mock.patch.object(record_adapter, "make_robot", make_robot),
            mock.patch.object(record_adapter, "call_record_loop", call_record_loop),
            mock.patch.object(
                record_adapter, "video_encoding_manager", video_encoding_manager
            ),
            mock.patch.object(
                record_adapter, "make_keyboard_events", make_keyboard_events
            ),
        ):
            stack.enter_context(patch)
        try:
            exit_code = cli.main(argv)
        except BaseException as error:  # noqa: BLE001 -- the case under test
            raised = error
    events_path = next((tmp_path / "outputs").glob("*/events.jsonl"))
    return {
        "exit_code": exit_code,
        "raised": raised,
        "robot": captured["robot"],
        "base_at_encoder_flush": captured["base_at_encoder_flush"],
        "events": [
            json.loads(line) for line in events_path.read_text().splitlines() if line
        ],
        "events_path": events_path,
        "dataset_root": tmp_path / "dataset",
    }


def _of(result: dict[str, Any], name: str) -> list[dict[str, Any]]:
    return [event for event in result["events"] if event["event"] == name]


def _names(result: dict[str, Any]) -> list[str]:
    return [event["event"] for event in result["events"]]


def _assert_log_is_closed(result: dict[str, Any]) -> None:
    """The same invariant test_smoke_runner asserts on every failure path."""
    names = _names(result)
    assert names[-1] == "trial_end", names
    depth = 0
    for name in names:
        depth += 1 if name == "stage_start" else (-1 if name == "stage_end" else 0)
        assert depth in (0, 1), names
    assert depth == 0, names
    assert names.count("transition") == names.count("stage_start"), names


def test_both_stages_start_from_their_pose_and_the_gripper_is_never_commanded(
    tmp_path: Path,
) -> None:
    """The happy path, first stage included, base stopped before the boundary move."""
    targets = _posed_targets()
    base_when_arms_were_commanded: list[dict[str, float]] = []

    def instrument(robot: Any, captured: dict[str, Any]) -> None:
        original = robot.arms.left_arm.driver.set_arm_positions

        def set_arm_positions(*args: Any, **kwargs: Any) -> None:
            base_when_arms_were_commanded.append(dict(robot._base_velocity))
            return original(*args, **kwargs)

        robot.arms.left_arm.driver.set_arm_positions = set_arm_positions

    def leave_the_base_driving(call_number: int, captured: dict[str, Any]) -> None:
        # The previous stage's policy left the base moving: the boundary's
        # stop_base must zero it BEFORE the arms move.
        if call_number == 1:
            captured["robot"]._base_velocity = dict(DRIVING_BASE)

    result = _run(tmp_path, instrument=instrument, after_stage=leave_the_base_driving)

    assert result["raised"] is None, result["raised"]
    assert result["exit_code"] == cli.EXIT_OK
    assert _names(result) == [
        "trial_start",
        "stage_reposition",
        "stage_start",
        "stage_end",
        "transition",
        "stage_reposition",
        "stage_start",
        "stage_end",
        "transition",
        "trial_end",
    ]
    _assert_log_is_closed(result)

    moves = _of(result, "stage_reposition")
    assert [move["stage_id"] for move in moves] == ["s4", "s5"]
    assert [move["outcome"] for move in moves] == ["reached", "reached"]
    assert moves[0]["reason"].startswith("reached within")
    # Stage 0 has no boundary before it; stage 1 follows a primary stop.
    assert [move["boundary_stop_base_path"] for move in moves] == [None, "primary"]
    assert all(move["gripper_commanded"] is False for move in moves)
    assert [move["target"] for move in moves] == [targets["s4"], targets["s5"]]
    assert moves[0]["source"]["dataset_repo_id"] == "local/hand_written_smoke_pose"
    assert moves[1]["source"]["episode_index"] == 1
    assert all(move["goal_time_s"] == 3.0 for move in moves)
    assert all(isinstance(move["reposition_s"], float) for move in moves)
    # The expiry wait. Stage 0 follows connect(), which commanded nothing on the
    # mock. Stage 1 follows the boundary hold, a send_action the mock records
    # with the 0.1 s minimum window, so it waits at most that plus the margin
    # (less whatever the boundary bookkeeping already took), and the wait is
    # part of the move's own time.
    assert [move["expiry_wait_source"] for move in moves] == [
        "no_command",
        "recorded",
    ]
    assert moves[0]["expiry_wait_s"] == 0.0
    assert moves[1]["previous_goal_time_s"] == {"left_arm": 0.1, "right_arm": 0.1}
    longest = 0.1 + reposition.TRAJECTORY_EXPIRY_MARGIN_S
    assert 0.0 <= moves[1]["expiry_wait_s"] <= longest
    assert all(move["reposition_s"] >= move["expiry_wait_s"] for move in moves)
    assert all(move["max_abs_error_rad"] == 0.0 for move in moves)

    # The base was zeroed by the boundary's stop_base BEFORE the boundary move.
    assert base_when_arms_were_commanded == [STOPPED_BASE, STOPPED_BASE]

    robot = result["robot"]
    for arm, keys in (
        (robot.arms.left_arm, LEFT_ARM),
        (robot.arms.right_arm, RIGHT_ARM),
    ):
        commands = arm.driver.arm_position_commands
        assert [command["goal_positions"] for command in commands] == [
            [targets["s4"][key] for key in keys],
            [targets["s5"][key] for key in keys],
        ]
        assert arm.driver.gripper_commands == [], "그리퍼에 명령이 갔다"

    # No frame is recorded during a move: frame_idx is the same on both sides.
    stage_ends = _of(result, "stage_end")
    assert moves[0]["frame_idx"] == 0
    assert moves[1]["frame_idx"] == stage_ends[0]["frame_idx"]
    assert _of(result, "stage_start")[1]["frame_idx"] == stage_ends[0]["frame_idx"]

    # THE POLICY STARTED FROM THE POSE. The first recorded frame of each stage is
    # read before that stage's first action, so its observation.state is where
    # the move left the arms -- and the grippers are where they always were.
    info = json.loads((result["dataset_root"] / "meta" / "info.json").read_text())
    names = info["features"]["observation.state"]["names"]
    frames = pd.concat(
        [
            pd.read_parquet(path)
            for path in sorted(result["dataset_root"].glob("data/**/*.parquet"))
        ]
    ).sort_values("frame_index")
    states = [list(state) for state in frames["observation.state"]]
    boundary = int(stage_ends[0]["frame_idx"])
    for frame, stage_id in ((0, "s4"), (boundary, "s5")):
        recorded = dict(zip(names, states[frame], strict=True))
        for key, value in targets[stage_id].items():
            assert math.isclose(recorded[key], value, abs_tol=1e-6), (stage_id, key)
    seeds = {"left_left_carriage_joint.pos": 0.7, "right_left_carriage_joint.pos": 1.4}
    for state in (states[0], states[boundary], states[-1]):
        recorded = dict(zip(names, state, strict=True))
        for key, value in seeds.items():
            assert math.isclose(recorded[key], value, abs_tol=1e-6), key

    # The reader parses it, and the seam numbers keep their meaning.
    trial = aggregate.build_trial(aggregate.load_events(result["events_path"]))
    assert trial.completed is True
    assert len(trial.reposition_durations_s) == 2
    assert trial.reposition_failed_stage_ids == ()
    assert len(trial.transition_gaps_s) == 1
    assert len(trial.stop_base_durations_s) == 2


def test_a_pose_that_is_not_reached_ends_the_trial_without_running_the_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stalled joint at the boundary: the s5 policy never runs, s4 is kept."""
    monkeypatch.setattr(reposition, "SETTLE_TIMEOUT_S", 0.05)

    def stall_after_the_first_stage(call_number: int, captured: dict[str, Any]) -> None:
        if call_number == 1:
            captured["robot"].arms.right_arm.driver.stall_joint = "joint_4"

    result = _run(tmp_path, after_stage=stall_after_the_first_stage)

    assert result["raised"] is None, result["raised"]
    assert result["exit_code"] == cli.EXIT_ABORTED
    _assert_log_is_closed(result)
    moves = _of(result, "stage_reposition")
    assert [move["outcome"] for move in moves] == ["reached", "not_reached"]
    assert "right_joint_4.pos" in moves[1]["reason"]
    s5_end = _of(result, "stage_end")[1]
    assert s5_end["stage_id"] == "s5"
    assert s5_end["terminator"] == "error"
    assert s5_end["frames"] == 0
    assert s5_end["elapsed_s"] == 0.0
    assert "start pose not reached" in s5_end["reason"]
    assert _of(result, "transition")[-1]["to_stage_id"] is None
    trial_end = result["events"][-1]
    # An abort, not an exception: the buffer's state is known, so s4's frames
    # are saved as a partial chain.
    assert trial_end["reason"] == "aborted"
    assert trial_end["completed"] is False
    assert trial_end["episode_saved"] is True
    assert result["base_at_encoder_flush"] == STOPPED_BASE
    for arm in (result["robot"].arms.left_arm, result["robot"].arms.right_arm):
        assert arm.driver.gripper_commands == []

    trial = aggregate.build_trial(aggregate.load_events(result["events_path"]))
    assert trial.reposition_failed_stage_ids == ("s5",)
    assert len(trial.reposition_durations_s) == 1
    timing = aggregate.timing_summary([trial])
    assert timing["stage_reposition_failed_n"] == 1
    assert timing["per_stage"]["s5"]["not_run_n"] == 1


def test_a_driver_error_at_the_first_stage_aborts_with_an_honest_event(
    tmp_path: Path,
) -> None:
    def refuse(robot: Any, captured: dict[str, Any]) -> None:
        robot.arms.left_arm.driver.raise_on_set = RuntimeError(
            "controller refused goal"
        )

    result = _run(tmp_path, instrument=refuse)

    assert result["raised"] is None, result["raised"]
    assert result["exit_code"] == cli.EXIT_ABORTED
    _assert_log_is_closed(result)
    (move,) = _of(result, "stage_reposition")
    assert move["stage_id"] == "s4"
    assert move["outcome"] == "error"
    assert "controller refused goal" in move["reason"]
    assert result["events"][-1]["reason"] == "aborted_empty"
    # The right arm was never commanded once the left one failed.
    assert result["robot"].arms.right_arm.driver.arm_position_commands == []


def test_a_ctrl_c_during_the_move_reaches_the_operator_after_the_base_stops(
    tmp_path: Path,
) -> None:
    def interrupt_the_boundary_move(call_number: int, captured: dict[str, Any]) -> None:
        if call_number == 1:
            robot = captured["robot"]
            robot._base_velocity = dict(DRIVING_BASE)
            robot.arms.right_arm.driver.raise_on_set = KeyboardInterrupt()

    result = _run(tmp_path, after_stage=interrupt_the_boundary_move)

    assert isinstance(result["raised"], KeyboardInterrupt), result["raised"]
    assert result["base_at_encoder_flush"] == STOPPED_BASE
    _assert_log_is_closed(result)
    moves = _of(result, "stage_reposition")
    assert [move["outcome"] for move in moves] == ["reached", "interrupted"]
    assert result["events"][-1]["reason"] == "exception"
    assert result["events"][-1]["episode_saved"] is False


def test_the_move_is_refused_when_the_boundary_did_not_stop_the_base(
    tmp_path: Path,
) -> None:
    """Both stop paths fail at the boundary: the arms must not move for seconds."""

    class RefusingBase:
        def set_cmd_vel(self, x_vel: float, theta_vel: float) -> bool:
            return False

    def break_both_stop_paths(call_number: int, captured: dict[str, Any]) -> None:
        if call_number == 1:
            robot = captured["robot"]
            robot.base = RefusingBase()
            robot._base_velocity = dict(DRIVING_BASE)

            def dead_camera() -> dict[str, Any]:
                raise RuntimeError("RealSense async_read timed out (injected)")

            robot.get_observation = dead_camera

    result = _run(tmp_path, after_stage=break_both_stop_paths)

    assert result["raised"] is None, result["raised"]
    assert result["exit_code"] == cli.EXIT_BASE_STOP_FAILED
    _assert_log_is_closed(result)
    moves = _of(result, "stage_reposition")
    assert [move["outcome"] for move in moves] == ["reached", "refused"]
    assert moves[1]["boundary_stop_base_path"] == "failed"
    # Only stage 0's move ever reached the drivers.
    robot = result["robot"]
    assert len(robot.arms.left_arm.driver.arm_position_commands) == 1
    assert len(robot.arms.right_arm.driver.arm_position_commands) == 1


def test_an_abort_key_skips_the_move_and_aborts_as_before(tmp_path: Path) -> None:
    result = _run(tmp_path, preset_events={"stop_recording": True})

    assert result["exit_code"] == cli.EXIT_ABORTED
    (move,) = _of(result, "stage_reposition")
    assert move["outcome"] == "skipped"
    assert move["reposition_s"] == 0.0
    assert move["expiry_wait_s"] is None
    assert result["robot"].arms.left_arm.driver.arm_position_commands == []
    assert _of(result, "stage_end")[0]["terminator"] == "stop_recording"
    assert result["events"][-1]["reason"] == "aborted_empty"
    trial = aggregate.build_trial(aggregate.load_events(result["events_path"]))
    assert trial.reposition_failed_stage_ids == (), "a skip is not a failure"


def test_without_a_start_pose_the_bimanual_mock_runs_as_before(tmp_path: Path) -> None:
    """No start_pose on any stage -> no stage_reposition line and no driver call."""
    source = (CONFIGS / "smoke_mock.yaml").read_text()
    path = tmp_path / "bimanual.yaml"
    path.write_text(
        source.replace(
            "  include_base_in_state: true\n",
            "  include_base_in_state: true\n  arm_layout: bimanual\n",
            1,
        )
    )
    result = _run(tmp_path, config_path=path)
    assert result["exit_code"] == cli.EXIT_OK
    assert "stage_reposition" not in _names(result)
    assert result["robot"].arms.left_arm.driver.arm_position_commands == []


# --------------------------------------------------------------------------
# The gripper goal in force during the move: the previous stage's last command.

# The beaker's width pushing the closed fingers open: the measured gripper
# position ends up wider than the policy's last command, which is what a grip on
# an object looks like (the difference is the squeeze). The mock gripper would
# otherwise sit exactly at its command, and "hold at the measured position" and
# "keep the last command" would be the same number.
OBJECT_WIDENS_GRIPPER_M = 0.02


def _open_grippers_by_the_object(call_number: int, captured: dict[str, Any]) -> None:
    if call_number == 1:
        for key in GRIPPERS:
            captured["robot"]._joint_positions[key.removesuffix(".pos")] += (
                OBJECT_WIDENS_GRIPPER_M
            )


def _hold_after_stage_one(result: dict[str, Any]) -> tuple[dict, dict]:
    """(stage 1's last policy action, the boundary hold right after it)."""
    frames = int(_of(result, "stage_end")[0]["frames"])
    sent = result["robot"].sent_actions
    return sent[frames - 1], sent[frames]


def test_the_boundary_hold_keeps_the_last_gripper_command_before_a_move(
    tmp_path: Path,
) -> None:
    """A gripper closed on an object keeps squeezing through the start pose move.

    stop_base's hold sets every .pos key to its measured position; for a gripper
    on a beaker that is the beaker's width, i.e. no squeeze. Before a stage with
    a start pose the hold must carry the previous stage's last COMMAND instead.
    """
    result = _run(tmp_path, after_stage=_open_grippers_by_the_object)
    assert result["exit_code"] == cli.EXIT_OK, result["raised"]

    last_policy, hold = _hold_after_stage_one(result)
    assert hold["x.vel"] == hold["theta.vel"] == 0.0
    for key in GRIPPERS:
        # float32 round trip through the episode buffer, nothing more.
        assert math.isclose(hold[key], last_policy[key], abs_tol=1e-6), key
        assert not math.isclose(
            hold[key], last_policy[key] + OBJECT_WIDENS_GRIPPER_M, abs_tol=1e-6
        ), f"{key}: the hold re-anchored the gripper to its measured position"

    moves = _of(result, "stage_reposition")
    assert [move["gripper_goal_source"] for move in moves] == [
        reposition.GRIPPER_GOAL_UNCHANGED,
        reposition.GRIPPER_GOAL_PREVIOUS_COMMAND,
    ]
    assert moves[0]["gripper_goal"] is None
    assert set(moves[1]["gripper_goal"]) == set(GRIPPERS)
    for key in GRIPPERS:
        assert math.isclose(
            moves[1]["gripper_goal"][key], last_policy[key], abs_tol=1e-6
        )
    # Still: the move itself sent the grippers nothing.
    robot = result["robot"]
    assert robot.arms.left_arm.driver.gripper_commands == []
    assert robot.arms.right_arm.driver.gripper_commands == []


def test_a_boundary_without_a_following_start_pose_holds_as_before(
    tmp_path: Path,
) -> None:
    """No start pose after the boundary -> the hold is the measured one, as always."""
    source = (CONFIGS / "smoke_mock.yaml").read_text()
    path = tmp_path / "bimanual.yaml"
    path.write_text(
        source.replace(
            "  include_base_in_state: true\n",
            "  include_base_in_state: true\n  arm_layout: bimanual\n",
            1,
        )
    )
    result = _run(tmp_path, config_path=path, after_stage=_open_grippers_by_the_object)
    assert result["exit_code"] == cli.EXIT_OK, result["raised"]
    last_policy, hold = _hold_after_stage_one(result)
    for key in GRIPPERS:
        assert math.isclose(
            hold[key], last_policy[key] + OBJECT_WIDENS_GRIPPER_M, abs_tol=1e-9
        ), key


def test_the_hold_refuses_a_held_position_for_the_base() -> None:
    robot = _bimanual_robot()
    robot.connect()
    with pytest.raises(KeyError):
        transitions.build_hold_action(robot, held_positions={"x.vel": 0.0})
    with pytest.raises(KeyError):
        transitions.build_hold_action(robot, held_positions={"nope.pos": 0.0})
    with pytest.raises(ValueError):
        transitions.build_hold_action(robot, held_positions={GRIPPERS[0]: float("nan")})
    action = transitions.build_hold_action(robot, held_positions={GRIPPERS[0]: 0.5})
    assert action[GRIPPERS[0]] == 0.5
    assert action[GRIPPERS[1]] == robot.get_observation()[GRIPPERS[1]]


# --------------------------------------------------------------------------
# The reader: the residual keeps its meaning.


def _event(name: str, iso: str, **fields: Any) -> dict[str, Any]:
    return {
        "event": name,
        "schema_version": 1,
        "run_id": "r1",
        "wall_clock_iso": iso,
        "stage_id": fields.pop("stage_id", None),
        "frame_idx": fields.pop("frame_idx", 0),
        "terminator": fields.pop("terminator", None),
        "reason": fields.pop("reason", ""),
        **fields,
    }


def _boundary_events(reposition_s: float | None) -> list[dict[str, Any]]:
    """Two stages; the boundary's raw stage_end -> stage_start window is 3.010 s."""
    events = [
        _event("trial_start", "2026-10-19T10:00:00.000+09:00", stage_ids=["s4", "s5"]),
        _event(
            "stage_start",
            "2026-10-19T10:00:00.000+09:00",
            stage_id="s4",
            terminator="timeout",
            stage_index=0,
        ),
        _event(
            "stage_end",
            "2026-10-19T10:00:10.000+09:00",
            stage_id="s4",
            terminator="timeout",
            frame_idx=300,
            stage_index=0,
            elapsed_s=10.0,
            frames=300,
            hertz=30.0,
        ),
        _event(
            "transition",
            "2026-10-19T10:00:10.002+09:00",
            stage_id="s4",
            reason="stop_base",
            frame_idx=300,
            from_stage_id="s4",
            to_stage_id="s5",
            stop_base_s=0.03,
            stop_base_path="primary",
        ),
    ]
    if reposition_s is not None:
        events.append(
            _event(
                "stage_reposition",
                "2026-10-19T10:00:13.005+09:00",
                stage_id="s5",
                frame_idx=300,
                outcome="reached",
                reposition_s=reposition_s,
            )
        )
    events += [
        _event(
            "stage_start",
            "2026-10-19T10:00:13.010+09:00",
            stage_id="s5",
            terminator="timeout",
            frame_idx=300,
            stage_index=1,
        ),
        _event(
            "stage_end",
            "2026-10-19T10:00:23.010+09:00",
            stage_id="s5",
            terminator="timeout",
            frame_idx=600,
            stage_index=1,
            elapsed_s=10.0,
            frames=300,
            hertz=30.0,
        ),
        _event(
            "transition",
            "2026-10-19T10:00:23.012+09:00",
            stage_id="s5",
            reason="stop_base",
            frame_idx=600,
            from_stage_id="s5",
            to_stage_id=None,
            stop_base_s=0.03,
            stop_base_path="primary",
        ),
        _event(
            "trial_end",
            "2026-10-19T10:00:23.020+09:00",
            reason="completed",
            completed=True,
        ),
    ]
    return events


def test_the_reposition_time_is_taken_out_of_the_residual() -> None:
    """3.010 s window, 3.000 s of it the arm move: the residual is 0.010 s.

    Without the subtraction a posed run would report a seam three hundred times
    more expensive than an unposed one for the same bookkeeping, and pooling the
    two would average a definition change into one mean.
    """
    posed = aggregate.build_trial(_boundary_events(3.0))
    assert len(posed.transition_gaps_s) == 1
    assert math.isclose(posed.transition_gaps_s[0], 0.010, abs_tol=1e-9)
    assert math.isclose(posed.boundary_costs_s[0], 0.040, abs_tol=1e-9)
    assert posed.reposition_durations_s == (3.0,)

    timing = aggregate.timing_summary([posed])
    assert timing["stage_reposition_s"]["n"] == 1
    assert timing["stage_reposition_s"]["mean"] == 3.0
    assert timing["stage_reposition_failed_n"] == 0
    report = aggregate.format_report(aggregate.summarize([posed], {}))
    assert "start pose arm move" in report

    # A log without the event is read exactly as before (raw window 3.010 s).
    unposed = aggregate.build_trial(_boundary_events(None))
    assert math.isclose(unposed.transition_gaps_s[0], 3.010, abs_tol=1e-9)
    assert unposed.reposition_durations_s == ()
    assert "start pose" not in aggregate.format_report(
        aggregate.summarize([unposed], {})
    )


def test_a_sub_millisecond_negative_residual_is_rounding_not_the_clock(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Millisecond stamps minus a perf_counter move can dip just below zero.

    The mock move takes ~0.1 ms and the stamps are truncated to ms, so the raw
    window can read 0.000 s while reposition_s is 0.0001 s. That is rounding, and
    must neither warn about the clock nor drag the residual mean negative. A
    shortfall beyond one resolution step still warns and is kept.
    """
    within = aggregate.build_trial(_boundary_events(3.0104))
    assert within.transition_gaps_s == (0.0,)
    assert "clock adjusted" not in capsys.readouterr().out

    beyond = aggregate.build_trial(_boundary_events(3.0125))
    assert math.isclose(beyond.transition_gaps_s[0], -0.0025, abs_tol=1e-9)
    assert "clock adjusted" in capsys.readouterr().out
