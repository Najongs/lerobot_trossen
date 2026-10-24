"""The leader_reset stage: the leader arm hands the robot an object, anywhere in a chain.

Drives the real cli.main against tests/configs/smoke_mock_leader_reset.yaml --
a reset before the first policy stage and another between the two -- so
lerobot's real record_loop runs both the reset (leader, no dataset) and the
policy stages, against the bimanual mock robot and the mock leader arm.

What is asserted, and why each matters on the robot:

* the leader is ALIGNED to the follower before each hand-over, so the
  follower's first copied command is not a jump (the mid-chain case);
* the leader's base echo NEVER reaches the robot: every reset command has the
  base at 0.0;
* nothing is recorded during a reset, and the reset is kept off the
  success-metric axis;
* the operator's last gripper command is held through the following start pose
  move, and through a base_replay that comes straight after the reset (the
  base stage would otherwise re-send the earlier policy's gripper command);
* base torque is released and restored only on a robot connected with it, and
  a failed restore ends the trial before the next policy;
* Esc during a reset aborts the trial before any policy runs; the ceiling does
  not (decided 2026-10-19).

NOTHING HERE IS EVIDENCE ABOUT THE REAL LEADER: the mock aligns instantly and
has no gravity compensation to drift.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
from lerobot.robots import Robot
from lerobot.robots.config import RobotConfig
from stage_runner import aggregate, cli, config, mock_policy, preflight, record_adapter
from stage_runner.config import StageConfig, StartPoseConfig, TerminatorConfig
from stage_runner.mock_robot import MockRobot, MockRobotConfig
from stage_runner.mock_teleop import MockTeleop, MockTeleopConfig
from test_base_assets import _write_pose_dataset
from test_base_stages import _path

CONFIGS = Path(__file__).parent / "configs"
RESET_CONFIG = CONFIGS / "smoke_mock_leader_reset.yaml"
GRIPPERS = ["left_left_carriage_joint.pos", "right_left_carriage_joint.pos"]
BASE_KEYS = ("x.vel", "theta.vel")


class _FakeBase:
    """robot.base as leader_reset and zero_base_directly reach it."""

    def __init__(self, restore_ok: bool = True) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.restore_ok = restore_ok

    def set_cmd_vel(self, linear: float, angular: float) -> bool:
        self.calls.append(("set_cmd_vel", (linear, angular)))
        return True

    def enable_motor_torque(self, enable: bool) -> tuple[bool, str]:
        self.calls.append(("enable_motor_torque", enable))
        if enable and not self.restore_ok:
            return False, "mock refusal"
        return True, ""


def _run(
    tmp_path: Path,
    *,
    config_path: Path = RESET_CONFIG,
    instrument: Any = None,
    on_get_action: Any = None,
) -> dict[str, Any]:
    """Drive cli.main once; capture the robot, the leader and the key flags."""
    captured: dict[str, Any] = {"robot": None, "teleop": None, "events": None}
    real_make_robot = record_adapter.make_robot
    real_make_teleop = record_adapter.make_teleop
    real_events = record_adapter.make_keyboard_events

    def make_robot(robot_config: RobotConfig) -> Robot:
        robot = real_make_robot(robot_config)
        captured["robot"] = robot
        if instrument is not None:
            instrument(robot, captured)
        return robot

    def make_teleop(teleop_config: Any) -> Any:
        teleop = real_make_teleop(teleop_config)
        captured["teleop"] = teleop
        if on_get_action is not None:
            teleop.on_get_action = lambda n, t: on_get_action(n, t, captured)
        return teleop

    def make_keyboard_events() -> tuple[Any | None, dict[str, bool]]:
        listener, keyboard_events = real_events()
        captured["events"] = keyboard_events
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
        for name, replacement in (
            ("make_robot", make_robot),
            ("make_teleop", make_teleop),
            ("make_keyboard_events", make_keyboard_events),
        ):
            stack.enter_context(mock.patch.object(record_adapter, name, replacement))
        try:
            exit_code = cli.main(argv)
        except BaseException as error:  # noqa: BLE001 -- the case under test
            raised = error
    events_path = next((tmp_path / "outputs").glob("*/events.jsonl"))
    return {
        **captured,
        "exit_code": exit_code,
        "raised": raised,
        "events": [
            json.loads(line) for line in events_path.read_text().splitlines() if line
        ],
        "keyboard": captured["events"],
        "events_path": events_path,
    }


def _of(result: dict[str, Any], name: str) -> list[dict[str, Any]]:
    return [event for event in result["events"] if event["event"] == name]


def _names(result: dict[str, Any]) -> list[str]:
    return [event["event"] for event in result["events"]]


def _assert_log_is_closed(result: dict[str, Any]) -> None:
    names = _names(result)
    assert names[-1] == "trial_end", names
    depth = 0
    for name in names:
        depth += 1 if name == "stage_start" else (-1 if name == "stage_end" else 0)
        assert depth in (0, 1), names
    assert depth == 0, names
    assert names.count("transition") == names.count("stage_start"), names


# --------------------------------------------------------------------------
# The chain: reset -> s4 (posed) -> reset -> s5.


def test_a_reset_anywhere_hands_over_aligned_and_records_nothing(
    tmp_path: Path,
) -> None:
    follower_at_handover: list[dict[str, float]] = []

    def instrument(robot: Any, captured: dict[str, Any]) -> None:
        # Snapshot the follower each time the leader is aligned to it.
        real_align = MockTeleop.align_to

        def align_to(self: MockTeleop, positions: dict, goal_time: float) -> None:
            follower_at_handover.append(
                {
                    key: value
                    for key, value in robot.get_observation().items()
                    if key in positions
                }
            )
            real_align(self, positions, goal_time)

        captured["align_patch"] = mock.patch.object(MockTeleop, "align_to", align_to)
        captured["align_patch"].start()

    try:
        result = _run(tmp_path, instrument=instrument)
    finally:
        mock.patch.stopall()

    assert result["raised"] is None, result["raised"]
    assert result["exit_code"] == cli.EXIT_OK
    assert _names(result) == [
        "trial_start",
        "stage_start",
        "stage_reset",
        "stage_end",
        "transition",
        "stage_reposition",
        "stage_start",
        "stage_end",
        "transition",
        "stage_start",
        "stage_reset",
        "stage_end",
        "transition",
        "stage_start",
        "stage_end",
        "transition",
        "trial_end",
    ]
    _assert_log_is_closed(result)

    starts = _of(result, "stage_start")
    assert [start["executor"] for start in starts] == [
        "leader_reset",
        "lerobot_policy",
        "leader_reset",
        "lerobot_policy",
    ]

    # ALIGNED: each hand-over drove the leader to exactly where the follower was.
    teleop = result["teleop"]
    assert len(teleop.align_calls) == 2
    for (target, goal_time), follower in zip(
        teleop.align_calls, follower_at_handover, strict=True
    ):
        assert target == follower
        assert goal_time >= 2.0
    # Mid-chain, the follower was where s4's policy left it -- not the staged
    # pose the leader started in.
    resets = _of(result, "stage_reset")
    assert [reset["outcome"] for reset in resets] == ["timeout", "timeout"]
    assert resets[1]["align_target"] == teleop.align_calls[1][0]

    # THE BASE ECHO NEVER REACHED THE ROBOT. The mock leader reports 0.25 for
    # both base keys; every command the resets sent has 0.0.
    robot = result["robot"]
    assert all(reset["sent_actions"] > 0 for reset in resets)
    for action in robot.sent_actions:
        for key in BASE_KEYS:
            assert action[key] != 0.25, action

    # NOTHING RECORDED: a reset's frames are 0 and its frame_idx does not move.
    ends = {end["stage_id"]: end for end in _of(result, "stage_end")}
    assert ends["reset_first"]["frames"] == 0
    assert ends["reset_mid"]["frames"] == 0
    assert ends["reset_first"]["frame_idx"] == 0
    assert ends["reset_mid"]["frame_idx"] == ends["s4"]["frame_idx"]
    trial_end = _of(result, "trial_end")[0]
    assert trial_end["buffered_frames"] == ends["s4"]["frames"] + ends["s5"]["frames"]

    # No base torque switch on the mock: nothing to release.
    assert all(reset["base_torque"] == "not_released" for reset in resets)

    # The reader keeps the resets off the success axis.
    trial = aggregate.build_trial(aggregate.load_events(result["events_path"]))
    assert aggregate.stage_order([trial]) == ["s4", "s5"]
    labels = {(trial.run_id, "s4"): "success", (trial.run_id, "s5"): "fail"}
    summary = aggregate.summarize([trial], labels)
    assert summary["scored_trial_count"] == 1
    assert summary["per_stage_success_rate"] == {"s4": 1.0, "s5": 0.0}
    assert summary["timing"]["leader_reset_s"]["n"] == 2
    assert summary["timing"]["leader_reset_s"]["terminators"] == {"timeout": 2}
    assert set(summary["timing"]["per_stage"]) == {"s4", "s5"}


def test_the_operators_grip_is_held_through_the_next_start_pose_move(
    tmp_path: Path,
) -> None:
    """The reset's LAST gripper command, not the measured width, is the hold."""

    def operator_closes_on_the_tube(n: int, teleop: Any, captured: dict) -> None:
        # Mid-reset, the operator squeezes both grippers past the tube.
        teleop.positions["left_left_carriage_joint.pos"] = 0.011
        teleop.positions["right_left_carriage_joint.pos"] = 0.012

    result = _run(tmp_path, on_get_action=operator_closes_on_the_tube)
    assert result["exit_code"] == cli.EXIT_OK

    first_reset = _of(result, "stage_reset")[0]
    assert first_reset["gripper_goals"] == {
        "left_left_carriage_joint.pos": 0.011,
        "right_left_carriage_joint.pos": 0.012,
    }
    move = _of(result, "stage_reposition")[0]
    assert move["stage_id"] == "s4"
    assert move["gripper_goal_source"] == "previous_command"
    assert move["gripper_goal"] == first_reset["gripper_goals"]
    # And the boundary hold that put it in force carried exactly that.
    hold = _of(result, "transition")[0]["stop_action"]
    for key in GRIPPERS:
        assert hold[key] == first_reset["gripper_goals"][key]


RESET_THEN_REPLAY = """\
version: 1
robot:
  type: stage_runner_mock_robot
  id: smoke
  arm_layout: bimanual
  simulate_base: true
teleop:
  type: stage_runner_mock_teleop
  id: leader
dataset:
  repo_id: local/eval_reset_then_base
  fps: 30
  video: false
defaults:
  executor: lerobot_policy
  manual_ceiling_s: 5.0
output:
  root: /tmp/unused
stages:
  - id: s1
    name: mock_stage_one
    policy_path: mock://hold
    instruction: "mock stage one"
    terminator: {type: timeout, timeout_s: 0.5}
  - id: reset_mid
    name: hand the tube over
    executor: leader_reset
    terminator: {type: timeout, timeout_s: 0.5}
  - id: s3
    name: drive with the tube
    executor: base_replay
    terminator: {type: timeout, timeout_s: 10}
    replay: {dataset: local/pose, root: POSE, episode: 0, until_row: 29}
"""


def test_the_operators_grip_is_held_through_a_base_replay_after_the_reset(
    tmp_path: Path,
) -> None:
    """policy (grippers at A) -> reset (operator sets B) -> base_replay holds B.

    A reset records no frames, so the episode buffer's last action row is
    still s1's command A. Holding A would re-open the grippers on the tube the
    operator just closed them on, for the whole drive.
    """
    pose = _write_pose_dataset(
        tmp_path / "pose",
        {0: [[0.0] * 14 + [s, t] for s, t in _path(30, 0.001, 0.3)]},
    )
    chain = tmp_path / "chain.yaml"
    chain.write_text(RESET_THEN_REPLAY.replace("POSE", str(pose)))

    def operator_closes_on_the_tube(n: int, teleop: Any, captured: dict) -> None:
        teleop.positions["left_left_carriage_joint.pos"] = 0.011
        teleop.positions["right_left_carriage_joint.pos"] = 0.012

    result = _run(
        tmp_path, config_path=chain, on_get_action=operator_closes_on_the_tube
    )
    assert result["exit_code"] == cli.EXIT_OK, result["raised"]
    (reset,) = _of(result, "stage_reset")
    (replay,) = _of(result, "stage_base_replay")
    assert replay["outcome"] == "settled"
    assert reset["gripper_goals"] == {
        "left_left_carriage_joint.pos": 0.011,
        "right_left_carriage_joint.pos": 0.012,
    }
    assert replay["gripper_hold_source"] == "reset"
    for key in GRIPPERS:
        assert replay["held_arm_positions"][key] == reset["gripper_goals"][key]

    # Every frame the replay recorded carries B, not s1's A.
    starts = {e["stage_id"]: e["frame_idx"] for e in _of(result, "stage_start")}
    ends = {e["stage_id"]: e["frame_idx"] for e in _of(result, "stage_end")}
    assert ends["s3"] > starts["s3"] > 0
    robot = result["robot"]
    names = list(robot.action_features)
    import pyarrow.parquet as pq

    (data,) = sorted((tmp_path / "dataset" / "data").rglob("*.parquet"))
    actions = pq.read_table(data, columns=["action"]).column("action").to_pylist()
    for row in actions[starts["s3"] : ends["s3"]]:
        values = dict(zip(names, row, strict=True))
        for key in GRIPPERS:
            assert values[key] == pytest.approx(reset["gripper_goals"][key], abs=1e-6)
    s1_last = dict(zip(names, actions[ends["s1"] - 1], strict=True))
    assert all(
        s1_last[key] != pytest.approx(reset["gripper_goals"][key], abs=1e-3)
        for key in GRIPPERS
    ), "the test needs s1's goal A to differ from the operator's B"


RESET_POLICY_THEN_REPLAY = RESET_THEN_REPLAY.replace(
    """  - id: s3
    name: drive with the tube""",
    """  - id: s3
    name: release the tube
    policy_path: mock://hold
    instruction: "mock release"
    terminator: {type: timeout, timeout_s: 0.5}
  - id: s4
    name: drive away""",
)


def test_a_reset_grip_expires_once_a_policy_stage_records_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """reset (operator sets B) -> policy (opens to C) -> base_replay holds C, not B.

    The runner clears StageContext.reset_gripper_goals after any stage that
    recorded frames: from then on the buffer's last row is newer than the
    operator's grip. Without the clear the base stage would close the grippers
    back onto B -- here, onto a tube the policy just let go of -- for the whole
    drive, because base_loop consults the reset goals before the buffer.
    """
    pose = _write_pose_dataset(
        tmp_path / "pose",
        {0: [[0.0] * 14 + [s, t] for s, t in _path(30, 0.001, 0.3)]},
    )
    chain = tmp_path / "chain.yaml"
    text = RESET_POLICY_THEN_REPLAY.replace("POSE", str(pose))
    assert "id: s4" in text, "the chain rewrite did not apply"
    chain.write_text(text)
    opened = 0.031

    real_init = mock_policy.MockPolicy.__init__
    real_select = mock_policy.MockPolicy.select_action

    def init(self: Any, *args: Any, **kwargs: Any) -> None:
        real_init(self, *args, **kwargs)
        self._test_grippers = [
            index
            for index, name in enumerate(kwargs["action_names"])
            if name in GRIPPERS
        ]

    def select_action(self: Any, batch: dict[str, Any]) -> Any:
        # Every mock policy stage commands the grippers open at C.
        action = real_select(self, batch)
        for index in self._test_grippers:
            action[0, index] = opened
        return action

    monkeypatch.setattr(mock_policy.MockPolicy, "__init__", init)
    monkeypatch.setattr(mock_policy.MockPolicy, "select_action", select_action)

    def operator_closes_on_the_tube(n: int, teleop: Any, captured: dict) -> None:
        teleop.positions["left_left_carriage_joint.pos"] = 0.011
        teleop.positions["right_left_carriage_joint.pos"] = 0.012

    result = _run(
        tmp_path, config_path=chain, on_get_action=operator_closes_on_the_tube
    )
    assert result["exit_code"] == cli.EXIT_OK, result["raised"]
    (reset,) = _of(result, "stage_reset")
    (replay,) = _of(result, "stage_base_replay")
    assert replay["outcome"] == "settled"
    assert all(
        reset["gripper_goals"][key] != pytest.approx(opened, abs=1e-3)
        for key in GRIPPERS
    ), "the test needs the operator's B to differ from the policy's C"
    assert replay["gripper_hold_source"] == "recorded"
    for key in GRIPPERS:
        assert replay["held_arm_positions"][key] == pytest.approx(opened, abs=1e-6)


def test_the_right_arrow_ends_a_reset_and_the_chain_goes_on(tmp_path: Path) -> None:
    def press_right_arrow(n: int, teleop: Any, captured: dict) -> None:
        if n == 3:
            captured["events"]["exit_early"] = True

    result = _run(tmp_path, on_get_action=press_right_arrow)
    assert result["exit_code"] == cli.EXIT_OK
    ends = {end["stage_id"]: end for end in _of(result, "stage_end")}
    assert ends["reset_first"]["terminator"] == "manual"
    assert ends["reset_first"]["elapsed_s"] < 0.4
    assert ends["reset_mid"]["terminator"] == "timeout"
    assert {"s4", "s5"} <= set(ends)


def test_esc_during_a_reset_aborts_before_any_policy_runs(tmp_path: Path) -> None:
    def press_esc(n: int, teleop: Any, captured: dict) -> None:
        if n == 3:
            captured["events"]["stop_recording"] = True
            captured["events"]["exit_early"] = True

    result = _run(tmp_path, on_get_action=press_esc)
    assert result["raised"] is None, result["raised"]
    assert result["exit_code"] == cli.EXIT_ABORTED
    _assert_log_is_closed(result)
    assert [start["stage_id"] for start in _of(result, "stage_start")] == [
        "reset_first"
    ]
    assert _of(result, "stage_end")[0]["terminator"] == "stop_recording"
    assert _of(result, "trial_end")[0]["reason"] == "aborted_empty"


def test_a_leader_that_cannot_align_ends_the_trial_without_a_hand_over(
    tmp_path: Path,
) -> None:
    def fail_align(n: int, teleop: Any, captured: dict) -> None:
        teleop.raise_on_align = RuntimeError("leader controller refused the goal")

    result = _run(tmp_path, on_get_action=fail_align)
    assert result["exit_code"] == cli.EXIT_ABORTED
    _assert_log_is_closed(result)
    reset = _of(result, "stage_reset")[0]
    assert reset["outcome"] == "error"
    assert "refused the goal" in reset["reason"]
    assert _of(result, "stage_end")[0]["terminator"] == "error"
    assert [start["stage_id"] for start in _of(result, "stage_start")] == [
        "reset_first"
    ]


# --------------------------------------------------------------------------
# Base torque: released for the hand-over, restored before the next stage.


def _with_base(restore_ok: bool = True) -> Any:
    def instrument(robot: Any, captured: dict[str, Any]) -> None:
        robot.base = _FakeBase(restore_ok=restore_ok)
        robot.config.enable_base_motor_torque = True
        captured["base"] = robot.base

    return instrument


def test_base_torque_is_released_and_restored_with_a_zero_command_first(
    tmp_path: Path,
) -> None:
    captured_base: dict[str, Any] = {}

    def instrument(robot: Any, captured: dict[str, Any]) -> None:
        _with_base()(robot, captured)
        captured_base["base"] = robot.base

    result = _run(tmp_path, instrument=instrument)
    assert result["exit_code"] == cli.EXIT_OK
    calls = captured_base["base"].calls
    torque = [value for name, value in calls if name == "enable_motor_torque"]
    assert torque == [False, True, False, True]
    # Every torque switch is preceded by a zero velocity command.
    for index, (name, _) in enumerate(calls):
        if name == "enable_motor_torque":
            assert calls[index - 1] == ("set_cmd_vel", (0.0, 0.0))
    assert [reset["base_torque"] for reset in _of(result, "stage_reset")] == [
        "released",
        "released",
    ]


def test_a_base_whose_torque_cannot_be_restored_ends_the_trial(
    tmp_path: Path,
) -> None:
    result = _run(tmp_path, instrument=_with_base(restore_ok=False))
    assert result["exit_code"] == cli.EXIT_ABORTED
    _assert_log_is_closed(result)
    reset = _of(result, "stage_reset")[0]
    assert reset["base_torque"] == "restore_failed"
    assert "mock refusal" in reset["base_torque_error"]
    assert [start["stage_id"] for start in _of(result, "stage_start")] == [
        "reset_first"
    ]


def test_a_robot_connected_without_base_torque_is_left_alone(tmp_path: Path) -> None:
    def instrument(robot: Any, captured: dict[str, Any]) -> None:
        robot.base = _FakeBase()
        robot.config.enable_base_motor_torque = False
        captured["base"] = robot.base

    holder: dict[str, Any] = {}

    def keep(robot: Any, captured: dict[str, Any]) -> None:
        instrument(robot, captured)
        holder["base"] = robot.base

    result = _run(tmp_path, instrument=keep)
    assert result["exit_code"] == cli.EXIT_OK
    assert not any(name == "enable_motor_torque" for name, _ in holder["base"].calls)


# --------------------------------------------------------------------------
# Config and preflight.


def test_an_old_yaml_has_no_teleop_and_snapshots_without_it() -> None:
    import draccus

    smoke = config.parse_config(["--config_path", str(CONFIGS / "smoke_mock.yaml")])
    assert smoke.teleop is None
    text = config.resolved_config_text(smoke)
    assert "teleop" not in text
    full = draccus.dump(smoke)
    assert text == "".join(
        line
        for line in full.splitlines(keepends=True)
        if "start_pose: null" not in line and line != "teleop: null\n"
    )


def _reset_config(**stage_overrides: Any) -> Any:
    parsed = config.parse_config(["--config_path", str(RESET_CONFIG)])
    for key, value in stage_overrides.items():
        setattr(parsed.stages[0], key, value)
    return parsed


@pytest.mark.parametrize(
    ("overrides", "needle"),
    [
        ({"policy_path": "mock://sine"}, "runs no policy"),
        (
            {"start_pose": StartPoseConfig(positions={"left_joint_0.pos": 0.1})},
            "posed by hand",
        ),
    ],
)
def test_preflight_rejects_a_reset_stage_that_would_ignore_a_field(
    overrides: dict[str, Any], needle: str
) -> None:
    with pytest.raises(preflight.PreflightError, match=needle):
        preflight.check_stage_definitions(_reset_config(**overrides))


def test_preflight_rejects_a_reset_without_a_leader_arm() -> None:
    parsed = _reset_config()
    parsed.teleop = None
    with pytest.raises(preflight.PreflightError, match="needs the leader arm"):
        preflight.check_stage_definitions(parsed)


def test_preflight_rejects_a_chain_of_resets_only() -> None:
    parsed = _reset_config()
    parsed.stages = [
        StageConfig(
            id="only_reset",
            executor="leader_reset",
            terminator=TerminatorConfig(type="timeout", timeout_s=1.0),
        )
    ]
    with pytest.raises(preflight.PreflightError, match="at least one"):
        preflight.check_stage_definitions(parsed)


def test_preflight_matches_the_leader_keys_to_the_robot() -> None:
    parsed = _reset_config()
    robot = MockRobot(MockRobotConfig(id="unit", arm_layout="bimanual"))
    teleop = MockTeleop(MockTeleopConfig(id="unit"))
    preflight.check_leader_reset_stages(parsed, robot, teleop)

    flat = MockRobot(MockRobotConfig(id="unit"))
    with pytest.raises(preflight.PreflightError, match="not action features"):
        preflight.check_leader_reset_stages(parsed, flat, teleop)

    class NoAlign:
        action_features = teleop.action_features

    with pytest.raises(preflight.PreflightError, match="no align_to"):
        preflight.check_leader_reset_stages(parsed, robot, NoAlign())


def test_a_yaml_without_a_reset_never_builds_the_leader(tmp_path: Path) -> None:
    built: list[Any] = []
    with mock.patch.object(
        record_adapter, "make_teleop", lambda cfg: built.append(cfg)
    ):
        exit_code = cli.main(
            [
                "--config_path",
                str(CONFIGS / "smoke_mock.yaml"),
                f"--dataset.root={tmp_path / 'dataset'}",
                f"--output.root={tmp_path / 'outputs'}",
            ]
        )
    assert exit_code == cli.EXIT_OK
    assert built == []


# --------------------------------------------------------------------------
# The real leader arm's align_to, against a recording driver.


class _RecordingDriver:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any, Any]] = []

    def set_all_modes(self, mode: Any) -> None:
        self.calls.append(("set_all_modes", str(mode), None))

    def set_all_positions(
        self, goal: list[float], goal_time: float, blocking: bool
    ) -> None:
        self.calls.append(("set_all_positions", list(goal), (goal_time, blocking)))

    def set_all_external_efforts(
        self, efforts: list[float], goal_time: float, blocking: bool
    ) -> None:
        self.calls.append(
            ("set_all_external_efforts", list(efforts), (goal_time, blocking))
        )


def test_the_real_leader_aligns_both_arms_by_name_then_frees_them() -> None:
    from lerobot_teleoperator_trossen import (
        MobileAILeaderTeleop,
        MobileAILeaderTeleopConfig,
    )

    leader = MobileAILeaderTeleop(
        MobileAILeaderTeleopConfig(
            id="unit", left_arm_ip_address="0.0.0.0", right_arm_ip_address="0.0.0.1"
        )
    )
    leader.left_arm.driver = left = _RecordingDriver()
    leader.right_arm.driver = right = _RecordingDriver()
    names = leader.left_arm.config.joint_names
    positions = {
        f"{side}{name}.pos": (0.1 if side == "left_" else 0.2) + index / 100
        for side in ("left_", "right_")
        for index, name in enumerate(names)
    }

    leader.align_to(positions, 2.5)

    for driver, side, blocking in ((left, "left_", False), (right, "right_", True)):
        kinds = [call[0] for call in driver.calls]
        assert kinds == [
            "set_all_modes",
            "set_all_positions",
            "set_all_modes",
            "set_all_external_efforts",
        ]
        assert "position" in driver.calls[0][1]
        assert driver.calls[1][1] == [positions[f"{side}{name}.pos"] for name in names]
        assert driver.calls[1][2] == (2.5, blocking)
        assert "external_effort" in driver.calls[2][1]
        assert driver.calls[3][1] == [0.0] * len(names)
