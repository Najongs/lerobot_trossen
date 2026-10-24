"""The base_replay executor on the simulated base: outcomes, guards and chain behaviour.

Two harnesses:

* DIRECT -- base_stages.replay called on a StageContext built here (a real
  LeRobotDataset, the bimanual mock with the simulated base, an EventLog), for
  the guards: a fault is injected at a chosen tick through a hook on the mock
  slate, and the stage's own event and CSV are read back;
* CLI -- the real cli.main on a chain YAML, for what a stage means to the
  chain: frames continue across the boundary, the chain goes on after a
  converged replay and stops after a failed one, and a replay runs correctly
  at any position and more than once in one chain, each anchored at its own
  start.

What is asserted, and why each matters on the robot:

* a replay reaches the demo's final pose and settles, holding the arms where
  they were, and leaves a per-tick CSV and one event with every field;
* every guard ends the stage base_failed with the value that tripped it, and
  the next policy stage never starts: lost odometry, refused commands, an
  e-stop, a base that does not follow or follows the wrong way;
* a short odometry gap does NOT abort: the tracker commands zero on it;
* every exit zeroes the base directly and puts the real driver back as
  robot.base; an exception reaches run_trial unchanged.

NOTHING HERE IS EVIDENCE ABOUT THE REAL BASE: the mock follows its command with
no slip, no deadband and (by default) no lag.
"""

from __future__ import annotations

import contextlib
import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import numpy as np
import pytest
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from stage_runner import base_stages, cli, record_adapter
from stage_runner.base_assets import ReplayTargets
from stage_runner.base_loop import BaseProbe
from stage_runner.config import (
    BaseReplayConfig,
    DefaultsConfig,
    StageConfig,
    TerminatorConfig,
)
from stage_runner.context import StageContext
from stage_runner.events import EventLog
from stage_runner.mock_robot import MockRobot, MockRobotConfig, MockSlate
from test_base_assets import _write_pose_dataset

FPS = 30
GRIPPERS = ("left_left_carriage_joint.pos", "right_left_carriage_joint.pos")

# Every field the spec lists for stage_base_replay (plus outcome).
REPLAY_EVENT_FIELDS = {
    "outcome",
    "reason",
    "dataset",
    "root",
    "episode",
    "from_row",
    "until_row",
    "rows",
    "row_period_s",
    "anchor_odom",
    "nominal_dt",
    "final_target_s_m",
    "final_target_theta_deg",
    "final_s_m",
    "final_theta_deg",
    "final_error_m",
    "final_error_deg",
    "motion_s",
    "hold_s",
    "ticks",
    "odom_bad_ticks",
    "max_odom_bad_run",
    "update_failed_ticks",
    "cmd_refused_ticks",
    "max_cmd_refused_run",
    "max_tracking_error_m",
    "max_tracking_error_deg",
    "base_state_last",
    "tick_mean_s",
    "tick_p95_s",
    "held_arm_positions",
    "gripper_hold_source",
    "csv",
}


def _path(rows: int, step_m: float, step_deg: float) -> list[tuple[float, float]]:
    """A demo base path: cumulative forward distance and heading per row."""
    return [
        (step_m * (row + 1), math.radians(step_deg) * (row + 1)) for row in range(rows)
    ]


def _targets(path: list[tuple[float, float]]) -> ReplayTargets:
    s_m = np.array([s for s, _ in path], dtype=np.float64)
    theta = np.array([t for _, t in path], dtype=np.float64)
    return ReplayTargets(
        s_m=s_m,
        theta_rad=theta,
        source={"dataset": "local/pose", "root": "/nowhere", "rows": len(path)},
    )


# --------------------------------------------------------------------------
# DIRECT harness.


def _direct(
    tmp_path: Path,
    *,
    path: list[tuple[float, float]] | None = None,
    timeout_s: float = 20.0,
    **robot_overrides: Any,
) -> tuple[StageContext, StageConfig]:
    robot = MockRobot(
        MockRobotConfig(arm_layout="bimanual", simulate_base=True, **robot_overrides)
    )
    processors = record_adapter.make_processors()
    dataset = LeRobotDataset.create(
        "local/base_stages",
        FPS,
        root=tmp_path / "dataset",
        robot_type=robot.name,
        features=record_adapter.build_dataset_features(robot, processors, False),
        use_videos=False,
    )
    robot.connect()
    stage = StageConfig(
        id="s1",
        name="drive to the rack",
        executor="base_replay",
        terminator=TerminatorConfig(type="timeout", timeout_s=timeout_s),
        replay=BaseReplayConfig(dataset="local/pose", until_row=1),
    )
    context = StageContext(
        config=SimpleNamespace(
            dataset=SimpleNamespace(fps=FPS), defaults=DefaultsConfig()
        ),
        robot=robot,
        dataset=dataset,
        processors=processors,
        events={
            "exit_early": False,
            "stop_recording": False,
            "rerecord_episode": False,
        },
        bundles={},
        log=EventLog(tmp_path / "events.jsonl", "run"),
        run_directory=tmp_path,
        base_assets={"s1": _targets(path or _path(30, 0.01, 0.5))},
    )
    return context, stage


def _on_update(slate: MockSlate, at: int, action: Any) -> None:
    """Run ``action()`` once, just before the ``at``-th update_state call from now."""
    real = slate.update_state
    target = slate.update_calls + at
    fired = []

    def update_state() -> bool:
        if not fired and slate.update_calls + 1 >= target:
            fired.append(True)
            action()
        return real()

    slate.update_state = update_state


def _events(tmp_path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
        if line
    ]


def _replay_event(tmp_path: Path) -> dict[str, Any]:
    (event,) = [e for e in _events(tmp_path) if e["event"] == "stage_base_replay"]
    return event


def _csv(tmp_path: Path, stage_id: str = "s1") -> list[dict[str, str]]:
    with (tmp_path / "base" / f"{stage_id}_replay.csv").open() as handle:
        return list(csv.DictReader(handle))


def _run_direct(context: StageContext, stage: StageConfig) -> Any:
    slate = context.robot.base
    assert isinstance(slate, MockSlate)
    result = base_stages.replay(context, stage)
    assert context.robot.base is slate, "robot.base는 원래 드라이버로 복원"
    assert slate.commands[-1][:2] == (0.0, 0.0), "끝나면 베이스를 0으로"
    return result


# --------------------------------------------------------------------------
# Settling.


def test_a_replay_follows_the_demo_path_and_settles(tmp_path: Path) -> None:
    context, stage = _direct(tmp_path)
    robot = context.robot
    first = robot.get_observation()
    start = robot.base.true_pose()
    result = _run_direct(context, stage)

    assert result.terminated_by == "converged", result.reason
    assert result.frames == context.buffered_frame_count() > 0
    assert result.elapsed_s > 0.0
    end = robot.base.true_pose()
    turned = math.degrees(end[2] - start[2])
    driven = math.hypot(end[0] - start[0], end[1] - start[1])
    assert turned == pytest.approx(15.0, abs=1.0)
    assert driven == pytest.approx(0.30, abs=0.01)

    event = _replay_event(tmp_path)
    assert REPLAY_EVENT_FIELDS <= set(event), REPLAY_EVENT_FIELDS - set(event)
    assert event["outcome"] == "settled"
    assert event["terminator"] == "converged"
    assert event["final_error_m"] < 0.01 and event["final_error_deg"] < 1.0
    assert event["gripper_hold_source"] == "measured"
    assert event["csv"] == "base/s1_replay.csv"
    assert event["frame_idx"] == context.buffered_frame_count()

    # Every recorded arm joint is where the stage found it.
    names = list(context.dataset.features["action"]["names"])
    for row in context.dataset.episode_buffer["action"]:
        values = dict(zip(names, (float(v) for v in row), strict=True))
        for key in robot.action_features:
            if key.endswith(".pos"):
                assert values[key] == pytest.approx(first[key]), key
    assert set(context.dataset.episode_buffer["task"]) == {"drive to the rack"}

    rows = _csv(tmp_path)
    assert len(rows) == result.frames, "CSV 한 줄 = 기록된 프레임 하나"
    assert rows[0]["phase"] == "preroll"
    assert {row["phase"] for row in rows} == {"preroll", "motion", "hold"}


def test_targets_are_followed_in_time_not_per_tick(tmp_path: Path) -> None:
    # 30 rows at 1/21 s is 1.38 s of motion whatever the loop rate is.
    context, stage = _direct(tmp_path)
    _run_direct(context, stage)
    event = _replay_event(tmp_path)
    assert event["motion_s"] == pytest.approx(29 / 21, abs=0.1)


def test_a_replay_that_ends_far_from_the_target_is_missed_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(base_stages, "REPLAY_MAX_FINAL_ERROR_M", 1e-9)
    context, stage = _direct(tmp_path)
    result = _run_direct(context, stage)
    assert result.terminated_by == "base_failed"
    assert _replay_event(tmp_path)["outcome"] == "missed_target"


def test_a_base_that_never_reads_still_is_not_settled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(base_stages, "REPLAY_SETTLE_MAX_S", 0.3)
    context, stage = _direct(tmp_path, path=_path(5, 0.0, 0.0))
    context.robot.base.velocity_override = [0.02, 0.0]
    result = _run_direct(context, stage)
    # A timeout, not a failure: the align stage after it measures where it is.
    assert result.terminated_by == "timeout", result.reason
    assert _replay_event(tmp_path)["outcome"] == "not_settled"


# --------------------------------------------------------------------------
# Odometry.


def test_no_odometry_at_all_ends_before_any_motion(tmp_path: Path) -> None:
    context, stage = _direct(tmp_path)
    context.robot.base.pose_override = [float("nan"), 0.0, 0.0]
    result = _run_direct(context, stage)
    assert result.terminated_by == "base_failed"
    event = _replay_event(tmp_path)
    assert event["outcome"] == "no_odometry"
    assert event["anchor_odom"] is None
    assert all(cmd[:2] == (0.0, 0.0) for cmd in context.robot.base.commands)


def test_a_short_odometry_gap_commands_zero_and_does_not_abort(
    tmp_path: Path,
) -> None:
    context, stage = _direct(tmp_path)
    slate = context.robot.base
    # 0.5 s pre-roll is ~15 ticks; fail three refreshes in the middle of motion.
    _on_update(slate, 25, lambda: setattr(slate, "fail_updates", 3))
    result = _run_direct(context, stage)
    assert result.terminated_by == "converged", result.reason
    stale = [row for row in _csv(tmp_path) if row["update_ok"] == "False"]
    assert len(stale) == 3
    for row in stale:
        assert row["odom_ok"] == "False"
        assert float(row["cmd_x_vel"]) == 0.0 and float(row["cmd_theta_vel"]) == 0.0
    event = _replay_event(tmp_path)
    assert event["odom_bad_ticks"] == 3 and event["max_odom_bad_run"] == 3
    assert event["update_failed_ticks"] == 3


def test_a_long_odometry_gap_is_no_odometry(tmp_path: Path) -> None:
    context, stage = _direct(tmp_path)
    slate = context.robot.base
    _on_update(slate, 25, lambda: setattr(slate, "fail_updates", 20))
    result = _run_direct(context, stage)
    assert result.terminated_by == "base_failed"
    event = _replay_event(tmp_path)
    assert event["outcome"] == "no_odometry"
    assert event["max_odom_bad_run"] == base_stages.ODOM_BAD_LIMIT + 1


# --------------------------------------------------------------------------
# Base faults.


def test_refused_commands_are_base_refused(tmp_path: Path) -> None:
    context, stage = _direct(tmp_path)
    slate = context.robot.base
    _on_update(slate, 20, lambda: setattr(slate, "refuse_cmds", 20))
    result = _run_direct(context, stage)
    assert result.terminated_by == "base_failed"
    event = _replay_event(tmp_path)
    assert event["outcome"] == "base_refused"
    # The guard trips on the 11th refusal in a row; the exit's direct zero is
    # refused as well and counted (the runner's stop_base follows).
    assert f"{base_stages.CMD_REFUSED_LIMIT + 1} consecutive" in event["reason"]
    assert event["max_cmd_refused_run"] >= base_stages.CMD_REFUSED_LIMIT + 1
    assert event["base_zero_error"], "a refused exit zero is reported"


def test_an_estop_mid_replay_is_base_estop(tmp_path: Path) -> None:
    context, stage = _direct(tmp_path)
    robot = context.robot
    _on_update(
        robot.base,
        22,
        lambda: setattr(robot, "_last_base_state", SimpleNamespace(name="SYS_ESTOP")),
    )
    result = _run_direct(context, stage)
    assert result.terminated_by == "base_failed"
    event = _replay_event(tmp_path)
    assert event["outcome"] == "base_estop"
    assert event["base_state_last"] == "SYS_ESTOP"


@pytest.mark.parametrize(
    "hook",
    [
        pytest.param({"ignore_cmds": True}, id="stalled"),
        pytest.param({"command_gain": -1.0}, id="wrong-way"),
    ],
)
def test_a_base_that_does_not_follow_is_tracking_lost(
    tmp_path: Path, hook: dict[str, Any]
) -> None:
    context, stage = _direct(tmp_path, path=_path(60, 0.02, 1.0))
    for name, value in hook.items():
        setattr(context.robot.base, name, value)
    result = _run_direct(context, stage)
    assert result.terminated_by == "base_failed"
    event = _replay_event(tmp_path)
    assert event["outcome"] == "tracking_lost", event["reason"]
    assert (
        event["max_tracking_error_m"] > base_stages.TRACK_ERROR_M
        or event["max_tracking_error_deg"] > base_stages.TRACK_ERROR_DEG
    )


def test_the_stage_ceiling_ends_a_replay_that_runs_too_long(tmp_path: Path) -> None:
    context, stage = _direct(tmp_path, timeout_s=0.8)
    result = _run_direct(context, stage)
    assert result.terminated_by == "base_failed"
    assert _replay_event(tmp_path)["outcome"] == "stage_ceiling"


# --------------------------------------------------------------------------
# Keys.


@pytest.mark.parametrize(
    ("flags", "terminated_by"),
    [
        ({"stop_recording": True, "exit_early": True}, "stop_recording"),
        ({"rerecord_episode": True, "exit_early": True}, "rerecord_requested"),
        ({"exit_early": True}, "manual"),
    ],
)
def test_keys_during_the_replay(
    tmp_path: Path, flags: dict[str, bool], terminated_by: str
) -> None:
    context, stage = _direct(tmp_path)
    _on_update(context.robot.base, 25, lambda: context.events.update(flags))
    result = _run_direct(context, stage)
    assert result.terminated_by == terminated_by
    event = _replay_event(tmp_path)
    assert event["outcome"] == ("manual" if terminated_by == "manual" else "aborted")
    assert context.events["exit_early"] is False, "exit_early는 소비된다"


# --------------------------------------------------------------------------
# Exception path.


def test_an_exception_mid_replay_zeroes_the_base_and_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context, stage = _direct(tmp_path)
    robot = context.robot
    slate = robot.base
    real_send = robot.send_action
    calls = []
    stopped: list[BaseException] = []
    monkeypatch.setattr(
        base_stages,
        "stop_image_writer_after",
        lambda error, dataset: stopped.append(error),
    )

    def send_action(action: dict[str, float]) -> dict[str, float]:
        calls.append(True)
        if len(calls) == 25:
            raise OSError("arm bus gone")
        return real_send(action)

    robot.send_action = send_action
    with pytest.raises(OSError, match="arm bus gone") as raised:
        base_stages.replay(context, stage)
    assert robot.base is slate
    assert slate.commands[-1][:2] == (0.0, 0.0)
    assert stopped == [raised.value]
    event = _replay_event(tmp_path)
    assert event["outcome"] == "interrupted"
    assert event["terminator"] == "error"
    assert context.buffered_frame_count() == 24, "the failed tick records nothing"


# --------------------------------------------------------------------------
# CLI harness: chains.

CHAIN = """\
version: 1
robot:
  type: stage_runner_mock_robot
  id: smoke
  arm_layout: bimanual
  simulate_base: true
dataset:
  repo_id: local/eval_replay_chain
  fps: 30
  video: false
stages:
STAGES
"""

FIRST_REPLAY = """\
  - id: s1
    name: drive to the rack
    executor: base_replay
    terminator: {type: timeout, timeout_s: 10}
    replay: {dataset: local/pose, root: POSE, episode: 0, until_row: 29}
"""
FIRST_POLICY = """\
  - id: s2
    name: mock_stage_two
    policy_path: mock://hold
    instruction: pick up the tube
    terminator: {type: timeout, timeout_s: 0.3}
"""
SECOND_REPLAY = """\
  - id: s3
    name: drive to the shelf
    executor: base_replay
    terminator: {type: timeout, timeout_s: 10}
    replay: {dataset: local/pose, root: POSE, episode: 1, from_row: 10, until_row: 29}
"""
SECOND_POLICY = """\
  - id: s4
    name: mock_stage_four
    policy_path: mock://hold
    instruction: place the tube
    terminator: {type: timeout, timeout_s: 0.3}
"""
LEADING_POLICY = """\
  - id: s0
    name: mock_stage_zero
    policy_path: mock://hold
    instruction: hold the tube
    terminator: {type: timeout, timeout_s: 0.3}
"""


def _pose_episode(step_m: float, step_deg: float) -> list[list[float]]:
    return [[0.0] * 14 + [s, theta] for s, theta in _path(30, step_m, step_deg)]


def _chain(tmp_path: Path, *stages: str) -> Path:
    pose = _write_pose_dataset(
        tmp_path / "pose",
        {0: _pose_episode(0.01, 0.5), 1: _pose_episode(-0.005, -1.0)},
    )
    text = CHAIN.replace("STAGES\n", "".join(stages)).replace("POSE", str(pose))
    path = tmp_path / "chain.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def _cli(tmp_path: Path, config_path: Path, instrument: Any = None) -> dict[str, Any]:
    captured: dict[str, Any] = {}
    real_make_robot = record_adapter.make_robot

    def make_robot(robot_config: Any) -> Any:
        robot = real_make_robot(robot_config)
        captured["robot"] = robot
        captured["slate"] = robot.base
        if instrument is not None:
            instrument(robot)
        return robot

    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.object(record_adapter, "make_robot", make_robot))
        exit_code = cli.main(
            [
                "--config_path",
                str(config_path),
                f"--dataset.root={tmp_path / 'dataset'}",
                f"--output.root={tmp_path / 'outputs'}",
            ]
        )
    run_directory = next((tmp_path / "outputs").glob("*/"))
    events = [
        json.loads(line)
        for line in (run_directory / "events.jsonl").read_text().splitlines()
        if line
    ]
    return {
        **captured,
        "exit_code": exit_code,
        "events": events,
        "run_directory": run_directory,
    }


def _of(result: dict[str, Any], name: str) -> list[dict[str, Any]]:
    return [event for event in result["events"] if event["event"] == name]


def _assert_frames_match_the_log(result: dict[str, Any]) -> None:
    starts = {e["stage_id"]: e for e in _of(result, "stage_start")}
    for end in _of(result, "stage_end"):
        delta = end["frame_idx"] - starts[end["stage_id"]]["frame_idx"]
        assert end["frames"] == delta, end


def test_two_replays_in_one_chain_each_anchor_at_their_own_start(
    tmp_path: Path,
) -> None:
    result = _cli(
        tmp_path,
        _chain(tmp_path, FIRST_REPLAY, FIRST_POLICY, SECOND_REPLAY, SECOND_POLICY),
    )
    assert result["exit_code"] == cli.EXIT_OK
    assert [e["stage_id"] for e in _of(result, "stage_start")] == [
        "s1",
        "s2",
        "s3",
        "s4",
    ]
    _assert_frames_match_the_log(result)
    first, second = _of(result, "stage_base_replay")
    assert (first["stage_id"], first["outcome"]) == ("s1", "settled")
    assert (second["stage_id"], second["outcome"]) == ("s3", "settled")
    # The second replay is episode 1 rows 10..29, rebased on row 9: it drives
    # back 10 cm and turns -20 deg FROM WHERE THE FIRST ONE LEFT THE BASE.
    assert second["final_target_s_m"] == pytest.approx(-0.10)
    assert second["final_target_theta_deg"] == pytest.approx(-20.0)
    assert second["final_error_m"] < 0.01 and second["final_error_deg"] < 1.0
    assert first["anchor_odom"] != second["anchor_odom"]
    pose = result["slate"].true_pose()
    assert math.degrees(pose[2]) == pytest.approx(15.0 - 20.0, abs=1.5)
    assert (result["run_directory"] / "base" / "s1_replay.csv").is_file()
    assert (result["run_directory"] / "base" / "s3_replay.csv").is_file()
    assert result["robot"].base is result["slate"]


def test_a_replay_after_a_recorded_stage_holds_the_recorded_grippers(
    tmp_path: Path,
) -> None:
    result = _cli(
        tmp_path, _chain(tmp_path, LEADING_POLICY, FIRST_REPLAY, FIRST_POLICY)
    )
    assert result["exit_code"] == cli.EXIT_OK
    (event,) = _of(result, "stage_base_replay")
    assert event["gripper_hold_source"] == "recorded"
    # The grippers the replay held are those of the policy's last action.
    robot = result["robot"]
    s0_end = next(e for e in _of(result, "stage_end") if e["stage_id"] == "s0")
    last_policy_action = robot.sent_actions[s0_end["frame_idx"] - 1]
    for key in GRIPPERS:
        assert event["held_arm_positions"][key] == pytest.approx(
            last_policy_action[key], abs=1e-6
        )


def test_a_failed_replay_stops_the_chain_before_the_policy(tmp_path: Path) -> None:
    def instrument(robot: Any) -> None:
        robot.base.ignore_cmds = True

    result = _cli(tmp_path, _chain(tmp_path, FIRST_REPLAY, FIRST_POLICY), instrument)
    assert result["exit_code"] == cli.EXIT_ABORTED
    assert [e["stage_id"] for e in _of(result, "stage_start")] == ["s1"]
    (end,) = _of(result, "stage_end")
    assert end["terminator"] == "base_failed"
    assert _of(result, "stage_base_replay")[0]["outcome"] == "tracking_lost"
    _assert_frames_match_the_log(result)
    assert isinstance(result["robot"].base, MockSlate)
    assert not isinstance(result["robot"].base, BaseProbe)


# --------------------------------------------------------------------------
# Preflight: the tracker the executors import.


def test_preflight_refuses_a_base_stage_without_the_tracker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from stage_runner import preflight
    from test_base_preflight import _base_robot, _config

    preflight.check_base_stages(_config(), _base_robot())
    monkeypatch.setattr(base_stages, "TRACKER_MODULE", "no_such_tracker_module")
    with pytest.raises(preflight.PreflightError, match="no_such_tracker_module"):
        preflight.check_base_stages(_config(), _base_robot())
