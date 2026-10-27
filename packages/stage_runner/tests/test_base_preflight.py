"""Preflight for the base_replay / base_align stages: every refusal, and what stays legal.

The robot-independent half (check_stage_definitions) and the robot-dependent
half (check_base_stages) are tested against hand-built configs; the cli wiring
is tested by running cli.main on a mock chain with TWO move bundles (replay ->
align -> policy -> replay -> align -> policy) whose stage bodies are stand-ins,
so what is asserted is that every base stage reaches its executor with its own
assets loaded before connect.

Each refusal message is matched on the words that tell the operator what to
change, since that sentence is the whole point of a preflight refusal.
"""

from __future__ import annotations

import contextlib
import copy
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import cv2
import pytest
from lerobot.cameras.configs import ColorMode, Cv2Rotation
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from stage_runner import (
    base_assets,
    base_stages,
    cli,
    config,
    mock_scene,
    preflight,
    record_adapter,
)
from stage_runner.config import (
    BaseAlignConfig,
    BaseReplayConfig,
    StageConfig,
    StartPoseConfig,
    TerminatorConfig,
)
from stage_runner.mock_robot import MockRobot, MockRobotConfig
from stage_runner.results import StageResult
from test_base_assets import _pose_rows, _write_pose_dataset


def _replay_stage(**overrides: Any) -> StageConfig:
    values: dict[str, Any] = {
        "id": "s1",
        "name": "drive to the rack",
        "executor": "base_replay",
        "terminator": TerminatorConfig(type="timeout", timeout_s=30.0),
        "replay": BaseReplayConfig(dataset="local/pose", until_row=297),
    }
    values.update(overrides)
    return StageConfig(**values)


def _align_block(**overrides: Any) -> BaseAlignConfig:
    values: dict[str, Any] = {
        "reference_image": "/tmp/reference.png",
        "landmark_roi": [285, 197, 93, 61],
        "landmark_distance_m": 0.60,
        "camera_matrix": [[392.6, 0.0, 322.7], [0.0, 392.0, 236.2], [0.0, 0.0, 1.0]],
        "dist_coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
    }
    values.update(overrides)
    return BaseAlignConfig(**values)


def _align_stage(**overrides: Any) -> StageConfig:
    values: dict[str, Any] = {
        "id": "s1_align",
        "name": "align to the rack",
        "executor": "base_align",
        "terminator": TerminatorConfig(type="timeout", timeout_s=60.0),
        "align": _align_block(),
    }
    values.update(overrides)
    return StageConfig(**values)


def _policy_stage(**overrides: Any) -> StageConfig:
    values: dict[str, Any] = {
        "id": "s2",
        "name": "mock_stage_two",
        "policy_path": "mock://hold",
        "instruction": "pick up the tube",
        "terminator": TerminatorConfig(type="timeout", timeout_s=1.0),
    }
    values.update(overrides)
    return StageConfig(**values)


def _config(*stages: StageConfig) -> config.StageRunnerConfig:
    return config.StageRunnerConfig(
        robot=MockRobotConfig(id="unit"),
        stages=list(stages) or [_replay_stage(), _align_stage(), _policy_stage()],
    )


def _problems(parsed: config.StageRunnerConfig) -> str:
    with pytest.raises(preflight.PreflightError) as caught:
        preflight.check_stage_definitions(parsed)
    return str(caught.value)


def test_a_replay_align_policy_chain_is_accepted() -> None:
    preflight.check_stage_definitions(_config())


def test_a_chain_may_repeat_base_stages_with_their_own_blocks() -> None:
    preflight.check_stage_definitions(
        _config(
            _replay_stage(),
            _align_stage(),
            _policy_stage(),
            _replay_stage(
                id="s3",
                name="drive to the shelf",
                replay=BaseReplayConfig(
                    dataset="local/pose", episode=2, from_row=40, until_row=90
                ),
            ),
            _align_stage(
                id="s3_align",
                name="align to the shelf",
                align=_align_block(
                    reference_image="/tmp/shelf.png", landmark_roi=[1, 2, 3, 4]
                ),
            ),
            _policy_stage(
                id="s4", name="mock_stage_four", instruction="place the tube"
            ),
        )
    )


@pytest.mark.parametrize(
    ("stage", "needle"),
    [
        (_replay_stage(policy_path="mock://hold"), "runs no policy.*Delete it"),
        (
            _align_stage(
                start_pose=StartPoseConfig(positions={"left_joint_0.pos": 0.1})
            ),
            "policy stage that follows",
        ),
        (_align_stage(instruction="align"), "`instruction` would be ignored"),
        (_replay_stage(name=""), "`name` is empty"),
        (
            _replay_stage(terminator=TerminatorConfig(type="manual", timeout_s=30.0)),
            "needs `terminator.type: timeout`.*runaway ceiling",
        ),
        (_replay_stage(replay=None), "needs a `replay:` block"),
        (_replay_stage(align=_align_block()), "ignores `align:`"),
        (
            _replay_stage(replay=BaseReplayConfig(until_row=10)),
            "`replay.dataset` is empty",
        ),
        (
            _replay_stage(
                replay=BaseReplayConfig(dataset="d", episode=-1, until_row=10)
            ),
            "`replay.episode: -1`",
        ),
        (_replay_stage(replay=BaseReplayConfig(dataset="d")), "has no default"),
        (
            _replay_stage(
                replay=BaseReplayConfig(dataset="d", from_row=5, until_row=5)
            ),
            "0 <= from_row < until_row",
        ),
        (
            _replay_stage(
                replay=BaseReplayConfig(dataset="d", until_row=9, row_period_s=0.0)
            ),
            "row_period_s: 0.0",
        ),
        (
            _replay_stage(
                replay=BaseReplayConfig(
                    dataset="d", until_row=9, row_period_s=float("nan")
                )
            ),
            "row_period_s: nan",
        ),
        (_align_stage(align=None), "needs an `align:` block"),
        (
            _align_stage(replay=BaseReplayConfig(dataset="d", until_row=9)),
            "ignores `replay:`",
        ),
        (
            _align_stage(align=_align_block(reference_dataset="org/demo", frame=3)),
            "exactly one of",
        ),
        (_align_stage(align=_align_block(reference_image=None)), "exactly one of"),
        (
            _align_stage(
                align=_align_block(reference_image=None, reference_dataset="org/demo")
            ),
            "arrival frame",
        ),
        (_align_stage(align=_align_block(landmark_roi=[1, 2, 3])), "landmark_roi"),
        (_align_stage(align=_align_block(landmark_roi=[-1, 2, 3, 4])), "landmark_roi"),
        (_align_stage(align=_align_block(landmark_roi=[1, 2, 0, 4])), "landmark_roi"),
        (
            _align_stage(align=_align_block(landmark_distance_m=0.0)),
            "landmark_distance_m",
        ),
        (_align_stage(align=_align_block(yaw_tol_deg=0.0)), "yaw_tol_deg: 0.0"),
        (_align_stage(align=_align_block(dist_tol_cm=-1.0)), "dist_tol_cm: -1.0"),
        (_align_stage(align=_align_block(max_moves=0)), "max_moves: 0"),
        (_align_stage(align=_align_block(camera_up=[0.0, 2.0, 0.0])), "norm"),
        (_align_stage(align=_align_block(camera_up=[0.0, 1.0])), "camera_up"),
        (_align_stage(align=_align_block(dist_coeffs=None)), "both"),
        (
            _align_stage(align=_align_block(camera_matrix=[[1.0, 0.0], [0.0, 1.0]])),
            "finite 3x3",
        ),
        (_align_stage(align=_align_block(dist_coeffs=[0.0] * 6)), "4, 5, 8, 12 or 14"),
        (_align_stage(align=_align_block(turn_kp=0.0)), "turn_kp"),
        (_align_stage(align=_align_block(min_turn_rate_rad_s=-0.1)), "min_turn_rate"),
    ],
)
def test_each_base_stage_refusal(stage: StageConfig, needle: str) -> None:
    other = _align_stage() if stage.executor == "base_replay" else _replay_stage()
    message = _problems(_config(stage, other, _policy_stage()))
    assert stage.id in message
    assert re.search(needle, message), message


def test_a_replay_stage_may_carry_the_next_start_pose() -> None:
    # The arms are posed for the next policy stage before the base moves
    # (robot PC 2026-10-25, t03-t06); only base_align still refuses it.
    replay = _replay_stage(
        start_pose=StartPoseConfig(positions={"left_joint_0.pos": 0.1})
    )
    preflight.check_stage_definitions(_config(replay, _align_stage(), _policy_stage()))


def test_the_align_ceiling_floor_names_its_number() -> None:
    message = _problems(
        _config(
            _replay_stage(),
            _align_stage(terminator=TerminatorConfig(type="timeout", timeout_s=56.0)),
            _policy_stage(),
        )
    )
    assert "at least 57 s" in message
    assert "align.max_moves" in message
    # Fewer moves, shorter floor: 56.0 s is enough for 5.
    shorter = _align_stage(
        terminator=TerminatorConfig(type="timeout", timeout_s=56.0),
        align=_align_block(max_moves=5),
    )
    assert base_stages.align_ceiling_floor_s(5) < 56.0
    preflight.check_stage_definitions(
        _config(_replay_stage(), shorter, _policy_stage())
    )


@pytest.mark.parametrize(
    ("stage", "block"),
    [
        (_policy_stage(replay=BaseReplayConfig(dataset="d", until_row=9)), "replay"),
        (
            StageConfig(
                id="reset",
                name="hand over",
                executor="leader_reset",
                terminator=TerminatorConfig(type="timeout", timeout_s=1.0),
                align=_align_block(),
            ),
            "align",
        ),
    ],
)
def test_a_base_block_on_another_executor_is_refused(
    stage: StageConfig, block: str
) -> None:
    parsed = _config(stage, _policy_stage(id="s9"))
    parsed.teleop = object()  # the leader_reset branch's own requirement
    message = _problems(parsed)
    assert f"ignores `{block}:`" in message
    assert "Set the stage's `executor:`" in message


def test_a_base_name_that_is_another_stages_task_string_is_refused() -> None:
    message = _problems(
        _config(_replay_stage(name="pick up the tube"), _align_stage(), _policy_stage())
    )
    assert "'s1'" in message and "could not be told apart" in message
    # A policy stage's task string is its instruction, else its name.
    message = _problems(
        _config(
            _replay_stage(name="mock_stage_two"),
            _align_stage(),
            _policy_stage(instruction=""),
        )
    )
    assert "could not be told apart" in message
    # And two base stages may not share a name either.
    message = _problems(
        _config(_replay_stage(name="base"), _align_stage(name="base"), _policy_stage())
    )
    assert "'s1'" in message and "'s1_align'" in message


def test_policy_stages_sharing_an_instruction_are_still_accepted() -> None:
    preflight.check_stage_definitions(
        _config(
            _policy_stage(id="a", name="a"),
            _policy_stage(id="b", name="b"),
            _replay_stage(),
        )
    )


def test_a_chain_of_base_stages_only_is_not_an_eval() -> None:
    message = _problems(_config(_replay_stage(), _align_stage()))
    assert "at least one" in message
    assert "`base_align`, `base_replay`" in message


# --------------------------------------------------------------------------
# The robot-dependent half.


def _base_robot(**overrides: Any) -> MockRobot:
    values: dict[str, Any] = {
        "id": "unit",
        "arm_layout": "bimanual",
        "simulate_base": True,
        "scene_camera": "cam_high",
    }
    values.update(overrides)
    return MockRobot(MockRobotConfig(**values))


def test_a_mock_with_a_base_and_the_scene_camera_passes() -> None:
    preflight.check_base_stages(_config(), _base_robot())


def test_a_chain_without_base_stages_does_not_look_at_the_base() -> None:
    preflight.check_base_stages(_config(_policy_stage()), SimpleNamespace())


@pytest.mark.parametrize(
    ("robot_overrides", "needle"),
    [
        ({"simulate_base": False}, "has no base with"),
        ({"enable_base_motor_torque": False}, "enable_base_motor_torque: true"),
        ({"scene_camera": None}, "is not a camera of robot"),
    ],
)
def test_a_robot_that_cannot_run_the_base_stages_is_refused(
    robot_overrides: dict[str, Any], needle: str
) -> None:
    with pytest.raises(preflight.PreflightError, match=needle):
        preflight.check_base_stages(_config(), _base_robot(**robot_overrides))


def test_a_base_without_velocity_keys_is_refused() -> None:
    robot = _base_robot()
    fake = SimpleNamespace(
        name="fake",
        action_features={key: float for key in robot.action_features if key != "x.vel"},
        base=robot.base,
        config=robot.config,
        cameras=robot.cameras,
    )
    with pytest.raises(
        preflight.PreflightError, match=r"no base velocity command \['x.vel'\]"
    ):
        preflight.check_base_stages(_config(), fake)


def _realsense_robot(**camera: Any) -> SimpleNamespace:
    robot = _base_robot()
    return SimpleNamespace(
        name="mobileai",
        action_features=robot.action_features,
        base=robot.base,
        cameras={"cam_high": object()},
        config=SimpleNamespace(
            enable_base_motor_torque=True,
            cameras={
                "cam_high": RealSenseCameraConfig(
                    serial_number_or_name="0123",
                    fps=30,
                    width=640,
                    height=480,
                    **camera,
                )
            },
        ),
    )


def test_a_realsense_is_its_own_intrinsics_source() -> None:
    no_intrinsics = _config(
        _replay_stage(),
        _align_stage(align=_align_block(camera_matrix=None, dist_coeffs=None)),
        _policy_stage(),
    )
    preflight.check_base_stages(no_intrinsics, _realsense_robot())
    # The mock has no camera config entry to read intrinsics from.
    with pytest.raises(preflight.PreflightError, match="no intrinsics source"):
        preflight.check_base_stages(no_intrinsics, _base_robot())


@pytest.mark.parametrize(
    ("camera", "needle"),
    [
        ({"rotation": Cv2Rotation.ROTATE_90}, "rotation to 0"),
        ({"color_mode": ColorMode.BGR}, "color_mode to rgb"),
    ],
)
def test_a_rotated_or_bgr_camera_is_refused(
    camera: dict[str, Any], needle: str
) -> None:
    with pytest.raises(preflight.PreflightError, match=needle):
        preflight.check_base_stages(_config(), _realsense_robot(**camera))


# --------------------------------------------------------------------------
# cli: two move bundles, every base stage reaching its executor with its assets.

YAML = """\
version: 1
robot:
  type: stage_runner_mock_robot
  id: smoke
  arm_layout: bimanual
  simulate_base: true
  scene_camera: cam_high
dataset:
  repo_id: local/eval_two_bundles
  fps: 30
  video: false
stages:
  - id: s1
    name: drive to the rack
    executor: base_replay
    terminator: {type: timeout, timeout_s: 20}
    replay: {dataset: local/pose, root: POSE, episode: 0, until_row: 20}
  - id: s1_align
    name: align to the rack
    executor: base_align
    terminator: {type: timeout, timeout_s: 60}
    align:
      reference_image: RACK
      landmark_roi: [285, 197, 93, 61]
      landmark_distance_m: 0.60
      camera_matrix: [[392.6346, 0, 322.7227], [0, 392.0662, 236.2555], [0, 0, 1]]
      dist_coeffs: [0, 0, 0, 0, 0]
  - id: s2
    name: mock_stage_two
    policy_path: mock://hold
    instruction: pick up the tube
    terminator: {type: timeout, timeout_s: 0.3}
  - id: s3
    name: drive to the shelf
    executor: base_replay
    terminator: {type: timeout, timeout_s: 20}
    replay: {dataset: local/pose, root: POSE, episode: 1, from_row: 5, until_row: 25}
  - id: s3_align
    name: align to the shelf
    executor: base_align
    terminator: {type: timeout, timeout_s: 60}
    align:
      reference_image: SHELF
      landmark_roi: [285, 197, 93, 61]
      landmark_distance_m: 0.45
      camera_matrix: [[392.6346, 0, 322.7227], [0, 392.0662, 236.2555], [0, 0, 1]]
      dist_coeffs: [0, 0, 0, 0, 0]
  - id: s4
    name: mock_stage_four
    policy_path: mock://hold
    instruction: place the tube
    terminator: {type: timeout, timeout_s: 0.3}
"""


def _write_chain(tmp_path: Path) -> Path:
    pose = _write_pose_dataset(
        tmp_path / "pose", {0: _pose_rows(0, 30), 1: _pose_rows(1, 30)}
    )
    pictures = {}
    for name, yaw in (("RACK", 0.0), ("SHELF", 5.0)):
        path = tmp_path / f"{name.lower()}.png"
        image = mock_scene.render_scene(view_yaw_deg=yaw)
        assert cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        pictures[name] = str(path)
    text = YAML.replace("POSE", str(pose))
    for name, path in pictures.items():
        text = text.replace(name, path)
    path = tmp_path / "chain.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def _run_cli(tmp_path: Path, config_path: Path) -> int:
    return cli.main(
        [
            "--config_path",
            str(config_path),
            f"--dataset.root={tmp_path / 'dataset'}",
            f"--output.root={tmp_path / 'outputs'}",
        ]
    )


def test_every_base_stage_of_two_move_bundles_gets_its_own_assets(
    tmp_path: Path,
) -> None:
    config_path = _write_chain(tmp_path)
    seen: list[tuple[str, str, Any]] = []
    connected_at_load: list[bool] = []
    real_load = base_assets.load_base_assets

    def load(cfg: Any, robot: Any) -> Any:
        connected_at_load.append(robot.is_connected)
        return real_load(cfg, robot)

    def stand_in(kind: str):
        def run(context: Any, stage: StageConfig) -> StageResult:
            asset = context.base_assets[stage.id]
            seen.append((kind, stage.id, copy.copy(asset.source)))
            return StageResult(terminated_by="converged", elapsed_s=0.01, frames=0)

        return run

    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.object(base_assets, "load_base_assets", load))
        stack.enter_context(
            mock.patch.object(base_stages, "replay", stand_in("replay"))
        )
        stack.enter_context(mock.patch.object(base_stages, "align", stand_in("align")))
        exit_code = _run_cli(tmp_path, config_path)

    assert exit_code == cli.EXIT_OK
    assert connected_at_load == [False], "assets load before connect, once"
    assert [(kind, stage_id) for kind, stage_id, _ in seen] == [
        ("replay", "s1"),
        ("align", "s1_align"),
        ("replay", "s3"),
        ("align", "s3_align"),
    ]
    sources = {stage_id: source for _, stage_id, source in seen}
    assert (sources["s1"]["episode"], sources["s1"]["rows"]) == (0, 21)
    assert (
        sources["s3"]["episode"],
        sources["s3"]["from_row"],
        sources["s3"]["rows"],
    ) == (
        1,
        5,
        21,
    )
    assert sources["s1_align"]["path"].endswith("rack.png")
    assert sources["s3_align"]["path"].endswith("shelf.png")
    events_path = next((tmp_path / "outputs").glob("*/events.jsonl"))
    starts = [
        json.loads(line)["stage_id"]
        for line in events_path.read_text().splitlines()
        if json.loads(line)["event"] == "stage_start"
    ]
    assert starts == ["s1", "s1_align", "s2", "s3", "s3_align", "s4"]


def test_a_missing_asset_is_exit_2_before_connect(tmp_path: Path) -> None:
    config_path = _write_chain(tmp_path)
    text = config_path.read_text().replace("until_row: 25", "until_row: 30")
    config_path.write_text(text)
    connected: list[bool] = []
    real_make_robot = record_adapter.make_robot

    def make_robot(robot_config: Any) -> Any:
        robot = real_make_robot(robot_config)
        robot.connect = lambda *a, **k: connected.append(True)  # noqa: ARG005
        return robot

    with mock.patch.object(record_adapter, "make_robot", make_robot):
        exit_code = _run_cli(tmp_path, config_path)
    assert exit_code == cli.EXIT_PREFLIGHT
    assert connected == []
    assert not (tmp_path / "outputs").exists(), "no run directory for a refused run"
