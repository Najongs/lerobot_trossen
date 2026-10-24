"""The committed base-stage smoke chain, end to end through cli.main.

Drives tests/configs/smoke_mock_base_align.yaml -- two move bundles in one
process (s1 replay -> s1_align -> s2 policy -> s3 replay -> s3_align -> s4
policy, user decision A5) -- on the bimanual mock with the simulated base and
the scene camera. The YAML's __NAME__ tokens are filled here: the pose dataset
and the two reference PNGs are generated per run.

What is asserted, and why each matters on the robot:

* the chain converges (spec case 3): each align ends `aligned` against its own
  reference, the first in 2-4 moves from an offset of +6 deg and 3 cm, and both
  policies run;
* the task string of every recorded frame is the stage that recorded it, read
  back from the saved dataset's tasks.parquet (spec case 5) -- a replay or an
  align whose frames carried the policy's instruction would train the next
  policy on base motion;
* a right arrow during a replay accepts the base where it is, and the align that
  follows waits for the base to be still before its first capture (spec case
  9), so a frame taken while the base coasts is never measured;
* the stdlib aggregator reads the same log with the base stages off the score
  axis and their outcomes counted apart.

And the robot example, configs/chain_stage12_align.yaml, parses and clears the
gates that need no robot: the stage list, the two ceiling floors and the s2
start pose.

NOTHING HERE IS EVIDENCE ABOUT THE REAL ROBOT: the scene is a pure rotation with
dense synthetic texture, and the mock base follows its command exactly (with an
optional first-order lag).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import cv2
import pandas as pd
import pytest
from stage_runner import (
    aggregate,
    base_stages,
    cli,
    config,
    landmark_align,
    mock_scene,
    record_adapter,
)
from test_base_assets import _write_pose_dataset
from test_base_stages import _cli, _of, _on_update, _path
from test_smoke_runner import _assert_log_is_closed

CONFIGS = Path(__file__).parent / "configs"
SMOKE_CONFIG = CONFIGS / "smoke_mock_base_align.yaml"

# The geometry the YAML header states. The landmark sits 0.60 m straight ahead
# of the pose the mock starts at; a reference rendered at (yaw, distance) is the
# view from heading `yaw` at that distance from it.
# Episode 0 ends 1.5 cm forward at -3 deg: 0.585 m from the rack. The rack view
# is that pose turned +6 deg and 3 cm closer (spec case 3).
POSE_EPISODES = {
    0: (0.0005, -0.1),
    1: (-0.002, -0.5),
}
REFERENCES = {
    "RACK": (3.0, 0.555),
    "SHELF": (-8.0, 0.62),
}
# Task string of each stage, in chain order: a base stage's name, a policy
# stage's instruction.
TASKS = {
    "s1": "drive to the rack",
    "s1_align": "align to the rack",
    "s2": "pick up the tube",
    "s3": "drive to the shelf",
    "s3_align": "align to the shelf",
    "s4": "place the tube",
}


def _write_smoke_config(tmp_path: Path) -> Path:
    """smoke_mock_base_align.yaml with its __NAME__ tokens filled."""
    pose = _write_pose_dataset(
        tmp_path / "pose",
        {
            episode: [[0.0] * 14 + [s, t] for s, t in _path(30, step_m, step_deg)]
            for episode, (step_m, step_deg) in POSE_EPISODES.items()
        },
    )
    text = SMOKE_CONFIG.read_text(encoding="utf-8")
    text = text.replace("__POSE_ROOT__", str(pose))
    for name, (yaw, distance) in REFERENCES.items():
        image = mock_scene.render_scene(
            view_yaw_deg=yaw, landmark_scale=0.60 / distance
        )
        path = tmp_path / f"{name.lower()}.png"
        assert cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        roi = landmark_align.dark_blob_box(image)
        assert roi is not None, name
        text = text.replace(f"__{name}_PNG__", str(path)).replace(
            f"__{name}_ROI__", json.dumps(list(roi))
        )
    assert "__" not in text.split("\nstages:\n", 1)[1], "unfilled YAML token"
    path = tmp_path / "smoke_mock_base_align.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def _capture_keyboard(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Keep the keyboard events dict cli builds, so a test can press keys."""
    captured: dict[str, Any] = {}
    real = record_adapter.make_keyboard_events

    def make_keyboard_events() -> tuple[Any, dict[str, bool]]:
        listener, events = real()
        captured["events"] = events
        return listener, events

    monkeypatch.setattr(record_adapter, "make_keyboard_events", make_keyboard_events)
    return captured


def _frame_tasks(dataset_root: Path) -> list[str]:
    """The task string of every saved frame, in frame order."""
    tasks = pd.read_parquet(dataset_root / "meta" / "tasks.parquet")
    task_by_index = {int(row.task_index): str(task) for task, row in tasks.iterrows()}
    frames = pd.concat(
        [
            pd.read_parquet(path)
            for path in sorted(dataset_root.glob("data/**/*.parquet"))
        ]
    ).sort_values("frame_index")
    return [task_by_index[int(index)] for index in frames["task_index"]]


def test_the_smoke_yaml_states_the_geometry_the_test_renders(tmp_path: Path) -> None:
    """The YAML's distances and the rendered references are one geometry."""
    # parse_config resolves the mock robot type; cli.main registers the
    # lerobot plugins first, and so does this test.
    record_adapter.register_plugins()
    parsed = config.parse_config(["--config_path", str(_write_smoke_config(tmp_path))])
    stages = {stage.id: stage for stage in parsed.stages}
    assert [stage.id for stage in parsed.stages] == list(TASKS)
    assert stages["s1_align"].align.landmark_distance_m == REFERENCES["RACK"][1]
    assert stages["s3_align"].align.landmark_distance_m == REFERENCES["SHELF"][1]
    assert [stages[s].replay.episode for s in ("s1", "s3")] == [0, 1]
    for stage in parsed.stages:
        expected = stage.instruction or stage.name
        assert expected == TASKS[stage.id], stage.id


def test_the_smoke_chain_converges_and_records_each_stage_task(
    tmp_path: Path,
) -> None:
    result = _cli(tmp_path, _write_smoke_config(tmp_path))

    assert result["exit_code"] == cli.EXIT_OK
    assert [e["stage_id"] for e in _of(result, "stage_start")] == list(TASKS)
    _assert_log_is_closed(result)

    # Case 3: both aligns converge, each against its own reference; the first
    # from a +6 deg, 3 cm offset in 2-4 moves (a turn, a drive, perhaps one of
    # each again).
    replays = {e["stage_id"]: e for e in _of(result, "stage_base_replay")}
    aligns = {e["stage_id"]: e for e in _of(result, "stage_base_align")}
    assert {e["outcome"] for e in replays.values()} == {"settled"}
    assert {s: e["outcome"] for s, e in aligns.items()} == {
        "s1_align": "aligned",
        "s3_align": "aligned",
    }, {s: e["reason"] for s, e in aligns.items()}
    assert 2 <= aligns["s1_align"]["move_count"] <= 4, aligns["s1_align"]["moves"]
    kinds = {move["move"] for move in aligns["s1_align"]["moves"]}
    assert kinds == {"turn", "drive"}, aligns["s1_align"]["moves"]
    assert aligns["s1_align"]["reference"]["path"].endswith("rack.png")
    assert aligns["s3_align"]["reference"]["path"].endswith("shelf.png")
    ends = {e["stage_id"]: e for e in _of(result, "stage_end")}
    assert ends["s2"]["terminator"] == "timeout"
    assert ends["s4"]["terminator"] == "timeout"

    # Case 5: one episode, and every frame carries the task of the stage that
    # recorded it -- the switch lands exactly on each stage_end.frame_idx.
    dataset_root = tmp_path / "dataset"
    info = json.loads((dataset_root / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 1
    frame_tasks = _frame_tasks(dataset_root)
    starts = {e["stage_id"]: e for e in _of(result, "stage_start")}
    assert len(frame_tasks) == ends["s4"]["frame_idx"]
    for stage_id, task in TASKS.items():
        first, last = starts[stage_id]["frame_idx"], ends[stage_id]["frame_idx"]
        assert last > first, stage_id
        assert last - first == ends[stage_id]["frames"], stage_id
        assert frame_tasks[first:last] == [task] * (last - first), stage_id

    # The reader: base stages off the score axis, outcomes counted apart.
    trial = aggregate.build_trial(
        aggregate.load_events(result["run_directory"] / "events.jsonl")
    )
    assert aggregate.stage_order([trial]) == ["s2", "s4"]
    timing = aggregate.timing_summary([trial])
    assert timing["base_replay_s"]["outcomes"] == {"settled": 2}
    assert timing["base_align_s"]["outcomes"] == {"aligned": 2}
    assert timing["base_stage_failed_n"] == 0
    assert timing["base_align_s"]["frames"]["n"] == 2


def test_a_right_arrow_mid_replay_moves_on_after_the_base_is_still(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    keyboard = _capture_keyboard(monkeypatch)

    def instrument(robot: Any) -> None:
        # A base that coasts: a first-order lag, so it is still moving when the
        # replay is cut short and the stage's direct zero is sent.
        robot.base.lag_s = 0.4
        # About 1 s into s1 (15 pre-roll ticks, then the path), mid-path.
        _on_update(robot.base, 30, lambda: keyboard["events"].update(exit_early=True))

    result = _cli(tmp_path, _write_smoke_config(tmp_path), instrument)

    replays = {e["stage_id"]: e for e in _of(result, "stage_base_replay")}
    assert replays["s1"]["outcome"] == "manual"
    ends = {e["stage_id"]: e for e in _of(result, "stage_end")}
    assert ends["s1"]["terminator"] == "manual"
    # The chain moved on: the align ran and s2 started after it.
    assert [e["stage_id"] for e in _of(result, "stage_start")][:3] == [
        "s1",
        "s1_align",
        "s2",
    ]
    # The right arrow was consumed by the replay, not carried into the align.
    assert _of(result, "stage_base_align")[0]["outcome"] != "manual"

    # The align's first capture waited for the coasting base. The first hold
    # ticks still read motion; the capture comes only after CAPTURE_DELAY_S of
    # continuous stillness.
    path = result["run_directory"] / "base" / "s1_align_align.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    first_wait = [r for r in rows if r["state"] == "wait_capture" and r["step"] == "0"]

    def still(row: dict[str, str]) -> bool:
        return (
            abs(float(row["meas_x_vel"])) < base_stages.STILL_LINEAR_M_S
            and abs(float(row["meas_theta_vel"])) < base_stages.STILL_ANGULAR_RAD_S
        )

    assert first_wait and not still(first_wait[0]), first_wait[:3]
    (first,) = [
        e
        for e in _of(result, "stage_base_align_step")
        if e["stage_id"] == "s1_align" and e["kind"] == "measure" and e["step"] == 0
    ]
    assert first["still_wait_s"] > base_stages.CAPTURE_DELAY_S + 0.1, first
    tail = [r for r in first_wait if float(r["t"]) >= float(first_wait[-1]["t"]) - 0.25]
    assert tail and all(still(r) for r in tail), tail
    _assert_log_is_closed(result)


def test_the_robot_example_parses_and_clears_the_offline_gates() -> None:
    """configs/chain_stage12_align.yaml: the shape the robot run starts from.

    Only the checks that need no robot and no dataset: the stage list gate, the
    two ceiling floors base_assets and preflight enforce, and the s2 start pose
    (user decision A1: the start-pose fix is in, so the example carries it).
    The dataset and camera checks run on the robot PC, at preflight.
    """
    from stage_runner import preflight

    record_adapter.register_plugins()
    path = Path(__file__).parents[3] / "configs" / "chain_stage12_align.yaml"
    chain = config.parse_config(["--config_path", str(path)])
    preflight.check_stage_definitions(chain)
    replay, align, policy = chain.stages
    assert [s.executor for s in chain.stages] == [
        config.EXECUTOR_BASE_REPLAY,
        config.EXECUTOR_BASE_ALIGN,
        config.EXECUTOR_LEROBOT_POLICY,
    ]
    rows = replay.replay.until_row - replay.replay.from_row + 1
    assert rows == 298
    assert replay.terminator.timeout_s >= base_stages.replay_ceiling_floor_s(
        rows, replay.replay.row_period_s
    )
    assert align.terminator.timeout_s >= base_stages.align_ceiling_floor_s(
        align.align.max_moves
    )
    assert align.align.landmark_roi == [285, 197, 93, 61]
    assert align.align.coarse_heading is True
    assert align.align.camera_matrix is None, "intrinsics come from the device"
    assert policy.start_pose is not None
    assert len(policy.start_pose.positions) == 12
    assert policy.start_pose.source.policy_path == policy.policy_path
