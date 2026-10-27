"""The base_align executor on the simulated base and scene camera.

Two harnesses, as test_base_stages.py:

* DIRECT -- base_stages.align called on a StageContext built here (a real
  LeRobotDataset, the bimanual mock with the simulated base and the scene
  camera, an EventLog). The base is placed off the reference pose and the
  REAL estimator measures the rendered view; for the decision rules the
  estimator is replaced by a script of measurements, so each rule is pinned
  without depending on what the renderer happens to produce;
* CLI -- the real cli.main on a chain YAML with TWO move bundles (replay ->
  align -> policy, twice), each align against its own reference, ROI and
  landmark distance (user decision A5: one command runs the whole chain).

What is asserted, and why each matters on the robot:

* the field's decision order: heading first, distance only once the heading
  is inside tolerance, divergence against the error before the last
  correction of the same kind (a drive does not reset the turn baseline);
* the coarse heading fallback (A2): a coarse estimate turns and is measured
  again, never counts as aligned, counts toward max_moves, and more than two
  coarse turns end the stage; off, an unusable fine estimate aborts. A coarse
  heading inside tolerance with a usable distance outside it drives instead
  (robot PC 10/25, a far arrival that never got a fine heading);
* heading_only continues the chain (A3); every other failure stops it before
  the next policy, each with the value that tripped it (A4);
* the measurement runs in a worker thread WHILE the loop keeps recording hold
  frames with a zero base command, and Esc does not wait for it;
* every exit zeroes the base directly and puts the real driver back.

NOTHING HERE IS EVIDENCE ABOUT THE REAL ROBOT: the scene is a pure rotation
with dense synthetic texture (hundreds of inliers where the lab gives 10-60),
and the mock base follows its command exactly.
"""

from __future__ import annotations

import csv
import json
import math
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pytest
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from stage_runner import base_stages, cli, landmark_align, mock_scene, record_adapter
from stage_runner.base_assets import AlignReference
from stage_runner.base_loop import BaseProbe
from stage_runner.config import (
    BaseAlignConfig,
    DefaultsConfig,
    StageConfig,
    TerminatorConfig,
)
from stage_runner.context import StageContext
from stage_runner.events import EventLog
from stage_runner.mock_robot import MockRobot, MockRobotConfig, MockSlate
from test_base_assets import _write_pose_dataset
from test_base_stages import _cli, _of, _on_update, _path
from test_smoke_runner import _assert_log_is_closed

FPS = 30
ROI = list(mock_scene.DEFAULT_LANDMARK_ROI)
K = [list(row) for row in mock_scene.DEFAULT_CAMERA_MATRIX]

# Every field the spec lists for stage_base_align (plus outcome).
ALIGN_EVENT_FIELDS = {
    "outcome",
    "reason",
    "reference",
    "intrinsics",
    "field_intrinsics_delta",
    "camera_up",
    "landmark_roi",
    "auto_roi",
    "landmark_distance_m",
    "yaw_tol_deg",
    "dist_tol_cm",
    "max_moves",
    "turn_kp",
    "min_turn_rate_rad_s",
    "distance_width_fallback",
    "coarse_heading",
    "measurements",
    "moves",
    "move_count",
    "coarse_turns",
    "final_yaw_deg",
    "final_move_cm",
    "measure_s_total",
    "tick_mean_s",
    "tick_p95_s",
    "tick_p95_s_while_measuring",
    "cmd_refused_ticks",
    "update_failed_ticks",
    "base_state_last",
    "gripper_hold_source",
    "csv",
}
MEASURE_EVENT_FIELDS = {
    "kind",
    "step",
    "capture_frame_idx",
    "capture_wall_clock_iso",
    "still_wait_s",
    "yaw_deg",
    "inliers",
    "matches",
    "heading_valid",
    "heading_mode",
    "heading_yaw_deg",
    "coarse_yaw_deg",
    "scale",
    "scale_inliers",
    "scale_matches",
    "scale_source",
    "move_m",
    "distance_valid",
    "measure_s",
    "live_png",
    "overlay_png",
    "decision",
}
MOVE_EVENT_FIELDS = {
    "kind",
    "step",
    "move",
    "commanded",
    "achieved",
    "ratio",
    "ramp_s",
    "hold_s",
    "still_at_exit",
    "odom_bad_ticks",
    "max_tracking_error_m",
    "max_tracking_error_deg",
    "frame_idx_start",
    "frame_idx_end",
    "heading_mode",
}


# --------------------------------------------------------------------------
# DIRECT harness.


def _reference() -> np.ndarray:
    image = mock_scene.render_scene()
    image.setflags(write=False)
    return image


def _direct(
    tmp_path: Path,
    *,
    yaw_deg: float = 0.0,
    x_m: float = 0.0,
    timeout_s: float = 60.0,
    robot: dict[str, Any] | None = None,
    **align: Any,
) -> tuple[StageContext, StageConfig]:
    """A connected mock placed at (x_m, 0, yaw_deg) off the reference pose (0, 0, 0)."""
    mock = MockRobot(
        MockRobotConfig(
            arm_layout="bimanual",
            simulate_base=True,
            scene_camera="cam_high",
            **(robot or {}),
        )
    )
    processors = record_adapter.make_processors()
    dataset = LeRobotDataset.create(
        "local/base_align",
        FPS,
        root=tmp_path / "dataset",
        robot_type=mock.name,
        features=record_adapter.build_dataset_features(mock, processors, False),
        use_videos=False,
    )
    mock.connect()
    mock.base.place(x_m, 0.0, math.radians(yaw_deg))
    settings = {
        "reference_image": str(tmp_path / "reference.png"),
        "landmark_roi": ROI,
        "landmark_distance_m": 0.60,
        "camera_matrix": K,
        "dist_coeffs": [0.0] * 5,
        **align,
    }
    stage = StageConfig(
        id="s1_align",
        name="align to the rack",
        executor="base_align",
        terminator=TerminatorConfig(type="timeout", timeout_s=timeout_s),
        align=BaseAlignConfig(**settings),
    )
    context = StageContext(
        config=SimpleNamespace(
            dataset=SimpleNamespace(fps=FPS), defaults=DefaultsConfig()
        ),
        robot=mock,
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
        base_assets={
            "s1_align": AlignReference(
                image=_reference(), source={"kind": "image", "path": "reference.png"}
            )
        },
    )
    return context, stage


def _events(tmp_path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
        if line
    ]


def _align_event(tmp_path: Path) -> dict[str, Any]:
    (event,) = (e for e in _events(tmp_path) if e["event"] == "stage_base_align")
    return event


def _steps(tmp_path: Path) -> list[dict[str, Any]]:
    return [e for e in _events(tmp_path) if e["event"] == "stage_base_align_step"]


def _csv(tmp_path: Path) -> list[dict[str, str]]:
    with (tmp_path / "base" / "s1_align_align.csv").open() as handle:
        return list(csv.DictReader(handle))


def _no_align_thread(within_s: float = 5.0) -> bool:
    deadline = time.perf_counter() + within_s
    while time.perf_counter() < deadline:
        if not any(t.name.startswith("align-") for t in threading.enumerate()):
            return True
        time.sleep(0.05)
    return False


def _run(context: StageContext, stage: StageConfig) -> Any:
    slate = context.robot.base
    assert isinstance(slate, MockSlate)
    result = base_stages.align(context, stage)
    assert context.robot.base is slate, "robot.base는 원래 드라이버로 복원"
    assert slate.commands[-1][:2] == (0.0, 0.0), "끝나면 베이스를 0으로"
    assert result.frames == context.buffered_frame_count()
    assert _no_align_thread(), "측정 스레드가 남지 않는다"
    return result


@pytest.fixture
def quick(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shorter holds and capture gate, for the decision tests (not the timing ones)."""
    monkeypatch.setattr(base_stages, "MOVE_HOLD_S", 0.25)
    monkeypatch.setattr(base_stages, "CAPTURE_DELAY_S", 0.05)


def _measurement(
    yaw: float | None,
    *,
    mode: str | None = "fine",
    move: float | None = 0.0,
    inliers: int = 400,
) -> landmark_align.Measurement:
    """A scripted estimate: ``yaw`` is what the decision acts on (fine or coarse)."""
    fine = mode == "fine"
    return landmark_align.Measurement(
        yaw_deg=yaw if fine else (None if yaw is None else yaw + 15.0),
        inliers=inliers if fine else 5,
        matches=500,
        heading_valid=fine,
        coarse_yaw_deg=yaw if mode == "coarse" else None,
        heading_mode=mode,
        heading_yaw_deg=yaw if mode is not None else None,
        sift_scale=None if move is None else 1.0,
        scale=None if move is None else 1.0,
        scale_inliers=0 if move is None else 40,
        scale_matches=0 if move is None else 50,
        scale_source=None if move is None else "sift",
        move_m=move,
        distance_valid=move is not None,
        measure_s=0.01,
        live_png=None,
        overlay_png=None,
    )


def _script(
    monkeypatch: pytest.MonkeyPatch,
    measurements: list[landmark_align.Measurement],
    before: Any = None,
) -> list[int]:
    """Replace the estimator with ``measurements`` in order; returns the steps asked."""
    asked: list[int] = []

    def measure(reference, live, params, save_dir, step):
        asked.append(step)
        if before is not None:
            before(step)
        return measurements[step]

    monkeypatch.setattr(landmark_align, "measure", measure)
    return asked


# --------------------------------------------------------------------------
# Converging on the real estimator.


def test_an_offset_base_turns_then_drives_and_aligns(tmp_path: Path) -> None:
    # 6 deg left of the demo heading and 3 cm farther from the rack.
    context, stage = _direct(tmp_path, yaw_deg=6.0, x_m=-0.03)
    result = _run(context, stage)
    assert result.terminated_by == "converged", result.reason

    event = _align_event(tmp_path)
    assert ALIGN_EVENT_FIELDS <= set(event), ALIGN_EVENT_FIELDS - set(event)
    assert event["outcome"] == "aligned"
    assert event["auto_roi"] == ROI
    assert event["intrinsics"]["source"] == "yaml"
    assert event["csv"] == "base/s1_align_align.csv"
    assert event["gripper_hold_source"] == "measured"
    # Heading first, then the distance, then the check.
    assert [m["decision"] for m in event["measurements"]] == [
        "turn",
        "drive",
        "aligned",
    ]
    assert [m["move"] for m in event["moves"]] == ["turn", "drive"]
    assert event["move_count"] == 2
    turn, drive = event["moves"]
    assert turn["commanded"] == pytest.approx(-6.0, abs=0.3)
    assert drive["commanded"] == pytest.approx(0.03, abs=0.004)
    assert turn["ratio"] == pytest.approx(1.0, abs=0.1)

    x, _, heading = context.robot.base.true_pose()
    assert math.degrees(heading) == pytest.approx(0.0, abs=stage.align.yaw_tol_deg)
    assert x == pytest.approx(0.0, abs=0.01)

    steps = _steps(tmp_path)
    assert [(s["kind"], s["step"]) for s in steps] == [
        ("measure", 0),
        ("move", 0),
        ("measure", 1),
        ("move", 1),
        ("measure", 2),
    ]
    for step in steps:
        fields = (
            MEASURE_EVENT_FIELDS if step["kind"] == "measure" else MOVE_EVENT_FIELDS
        )
        assert fields <= set(step), fields - set(step)
    side = tmp_path / "base" / "s1_align"
    assert (side / "reference.png").is_file()
    for step in range(3):
        assert (side / f"s{step}_live.png").is_file()
        assert (side / f"s{step}_overlay.png").is_file()
    assert steps[0]["live_png"] == "base/s1_align/s0_live.png"

    rows = _csv(tmp_path)
    assert len(rows) == result.frames, "CSV 한 줄 = 기록된 프레임 하나"
    assert {row["state"] for row in rows} == {"wait_capture", "measuring", "moving"}
    assert set(context.dataset.episode_buffer["task"]) == {"align to the rack"}


def test_a_coarse_heading_turns_and_is_measured_again(tmp_path: Path) -> None:
    # 25 deg off: SIFT fits it but past MAX_ABS_YAW_DEG, so only the rack blob
    # gives a usable heading (A2, the field's heading_for_step).
    context, stage = _direct(tmp_path, yaw_deg=25.0)
    result = _run(context, stage)
    assert result.terminated_by == "converged", result.reason
    event = _align_event(tmp_path)
    first = event["measurements"][0]
    assert first["heading_mode"] == "coarse"
    assert first["heading_valid"] is False
    assert first["decision"] == "turn"
    assert event["moves"][0]["heading_mode"] == "coarse"
    assert event["coarse_turns"] == 1
    assert event["measurements"][-1]["heading_mode"] == "fine"
    assert event["outcome"] == "aligned"


@pytest.mark.parametrize(
    ("yaw_deg", "coarse_heading"),
    [
        pytest.param(25.0, False, id="coarse-off"),
        pytest.param(38.0, True, id="beyond-coarse"),
    ],
)
def test_an_unusable_heading_ends_the_stage_without_moving(
    tmp_path: Path, yaw_deg: float, coarse_heading: bool
) -> None:
    context, stage = _direct(tmp_path, yaw_deg=yaw_deg, coarse_heading=coarse_heading)
    result = _run(context, stage)
    # A measurement failure stops the base and goes on (user decision 2026-10-25).
    assert result.terminated_by == "converged"
    event = _align_event(tmp_path)
    assert event["outcome"] == "estimate_unusable", event["reason"]
    assert event["move_count"] == 0
    (measurement,) = event["measurements"]
    assert measurement["heading_mode"] is None
    assert measurement["decision"] == "estimate_unusable"
    assert all(cmd[:2] == (0.0, 0.0) for cmd in context.robot.base.commands)


# --------------------------------------------------------------------------
# Decision rules, on scripted estimates.


@pytest.mark.parametrize(
    ("script", "max_moves", "outcome", "decisions"),
    [
        pytest.param(
            [_measurement(5.0), _measurement(7.0)],
            6,
            "diverging",
            ["turn", "diverging"],
            id="heading-diverges",
        ),
        pytest.param(
            [_measurement(5.0)] * 3,
            2,
            "not_converged",
            ["turn", "turn", "not_converged"],
            id="max-moves",
        ),
        pytest.param(
            [_measurement(1.0, move=None)],
            6,
            "heading_only",
            ["heading_only"],
            id="heading-only",
        ),
        pytest.param(
            # The field rule: a drive does not reset the turn baseline, so a
            # heading worse than before the LAST TURN is diverging even after it.
            [
                _measurement(5.0),
                _measurement(0.5, move=0.03),
                _measurement(5.6),
            ],
            6,
            "diverging",
            ["turn", "drive", "diverging"],
            id="drive-keeps-turn-baseline",
        ),
        pytest.param(
            [_measurement(1.0, move=0.03), _measurement(0.5, move=0.04)],
            6,
            "diverging",
            ["drive", "diverging"],
            id="distance-diverges",
        ),
        pytest.param(
            # A coarse estimate turns even inside tolerance and never aligns.
            [_measurement(1.0, mode="coarse"), _measurement(0.2)],
            6,
            "aligned",
            ["turn", "aligned"],
            id="coarse-inside-tolerance-turns",
        ),
        pytest.param(
            # ... unless the distance is usable and outside tolerance: then it
            # drives first, and only the fine estimate after it aligns.
            [_measurement(1.0, mode="coarse", move=0.094), _measurement(0.2)],
            6,
            "aligned",
            ["drive", "aligned"],
            id="coarse-inside-tolerance-far-drives",
        ),
        pytest.param(
            # Without a usable distance a coarse heading inside tolerance turns.
            [_measurement(1.0, mode="coarse", move=None), _measurement(0.2)],
            6,
            "aligned",
            ["turn", "aligned"],
            id="coarse-inside-tolerance-no-distance-turns",
        ),
        pytest.param(
            # A coarse heading outside tolerance still turns first.
            [_measurement(5.0, mode="coarse", move=0.094), _measurement(0.2)],
            6,
            "aligned",
            ["turn", "aligned"],
            id="coarse-outside-tolerance-far-turns",
        ),
        pytest.param(
            # r09 on the robot PC (10/25): two coarse turns with the distance
            # past the 10 cm cap, then coarse -0.95 deg / +9.4 cm. It ended
            # estimate_unusable there; now it drives and measures again.
            [
                _measurement(-4.44, mode="coarse", move=None),
                _measurement(-2.04, mode="coarse", move=None),
                _measurement(-0.95, mode="coarse", move=0.094),
                _measurement(-0.4),
            ],
            6,
            "aligned",
            ["turn", "turn", "drive", "aligned"],
            id="r09-far-arrival-drives-after-coarse-turns",
        ),
        pytest.param(
            # Coarse drives are judged like any drive: a growing distance diverges.
            [
                _measurement(1.0, mode="coarse", move=0.05),
                _measurement(1.0, mode="coarse", move=0.06),
            ],
            6,
            "diverging",
            ["drive", "diverging"],
            id="coarse-drive-diverges",
        ),
        pytest.param(
            [_measurement(10.0, mode="coarse")] * 3,
            6,
            "estimate_unusable",
            ["turn", "turn", "estimate_unusable"],
            id="three-coarse",
        ),
        pytest.param(
            # The field (stage1_align) runs the divergence test on a coarse
            # estimate too: a wrong-sign coarse turn (a mirrored or different
            # blob) must not go unnoticed for up to MAX_COARSE_TURNS turns.
            [_measurement(10.0, mode="coarse"), _measurement(12.0, mode="coarse")],
            6,
            "diverging",
            ["turn", "diverging"],
            id="coarse-diverges",
        ),
        pytest.param(
            # ... and a coarse turn is the baseline for the fine estimate after it.
            [_measurement(10.0, mode="coarse"), _measurement(12.0)],
            6,
            "diverging",
            ["turn", "diverging"],
            id="fine-diverges-after-coarse",
        ),
        pytest.param(
            # Coarse turns count toward max_moves.
            [_measurement(10.0, mode="coarse"), _measurement(5.0)],
            1,
            "not_converged",
            ["turn", "not_converged"],
            id="coarse-counts-as-a-move",
        ),
        pytest.param(
            [_measurement(None, mode=None)],
            6,
            "estimate_unusable",
            ["estimate_unusable"],
            id="no-estimate",
        ),
    ],
)
def test_the_field_decision_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    quick: None,
    script: list[landmark_align.Measurement],
    max_moves: int,
    outcome: str,
    decisions: list[str],
) -> None:
    _script(monkeypatch, script)
    context, stage = _direct(tmp_path, max_moves=max_moves)
    result = _run(context, stage)
    event = _align_event(tmp_path)
    assert event["outcome"] == outcome, event["reason"]
    assert [m["decision"] for m in event["measurements"]] == decisions
    goes_on = (
        "aligned",
        "heading_only",
        "estimate_unusable",
        "diverging",
        "not_converged",
    )
    expected = "converged" if outcome in goes_on else "base_failed"
    assert result.terminated_by == expected
    # Every move turns by -yaw or drives by +move, as the field.
    for measurement, move in zip(event["measurements"], event["moves"], strict=False):
        if move["move"] == "turn":
            assert move["commanded"] == pytest.approx(-measurement["heading_yaw_deg"])
        else:
            assert move["commanded"] == pytest.approx(measurement["move_m"])


def test_the_moves_follow_the_field_sign_convention(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    quick: None,
) -> None:
    # +4 deg reads as "turned left": turn right by 4. +2 cm reads as "farther":
    # drive forward by 2 cm.
    _script(
        monkeypatch,
        [_measurement(4.0), _measurement(0.0, move=0.02), _measurement(0.0)],
    )
    context, stage = _direct(tmp_path)
    result = _run(context, stage)
    assert result.terminated_by == "converged", result.reason
    x, _, heading = context.robot.base.true_pose()
    assert math.degrees(heading) == pytest.approx(-4.0, abs=0.2)
    assert x == pytest.approx(0.02 * math.cos(math.radians(-4.0)), abs=0.003)


def test_the_small_turn_rate_floor_is_off_by_default_and_floors_when_on(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    quick: None,
) -> None:
    _script(monkeypatch, [_measurement(3.0), _measurement(0.0)])
    context, stage = _direct(tmp_path)
    _run(context, stage)
    assert _align_event(tmp_path)["moves"][0]["min_rate_ticks"] == 0

    floored = tmp_path / "floored"
    floored.mkdir()
    _script(monkeypatch, [_measurement(3.0), _measurement(0.0)])
    context, stage = _direct(floored, min_turn_rate_rad_s=0.9)
    _run(context, stage)
    move = _align_event(floored)["moves"][0]
    assert move["min_rate_ticks"] > 0
    floor_rows = [
        row
        for row in _csv(floored)
        if row["state"] == "moving" and abs(float(row["cmd_theta_vel"])) == 0.9
    ]
    assert len(floor_rows) == move["min_rate_ticks"]


# --------------------------------------------------------------------------
# The worker thread: hold frames while measuring, Esc does not wait.


def test_hold_frames_are_recorded_while_the_worker_measures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = landmark_align.measure
    threads: list[str] = []

    def slow(reference, live, params, save_dir, step):
        threads.append(threading.current_thread().name)
        time.sleep(0.4)
        return real(reference, live, params, save_dir, step)

    monkeypatch.setattr(landmark_align, "measure", slow)
    context, stage = _direct(tmp_path, yaw_deg=4.0)
    result = _run(context, stage)
    assert result.terminated_by == "converged", result.reason
    assert all(name.startswith("align-s1_align") for name in threads)

    rows = _csv(tmp_path)
    names = list(context.dataset.features["action"]["names"])
    x_index, theta_index = names.index("x.vel"), names.index("theta.vel")
    actions = context.dataset.episode_buffer["action"]
    measuring = [i for i, row in enumerate(rows) if row["state"] == "measuring"]
    per_step: dict[str, list[int]] = {}
    for i in measuring:
        per_step.setdefault(rows[i]["step"], []).append(i)
    for frames in per_step.values():
        assert len(frames) >= 0.4 * FPS * 0.7, "측정 중에도 프레임이 fps로 쌓인다"
    for i in measuring:
        assert float(actions[i][x_index]) == 0.0
        assert float(actions[i][theta_index]) == 0.0
    assert list(context.dataset.episode_buffer["frame_index"]) == list(
        range(result.frames)
    )
    # The frame captured is the one recorded on the capture tick: that row is
    # still wait_capture and the very next one is the step's first measuring
    # row. Only the second check pins the index -- the ticks before a capture
    # are wait_capture too, so an off-by-one index would pass the first alone.
    for measurement in _align_event(tmp_path)["measurements"]:
        capture = measurement["capture_frame_idx"]
        assert rows[capture]["state"] == "wait_capture", (
            "the frame captured is recorded"
        )
        assert rows[capture + 1]["state"] == "measuring"
        assert capture + 1 == per_step[rows[capture + 1]["step"]][0]


def test_esc_while_measuring_aborts_without_waiting_for_the_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context, stage = _direct(tmp_path)

    def press_esc(step: int) -> None:
        # Pressed while the measurement is still running: a loop that blocked
        # on the worker would only see it 1.0 s later.
        time.sleep(0.5)
        context.events.update(stop_recording=True, exit_early=True)
        time.sleep(1.0)

    _script(monkeypatch, [_measurement(0.0)], before=press_esc)
    started = time.perf_counter()
    result = _run(context, stage)
    assert result.terminated_by == "stop_recording"
    event = _align_event(tmp_path)
    assert event["outcome"] == "aborted"
    assert event["measurements"] == [], "the dropped measurement is not recorded"
    assert result.elapsed_s < 1.5
    assert time.perf_counter() - started < 5.0


def test_right_arrow_accepts_the_base_where_it_is(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    quick: None,
) -> None:
    context, stage = _direct(tmp_path)

    def press_right(step: int) -> None:
        if step == 1:
            context.events["exit_early"] = True

    _script(monkeypatch, [_measurement(5.0), _measurement(3.0)], before=press_right)
    result = _run(context, stage)
    assert result.terminated_by == "manual"
    event = _align_event(tmp_path)
    assert event["outcome"] == "manual"
    assert context.events["exit_early"] is False


# --------------------------------------------------------------------------
# Guards.


def test_a_base_that_never_reads_still_is_not_still(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(base_stages, "CAPTURE_STILL_MAX_S", 0.5)
    context, stage = _direct(tmp_path)
    context.robot.base.velocity_override = [0.02, 0.0]
    result = _run(context, stage)
    assert result.terminated_by == "base_failed"
    event = _align_event(tmp_path)
    assert event["outcome"] == "not_still", event["reason"]
    assert event["measurements"] == []


@pytest.mark.parametrize(
    ("hook", "outcome"),
    [
        pytest.param({"refuse_cmds": 20}, "base_refused", id="refused"),
        pytest.param({"fail_updates": 20}, "no_odometry", id="odometry"),
        pytest.param({"ignore_cmds": True}, "move_failed", id="stalled"),
    ],
)
def test_a_fault_during_a_move_ends_the_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    quick: None,
    hook: dict[str, Any],
    outcome: str,
) -> None:
    context, stage = _direct(tmp_path)
    slate = context.robot.base

    def fault(step: int) -> None:
        # From the next tick on: the decision starts the turn into the fault.
        _on_update(slate, 1, lambda: [setattr(slate, k, v) for k, v in hook.items()])

    _script(monkeypatch, [_measurement(8.0), _measurement(0.0)], before=fault)
    result = _run(context, stage)
    assert result.terminated_by == "base_failed"
    event = _align_event(tmp_path)
    assert event["outcome"] == outcome, event["reason"]
    if outcome == "move_failed":
        assert "did not move" in event["reason"]
        assert event["moves"][0]["ratio"] < base_stages.MIN_MOVE_FRACTION
    else:
        # The cut move is still recorded, with what cut it.
        assert event["moves"][-1]["ended_by"] == outcome


def test_a_base_that_stalls_past_the_tracking_limit_is_tracking_lost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The MOVING tracking_lost guard itself, not move_failed after the hold.

    The stalled case above turns 8 deg, which never exceeds
    MOVE_TRACK_ERROR_DEG (15), so it ends as move_failed once the hold is
    over. Lowering the limit to 2 deg makes a base that ignores every command
    fall behind the ramp by more than the limit for TRACK_ERROR_HOLD_S (0.5 s)
    well before the 1 s hold after the ramp ends, so only the guard can end
    this stage.
    """
    monkeypatch.setattr(base_stages, "MOVE_TRACK_ERROR_DEG", 2.0)
    context, stage = _direct(tmp_path)
    slate = context.robot.base

    def stall(step: int) -> None:
        _on_update(slate, 1, lambda: setattr(slate, "ignore_cmds", True))

    _script(monkeypatch, [_measurement(8.0), _measurement(0.0)], before=stall)
    result = _run(context, stage)
    assert result.terminated_by == "base_failed"
    event = _align_event(tmp_path)
    assert event["outcome"] == "tracking_lost", event["reason"]
    assert event["moves"][-1]["ended_by"] == "tracking_lost"
    assert "tracking error" in event["reason"]
    assert event["moves"][-1]["max_tracking_error_deg"] > 2.0


def test_an_estop_while_holding_is_base_estop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context, stage = _direct(tmp_path)
    robot = context.robot

    def estop(step: int) -> None:
        robot._last_base_state = SimpleNamespace(name="SYS_ESTOP")

    _script(monkeypatch, [_measurement(0.0)], before=estop)
    monkeypatch.setattr(
        landmark_align,
        "measure",
        _slow(landmark_align.measure, 0.3),
    )
    result = _run(context, stage)
    assert result.terminated_by == "base_failed"
    event = _align_event(tmp_path)
    assert event["outcome"] == "base_estop"
    assert event["base_state_last"] == "SYS_ESTOP"


def _slow(measure: Any, seconds: float) -> Any:
    def run(*args: Any) -> Any:
        result = measure(*args)
        time.sleep(seconds)
        return result

    return run


@pytest.mark.parametrize(
    ("track_error_deg", "outcomes"),
    [
        # The P term chases a target the base runs away from: tracking_lost
        # (or, if the hold ends first, move_failed) -- never a third move.
        pytest.param(None, {"tracking_lost", "move_failed"}, id="guarded"),
        pytest.param(1e9, {"move_failed"}, id="signed-ratio"),
    ],
)
def test_a_base_that_turns_the_wrong_way_is_stopped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    quick: None,
    track_error_deg: float | None,
    outcomes: set[str],
) -> None:
    if track_error_deg is not None:
        monkeypatch.setattr(base_stages, "MOVE_TRACK_ERROR_DEG", track_error_deg)
    _script(monkeypatch, [_measurement(4.0), _measurement(0.0)])
    context, stage = _direct(tmp_path)
    context.robot.base.command_gain = -1.0
    result = _run(context, stage)
    assert result.terminated_by == "base_failed"
    event = _align_event(tmp_path)
    assert event["outcome"] in outcomes, event["reason"]
    if track_error_deg is not None and event["outcome"] == "move_failed":
        assert "moved opposite to command" in event["reason"]
    if track_error_deg == 1e9:
        assert "moved opposite to command" in event["reason"]
        assert event["moves"][0]["ratio"] < 0.0


def test_a_tiny_coarse_turn_is_logged_not_judged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    quick: None,
) -> None:
    # A coarse estimate of 0.3 deg still turns (field rule). Below
    # MOVE_JUDGE_MIN_RAD the achieved fraction is noise and is not judged.
    _script(monkeypatch, [_measurement(0.3, mode="coarse"), _measurement(0.1)])
    context, stage = _direct(tmp_path)
    context.robot.base.ignore_cmds = True
    result = _run(context, stage)
    assert result.terminated_by == "converged", result.reason
    (move,) = _align_event(tmp_path)["moves"]
    assert move["judged"] is False


def test_the_stage_ceiling_ends_an_align_that_runs_too_long(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context, stage = _direct(tmp_path, timeout_s=0.6)
    monkeypatch.setattr(landmark_align, "measure", _slow(landmark_align.measure, 1.0))
    result = _run(context, stage)
    assert result.terminated_by == "base_failed"
    assert _align_event(tmp_path)["outcome"] == "stage_ceiling"


# --------------------------------------------------------------------------
# Intrinsics.


def test_no_intrinsics_source_is_intrinsics_unavailable(tmp_path: Path) -> None:
    # The mock camera is not a RealSense; preflight refuses this YAML, so the
    # stage's own check is what a direct caller meets.
    context, stage = _direct(tmp_path, camera_matrix=None, dist_coeffs=None)
    result = _run(context, stage)
    assert result.terminated_by == "base_failed"
    event = _align_event(tmp_path)
    assert event["outcome"] == "intrinsics_unavailable"
    assert result.frames == 0, "nothing moves and nothing is recorded"


@pytest.mark.parametrize(
    ("size", "outcome"),
    [((640, 480), "aligned"), ((848, 480), "intrinsics_unavailable")],
)
def test_device_intrinsics_are_used_only_for_the_reference_size(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    size: tuple[int, int],
    outcome: str,
) -> None:
    def read(camera: Any) -> dict[str, Any]:
        return {
            "camera_matrix": K,
            "dist_coeffs": [0.0] * 5,
            "model": "distortion.inverse_brown_conrady",
            "width": size[0],
            "height": size[1],
        }

    monkeypatch.setattr(base_stages, "device_intrinsics", read)
    context, stage = _direct(tmp_path, camera_matrix=None, dist_coeffs=None)
    _run(context, stage)
    event = _align_event(tmp_path)
    assert event["outcome"] == outcome, event["reason"]
    if outcome == "aligned":
        assert event["intrinsics"]["source"] == "device"
        assert event["field_intrinsics_delta"]["max_abs_camera_matrix"] == 0.0


def test_an_roi_other_than_the_field_box_is_logged(tmp_path: Path) -> None:
    context, stage = _direct(tmp_path, landmark_roi=[280, 190, 100, 70])
    _run(context, stage)
    event = _align_event(tmp_path)
    assert event["landmark_roi"] == [280, 190, 100, 70]
    assert event["auto_roi"] == ROI


# --------------------------------------------------------------------------
# Exception path.


def test_an_exception_mid_align_zeroes_the_base_and_propagates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    quick: None,
) -> None:
    _script(monkeypatch, [_measurement(8.0), _measurement(0.0)])
    context, stage = _direct(tmp_path)
    robot = context.robot
    slate = robot.base
    real_send = robot.send_action
    sent: list[bool] = []
    stopped: list[BaseException] = []
    monkeypatch.setattr(
        base_stages, "stop_image_writer_after", lambda e, d: stopped.append(e)
    )

    def send_action(action: dict[str, float]) -> dict[str, float]:
        sent.append(True)
        if len(sent) == 20:
            raise OSError("arm bus gone")
        return real_send(action)

    robot.send_action = send_action
    with pytest.raises(OSError, match="arm bus gone") as raised:
        base_stages.align(context, stage)
    assert robot.base is slate
    assert slate.commands[-1][:2] == (0.0, 0.0)
    assert stopped == [raised.value]
    assert _align_event(tmp_path)["outcome"] == "interrupted"
    assert _align_event(tmp_path)["terminator"] == "error"
    assert context.buffered_frame_count() == 19, "the failed tick records nothing"
    assert _no_align_thread()


def test_a_worker_exception_surfaces_on_the_main_thread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*args: Any) -> Any:
        raise ValueError("sift exploded")

    monkeypatch.setattr(landmark_align, "measure", broken)
    context, stage = _direct(tmp_path)
    slate = context.robot.base
    with pytest.raises(ValueError, match="sift exploded"):
        base_stages.align(context, stage)
    assert context.robot.base is slate
    assert not isinstance(context.robot.base, BaseProbe)
    assert _align_event(tmp_path)["outcome"] == "interrupted"


# --------------------------------------------------------------------------
# CLI: the whole chain, two move bundles in one process (A5).

CHAIN = """\
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
    terminator: {type: timeout, timeout_s: 10}
    replay: {dataset: local/pose, root: POSE, episode: 0, until_row: 29}
  - id: s1_align
    name: align to the rack
    executor: base_align
    terminator: {type: timeout, timeout_s: 60}
    align:
      reference_image: RACK_PNG
      landmark_roi: RACK_ROI
      landmark_distance_m: RACK_DISTANCE
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
    terminator: {type: timeout, timeout_s: 10}
    replay: {dataset: local/pose, root: POSE, episode: 1, until_row: 29}
  - id: s3_align
    name: align to the shelf
    executor: base_align
    terminator: {type: timeout, timeout_s: 60}
    align:
      reference_image: SHELF_PNG
      landmark_roi: SHELF_ROI
      landmark_distance_m: SHELF_DISTANCE
      yaw_tol_deg: 1.5
      camera_matrix: [[392.6346, 0, 322.7227], [0, 392.0662, 236.2555], [0, 0, 1]]
      dist_coeffs: [0, 0, 0, 0, 0]
  - id: s4
    name: mock_stage_four
    policy_path: mock://hold
    instruction: place the tube
    terminator: {type: timeout, timeout_s: 0.3}
"""

# The mock scene: the landmark sits 0.60 m straight ahead of the pose the
# robot starts at (scene_reference_pose = 0). A reference image rendered at
# (yaw, scale) is the view from heading `yaw` at distance 0.60 / scale.
# Bundle 1: the demo path drives 3 cm and turns +9 deg; its arrival view is at
# +3 deg, 0.555 m from the rack. Bundle 2: the path drives back 6 cm and turns
# -15 deg from wherever align 1 left the base; its arrival is at -8 deg, 0.62 m.
BUNDLES = {
    "RACK": (3.0, 0.555),
    "SHELF": (-8.0, 0.62),
}


def _write_two_bundles(tmp_path: Path, **overrides: str) -> Path:
    pose = _write_pose_dataset(
        tmp_path / "pose",
        {
            0: [[0.0] * 14 + [s, t] for s, t in _path(30, 0.001, 0.3)],
            1: [[0.0] * 14 + [s, t] for s, t in _path(30, -0.002, -0.5)],
        },
    )
    text = CHAIN.replace("POSE", str(pose))
    for name, (yaw, distance) in BUNDLES.items():
        image = mock_scene.render_scene(
            view_yaw_deg=yaw, landmark_scale=0.60 / distance
        )
        path = tmp_path / f"{name.lower()}.png"
        assert cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        roi = landmark_align.dark_blob_box(image)
        assert roi is not None
        text = (
            text.replace(f"{name}_PNG", str(path))
            .replace(f"{name}_ROI", json.dumps(list(roi)))
            .replace(f"{name}_DISTANCE", str(distance))
        )
    for old, new in overrides.items():
        text = text.replace(old, new)
    path = tmp_path / "chain.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def _landmark_view(slate: MockSlate) -> tuple[float, float]:
    """(heading deg, distance to the landmark m) of the simulated base."""
    x, y, heading = slate.true_pose()
    return math.degrees(heading), math.hypot(0.60 - x, y)


def test_two_move_bundles_run_end_to_end_in_one_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    robots: list[Any] = []
    views: dict[str, tuple[float, float]] = {}
    real_emit = EventLog.emit

    def emit(self: EventLog, event: str, **fields: Any) -> dict[str, Any]:
        # Where each align left the base, read when its stage_end is written
        # (after the boundary stop).
        if event == "stage_end" and fields.get("stage_id", "").endswith("_align"):
            views[fields["stage_id"]] = _landmark_view(robots[0].base)
        return real_emit(self, event, **fields)

    monkeypatch.setattr(EventLog, "emit", emit)
    result = _cli(tmp_path, _write_two_bundles(tmp_path), robots.append)

    assert result["exit_code"] == cli.EXIT_OK
    assert [e["stage_id"] for e in _of(result, "stage_start")] == [
        "s1",
        "s1_align",
        "s2",
        "s3",
        "s3_align",
        "s4",
    ]
    _assert_log_is_closed(result)
    starts = {e["stage_id"]: e for e in _of(result, "stage_start")}
    for end in _of(result, "stage_end"):
        assert end["frames"] == end["frame_idx"] - starts[end["stage_id"]]["frame_idx"]

    replays = {e["stage_id"]: e for e in _of(result, "stage_base_replay")}
    aligns = {e["stage_id"]: e for e in _of(result, "stage_base_align")}
    assert {e["outcome"] for e in replays.values()} == {"settled"}
    assert [aligns[s]["outcome"] for s in ("s1_align", "s3_align")] == [
        "aligned",
        "aligned",
    ]
    # Each align measured against ITS reference with ITS ROI and distance.
    assert aligns["s1_align"]["reference"]["path"].endswith("rack.png")
    assert aligns["s3_align"]["reference"]["path"].endswith("shelf.png")
    assert aligns["s1_align"]["landmark_distance_m"] == 0.555
    assert aligns["s3_align"]["landmark_distance_m"] == 0.62
    assert aligns["s3_align"]["yaw_tol_deg"] == 1.5
    for stage_id in ("s1_align", "s3_align"):
        assert aligns[stage_id]["move_count"] >= 1
        assert aligns[stage_id]["gripper_hold_source"] == "recorded"
    # Each align left the base at its own demo arrival.
    for stage_id, name in (("s1_align", "RACK"), ("s3_align", "SHELF")):
        heading, distance = views[stage_id]
        yaw, target = BUNDLES[name]
        assert heading == pytest.approx(yaw, abs=2.0), stage_id
        assert distance == pytest.approx(target, abs=0.015), stage_id
    # The second replay anchored where the first align stopped.
    assert replays["s3"]["anchor_odom"] != replays["s1"]["anchor_odom"]
    run_directory = result["run_directory"]
    for stage_id in ("s1_align", "s3_align"):
        assert (run_directory / "base" / f"{stage_id}_align.csv").is_file()
        assert (run_directory / "base" / stage_id / "reference.png").is_file()
    assert result["robot"].base is result["slate"]
    assert _no_align_thread()


def test_a_failed_align_measurement_goes_on_to_the_policy(tmp_path: Path) -> None:
    # The first align cannot read a heading at all (coarse off, 25 deg away).
    # A measurement failure stops the base where it is and the chain goes on
    # (user decision 2026-10-25); the outcome stays in stage_base_align.
    config_path = _write_two_bundles(
        tmp_path,
        **{
            "      dist_coeffs: [0, 0, 0, 0, 0]\n  - id: s2": (
                "      dist_coeffs: [0, 0, 0, 0, 0]\n      coarse_heading: false\n"
                "  - id: s2"
            )
        },
    )

    def instrument(robot: Any) -> None:
        robot.base.place(0.0, 0.0, math.radians(16.0))

    result = _cli(tmp_path, config_path, instrument)
    assert result["exit_code"] == cli.EXIT_OK
    starts = [e["stage_id"] for e in _of(result, "stage_start")]
    assert starts == ["s1", "s1_align", "s2", "s3", "s3_align", "s4"]
    event = _of(result, "stage_base_align")[0]
    assert event["stage_id"] == "s1_align"
    assert event["outcome"] == "estimate_unusable", event["reason"]
    assert event["coarse_heading"] is False
    assert len(event["measurements"]) == 1
    (align_end,) = (e for e in _of(result, "stage_end") if e["stage_id"] == "s1_align")
    assert align_end["terminator"] == "converged"
    _assert_log_is_closed(result)
    assert isinstance(result["robot"].base, MockSlate)
