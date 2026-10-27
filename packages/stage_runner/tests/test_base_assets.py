"""base_assets: the replay targets and align references every base stage needs, read before connect.

Everything here runs offline against synthetic datasets in tmp_path:

* a POSE dataset written as raw parquet (meta/info.json + episodes split over two
  files), the shape derive_pose.py leaves on the robot PC;
* reference datasets written by LeRobotDataset.create, one with the camera as an
  IMAGE feature and one as a VIDEO feature, so the frame lerobot itself would
  return is the yardstick;
* a reference PNG.

And one check against the real 10/24 demo copy, skipped unless
STAGE_RUNNER_DEMO_ROOT is set: the loader returns the frame the parity test
(and the field script) decoded.

What matters on the robot: nothing is downloaded, a wrong dataset is refused
with the fix before the robot is powered, the selected frame is the one the
field aligned against, and any number of base stages -- each with its own
dataset, rows and reference -- load in one pass.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from stage_runner import base_assets, base_stages, config, landmark_align, mock_scene
from stage_runner.config import (
    BaseAlignConfig,
    BaseReplayConfig,
    StageConfig,
    TerminatorConfig,
)
from stage_runner.mock_robot import MockRobot, MockRobotConfig
from stage_runner.preflight import PreflightError

CAMERA_KEY = "observation.images.cam_high"
ARM = [f"joint_{index}.pos" for index in range(14)]
POSE_NAMES = [*ARM, "x.pos", "theta.pos"]
VELOCITY_NAMES = [*ARM, "x.vel", "theta.vel"]


def _pose_rows(episode: int, length: int) -> list[list[float]]:
    """A cumulative path: 1 cm and (1 + episode) mrad per row, arms at the row index."""
    return [
        [float(row)] * len(ARM) + [0.01 * (row + 1), 0.001 * (episode + 1) * (row + 1)]
        for row in range(length)
    ]


def _write_pose_dataset(
    root: Path,
    episodes: dict[int, list[list[float]]],
    names: list[str] = POSE_NAMES,
    *,
    files: int = 2,
) -> Path:
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v3.0",
                "total_episodes": len(episodes),
                "features": {
                    "action": {
                        "dtype": "float32",
                        "shape": [len(names)],
                        "names": names,
                    }
                },
            }
        )
    )
    chunk = root / "data" / "chunk-000"
    chunk.mkdir(parents=True)
    items = sorted(episodes.items())
    for file_index in range(files):
        episode_column, frame_column, action_column = [], [], []
        for episode, rows in items[file_index::files]:
            # Written in reverse frame order: the loader must sort, not trust order.
            for frame_index in reversed(range(len(rows))):
                episode_column.append(episode)
                frame_column.append(frame_index)
                action_column.append(rows[frame_index])
        table = pa.table(
            {
                "episode_index": pa.array(episode_column, pa.int64()),
                "frame_index": pa.array(frame_column, pa.int64()),
                "action": pa.array(action_column, pa.list_(pa.float32())),
            }
        )
        pq.write_table(table, chunk / f"file-{file_index:03d}.parquet")
    return root


@pytest.fixture
def pose_root(tmp_path: Path) -> Path:
    return _write_pose_dataset(
        tmp_path / "pose", {0: _pose_rows(0, 40), 1: _pose_rows(1, 30)}
    )


def _replay(root: Path, **overrides) -> BaseReplayConfig:
    values = {"dataset": "local/pose", "root": str(root), "episode": 0, "until_row": 20}
    values.update(overrides)
    return BaseReplayConfig(**values)


# --------------------------------------------------------------------------
# Replay targets.


def test_raw_rows_are_the_targets_from_row_zero(pose_root: Path) -> None:
    targets = base_assets.load_replay_targets(_replay(pose_root, until_row=20))
    assert targets.rows == 21
    expected = np.array(_pose_rows(0, 40), dtype=np.float32)[:21]
    np.testing.assert_array_equal(targets.s_m, expected[:, -2].astype(np.float64))
    np.testing.assert_array_equal(targets.theta_rad, expected[:, -1].astype(np.float64))
    assert targets.s_m.dtype == np.float64
    assert targets.source["rebased_on_row"] is None
    assert targets.source["episode_rows"] == 40
    assert targets.source["final_target_s_m"] == pytest.approx(0.21)
    with pytest.raises(ValueError):
        targets.s_m[0] = 1.0


def test_a_later_piece_is_rebased_on_the_row_before_it(pose_root: Path) -> None:
    targets = base_assets.load_replay_targets(
        _replay(pose_root, episode=1, from_row=10, until_row=29)
    )
    assert targets.rows == 20
    # Row 9 of episode 1 is s = 0.10 m, theta = 0.020 rad: the piece starts at 0.
    np.testing.assert_allclose(targets.s_m, 0.01 * np.arange(1, 21), atol=1e-6)
    np.testing.assert_allclose(targets.theta_rad, 0.002 * np.arange(1, 21), atol=1e-6)
    assert targets.source["rebased_on_row"] == 9


def test_a_missing_dataset_is_refused_with_the_fix(tmp_path: Path) -> None:
    with pytest.raises(base_assets._AssetProblem) as caught:
        base_assets.load_replay_targets(_replay(tmp_path / "absent"))
    message = str(caught.value)
    assert str(tmp_path / "absent") in message
    assert "never downloads" in message
    assert "derive_pose.py" in message and "replay.root" in message


def test_the_root_defaults_to_the_lerobot_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(base_assets, "HF_LEROBOT_HOME", Path("/cache"))
    assert base_assets.dataset_root("org/name", None) == Path("/cache/org/name")
    assert base_assets.dataset_root("org/name", "/x") == Path("/x")


def test_a_velocity_dataset_is_refused_naming_derive_pose(tmp_path: Path) -> None:
    root = _write_pose_dataset(tmp_path / "vel", {0: _pose_rows(0, 10)}, VELOCITY_NAMES)
    with pytest.raises(
        base_assets._AssetProblem, match="velocity dataset.*derive_pose"
    ):
        base_assets.load_replay_targets(_replay(root, until_row=5))


@pytest.mark.parametrize(
    ("overrides", "needle"),
    [
        ({"episode": 7}, "episode 7 is not in"),
        ({"until_row": 40}, "past the end of episode 0.*40 rows"),
        ({"from_row": 20, "until_row": 20}, "from_row < until_row"),
    ],
)
def test_rows_outside_the_episode_are_refused(
    pose_root: Path, overrides: dict, needle: str
) -> None:
    with pytest.raises(base_assets._AssetProblem, match=needle):
        base_assets.load_replay_targets(_replay(pose_root, **overrides))


def test_non_finite_targets_and_gaps_are_refused(tmp_path: Path) -> None:
    rows = _pose_rows(0, 10)
    rows[4][-1] = float("nan")
    root = _write_pose_dataset(tmp_path / "nan", {0: rows}, files=1)
    with pytest.raises(base_assets._AssetProblem, match="non-finite"):
        base_assets.load_replay_targets(_replay(root, until_row=8))
    # A non-finite value OUTSIDE the replayed rows is no reason to refuse.
    assert base_assets.load_replay_targets(_replay(root, until_row=3)).rows == 4

    gappy = _write_pose_dataset(tmp_path / "gap", {0: _pose_rows(0, 10)}, files=1)
    path = next((gappy / "data").rglob("*.parquet"))
    table = pq.read_table(path)
    pq.write_table(table.filter(pa.compute.not_equal(table["frame_index"], 3)), path)
    with pytest.raises(base_assets._AssetProblem, match="not 0..8"):
        base_assets.load_replay_targets(_replay(gappy, until_row=5))


# --------------------------------------------------------------------------
# Align references.


def _write_reference_dataset(root: Path, kind: str) -> tuple[Path, list[np.ndarray]]:
    """Two episodes of 12 random 64x48 frames, the camera stored as `kind`."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset.create(
        repo_id=f"local/reference_{kind}",
        fps=30,
        root=root,
        use_videos=kind == "video",
        features={
            CAMERA_KEY: {
                "dtype": kind,
                "shape": (48, 64, 3),
                "names": ["height", "width", "channels"],
            },
            "action": {
                "dtype": "float32",
                "shape": (2,),
                "names": ["x.vel", "theta.vel"],
            },
        },
    )
    rng = np.random.default_rng(0)
    written: list[np.ndarray] = []
    for _ in range(2):
        for _ in range(12):
            image = rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)
            written.append(image)
            dataset.add_frame(
                {CAMERA_KEY: image, "action": np.zeros(2, np.float32), "task": "demo"}
            )
        dataset.save_episode()
    dataset.finalize()
    return root, written


def _lerobot_frame(root: Path, repo_id: str, index: int) -> np.ndarray:
    """What heading_align.demo_frame did: LeRobotDataset[index], x255, truncated."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    item = LeRobotDataset(repo_id, root=root)[index][CAMERA_KEY]
    return (item.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def _align(**overrides) -> BaseAlignConfig:
    values = {"landmark_roi": [10, 10, 20, 10], "landmark_distance_m": 0.6}
    values.update(overrides)
    return BaseAlignConfig(**values)


@pytest.mark.parametrize("kind", ["image", "video"])
def test_a_dataset_frame_is_the_one_lerobot_returns(tmp_path: Path, kind: str) -> None:
    root, written = _write_reference_dataset(tmp_path / kind, kind)
    # Episode 1 frame 5: its video timestamp is episode 1's from_timestamp (0.4 s
    # into the shared file) plus the row's own timestamp.
    reference = base_assets.load_align_reference(
        _align(
            reference_dataset=f"local/reference_{kind}",
            reference_root=str(root),
            episode=1,
            frame=5,
        )
    )
    expected = _lerobot_frame(root, f"local/reference_{kind}", 12 + 5)
    np.testing.assert_array_equal(reference.image, expected)
    assert reference.image.flags.c_contiguous and not reference.image.flags.writeable
    assert reference.source["kind"] == "dataset"
    assert reference.source["timestamp"] == pytest.approx(5 / 30)
    assert reference.source["shape"] == [48, 64, 3]
    if kind == "image":
        # PNG is lossless, so the stored frame comes back exactly.
        np.testing.assert_array_equal(reference.image, written[12 + 5])
    else:
        assert reference.source["from_timestamp"] == pytest.approx(12 / 30)
        assert reference.source["video_timestamp"] == pytest.approx(17 / 30)


def test_a_reference_dataset_without_the_frame_or_camera_is_refused(
    tmp_path: Path,
) -> None:
    root, _ = _write_reference_dataset(tmp_path / "image", "image")
    common = {"reference_dataset": "local/reference_image", "reference_root": str(root)}
    with pytest.raises(base_assets._AssetProblem, match="frame: 12.*12 frames"):
        base_assets.load_align_reference(_align(**common, episode=0, frame=12))
    with pytest.raises(base_assets._AssetProblem, match="episode: 2.*2 episodes"):
        base_assets.load_align_reference(_align(**common, episode=2, frame=0))
    with pytest.raises(
        base_assets._AssetProblem, match="no observation.images.cam_left"
    ):
        base_assets.load_align_reference(_align(**common, camera="cam_left", frame=0))
    with pytest.raises(base_assets._AssetProblem, match="never downloads"):
        base_assets.load_align_reference(
            _align(
                reference_dataset="local/x",
                reference_root=str(tmp_path / "no"),
                frame=0,
            )
        )


def test_a_video_dataset_with_partial_metadata_is_refused_not_downloaded(
    tmp_path: Path,
) -> None:
    root, _ = _write_reference_dataset(tmp_path / "video", "video")
    (root / "meta" / "tasks.parquet").unlink()
    with pytest.raises(base_assets._AssetProblem, match="tasks.parquet.*never does"):
        base_assets.load_align_reference(
            _align(
                reference_dataset="local/reference_video",
                reference_root=str(root),
                frame=0,
            )
        )


def test_a_reference_image_is_read_as_rgb(tmp_path: Path) -> None:
    rgb = np.random.default_rng(1).integers(0, 255, (48, 64, 3), dtype=np.uint8)
    path = tmp_path / "reference.png"
    assert cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    reference = base_assets.load_align_reference(_align(reference_image=str(path)))
    np.testing.assert_array_equal(reference.image, rgb)
    assert reference.source == {
        "kind": "image",
        "path": str(path),
        "shape": [48, 64, 3],
    }
    with pytest.raises(base_assets._AssetProblem, match="cannot read"):
        base_assets.load_align_reference(
            _align(reference_image=str(tmp_path / "x.png"))
        )


def test_the_read_only_reference_goes_through_the_estimator() -> None:
    reference = base_assets._owned_rgb(mock_scene.render_scene())
    params = landmark_align.AlignParams(
        camera_matrix=np.array(mock_scene.DEFAULT_CAMERA_MATRIX),
        dist_coeffs=np.zeros(5),
        up=np.array(mock_scene.DEFAULT_UP) / np.linalg.norm(mock_scene.DEFAULT_UP),
        landmark_roi=mock_scene.DEFAULT_LANDMARK_ROI,
        landmark_distance_m=0.6,
    )
    measurement = landmark_align.measure(
        reference, mock_scene.render_scene(view_yaw_deg=3.0), params, None, 0
    )
    assert measurement.heading_usable


# --------------------------------------------------------------------------
# load_base_assets: the whole chain, against the robot.


def _robot() -> MockRobot:
    return MockRobot(
        MockRobotConfig(
            id="unit",
            arm_layout="bimanual",
            simulate_base=True,
            scene_camera="cam_high",
        )
    )


def _chain(*stages: StageConfig) -> config.StageRunnerConfig:
    return config.StageRunnerConfig(
        robot=MockRobotConfig(id="unit"), stages=list(stages)
    )


def _base_stage(
    stage_id: str, executor: str, timeout_s: float = 60.0, **blocks
) -> StageConfig:
    return StageConfig(
        id=stage_id,
        name=f"{stage_id} base",
        executor=executor,
        terminator=TerminatorConfig(type="timeout", timeout_s=timeout_s),
        **blocks,
    )


def _reference_png(tmp_path: Path, name: str, yaw: float) -> Path:
    path = tmp_path / name
    image = mock_scene.render_scene(view_yaw_deg=yaw)
    assert cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return path


def test_a_chain_without_base_stages_loads_nothing() -> None:
    policy = StageConfig(id="s1", policy_path="mock://hold")
    assert base_assets.load_base_assets(_chain(policy), _robot()) == {}


def test_two_move_bundles_load_their_own_sources_in_one_pass(
    tmp_path: Path, pose_root: Path
) -> None:
    other = _write_pose_dataset(tmp_path / "pose2", {0: _pose_rows(4, 50)}, files=1)
    video_root, _ = _write_reference_dataset(tmp_path / "video", "video")
    first_png = _reference_png(tmp_path, "first.png", 0.0)
    chain = _chain(
        _base_stage("s1", "base_replay", replay=_replay(pose_root, until_row=20)),
        _base_stage(
            "s1_align",
            "base_align",
            align=_align(
                reference_image=str(first_png), landmark_roi=[285, 197, 93, 61]
            ),
        ),
        StageConfig(id="s2", policy_path="mock://hold"),
        _base_stage(
            "s3", "base_replay", replay=_replay(other, from_row=5, until_row=49)
        ),
        _base_stage(
            "s3_again",
            "base_replay",
            replay=_replay(pose_root, episode=1, from_row=3, until_row=9),
        ),
        _base_stage(
            "s3_align",
            "base_align",
            align=_align(
                reference_dataset="local/reference_video",
                reference_root=str(video_root),
                episode=1,
                frame=3,
            ),
        ),
    )
    robot = _robot()
    # The video reference is 64x48 and the scene camera 640x480: refused, and
    # the refusal names the one stage it belongs to.
    with pytest.raises(PreflightError) as caught:
        base_assets.load_base_assets(chain, robot)
    lines = str(caught.value).splitlines()[1:]
    assert len(lines) == 1 and "'s3_align'" in lines[0] and "64x48" in lines[0]
    assert base_assets.load_align_reference(chain.stages[-1].align).image.shape == (
        48,
        64,
        3,
    )

    second_png = _reference_png(tmp_path, "second.png", 4.0)
    chain.stages[-1].align = _align(reference_image=str(second_png))
    assets = base_assets.load_base_assets(chain, robot)
    assert set(assets) == {"s1", "s1_align", "s3", "s3_again", "s3_align"}
    assert (
        assets["s1"].rows == 21
        and assets["s3"].rows == 45
        and assets["s3_again"].rows == 7
    )
    assert assets["s3"].source["root"] == str(other)
    assert assets["s3_again"].source["episode"] == 1
    assert assets["s1_align"].source["path"] == str(first_png)
    assert assets["s3_align"].source["path"] == str(second_png)
    assert not np.array_equal(assets["s1_align"].image, assets["s3_align"].image)


def test_every_problem_of_the_chain_is_reported_at_once(
    tmp_path: Path, pose_root: Path
) -> None:
    chain = _chain(
        _base_stage("s1", "base_replay", replay=_replay(tmp_path / "absent")),
        _base_stage(
            "s1_align",
            "base_align",
            align=_align(
                reference_image=str(_reference_png(tmp_path, "r.png", 0.0)),
                landmark_roi=[600, 400, 93, 61],
            ),
        ),
        _base_stage(
            "s3", "base_replay", timeout_s=5.0, replay=_replay(pose_root, until_row=39)
        ),
        _base_stage(
            "s3_align",
            "base_align",
            align=_align(reference_image=str(tmp_path / "r.png"), camera="cam_low"),
        ),
    )
    with pytest.raises(PreflightError) as caught:
        base_assets.load_base_assets(chain, _robot())
    message = str(caught.value)
    lines = message.splitlines()[1:]
    assert len(lines) == 4, message
    assert "'s1'" in lines[0] and "no dataset at" in lines[0]
    assert "'s1_align'" in lines[1] and "reaches outside the 640x480" in lines[1]
    # 40 rows at 1/21 s + 0.5 + 4 + 2 = 8.36 s -> at least 9.
    assert "'s3'" in lines[2] and "at least 9" in lines[2]
    assert "'s3_align'" in lines[3] and "no `cam_low` camera" in lines[3]


def test_the_replay_ceiling_floor_is_the_executors_own() -> None:
    needed = base_stages.replay_ceiling_floor_s(40, 1.0 / 21.0)
    targets = base_assets.ReplayTargets(np.zeros(40), np.zeros(40), {})
    stage = _base_stage("s", "base_replay", timeout_s=needed, replay=BaseReplayConfig())
    assert base_assets._ceiling_problems(stage, targets) == []
    stage.terminator.timeout_s = needed - 0.01
    assert base_assets._ceiling_problems(stage, targets)


def test_the_width_fallback_with_another_landmark_roi_is_refused(
    tmp_path: Path,
) -> None:
    png = _reference_png(tmp_path, "r.png", 0.0)
    rack = list(
        landmark_align.dark_blob_box(
            cv2.cvtColor(cv2.imread(str(png)), cv2.COLOR_BGR2RGB)
        )
    )
    on_rack = _align(
        reference_image=str(png), landmark_roi=rack, distance_width_fallback=True
    )
    stage = _base_stage("s1_align", "base_align", align=on_rack)
    assert set(base_assets.load_base_assets(_chain(stage), _robot())) == {"s1_align"}
    stage.align = _align(
        reference_image=str(png),
        landmark_roi=[60, 120, 85, 60],
        distance_width_fallback=True,
    )
    with pytest.raises(PreflightError) as caught:
        base_assets.load_base_assets(_chain(stage), _robot())
    lines = str(caught.value).splitlines()[1:]
    assert len(lines) == 1 and "'s1_align'" in lines[0]
    assert "distance_width_fallback" in lines[0] and str(rack) in lines[0]
    # Off, the same ROI is fine: the fallback is the only thing tied to it.
    stage.align = _align(reference_image=str(png), landmark_roi=[60, 120, 85, 60])
    assert set(base_assets.load_base_assets(_chain(stage), _robot())) == {"s1_align"}


def test_replay_target_limits_sit_below_the_tracker_clamp() -> None:
    # The refusal is only worth having if it trips before the tracker's own
    # saturation, and it must let the 10/24 replay (0.49 m/s, 0.57 rad/s) run.
    assert 0.49 < base_stages.REPLAY_TARGET_MAX_X_VEL_M_S < base_stages.REPLAY_MAX_X_VEL
    assert (
        0.57
        < base_stages.REPLAY_TARGET_MAX_THETA_VEL_RAD_S
        < base_stages.REPLAY_MAX_THETA_VEL
    )


@pytest.mark.parametrize(
    ("row_period_s", "per_row_m", "refused"),
    [
        # 1 cm per row at the field's 1/21 s: 0.21 m/s.
        pytest.param(1.0 / 21.0, 0.01, False, id="field-rate"),
        # The same rows with a typo in row_period_s: 1.0 m/s.
        pytest.param(0.01, 0.01, True, id="row-period-typo"),
        # A denser dataset at the field's rate: 3 cm per row, 0.63 m/s.
        pytest.param(1.0 / 21.0, 0.03, True, id="dense-rows"),
    ],
)
def test_a_replay_faster_than_the_robot_has_run_is_refused(
    tmp_path: Path, row_period_s: float, per_row_m: float, refused: bool
) -> None:
    rows = [
        [0.0] * len(ARM) + [per_row_m * (row + 1), 0.001 * (row + 1)]
        for row in range(40)
    ]
    root = _write_pose_dataset(tmp_path / "pose", {0: rows}, files=1)
    stage = _base_stage(
        "s1",
        "base_replay",
        replay=_replay(root, until_row=39, row_period_s=row_period_s),
    )
    if not refused:
        assert set(base_assets.load_base_assets(_chain(stage), _robot())) == {"s1"}
        return
    with pytest.raises(PreflightError) as caught:
        base_assets.load_base_assets(_chain(stage), _robot())
    message = str(caught.value)
    assert "'s1'" in message and "above the replay limits" in message
    # The period it names brings the peak back to the limit.
    fitting = float(message.rsplit("at least ", 1)[1].split(" ")[0])
    targets = base_assets.load_replay_targets(stage.replay)
    x_vel, theta_vel = base_assets.target_peak_rates(targets, fitting)
    assert x_vel <= base_stages.REPLAY_TARGET_MAX_X_VEL_M_S * 1.001
    assert theta_vel <= base_stages.REPLAY_TARGET_MAX_THETA_VEL_RAD_S * 1.001


# --------------------------------------------------------------------------
# The real demo, when a local copy is available.

DEMO_ROOT = os.environ.get("STAGE_RUNNER_DEMO_ROOT")


@pytest.mark.skipif(DEMO_ROOT is None, reason="STAGE_RUNNER_DEMO_ROOT not set")
def test_the_demo_arrival_frame_is_the_one_the_field_aligned_against() -> None:
    from test_landmark_align import ARRIVAL_FRAME, FIELD_ROI, _demo_frame

    root = Path(DEMO_ROOT)
    reference = base_assets.load_align_reference(
        BaseAlignConfig(
            reference_dataset="kiroaiseoul/test_replay_timing_1024",
            reference_root=str(root),
            episode=0,
            frame=ARRIVAL_FRAME[0],
        )
    )
    np.testing.assert_array_equal(
        reference.image, _demo_frame(root, 0, ARRIVAL_FRAME[0])
    )
    assert landmark_align.dark_blob_box(reference.image) == FIELD_ROI
    assert reference.source["video_timestamp"] == pytest.approx(302 / 30, abs=1e-3)
