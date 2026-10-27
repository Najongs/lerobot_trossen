"""The heading and distance estimates base_align acts on, against a synthetic scene.

landmark_align is a port of the field script that aligned stage 1 on the robot
on 2026-10-24 (heading_align.py); mock_scene renders a camera view whose yaw and
landmark scale are known exactly. No lerobot import outside the opt-in parity
test: these must run on any interpreter with cv2 and numpy.

What is asserted, and why each matters on the robot:

* the SIGN: a view turned left reads positive, and the correction is -yaw. A
  sign error turns the base the wrong way (the first 10/24 trial did, +12.98
  deg) -- the divergence guard is the backstop, this is the pin;
* the arms are masked: unmasked on a plain background they read as zero shift
  and the base would stop turning short;
* the usability limits (15 inliers, 20 deg, 0.10 m) and the coarse rack-blob
  fallback that takes over when the fine estimate is not usable -- and is off
  when asked to be;
* the distance sign and size: a landmark that looks bigger means the base is
  too close, so the move is backward;
* the same pair gives the same answer twice (the RANSAC is re-seeded), so a
  re-run of a measurement is not a different decision;
* the cv2 that lerobot's pin brings has SIFT, since stage_runner declares no
  OpenCV dependency of its own.

NOTHING HERE IS EVIDENCE ABOUT THE REAL CAMERA except the parity test, which
runs only with STAGE_RUNNER_DEMO_ROOT set to a local copy of the 10/24 demo
dataset (kiroaiseoul/test_replay_timing_1024) and checks this port reproduces
the field script's numbers on its frames.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import pytest
from stage_runner import landmark_align, mock_scene

K = np.array(mock_scene.DEFAULT_CAMERA_MATRIX)
ZERO_DIST = np.zeros(5)
UP = np.array(mock_scene.DEFAULT_UP) / np.linalg.norm(mock_scene.DEFAULT_UP)
ROI = mock_scene.DEFAULT_LANDMARK_ROI
DISTANCE_M = 0.60


def _params(**overrides) -> landmark_align.AlignParams:
    values = {
        "camera_matrix": K,
        "dist_coeffs": ZERO_DIST,
        "up": UP,
        "landmark_roi": ROI,
        "landmark_distance_m": DISTANCE_M,
    }
    values.update(overrides)
    return landmark_align.AlignParams(**values)


@pytest.fixture(scope="module")
def reference() -> np.ndarray:
    return mock_scene.render_scene()


def _no_mask(image: np.ndarray) -> np.ndarray:
    return np.full(image.shape[:2], 255, np.uint8)


# ---------------------------------------------------------------------------
# heading
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("yaw", [-12.0, -6.0, -2.0, 1.5, 6.0, 12.0])
def test_fine_yaw_recovers_the_turn_with_its_sign(reference, yaw):
    live = mock_scene.render_scene(view_yaw_deg=yaw)
    m = landmark_align.measure(reference, live, _params(), None, 0)
    assert m.yaw_deg == pytest.approx(yaw, abs=0.3), "요 추정 오차가 0.3°를 넘음"
    assert np.sign(m.yaw_deg) == np.sign(yaw), "요 부호가 반대 — 반대로 돈다"
    assert m.inliers >= landmark_align.MIN_INLIERS
    assert m.heading_valid and m.heading_mode == landmark_align.HEADING_FINE
    assert m.heading_yaw_deg == m.yaw_deg
    assert m.coarse_yaw_deg is None, (
        "fine이 쓸 만하면 coarse는 계산하지 않는다(현장과 같게)"
    )


def test_sign_convention_left_turn_is_positive_and_corrected_by_minus_yaw(reference):
    # Rendering a view turned LEFT (CCW from above) moves the scene RIGHT in
    # the image, as a camera turning left sees it; the estimate must call that
    # positive, the same sign as odometry theta, so turning by -yaw undoes it.
    live = mock_scene.render_scene(view_yaw_deg=5.0)
    box_ref, box_live = (
        landmark_align.dark_blob_box(reference),
        landmark_align.dark_blob_box(live),
    )
    assert box_live[0] > box_ref[0], "왼쪽으로 돈 화면에서 랙이 오른쪽으로 가야 한다"
    yaw, _, _ = landmark_align.yaw_between(reference, live, K, ZERO_DIST, UP)
    corrected = mock_scene.render_scene(view_yaw_deg=5.0 + (-yaw))
    residual, _, _ = landmark_align.yaw_between(reference, corrected, K, ZERO_DIST, UP)
    assert abs(residual) < 0.3


def test_the_arms_are_masked_and_would_pull_the_yaw_to_zero_if_not(
    reference, monkeypatch
):
    # A plain room: few background features, so the arms -- which move with
    # the camera -- outnumber them. Masked, the yaw is right; unmasked, the
    # RANSAC takes the arms' zero-shift consensus (trap 1).
    sparse_ref = mock_scene.render_scene(background_density=0.2)
    sparse_live = mock_scene.render_scene(view_yaw_deg=6.0, background_density=0.2)
    mask = landmark_align.robot_mask(sparse_ref)
    for x, w, h in mock_scene._ARMS:
        assert (mask[0:h, x : x + w] == 0).all(), "팔 영역이 마스크되지 않음"
    masked = landmark_align.yaw_between(sparse_ref, sparse_live, K, ZERO_DIST, UP)
    assert masked[0] == pytest.approx(6.0, abs=0.3)
    monkeypatch.setattr(landmark_align, "robot_mask", _no_mask)
    unmasked = landmark_align.yaw_between(sparse_ref, sparse_live, K, ZERO_DIST, UP)
    assert abs(unmasked[0]) < 1.0, (
        "마스크 없이도 맞으면 이 시험이 팔 함정을 재현하지 못한 것"
    )


def test_a_fit_beyond_twenty_degrees_is_not_usable_and_falls_to_coarse(reference):
    live = mock_scene.render_scene(view_yaw_deg=25.0)
    m = landmark_align.measure(reference, live, _params(), None, 0)
    assert m.yaw_deg == pytest.approx(25.0, abs=0.5)
    assert not m.heading_valid, "20°를 넘는 fine 추정은 쓰면 안 된다"
    assert m.heading_mode == landmark_align.HEADING_COARSE
    assert m.heading_yaw_deg == m.coarse_yaw_deg
    # Rough by construction (pitch ignored), but the right side and size.
    assert 15.0 < m.coarse_yaw_deg < 30.0


def test_too_few_inliers_falls_to_the_rack_blob(reference):
    # A badly blurred view: SIFT keeps a handful of inliers, the dark rack blob
    # is still there. 10/24 f1 stopped here (7 inliers at +15.3 deg) before the
    # coarse fallback was added.
    live = cv2.GaussianBlur(mock_scene.render_scene(view_yaw_deg=6.0), (0, 0), 14)
    m = landmark_align.measure(reference, live, _params(), None, 0)
    assert m.inliers < landmark_align.MIN_INLIERS and not m.heading_valid
    assert m.heading_mode == landmark_align.HEADING_COARSE
    assert m.heading_yaw_deg == pytest.approx(6.0, abs=2.0)


def test_coarse_off_leaves_an_unusable_fine_estimate_unusable(reference):
    live = cv2.GaussianBlur(mock_scene.render_scene(view_yaw_deg=6.0), (0, 0), 14)
    m = landmark_align.measure(reference, live, _params(coarse_heading=False), None, 0)
    assert m.heading_mode is None and m.heading_yaw_deg is None
    assert m.coarse_yaw_deg is None
    assert not m.heading_usable


def test_coarse_beyond_thirty_degrees_is_not_usable(reference, monkeypatch):
    monkeypatch.setattr(landmark_align, "yaw_between", lambda *a, **k: None)
    monkeypatch.setattr(landmark_align, "coarse_yaw_from_rack", lambda *a, **k: 31.0)
    m = landmark_align.measure(reference, reference, _params(), None, 0)
    assert m.coarse_yaw_deg == 31.0 and m.heading_mode is None


def test_no_features_gives_no_heading_and_no_distance(reference):
    flat = np.full_like(reference, 150)
    m = landmark_align.measure(reference, flat, _params(), None, 0)
    assert m.yaw_deg is None and m.inliers == 0 and not m.heading_valid
    assert m.heading_mode is None, "랙도 없으면 coarse도 없다"
    assert m.move_m is None and not m.distance_valid and m.scale_source is None


def test_dark_blob_box_finds_the_landmark_sprite(reference):
    assert landmark_align.dark_blob_box(reference) == ROI
    # The arms touch the top edge and must never be taken for the rack.
    no_rack = mock_scene.render_scene()
    no_rack[100:400] = 200
    assert landmark_align.dark_blob_box(no_rack) is None


# ---------------------------------------------------------------------------
# distance
# ---------------------------------------------------------------------------


def test_a_bigger_landmark_means_too_close_so_the_move_is_backward(reference):
    m = landmark_align.measure(
        reference, mock_scene.render_scene(landmark_scale=1.05), _params(), None, 0
    )
    assert m.scale_source == landmark_align.SCALE_SIFT
    assert m.scale_inliers >= landmark_align.MIN_SCALE_INLIERS
    assert m.move_m == pytest.approx(-0.0286, abs=0.002)
    assert m.distance_valid


def test_a_smaller_landmark_means_too_far_so_the_move_is_forward(reference):
    m = landmark_align.measure(
        reference, mock_scene.render_scene(landmark_scale=0.95), _params(), None, 0
    )
    assert m.move_m == pytest.approx(+0.0316, abs=0.002)
    assert m.distance_valid


def test_distance_move_formula():
    assert landmark_align.distance_move(1.0, 0.6) == 0.0
    assert landmark_align.distance_move(0.5, 0.6) == pytest.approx(0.6)


def test_a_move_beyond_ten_centimetres_is_not_usable(reference, monkeypatch):
    monkeypatch.setattr(landmark_align, "landmark_scale", lambda *a, **k: (0.8, 30, 40))
    m = landmark_align.measure(reference, reference, _params(), None, 0)
    assert m.move_m == pytest.approx(0.15) and not m.distance_valid


def test_too_few_scale_inliers_is_not_usable_without_the_width_fallback(
    reference, monkeypatch
):
    monkeypatch.setattr(landmark_align, "landmark_scale", lambda *a, **k: (1.02, 5, 9))
    m = landmark_align.measure(
        reference, mock_scene.render_scene(landmark_scale=1.05), _params(), None, 0
    )
    assert m.sift_scale == 1.02 and m.scale_inliers == 5
    assert m.scale is None and m.move_m is None and not m.distance_valid


def test_the_width_fallback_uses_the_live_rack_box_when_on(reference, monkeypatch):
    monkeypatch.setattr(landmark_align, "landmark_scale", lambda *a, **k: None)
    live = mock_scene.render_scene(landmark_scale=1.05)
    m = landmark_align.measure(reference, live, _params(width_fallback=True), None, 0)
    assert m.scale_source == landmark_align.SCALE_BOX_WIDTH
    assert m.scale == landmark_align.dark_blob_box(live)[2] / ROI[2]
    assert m.move_m < 0 and m.distance_valid


def test_the_width_fallback_is_off_for_a_landmark_that_is_not_the_rack(
    reference, monkeypatch
):
    # The live box the fallback measures is always the dark rack blob. With
    # another landmark's ROI the field (--roi given) printed DIST_UNRELIABLE
    # and did not move; dividing the rack width by that ROI's width gave a
    # 5 cm drive on a frame whose true distance error is 0.
    monkeypatch.setattr(landmark_align, "landmark_scale", lambda *a, **k: (1.0, 3, 9))
    other = (60, 120, 85, 60)
    assert landmark_align.dark_blob_box(reference) != other
    live = mock_scene.render_scene(view_yaw_deg=0.5)
    m = landmark_align.measure(
        reference, live, _params(landmark_roi=other, width_fallback=True), None, 0
    )
    assert m.scale is None and m.scale_source is None
    assert m.move_m is None and not m.distance_valid


# ---------------------------------------------------------------------------
# measure as a whole
# ---------------------------------------------------------------------------


def test_the_same_pair_gives_the_same_answer_twice(reference):
    live = mock_scene.render_scene(view_yaw_deg=4.0, landmark_scale=0.97)
    first = landmark_align.measure(reference, live, _params(), None, 0).as_event()
    second = landmark_align.measure(reference, live, _params(), None, 0).as_event()
    first.pop("measure_s"), second.pop("measure_s")
    assert first == second


def test_measure_writes_the_live_and_overlay_pngs(reference, tmp_path):
    live = mock_scene.render_scene(view_yaw_deg=3.0)
    m = landmark_align.measure(reference, live, _params(), tmp_path / "s1", 2)
    assert m.live_png == str(tmp_path / "s1" / "s2_live.png")
    assert m.overlay_png == str(tmp_path / "s1" / "s2_overlay.png")
    written = cv2.cvtColor(cv2.imread(m.live_png), cv2.COLOR_BGR2RGB)
    assert np.array_equal(written, live), "PNG는 RGB 원본 그대로여야 한다"


def test_an_unwritable_png_does_not_end_the_measurement(reference, tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("not a directory")
    m = landmark_align.measure(reference, reference, _params(), blocker, 0)
    assert m.live_png is None and m.overlay_png is None
    assert m.heading_valid


def test_as_event_is_plain_python(reference):
    event = landmark_align.measure(reference, reference, _params(), None, 0).as_event()
    for value in event.values():
        assert value is None or isinstance(value, (bool, int, float, str))


def test_the_cv2_lerobot_brings_has_sift_inside_its_pin():
    # stage_runner declares no OpenCV dependency: cv2 arrives through lerobot
    # 0.4.4's opencv-python-headless>=4.9,<4.13 (uv.lock). A lerobot bump that
    # dropped or moved it would break base_align on import, not in a test.
    major, minor = (int(part) for part in cv2.__version__.split(".")[:2])
    assert (4, 9) <= (major, minor) < (4, 13), cv2.__version__
    assert callable(cv2.SIFT_create)


# ---------------------------------------------------------------------------
# parity with the field script on the real demo frames (opt-in)
# ---------------------------------------------------------------------------

DEMO_ROOT = os.environ.get("STAGE_RUNNER_DEMO_ROOT")
# The field's ARRIVAL_TICK (heading_align.py:37): demo first base motion end + 5.
ARRIVAL_FRAME = {0: 302, 1: 263, 2: 253, 3: 231}
# heading_align.py's own output on these frames (recomputed 2026-10-24 with the
# field script): yaw deg and distance cm of episodes 1-3 against episode 0.
FIELD_YAW_DEG = {1: 1.643, 2: 0.956, 3: 1.121}
FIELD_DIST_CM = {1: 1.5, 2: 1.4, 3: 1.5}
FIELD_ROI = (285, 197, 93, 61)


def _demo_frame(root: Path, episode: int, frame: int) -> np.ndarray:
    """The frame the field decoded, selected by the parquet ``timestamp``."""
    import pyarrow.parquet as pq
    from lerobot.datasets.video_utils import decode_video_frames

    key = "observation.images.cam_high"
    rows = pq.read_table(
        min(root.glob("data/**/*.parquet")),
        columns=["episode_index", "frame_index", "timestamp"],
    ).to_pandas()
    row = rows[(rows.episode_index == episode) & (rows.frame_index == frame)]
    episodes = pq.read_table(min(root.glob("meta/episodes/**/*.parquet"))).to_pandas()
    start = float(
        episodes.loc[
            episodes.episode_index == episode, f"videos/{key}/from_timestamp"
        ].iloc[0]
    )
    video = min(root.glob(f"videos/{key}/**/*.mp4"))
    tensor = decode_video_frames(video, [start + float(row.timestamp.iloc[0])], 1e-4)[0]
    return (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


@pytest.mark.skipif(DEMO_ROOT is None, reason="STAGE_RUNNER_DEMO_ROOT not set")
def test_parity_with_the_field_script_on_the_demo_frames():
    root = Path(DEMO_ROOT)
    reference = _demo_frame(root, 0, ARRIVAL_FRAME[0])
    auto_roi = landmark_align.dark_blob_box(reference)
    print(f"dark_blob_box(ep0 frame 302) = {auto_roi}; robot YAML ROI = {FIELD_ROI}")
    assert auto_roi == FIELD_ROI, "YAML ROI가 현장이 매번 계산하던 자동 박스와 다르다"
    params = landmark_align.AlignParams(
        camera_matrix=landmark_align.FIELD_K,
        dist_coeffs=landmark_align.FIELD_DIST,
        up=landmark_align.FIELD_UP,
        landmark_roi=FIELD_ROI,
        landmark_distance_m=0.60,
    )
    for episode in (1, 2, 3):
        live = _demo_frame(root, episode, ARRIVAL_FRAME[episode])
        m = landmark_align.measure(reference, live, params, None, 0)
        print(
            f"ep{episode}: yaw {m.yaw_deg:+.3f} ({m.inliers}) move {m.move_m * 100:+.2f} cm"
        )
        assert m.heading_mode == landmark_align.HEADING_FINE
        assert m.yaw_deg == pytest.approx(FIELD_YAW_DEG[episode], abs=0.2)
        assert m.move_m * 100 == pytest.approx(FIELD_DIST_CM[episode], abs=0.2)
