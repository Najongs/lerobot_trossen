"""A synthetic cam_high scene for the base_align stage's tests and smoke runs.

``render_scene`` draws what a camera on the base would see after turning by
``view_yaw_deg`` about the robot's vertical axis and driving so a landmark
looks ``landmark_scale`` times its reference size. It is NOT a renderer of the
lab: it reproduces exactly the image properties the estimators in
``landmark_align`` depend on, and nothing else.

* A bright, densely textured background (gray 90-230), so SIFT finds plenty
  of features and none of them is dark enough to look like the rack or an arm.
* A pure camera rotation, applied as the homography
  ``K * Rodrigues(-yaw * up) * K^-1`` -- the model ``yaw_between`` fits, with the
  same sign: positive yaw = turned left (CCW from above), so world points move
  right in the image. The background is drawn on a canvas larger than the frame
  so the rotated view has no empty border (a black border would be a dark blob).
* A dark textured landmark sprite (gray 5-50) at ``landmark_roi`` in the
  reference, scaled about its centre by ``landmark_scale``, sitting on a flat
  bright mat so the padded reference crop of ``landmark_scale`` holds only
  landmark features (a textured surround would not scale and would pull the
  similarity toward 1). ``dark_blob_box`` finds exactly that sprite at yaw 0.
* The robot's arms: dark textured blobs touching the top edge, composited
  AFTER the rotation because they move with the camera. ``robot_mask`` removes
  them; unmasked they read as zero shift (trap 1 of landmark_align).
* No lens distortion: give the estimators zero distortion coefficients.

NOTHING HERE IS EVIDENCE ABOUT THE ROBOT'S CAMERA: real frames have a sideways
offset folded into the yaw, lighting changes and far fewer features (white
table, grey wall -- 10-60 matches on 10/24). Only cv2 and numpy are imported,
and only by the tests and by mock_robot's opt-in scene camera.
"""

from functools import lru_cache

import cv2
import numpy as np

# The 10/24 cam_high intrinsics and vertical axis (landmark_align.FIELD_*),
# copied so this module can be imported on its own.
DEFAULT_CAMERA_MATRIX: tuple[tuple[float, ...], ...] = (
    (392.6346, 0.0, 322.7227),
    (0.0, 392.0662, 236.2555),
    (0.0, 0.0, 1.0),
)
DEFAULT_UP: tuple[float, float, float] = (-0.06288349, -0.84574326, -0.52987169)
# The rack's dark_blob_box in demo episode 0 frame 302, the robot YAML's ROI.
DEFAULT_LANDMARK_ROI: tuple[int, int, int, int] = (285, 197, 93, 61)

# Canvas margin around the frame (px). A 30 deg turn about the tilted vertical
# axis moves image points by up to ~230 px; the rest is headroom.
_MARGIN: int = 360
# Flat mat around the landmark (px per side), wide enough for the ROI_PAD=25
# crop plus a 20 % larger sprite.
_MAT: int = 45
_MAT_GRAY: int = 205
# The arms: two blobs hanging from the top edge, (x, w, h) in the frame.
_ARMS: tuple[tuple[int, int, int], ...] = ((20, 190, 120), (430, 190, 120))


@lru_cache(maxsize=8)
def _background(seed: int, width: int, height: int, density: float) -> np.ndarray:
    """Bright textured canvas, gray 90-230, (H + 2M, W + 2M) uint8."""
    rng = np.random.default_rng(seed)
    h, w = height + 2 * _MARGIN, width + 2 * _MARGIN
    canvas = np.full((h, w), 160, np.uint8)
    for _ in range(int(density * h * w / 450)):
        gray = int(rng.integers(90, 231))
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        size = int(rng.integers(3, 14))
        if rng.random() < 0.5:
            cv2.circle(canvas, (x, y), size, gray, -1)
        else:
            x2 = x + int(rng.integers(3, 20))
            y2 = y + int(rng.integers(3, 20))
            cv2.rectangle(canvas, (x, y), (x2, y2), gray, -1)
    canvas = cv2.GaussianBlur(canvas, (3, 3), 0.8)
    return canvas


@lru_cache(maxsize=8)
def _dark_texture(
    seed: int, width: int, height: int, low: int, high: int
) -> np.ndarray:
    """Dark textured patch, gray in [low, high]: the landmark or an arm."""
    rng = np.random.default_rng(seed)
    patch = np.full((height, width), (low + high) // 2, np.uint8)
    for _ in range(max(1, width * height // 60)):
        gray = int(rng.integers(low, high + 1))
        x, y = int(rng.integers(0, width)), int(rng.integers(0, height))
        size = int(rng.integers(1, 5))
        if rng.random() < 0.5:
            cv2.circle(patch, (x, y), size, gray, -1)
        else:
            cv2.rectangle(
                patch,
                (x, y),
                (x + int(rng.integers(2, 8)), y + int(rng.integers(2, 8))),
                gray,
                -1,
            )
    # Slightly blurred so its features survive a sub-pixel rescale; aliased
    # one-pixel edges would make the landmark scale a property of the
    # interpolation instead of the geometry.
    patch = cv2.GaussianBlur(patch, (3, 3), 0.8)
    return np.clip(patch, low, high).astype(np.uint8)


def render_scene(
    *,
    view_yaw_deg: float = 0.0,
    landmark_scale: float = 1.0,
    camera_matrix: np.ndarray | None = None,
    up: np.ndarray | None = None,
    width: int = 640,
    height: int = 480,
    seed: int = 0,
    landmark_roi: tuple[int, int, int, int] = DEFAULT_LANDMARK_ROI,
    arms: bool = True,
    background_density: float = 1.0,
) -> np.ndarray:
    """The view after turning ``view_yaw_deg`` (deg, + = left) with the landmark
    at ``landmark_scale`` times its reference size. uint8 RGB (H, W, 3).

    ``render_scene()`` with the defaults is the reference view.
    ``background_density`` scales how many shapes the background has; a low
    value is a plain room, where the arms -- if not masked -- outnumber the
    background's features (the case trap 1 is about).
    """
    k = np.asarray(
        DEFAULT_CAMERA_MATRIX if camera_matrix is None else camera_matrix, float
    )
    axis = np.asarray(DEFAULT_UP if up is None else up, float)
    axis = axis / np.linalg.norm(axis)

    canvas = _background(seed, width, height, float(background_density)).copy()
    x, y, w, h = landmark_roi
    mat = np.full((h + 2 * _MAT, w + 2 * _MAT), _MAT_GRAY, np.uint8)
    mat[_MAT : _MAT + h, _MAT : _MAT + w] = _dark_texture(seed + 1, w, h, 5, 50)
    if landmark_scale != 1.0:
        # Scaled about the ROI centre with a sub-pixel affine, as an approach
        # scales the object; a resize to whole pixels would put up to half a
        # pixel of error per side into the scale the test then checks.
        centre = ((mat.shape[1] - 1) / 2.0, (mat.shape[0] - 1) / 2.0)
        scaling = cv2.getRotationMatrix2D(centre, 0.0, float(landmark_scale))
        mat = cv2.warpAffine(
            mat,
            scaling,
            (mat.shape[1], mat.shape[0]),
            flags=cv2.INTER_AREA if landmark_scale < 1.0 else cv2.INTER_LINEAR,
            borderValue=_MAT_GRAY,
        )
    top, left = y + _MARGIN - _MAT, x + _MARGIN - _MAT
    canvas[top : top + mat.shape[0], left : left + mat.shape[1]] = mat

    # Canvas pixel -> reference-view pixel is a shift by the margin, so the
    # reference intrinsics on the canvas are K with the principal point moved.
    k_canvas = k.copy()
    k_canvas[0, 2] += _MARGIN
    k_canvas[1, 2] += _MARGIN
    rotation, _ = cv2.Rodrigues(-np.radians(view_yaw_deg) * axis)
    homography = k @ rotation @ np.linalg.inv(k_canvas)
    gray = cv2.warpPerspective(
        canvas,
        homography,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )

    if arms:
        for index, (ax, aw, ah) in enumerate(_ARMS):
            gray[0:ah, ax : ax + aw] = _dark_texture(seed + 2 + index, aw, ah, 0, 60)
    return np.ascontiguousarray(np.repeat(gray[:, :, None], 3, axis=2))
