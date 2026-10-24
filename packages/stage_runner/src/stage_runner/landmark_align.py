"""Heading and distance of the live cam_high view against a demo's arrival view.

Pure functions over two uint8 RGB images (H, W, 3); the only dependencies are
``cv2`` and ``numpy``. The base_align stage calls ``measure`` from a worker
thread, so nothing here touches a camera, the robot or a dataset.

PROVENANCE. Every estimate is a port of the field script that aligned stage 1 on
the robot on 2026-10-24 (``heading_align.py``, driven by ``stage1_align.py`` and
``align_loop.py``; in the vault under ``.claude/scripts/heading-align/``). The
bodies are kept as they ran there -- the SIFT parameters, the Lowe ratio, the
minimum match counts, the RANSAC that is re-seeded on every call and refits the
largest inlier set -- with one change: the intrinsics ``K``/``DIST`` and the
robot's vertical axis ``UP`` were module constants there and are arguments here,
because the stage reads the intrinsics off the RealSense profile (or the YAML)
at stage start. The 10/24 values are kept as ``FIELD_*`` for logging the
difference, not for computing with.

SIGN CONVENTION. ``yaw_deg`` is live heading minus demo heading, in degrees.
POSITIVE MEANS THE LIVE VIEW IS TURNED LEFT (counter-clockwise seen from above),
the same sign as odometry theta, so the correction is a turn of ``-yaw_deg``.
The minus in ``yaw_between`` is the one checked against odometry on the robot
(10/24 e1: measured +4.68, turned -4.684, next measurement +2.15). Do not copy
the sign from ``view_align_rot.py``, which pairs ``+dot`` with a runtime flip.

``move_m`` is how far to DRIVE FORWARD: positive means the live view is farther
from the landmark than the demo was (the landmark looks smaller).

HEADING, FINE THEN COARSE. The fine estimate is a pure rotation fitted to SIFT
bearings with the robot's own arms masked out. It is usable with at least
``MIN_INLIERS`` inliers and ``|yaw| <= MAX_ABS_YAW_DEG``. When it is not, and
``AlignParams.coarse_heading`` is on, the coarse estimate is the horizontal
bearing of the dark rack blob in each image (``coarse_yaw_from_rack``), usable up
to ``MAX_ABS_COARSE_YAW_DEG``. That is ``heading_for_step`` from the field,
added to stage1_align after run f1 stopped at 7 inliers / +15.3 deg. A coarse
estimate folds any sideways offset into the angle and ignores the camera pitch:
it only gets the base close enough for the fine estimate, so the caller must
turn on it and measure again, and must never treat it as "inside tolerance".

TRAPS THE FIELD RAN INTO, kept here because each one silently gives a wrong
number rather than an error:

1. The arms move with the camera, so unmasked they read as zero shift and pull
   the yaw toward 0. ``robot_mask`` removes every dark blob touching the top
   edge (and the distance match masks the live frame the same way).
2. The thresholds (gray 70 for the mask, 55 for the rack) are tuned on
   ``COLOR_RGB2GRAY``. A BGR stream changes them; the caller must pass RGB.
3. Read distance only once the heading is aligned: a frame 10 deg off read
   -1.9 cm where the aligned frame read +1.1 cm (the heading error changes the
   landmark's apparent size).
4. Driving changes the heading (a 2.3 cm drive moved it by -2.5 deg), so the
   heading has to be measured again after every drive.
5. Around 20 deg off the inliers drop to about 8; that is what the coarse
   fallback is for.
6. ``yaw ~ 0`` is not "true heading 0": a sideways offset of the base folds into
   the yaw (a cross-episode cycle check disagreed by 3-5 deg). Check the overlay.
7. The overlay is red = reference, cyan = live; for the black rack the caption
   reads inverted (the dark rack is where the OTHER image is bright).
8. Hand-placed objects (the rack itself, a tissue box) pull both estimates if
   they moved from their demo position.

DISTANCE. ``landmark_scale`` matches SIFT inside the padded landmark ROI of the
reference against the whole live frame and fits a similarity by RANSAC; the
scale ``s`` turns into a forward move with the landmark's distance at the demo
arrival: ``d * (1/s - 1)``. When SIFT gives fewer than ``MIN_SCALE_INLIERS``
inliers the field fell back on the rack box width (``live box w / roi w``) --
that branch was live on 10/24 because no ROI was passed, and run f2 aligned on
it. Here it runs only when ``AlignParams.width_fallback`` is on AND the ROI is
the reference's own ``dark_blob_box`` (the field never used it with --roi).
"""

import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Fine heading usable: at least this many RANSAC inliers ...
MIN_INLIERS: int = 15
# ... and no further off than this (deg). Past it the field stopped trusting the
# fit (inliers fall to ~8 near 20 deg) and used the coarse estimate instead.
MAX_ABS_YAW_DEG: float = 20.0
# Coarse (rack blob) heading usable up to this (deg); heading_align.py:167.
MAX_ABS_COARSE_YAW_DEG: float = 30.0
# A fine fit with fewer than MIN_INLIERS but at least this many is still turned
# on as a COARSE estimate when no rack blob is in view (beaker table, fridge,
# shelf). Robot PC 2026-10-25: t02 and t04 stopped at 13 matches otherwise.
MIN_WEAK_FINE_INLIERS: int = 8
# Distance usable from SIFT with at least this many similarity inliers.
MIN_SCALE_INLIERS: int = 8
# A distance correction larger than this is refused (m): the shell-era `move`
# refused > 0.10 m, and a bigger number is far more likely a bad match.
MAX_ABS_MOVE_M: float = 0.10

# The rotation RANSAC, as run on 10/24. Offline, thresholds 0.25-1.0 deg agreed
# within 0.3 deg except one pair (1.4 deg); the field ran 1.0.
RANSAC_THRESHOLD_DEG: float = 1.0
RANSAC_ITERATIONS: int = 2000
# Re-seeded on every call, so the same pair always gives the same answer.
RANSAC_SEED: int = 0
LOWE_RATIO: float = 0.8
# Fewer good matches than this and no rotation is fitted at all.
MIN_ROTATION_MATCHES: int = 8
# Fewer good matches than this and no similarity is fitted at all.
MIN_SCALE_MATCHES: int = 6

# robot_mask: near-black (< 70) blobs touching the top edge with more than 2000
# px are the arms; dilated 25 x 25 so their edges go too.
MASK_GRAY_MAX: int = 70
MASK_MIN_AREA: int = 2000
MASK_DILATE: int = 25

# landmark_scale: the reference crop is the ROI padded by this much (px).
ROI_PAD: int = 25
AFFINE_REPROJ_PX: float = 2.0

# dark_blob_box, verbatim from heading_align.py:139-151: gray < 55, closed 7 x 7,
# area 1500-40000, not touching the top, top >= 100, bottom <= 400, w >= 40.
BLOB_GRAY_MAX: int = 55
BLOB_CLOSE: int = 7
BLOB_MIN_AREA: int = 1500
BLOB_MAX_AREA: int = 40000
BLOB_MIN_Y: int = 100
BLOB_MAX_BOTTOM: int = 400
BLOB_MIN_W: int = 40

# The 10/24 constants of heading_align.py. FOR LOGGING ONLY: the stage computes
# with the intrinsics read off the device at stage start (or given in YAML) and
# logs how far they are from these.
FIELD_K: np.ndarray = np.array(
    [[392.6346, 0.0, 322.7227], [0.0, 392.0662, 236.2555], [0.0, 0.0, 1.0]]
)
FIELD_DIST: np.ndarray = np.array(
    [-0.0522401, 0.0587252, 0.0001675, 0.0009436, -0.0198832]
)
# Robot vertical axis in camera coordinates, from essential-matrix poses of the
# replay run's in-place turns (view_align.py); camera pitch asin(-up[2]) ~ 32 deg.
FIELD_UP: np.ndarray = np.array([-0.06288349, -0.84574326, -0.52987169])

# heading_mode values. Closed, because the stage's decision and the events
# reader branch on them.
HEADING_FINE: str = "fine"
HEADING_COARSE: str = "coarse"
HEADING_MODES: tuple[str, ...] = (HEADING_FINE, HEADING_COARSE)

# scale_source values. Closed for the same reason.
SCALE_SIFT: str = "sift"
SCALE_BOX_WIDTH: str = "box_width"
SCALE_SOURCES: tuple[str, ...] = (SCALE_SIFT, SCALE_BOX_WIDTH)


@dataclass(frozen=True)
class AlignParams:
    """Everything ``measure`` needs besides the two images."""

    # 3 x 3 pinhole matrix of the live camera, the one the reference was taken
    # with too (same camera, same resolution).
    camera_matrix: np.ndarray
    # OpenCV distortion coefficients (4, 5, 8, 12 or 14 values).
    dist_coeffs: np.ndarray
    # Unit vector: the robot's vertical axis in camera coordinates.
    up: np.ndarray
    # (x, y, w, h) of the fixed landmark in the REFERENCE image.
    landmark_roi: tuple[int, int, int, int]
    # Camera-to-landmark distance at the demo arrival (m); scales the move.
    landmark_distance_m: float
    # Rack-box-width distance when SIFT scale has too few inliers (field f2).
    width_fallback: bool = False
    # Rack-blob heading when the fine estimate is not usable (field f3 on).
    coarse_heading: bool = True


@dataclass(frozen=True)
class Measurement:
    """One live frame against the reference: heading, distance, side files.

    ``yaw_deg``/``inliers``/``matches`` are the FINE estimate as fitted (None /
    0 when nothing could be fitted) and ``heading_valid`` says whether it is
    usable. ``heading_mode`` and ``heading_yaw_deg`` are what a decision acts on:
    the fine estimate when it is usable, else the coarse one when that is on and
    usable, else None. ``coarse_yaw_deg`` is filled only when the coarse
    estimate was computed (fine not usable, coarse on).

    ``sift_scale`` is the raw similarity scale whatever its inlier count;
    ``scale`` is the scale the move was computed from (None when distance is not
    usable), and ``scale_source`` says where it came from.
    """

    yaw_deg: float | None
    inliers: int
    matches: int
    heading_valid: bool
    coarse_yaw_deg: float | None
    heading_mode: str | None
    heading_yaw_deg: float | None
    sift_scale: float | None
    scale: float | None
    scale_inliers: int
    scale_matches: int
    scale_source: str | None
    move_m: float | None
    distance_valid: bool
    measure_s: float
    live_png: str | None
    overlay_png: str | None

    @property
    def heading_usable(self) -> bool:
        """True when there is SOME heading to act on (fine or coarse)."""
        return self.heading_mode is not None

    def as_event(self) -> dict[str, Any]:
        """Plain Python values only, for an events.jsonl line."""
        return asdict(self)


# The arms are masked by two fixed top-corner boxes (user-approved 2026-10-25)
# instead of "every dark blob touching the top edge", which also swallowed the
# black fridge (merged with the arms) and left 15-19 matches in front of it. Boxes as fractions of the 640x480 frame: left
# x < 230, right x >= 420, both y < 140 (covers the gripper tips, which reach
# x ~215, and the mock scene's arms). On t05's fridge frames this gave 30-35
# matches and estimates that follow the turn; rack and beaker frames unchanged
# or better.
ARM_BOX_LEFT_X: float = 230 / 640
ARM_BOX_RIGHT_X: float = 420 / 640
ARM_BOX_BOTTOM_Y: float = 140 / 480


def robot_mask(image: np.ndarray) -> np.ndarray:
    """255 everywhere except the robot's arms -- two fixed top-corner boxes."""
    height, width = image.shape[:2]
    mask = np.full((height, width), 255, np.uint8)
    bottom = int(round(ARM_BOX_BOTTOM_Y * height))
    mask[:bottom, : int(round(ARM_BOX_LEFT_X * width))] = 0
    mask[:bottom, int(round(ARM_BOX_RIGHT_X * width)) :] = 0
    return mask


def robot_mask_dark_blobs(image: np.ndarray) -> np.ndarray:
    """255 everywhere except the robot's arms (dark blobs touching the top edge).

    heading_align.py:87-96, verbatim. Kept for comparison; unused while the
    fixed-box patch above is in.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    dark = (gray < MASK_GRAY_MAX).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    robot = np.zeros_like(dark)
    for i in range(1, count):
        if (
            stats[i, cv2.CC_STAT_TOP] == 0
            and stats[i, cv2.CC_STAT_AREA] > MASK_MIN_AREA
        ):
            robot[labels == i] = 1
    robot = cv2.dilate(robot, np.ones((MASK_DILATE, MASK_DILATE), np.uint8))
    return ((1 - robot) * 255).astype(np.uint8)


def bearings(
    points: np.ndarray, camera_matrix: np.ndarray, dist_coeffs: np.ndarray
) -> np.ndarray:
    """Unit bearing vectors of pixel points (heading_align.py:99-102)."""
    normalized = cv2.undistortPoints(
        points.reshape(-1, 1, 2), camera_matrix, dist_coeffs
    ).reshape(-1, 2)
    rays = np.c_[normalized, np.ones(len(normalized))]
    return rays / np.linalg.norm(rays, axis=1, keepdims=True)


def kabsch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation R with b ~ R a (heading_align.py:105-108)."""
    u, _, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    return vt.T @ np.diag([1, 1, d]) @ u.T


def yaw_between(
    reference: np.ndarray,
    live: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    up: np.ndarray,
    threshold_deg: float = RANSAC_THRESHOLD_DEG,
    iterations: int = RANSAC_ITERATIONS,
) -> tuple[float, int, int] | None:
    """(yaw_deg, inliers, good matches), or None when nothing could be fitted.

    heading_align.py:111-136 with K, DIST and UP as arguments. The RANSAC stays
    pure Python for parity with the field (a vectorised one would not draw the
    same samples); its cost on the robot is logged by the stage as measure_s.
    """
    sift = cv2.SIFT_create(nfeatures=6000, contrastThreshold=0.01, edgeThreshold=20)
    gray_ref, gray_live = (
        cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) for x in (reference, live)
    )
    kp_ref, des_ref = sift.detectAndCompute(gray_ref, robot_mask(reference))
    kp_live, des_live = sift.detectAndCompute(gray_live, robot_mask(live))
    if des_ref is None or des_live is None:
        return None
    pairs = cv2.BFMatcher().knnMatch(des_ref, des_live, k=2)
    good = [
        p[0]
        for p in pairs
        if len(p) == 2 and p[0].distance < LOWE_RATIO * p[1].distance
    ]
    if len(good) < MIN_ROTATION_MATCHES:
        return None
    a = bearings(
        np.float32([kp_ref[g.queryIdx].pt for g in good]), camera_matrix, dist_coeffs
    )
    b = bearings(
        np.float32([kp_live[g.trainIdx].pt for g in good]), camera_matrix, dist_coeffs
    )
    cos_threshold = np.cos(np.radians(threshold_deg))
    rng = np.random.default_rng(RANSAC_SEED)
    best = None
    for _ in range(iterations):
        pick = rng.choice(len(a), 2, replace=False)
        rotation = kabsch(a[pick], b[pick])
        inliers = np.einsum("ij,ij->i", (rotation @ a.T).T, b) > cos_threshold
        if best is None or inliers.sum() > best.sum():
            best = inliers
    rotation = kabsch(a[best], b[best])
    # sign fixed against odometry on the 10/24 replay run (in-place turns)
    yaw = -float(np.degrees(np.dot(cv2.Rodrigues(rotation)[0].ravel(), up)))
    return yaw, int(best.sum()), len(good)


def heading_valid(result: tuple[float, int, int] | None) -> bool:
    """The fine estimate is usable: fitted, enough inliers, not too far off."""
    if result is None:
        return False
    yaw, inliers, _ = result
    return inliers >= MIN_INLIERS and abs(yaw) <= MAX_ABS_YAW_DEG


def dark_blob_box(image: np.ndarray) -> tuple[int, int, int, int] | None:
    """Largest near-black blob not touching the top edge (the arms do) - the stage-2 tube rack.

    heading_align.py:139-151, verbatim. (x, y, w, h) or None.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    dark = cv2.morphologyEx(
        (gray < BLOB_GRAY_MAX).astype(np.uint8),
        cv2.MORPH_CLOSE,
        np.ones((BLOB_CLOSE, BLOB_CLOSE), np.uint8),
    )
    count, _, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    best = None
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if (
            y == 0
            or area < BLOB_MIN_AREA
            or area > BLOB_MAX_AREA
            or y < BLOB_MIN_Y
            or y + h > BLOB_MAX_BOTTOM
            or w < BLOB_MIN_W
        ):
            continue
        if best is None or area > best[4]:
            best = (int(x), int(y), int(w), int(h), int(area))
    return best[:4] if best else None


def coarse_yaw_from_rack(
    reference: np.ndarray, live: np.ndarray, camera_matrix: np.ndarray
) -> float | None:
    """Rough heading difference from where the rack blob sits horizontally.

    Degrees, same sign as ``yaw_deg``; heading_align.py:154-164 with K as an
    argument. For when SIFT finds too few matches far from the reference. A
    sideways offset of the base is folded in, and the camera pitch is ignored,
    so it only gets the base close enough for the fine estimate (10/24: it
    agreed with SIFT within 2 deg on 10 frames that were 7-27 deg off).
    """
    box_ref, box_live = dark_blob_box(reference), dark_blob_box(live)
    if box_ref is None or box_live is None:
        return None

    def angle(box: tuple[int, int, int, int]) -> float:
        centre = box[0] + box[2] / 2.0
        return float(
            np.degrees(np.arctan((centre - camera_matrix[0, 2]) / camera_matrix[0, 0]))
        )

    return angle(box_live) - angle(box_ref)


def heading_for_step(
    reference: np.ndarray, live: np.ndarray, params: AlignParams
) -> tuple[tuple[float, int, int] | None, float | None, str | None, float | None]:
    """(fine result, coarse yaw, mode, yaw to act on) -- heading_align.py:170-185.

    mode "fine" = the SIFT rotation is usable; "coarse" = it is not but the rack
    blob is (computed only then, and only with ``params.coarse_heading`` on);
    None = neither. A fine fit that is usable on inliers but beyond
    MAX_ABS_YAW_DEG also falls through to coarse, as in the field.
    """
    fine = yaw_between(
        reference, live, params.camera_matrix, params.dist_coeffs, params.up
    )
    if heading_valid(fine):
        assert fine is not None
        return fine, None, HEADING_FINE, fine[0]
    if not params.coarse_heading:
        return fine, None, None, None
    coarse = coarse_yaw_from_rack(reference, live, params.camera_matrix)
    if coarse is not None and abs(coarse) <= MAX_ABS_COARSE_YAW_DEG:
        return fine, coarse, HEADING_COARSE, coarse
    # No rack in view (beaker table, fridge, shelf): a fine fit short of
    # MIN_INLIERS but with MIN_WEAK_FINE_INLIERS is turned on as a coarse
    # estimate and measured again -- the same turn-then-remeasure as the rack
    # blob, bounded by MAX_COARSE_TURNS and never counted as aligned
    # (user-approved 2026-10-25).
    if (
        fine is not None
        and fine[1] >= MIN_WEAK_FINE_INLIERS
        and abs(fine[0]) <= MAX_ABS_YAW_DEG
    ):
        return fine, fine[0], HEADING_COARSE, fine[0]
    return fine, coarse, None, None


def landmark_scale(
    reference: np.ndarray,
    live: np.ndarray,
    roi: tuple[int, int, int, int],
    pad: int = ROI_PAD,
) -> tuple[float, int, int] | None:
    """Apparent size of a fixed object, live / reference: (scale, inliers, good).

    heading_align.py:188-212, verbatim. SIFT inside the padded roi of the
    reference (NOT masked), matched against the whole live frame (robot masked),
    similarity transform by RANSAC. < 1 means the camera is farther than in the
    demo.
    """
    x, y, w, h = roi
    x0, y0 = max(0, x - pad), max(0, y - pad)
    crop = cv2.cvtColor(
        reference[y0 : y + h + pad, x0 : x + w + pad], cv2.COLOR_RGB2GRAY
    )
    sift = cv2.SIFT_create(nfeatures=3000, contrastThreshold=0.01)
    kp_ref, des_ref = sift.detectAndCompute(crop, None)
    gray_live = cv2.cvtColor(live, cv2.COLOR_RGB2GRAY)
    kp_live, des_live = sift.detectAndCompute(gray_live, robot_mask(live))
    if des_ref is None or des_live is None:
        return None
    pairs = cv2.BFMatcher().knnMatch(des_ref, des_live, k=2)
    good = [
        p[0]
        for p in pairs
        if len(p) == 2 and p[0].distance < LOWE_RATIO * p[1].distance
    ]
    if len(good) < MIN_SCALE_MATCHES:
        return None
    pa = np.float32([kp_ref[g.queryIdx].pt for g in good]) + np.float32([x0, y0])
    pb = np.float32([kp_live[g.trainIdx].pt for g in good])
    matrix, inliers = cv2.estimateAffinePartial2D(
        pa, pb, method=cv2.RANSAC, ransacReprojThreshold=AFFINE_REPROJ_PX
    )
    if matrix is None:
        return None
    return float(np.hypot(matrix[0, 0], matrix[1, 0])), int(inliers.sum()), len(good)


def distance_move(scale: float, landmark_distance_m: float) -> float:
    """Forward move (m) that brings the landmark to its demo size: d * (1/s - 1)."""
    return landmark_distance_m * (1.0 / scale - 1.0)


def overlay(reference: np.ndarray, live: np.ndarray) -> np.ndarray:
    """Red = reference, cyan = live (heading_align.py:215-220)."""
    out = np.zeros_like(reference)
    out[..., 0] = reference.mean(axis=2)
    out[..., 1] = live.mean(axis=2)
    out[..., 2] = live.mean(axis=2)
    return out


def write_png(path: Path, image: np.ndarray) -> bool:
    """Write an RGB image as PNG (cv2 wants BGR). True = written.

    NEVER RAISES: the PNGs are evidence for a person reading the run afterwards,
    not an input to any decision, so a full disk must not end a trial in the
    middle of a base move. A failure is logged and the event gets no path.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        written = bool(cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)))
    except Exception:
        logger.exception(f"could not write {path}")
        return False
    if not written:
        logger.warning(f"could not write {path} (cv2.imwrite returned False)")
    return written


def measure(
    reference: np.ndarray,
    live: np.ndarray,
    params: AlignParams,
    save_dir: Path | None,
    step: int,
) -> Measurement:
    """Heading and distance of ``live`` against ``reference``, plus side files.

    Both estimates are computed on every call, as the field did; the caller acts
    on the distance only once the heading is inside tolerance (trap 3). With
    ``save_dir`` set, ``s<step>_live.png`` and ``s<step>_overlay.png`` are
    written there.
    """
    started = time.perf_counter()
    live_png = overlay_png = None
    if save_dir is not None:
        live_path = save_dir / f"s{step}_live.png"
        overlay_path = save_dir / f"s{step}_overlay.png"
        if write_png(live_path, live):
            live_png = str(live_path)
        if write_png(overlay_path, overlay(reference, live)):
            overlay_png = str(overlay_path)

    fine, coarse, mode, heading_yaw = heading_for_step(reference, live, params)

    sift = landmark_scale(reference, live, params.landmark_roi)
    sift_scale = None if sift is None else sift[0]
    scale_inliers = 0 if sift is None else sift[1]
    scale_matches = 0 if sift is None else sift[2]
    scale: float | None = None
    scale_source: str | None = None
    if sift is not None and sift[1] >= MIN_SCALE_INLIERS:
        scale, scale_source = sift[0], SCALE_SIFT
    elif params.width_fallback and dark_blob_box(reference) == tuple(
        params.landmark_roi
    ):
        # distance_estimate's fallback (heading_align.py:231-235): the live rack
        # box width against the reference ROI width. ONLY when the ROI is the
        # reference's own dark_blob_box, as in the field, which skipped the
        # fallback whenever --roi was given: the live box measured here is
        # always the dark rack blob, so against any other landmark's width the
        # ratio is not a scale at all, and it would still drive the base.
        live_box = dark_blob_box(live)
        if live_box is not None:
            scale, scale_source = live_box[2] / params.landmark_roi[2], SCALE_BOX_WIDTH
    move_m = None if scale is None else distance_move(scale, params.landmark_distance_m)
    distance_valid = move_m is not None and abs(move_m) <= MAX_ABS_MOVE_M

    return Measurement(
        yaw_deg=None if fine is None else fine[0],
        inliers=0 if fine is None else fine[1],
        matches=0 if fine is None else fine[2],
        heading_valid=heading_valid(fine),
        coarse_yaw_deg=coarse,
        heading_mode=mode,
        heading_yaw_deg=heading_yaw,
        sift_scale=sift_scale,
        scale=scale,
        scale_inliers=scale_inliers,
        scale_matches=scale_matches,
        scale_source=scale_source,
        move_m=move_m,
        distance_valid=distance_valid,
        measure_s=time.perf_counter() - started,
        live_png=live_png,
        overlay_png=overlay_png,
    )
