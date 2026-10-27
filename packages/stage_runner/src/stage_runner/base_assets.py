"""Replay targets and align references, read from local files before connect.

cli.main calls :func:`load_base_assets` inside its preflight block, after
run_preflight and before ``robot.connect()``, and hands the result to every
stage through ``StageContext.base_assets``, keyed by stage id. So:

* a missing dataset, a velocity dataset where a pose dataset was meant, a row
  range past the episode or a reference image of the wrong size stops the run
  with exit code 2 while the robot is still unpowered, instead of in the middle
  of a chain with a tube in the gripper;
* every base stage in the chain -- any number of them, each with its own
  dataset, episode, rows, reference and ROI -- has its numbers in memory before
  the first stage starts, and no stage opens a dataset or decodes a video
  between two others.

NOTHING HERE DOWNLOADS. The datasets are read from ``root`` (or
``HF_LEROBOT_HOME / repo``) exactly as they sit on disk; a dataset that is not
there is refused with the fix, because a hub download mid-preflight on the robot
PC is the several-minute stall this package keeps out of its start-up (and the
field scripts ran with HF_HUB_OFFLINE=1 for the same reason).

Every refusal is collected and raised once as ``PreflightError``, which cli.main
turns into exit code 2, in the same one-pass-fixes-the-YAML style as preflight.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from lerobot.utils.constants import HF_LEROBOT_HOME

from stage_runner import base_stages
from stage_runner.config import (
    EXECUTOR_BASE_ALIGN,
    EXECUTOR_BASE_REPLAY,
    BaseAlignConfig,
    BaseReplayConfig,
    StageConfig,
    StageRunnerConfig,
)
from stage_runner.preflight import PreflightError

if TYPE_CHECKING:
    from lerobot.robots import Robot

logger = logging.getLogger(__name__)

INFO_PATH: str = "meta/info.json"
# The two pose labels derive_pose.py writes in place of x.vel / theta.vel.
POSE_X_KEY: str = "x.pos"
POSE_THETA_KEY: str = "theta.pos"
# The files LeRobotDatasetMetadata.load_metadata reads. When any is missing it
# does not fail -- it DOWNLOADS meta/ from the hub (lerobot_dataset.py:103-113),
# so they are checked here first and a missing one is a refusal instead.
_METADATA_FILES: tuple[str, ...] = (INFO_PATH, "meta/tasks.parquet")
_EPISODES_DIR: str = "meta/episodes"
# Seek tolerance for the reference frame. The frame is addressed by its own
# stored timestamp, so the decoder should land on it exactly; this is the value
# the 10/24 parity check used.
VIDEO_TOLERANCE_S: float = 1e-4


class _AssetProblem(Exception):
    """One refusal for one stage; collected into a single PreflightError."""


@dataclass(frozen=True)
class ReplayTargets:
    """A replay stage's targets: forward distance and heading per row.

    Relative to the base pose at the stage's start, which the executor anchors
    its odometry to -- so the same targets mean the same motion wherever in the
    chain (and however often) the stage runs.
    """

    # float64, (rows,): cumulative forward distance (m) from the stage anchor.
    s_m: np.ndarray
    # float64, (rows,): unwrapped heading (rad) from the stage anchor.
    theta_rad: np.ndarray
    # dataset, root, episode, from_row, until_row, rows and the final targets;
    # copied into the stage event so a run says which demo it replayed.
    source: dict[str, Any]

    @property
    def rows(self) -> int:
        return int(self.s_m.shape[0])


@dataclass(frozen=True)
class AlignReference:
    """An align stage's reference view: the demo's arrival frame, uint8 RGB (H, W, 3).

    The array is an owned, C-contiguous, READ-ONLY copy: the align worker reads it
    on every measurement, possibly in several stages of one chain, and nothing
    may write into it.
    """

    image: np.ndarray
    # kind (dataset | image), the repo / root / episode / frame / timestamp used,
    # or the path, and the shape.
    source: dict[str, Any]


def dataset_root(repo_id: str, root: str | None) -> Path:
    """``Path(root)`` if given, else ``HF_LEROBOT_HOME / repo_id`` -- lerobot's own rule."""
    if root:
        return Path(root).expanduser()
    return HF_LEROBOT_HOME / repo_id


def _read_info(root: Path, what: str, fix: str) -> dict[str, Any]:
    info_path = root / INFO_PATH
    if not info_path.is_file():
        raise _AssetProblem(
            f"no dataset at {root} ({INFO_PATH} is missing). The robot never "
            f"downloads at run time: {fix}, or set `{what}` to where it is."
        )
    try:
        return json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise _AssetProblem(f"cannot read {info_path}: {error}") from error


def _feature_names(info: Mapping[str, Any], key: str) -> list[str]:
    feature = (info.get("features") or {}).get(key) or {}
    names = feature.get("names")
    # v2.x datasets sometimes nest the list one level down ({"motors": [...]}),
    # the same tolerance start_pose._state_names has.
    if isinstance(names, Mapping) and len(names) == 1:
        names = next(iter(names.values()))
    if not isinstance(names, list):
        return []
    return [str(name) for name in names]


def _data_files(root: Path) -> list[Path]:
    files = sorted((root / "data").glob("**/*.parquet"))
    if not files:
        raise _AssetProblem(f"no parquet files under {root / 'data'}")
    return files


def _read_columns(root: Path, columns: list[str], cache: dict[Any, Any]) -> Any:
    """Every data file of ``root``, these columns only, concatenated. Cached per call.

    The cache is what lets a chain replay several pieces of one dataset (or
    align twice against frames of one dataset) while reading its parquet once.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    key = (str(root), tuple(columns))
    if key not in cache:
        try:
            tables = [
                pq.read_table(path, columns=columns) for path in _data_files(root)
            ]
        except _AssetProblem:
            raise
        except Exception as error:
            raise _AssetProblem(
                f"cannot read columns {columns} of {root / 'data'}: {error}"
            ) from error
        cache[key] = pa.concat_tables(tables)
    return cache[key]


def _episode_rows(table: Any, episode: int, root: Path) -> Any:
    """The rows of one episode, sorted by frame_index, which must be 0..n-1."""
    import pyarrow.compute as pc

    rows = table.filter(pc.equal(table["episode_index"], episode))
    if rows.num_rows == 0:
        present = sorted(set(table["episode_index"].to_pylist()))
        span = f"{present[0]}..{present[-1]}" if present else "none"
        raise _AssetProblem(
            f"episode {episode} is not in {root} (episodes there: {span})"
        )
    rows = rows.sort_by("frame_index")
    frames = np.asarray(rows["frame_index"].to_pylist())
    if not np.array_equal(frames, np.arange(len(frames))):
        raise _AssetProblem(
            f"episode {episode} of {root}: frame_index is not 0..{len(frames) - 1} "
            "(missing or repeated frames), so its rows cannot be addressed"
        )
    return rows


def load_replay_targets(
    replay: BaseReplayConfig, cache: dict[Any, Any] | None = None
) -> ReplayTargets:
    """Read rows ``from_row..until_row`` (inclusive) of the episode's x.pos / theta.pos.

    From the ``action`` column, chosen BY NAME from meta/info.json. With
    ``from_row == 0`` the targets are the raw rows, exactly what stage1_align
    replayed on 10/24 (the demo's cumulative pose starts at its first command).
    With ``from_row > 0`` they are rebased onto row ``from_row - 1``, the pose the
    demo's base was at when that piece started: x.pos is a cumulative path length
    and theta.pos is unwrapped, so a difference is the motion of the piece.
    """
    cache = {} if cache is None else cache
    root = dataset_root(replay.dataset, replay.root)
    info = _read_info(
        root,
        "replay.root",
        "make the pose copy there with derive_pose.py (vault "
        ".claude/scripts/heading-align)",
    )
    names = _feature_names(info, "action")
    if POSE_X_KEY not in names or POSE_THETA_KEY not in names:
        raise _AssetProblem(
            f"{root}: the action names {names} have no {POSE_X_KEY} / "
            f"{POSE_THETA_KEY}, so this is a velocity dataset; make the pose copy "
            "with derive_pose.py (needs the acquisition basevel CSV for the real "
            "dt) and point `replay.dataset` at it."
        )
    rows = _episode_rows(
        _read_columns(root, ["episode_index", "frame_index", "action"], cache),
        replay.episode,
        root,
    )
    length = rows.num_rows
    if replay.until_row >= length:
        raise _AssetProblem(
            f"`replay.until_row: {replay.until_row}` is past the end of episode "
            f"{replay.episode} of {root} ({length} rows, the last is {length - 1})."
        )
    if not 0 <= replay.from_row < replay.until_row:
        raise _AssetProblem(
            f"`replay.from_row: {replay.from_row}` / `until_row: "
            f"{replay.until_row}` must satisfy 0 <= from_row < until_row."
        )
    action = np.asarray(rows["action"].to_pylist(), dtype=np.float64)
    if action.ndim != 2 or action.shape[1] != len(names):
        raise _AssetProblem(
            f"episode {replay.episode} of {root}: the action rows are "
            f"{action.shape}, but {INFO_PATH} names {len(names)} values"
        )
    x_column, theta_column = names.index(POSE_X_KEY), names.index(POSE_THETA_KEY)
    piece = action[replay.from_row : replay.until_row + 1]
    s_m = piece[:, x_column].copy()
    theta_rad = piece[:, theta_column].copy()
    if replay.from_row > 0:
        before = action[replay.from_row - 1]
        s_m -= before[x_column]
        theta_rad -= before[theta_column]
    if not (np.isfinite(s_m).all() and np.isfinite(theta_rad).all()):
        raise _AssetProblem(
            f"episode {replay.episode} of {root}: rows {replay.from_row}.."
            f"{replay.until_row} hold a non-finite {POSE_X_KEY} or {POSE_THETA_KEY}"
        )
    s_m.setflags(write=False)
    theta_rad.setflags(write=False)
    return ReplayTargets(
        s_m=s_m,
        theta_rad=theta_rad,
        source={
            "dataset": replay.dataset,
            "root": str(root),
            "episode": replay.episode,
            "episode_rows": length,
            "from_row": replay.from_row,
            "until_row": replay.until_row,
            "rows": int(s_m.shape[0]),
            "rebased_on_row": replay.from_row - 1 if replay.from_row > 0 else None,
            "final_target_s_m": float(s_m[-1]),
            "final_target_theta_deg": math.degrees(float(theta_rad[-1])),
        },
    )


def _owned_rgb(image: np.ndarray) -> np.ndarray:
    array = np.array(image, dtype=np.uint8, order="C", copy=True)
    if array.ndim != 3 or array.shape[2] != 3:
        raise _AssetProblem(f"the reference is {array.shape}, not an (H, W, 3) image")
    array.setflags(write=False)
    return array


def _load_reference_image(path_text: str) -> AlignReference:
    import cv2

    path = Path(path_text).expanduser()
    image = cv2.imread(str(path), cv2.IMREAD_COLOR) if path.is_file() else None
    if image is None:
        raise _AssetProblem(
            f"cannot read `align.reference_image: {path_text}` (missing, or not an "
            "image file cv2 can decode)"
        )
    rgb = _owned_rgb(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return AlignReference(
        image=rgb,
        source={"kind": "image", "path": str(path), "shape": list(rgb.shape)},
    )


def _load_reference_frame(
    align: BaseAlignConfig, cache: dict[Any, Any]
) -> AlignReference:
    """One frame of one camera of a local dataset, as LeRobotDataset would return it.

    The frame is selected by the parquet row's own ``timestamp`` (the value
    ``LeRobotDataset.__getitem__`` uses, lerobot_dataset.py:1099 / :1060), so it
    is the frame heading_align.demo_frame decoded on 10/24. Video frames are
    scaled by 255 and TRUNCATED to uint8, as demo_frame did; image frames are the
    stored PNG decoded directly.
    """
    root = dataset_root(align.reference_dataset, align.reference_root)
    info = _read_info(
        root,
        "align.reference_root",
        "copy the demo dataset there (with its videos) on a machine that can reach "
        "the hub",
    )
    key = f"observation.images.{align.camera}"
    feature = (info.get("features") or {}).get(key)
    if feature is None:
        cameras = sorted(
            name
            for name, value in (info.get("features") or {}).items()
            if isinstance(value, Mapping) and value.get("dtype") in ("video", "image")
        )
        raise _AssetProblem(
            f"{root} has no {key} feature (cameras there: {cameras}); set "
            "`align.camera` to the camera the reference was taken with."
        )
    dtype = feature.get("dtype")
    if dtype not in ("video", "image"):
        raise _AssetProblem(f"{root}: {key} is a {dtype!r} feature, not a camera")
    total = info.get("total_episodes")
    if isinstance(total, int) and not 0 <= align.episode < total:
        raise _AssetProblem(
            f"`align.episode: {align.episode}` is not in {root} ({total} episodes)"
        )

    columns = ["episode_index", "frame_index", "timestamp"]
    if dtype == "image":
        columns.append(key)
    rows = _episode_rows(_read_columns(root, columns, cache), align.episode, root)
    if not 0 <= align.frame < rows.num_rows:
        raise _AssetProblem(
            f"`align.frame: {align.frame}` is not in episode {align.episode} of "
            f"{root} ({rows.num_rows} frames)"
        )
    row = rows.slice(align.frame, 1)
    timestamp = float(row["timestamp"][0].as_py())
    source: dict[str, Any] = {
        "kind": "dataset",
        "dataset": align.reference_dataset,
        "root": str(root),
        "feature": key,
        "episode": align.episode,
        "frame": align.frame,
        "timestamp": timestamp,
    }

    if dtype == "image":
        import cv2

        value = row[key][0].as_py()
        data = value.get("bytes") if isinstance(value, Mapping) else None
        decoded = (
            cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if data
            else None
        )
        if decoded is None:
            raise _AssetProblem(
                f"{root}: episode {align.episode} frame {align.frame} of {key} holds "
                "no decodable image bytes"
            )
        image = _owned_rgb(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB))
    else:
        image, video_source = _decode_video_frame(align, root, key, timestamp, cache)
        source.update(video_source)
    source["shape"] = list(image.shape)
    return AlignReference(image=image, source=source)


def _decode_video_frame(
    align: BaseAlignConfig,
    root: Path,
    key: str,
    timestamp: float,
    cache: dict[Any, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    missing = [name for name in _METADATA_FILES if not (root / name).is_file()]
    if not any((root / _EPISODES_DIR).glob("**/*.parquet")):
        missing.append(f"{_EPISODES_DIR}/**/*.parquet")
    if missing:
        raise _AssetProblem(
            f"{root} lacks {missing}; lerobot would download meta/ from the hub to "
            "find the video, which the robot never does. Copy the whole dataset."
        )
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.datasets.video_utils import decode_video_frames

    meta_key = ("meta", str(root))
    try:
        if meta_key not in cache:
            cache[meta_key] = LeRobotDatasetMetadata(align.reference_dataset, root=root)
        meta = cache[meta_key]
        path = root / meta.get_video_file_path(align.episode, key)
        from_timestamp = float(
            meta.episodes[align.episode][f"videos/{key}/from_timestamp"]
        )
    except Exception as error:
        raise _AssetProblem(
            f"cannot locate the {key} video of episode {align.episode} in {root}: "
            f"{error!r}"
        ) from error
    if not path.is_file():
        raise _AssetProblem(f"the video {path} is missing; copy the dataset's videos")
    video_timestamp = from_timestamp + timestamp
    try:
        frames = decode_video_frames(path, [video_timestamp], VIDEO_TOLERANCE_S)
    except Exception as error:
        raise _AssetProblem(
            f"cannot decode {path} at {video_timestamp:.4f} s: {error!r}"
        ) from error
    array = (frames[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return _owned_rgb(array), {
        "video": str(path),
        "from_timestamp": from_timestamp,
        "video_timestamp": video_timestamp,
    }


def load_align_reference(
    align: BaseAlignConfig, cache: dict[Any, Any] | None = None
) -> AlignReference:
    """The reference view of one align stage: the image file, else the dataset frame."""
    cache = {} if cache is None else cache
    if align.reference_image:
        return _load_reference_image(align.reference_image)
    return _load_reference_frame(align, cache)


def _robot_problems(
    align: BaseAlignConfig, reference: AlignReference, robot: Robot
) -> list[str]:
    """The reference against what this robot's camera streams, and the ROI against it."""
    problems: list[str] = []
    height, width = reference.image.shape[:2]
    feature = robot.observation_features.get(align.camera)
    if feature is None:
        problems.append(
            f"robot {robot.name!r} has no `{align.camera}` camera observation"
        )
    elif tuple(feature) != tuple(reference.image.shape):
        problems.append(
            f"the reference is {width}x{height}, but robot camera `{align.camera}` "
            f"streams {tuple(feature)} (H, W, C); the estimator compares pixels, "
            "so the two must be the same size"
        )
    roi = align.landmark_roi
    if len(roi) == 4:
        x, y, w, h = roi
        if x < 0 or y < 0 or x + w > width or y + h > height:
            problems.append(
                f"`align.landmark_roi: {list(roi)}` reaches outside the "
                f"{width}x{height} reference"
            )
    return problems


def _fallback_problems(align: BaseAlignConfig, reference: AlignReference) -> list[str]:
    """The width fallback is only a scale when the ROI is the reference's rack blob.

    landmark_align.measure measures the live dark_blob_box, which is always the
    dark rack, and divides its width by the ROI's; the field used that only with
    no --roi, i.e. with the ROI = dark_blob_box(reference). measure itself skips
    the fallback otherwise, so this refusal changes nothing the base does: it
    tells the operator before connect that the option they turned on is dead.
    """
    if not align.distance_width_fallback or len(align.landmark_roi) != 4:
        return []
    from stage_runner import landmark_align

    auto = landmark_align.dark_blob_box(reference.image)
    if auto is not None and tuple(auto) == tuple(align.landmark_roi):
        return []
    return [
        f"`align.distance_width_fallback: true` needs `landmark_roi` to be the "
        f"reference's dark rack box {None if auto is None else list(auto)}, but it "
        f"is {list(align.landmark_roi)}: the fallback compares the live rack box "
        "width with the ROI width, which is no scale for another landmark. Set "
        "distance_width_fallback: false, or use that box as the ROI."
    ]


def load_base_assets(
    config: StageRunnerConfig, robot: Robot
) -> dict[str, ReplayTargets | AlignReference]:
    """Load every base stage's targets or reference, keyed by stage id.

    Empty for a chain without base stages, which then reads nothing. Runs after
    run_preflight, so each base stage has its block; a stage without one is
    skipped here (preflight has already refused it).
    """
    assets: dict[str, ReplayTargets | AlignReference] = {}
    problems: list[str] = []
    # Shared by every stage of this call: one parquet read per dataset however
    # many stages use it.
    cache: dict[Any, Any] = {}
    for index, stage in enumerate(config.stages):
        where = f"stages[{index}] ({stage.id!r})"
        try:
            if stage.executor == EXECUTOR_BASE_REPLAY and stage.replay is not None:
                targets = load_replay_targets(stage.replay, cache)
                problems.extend(
                    f"{where}: {problem}"
                    for problem in _ceiling_problems(stage, targets)
                )
                problems.extend(
                    f"{where}: {problem}" for problem in _rate_problems(stage, targets)
                )
                assets[stage.id] = targets
                logger.info(
                    f"stage {stage.id!r}: {targets.rows} replay targets from "
                    f"{targets.source['root']} episode {stage.replay.episode} rows "
                    f"{stage.replay.from_row}..{stage.replay.until_row} (final "
                    f"{targets.source['final_target_s_m']:+.3f} m, "
                    f"{targets.source['final_target_theta_deg']:+.1f} deg)"
                )
            elif stage.executor == EXECUTOR_BASE_ALIGN and stage.align is not None:
                reference = load_align_reference(stage.align, cache)
                problems.extend(
                    f"{where}: {problem}"
                    for problem in _robot_problems(stage.align, reference, robot)
                )
                problems.extend(
                    f"{where}: {problem}"
                    for problem in _fallback_problems(stage.align, reference)
                )
                assets[stage.id] = reference
                logger.info(f"stage {stage.id!r}: align reference {reference.source}")
        except _AssetProblem as error:
            problems.append(f"{where}: {error}")
    if problems:
        raise PreflightError(
            "cannot load the base stage assets:\n  - " + "\n  - ".join(problems)
        )
    return assets


def _ceiling_problems(stage: StageConfig, targets: ReplayTargets) -> list[str]:
    """A replay ceiling too short for its own targets would abort a healthy replay."""
    timeout_s = stage.terminator.timeout_s
    period = stage.replay.row_period_s if stage.replay is not None else 0.0
    needed = base_stages.replay_ceiling_floor_s(targets.rows, period)
    if timeout_s is None or timeout_s >= needed:
        return []
    return [
        f"`terminator.timeout_s: {timeout_s}` is shorter than this replay needs "
        f"({targets.rows} rows at {period:.4f} s, plus "
        f"{base_stages.PREROLL_S} s pre-roll and {base_stages.REPLAY_SETTLE_MAX_S} s "
        f"settle): raise terminator.timeout_s to at least {math.ceil(needed)}."
    ]


def target_peak_rates(
    targets: ReplayTargets, row_period_s: float
) -> tuple[float, float]:
    """The fastest the replay's targets move: (m/s, rad/s) over one row step."""
    if targets.rows < 2:
        return 0.0, 0.0
    step_m = float(np.max(np.abs(np.diff(targets.s_m))))
    step_rad = float(np.max(np.abs(np.diff(targets.theta_rad))))
    return step_m / row_period_s, step_rad / row_period_s


def _rate_problems(stage: StageConfig, targets: ReplayTargets) -> list[str]:
    """Targets faster than the robot has replayed would make a full-speed drive.

    The target at time t is row (t - t0) / row_period_s, so its speed is the
    largest row step over row_period_s; above base_stages'
    REPLAY_TARGET_MAX_*_VEL the replay is refused (see there for why), naming
    the row_period_s that would bring it back under the limits.
    """
    if stage.replay is None:
        return []
    period = float(stage.replay.row_period_s)
    x_vel, theta_vel = target_peak_rates(targets, period)
    x_limit = base_stages.REPLAY_TARGET_MAX_X_VEL_M_S
    theta_limit = base_stages.REPLAY_TARGET_MAX_THETA_VEL_RAD_S
    if x_vel <= x_limit and theta_vel <= theta_limit:
        return []
    fitting = period * max(x_vel / x_limit, theta_vel / theta_limit)
    return [
        f"at `replay.row_period_s: {period:.4f}` the targets peak at "
        f"{x_vel:.2f} m/s and {theta_vel:.2f} rad/s, above the replay limits "
        f"{x_limit:.2f} m/s / {theta_limit:.2f} rad/s (the fastest replay run on "
        f"the robot peaked at 0.49 m/s / 0.57 rad/s at 1/21 s per row): check "
        f"row_period_s and the dataset, or set row_period_s to at least "
        f"{fitting:.4f} to replay this demo slower."
    ]
