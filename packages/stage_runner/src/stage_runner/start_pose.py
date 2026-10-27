"""Compute each stage's arm start pose from its training data, OFFLINE.

    uv run python -m stage_runner.start_pose --config_path configs/chain_task45.yaml

prints, per stage, a ``start_pose:`` YAML block to paste under that stage in the
config. Run it on a machine with network access (or a local copy of the dataset):
the robot PC never downloads a dataset at run time, it only reads the numbers
this wrote.

THE FORMULA (user-approved 2026-10-19, so it is recorded here, not re-derived):

1. The dataset is the stage checkpoint's TRAINING dataset -- ``dataset.repo_id``
   in its ``train_config.json`` -- unless ``--dataset STAGE=REPO`` overrides it.
2. Take ``observation.state`` at ``frame_index == 0`` of every episode, and only
   its ARM JOINT columns, chosen BY NAME from
   ``meta/info.json features["observation.state"]["names"]``: ``.pos`` columns
   that are neither a gripper nor the base (reposition.is_arm_joint_key). Never by
   position: the same index means a different joint the moment
   include_base_in_state or include_velocity changes.
3. Standardise each joint by its across-episode standard deviation (a zero std is
   replaced by 1: every episode then has the same value, so the joint contributes
   0 to every distance either way), take the per-joint median, and pick the REAL
   episode with the smallest standardised L2 distance to it.
4. The pose is THAT EPISODE's first-frame arm values.

WHY A REAL EPISODE AND NOT THE MEDIAN ITSELF. The per-joint median is taken
joint by joint, so it can combine joint_1 from one demonstration with joint_2
from another into a pose no demonstration ever had -- and the policy has then
never seen the camera images or the state of that pose. The nearest real episode
is a pose the policy was trained to start from, chosen to be typical.

Why standardised: the joints vary by very different amounts across demos (a wrist
roll may wander by a radian where the shoulder moves by millimetres), and a raw
L2 would let the noisiest joint alone decide which episode is "typical".
"""

import argparse
import json
import logging
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from stage_runner.reposition import is_arm_joint_key

logger = logging.getLogger(__name__)

METHOD: str = "first_frame_nearest_to_median_std_l2"

# Copies of lerobot's own constants (utils/constants.py OBS_STATE,
# configs/train.py TRAIN_CONFIG_NAME, datasets/lerobot_dataset.py
# CODEBASE_VERSION). configs.train drags in the whole training config tree just to
# name a file, so these are written out and test_start_pose asserts they still
# equal lerobot's -- a rename upstream fails a test instead of a download.
OBS_STATE_KEY: str = "observation.state"
TRAIN_CONFIG_FILENAME: str = "train_config.json"
INFO_PATH: str = "meta/info.json"
# Only these are fetched from the hub: the schema and the frame tables. Not
# videos -- a first frame's state lives in data/, and the videos are most of a
# dataset's bytes.
DOWNLOAD_PATTERNS: tuple[str, ...] = (INFO_PATH, "data/**/*.parquet")

# The default when train_config.json names no revision: lerobot itself loads a
# dataset at its CODEBASE_VERSION tag in that case (LeRobotDatasetMetadata), so
# that tag is the revision the checkpoint was actually trained on.
DEFAULT_DATASET_REVISION: str = "v3.0"

# Where a block lands in the stage YAML: stage list items are "  - id: ...",
# their keys at four spaces (configs/chain_task45.yaml).
STAGE_KEY_INDENT: str = "    "


@dataclass(frozen=True)
class FirstFrames:
    """The first frame of every episode, arm joint columns only, in name order."""

    joint_names: tuple[str, ...]
    episode_indices: tuple[int, ...]
    values: np.ndarray  # shape (n_episodes, n_joints)
    # Episodes dropped because a first-frame arm value was not finite.
    dropped_episode_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class StartPosePick:
    positions: dict[str, float]
    episode_index: int
    n_episodes: int
    distance: float
    method: str = METHOD


def arm_joint_columns(names: Sequence[str]) -> list[int]:
    """Indices of the arm joint columns of observation.state, chosen BY NAME."""
    indices = [index for index, name in enumerate(names) if is_arm_joint_key(str(name))]
    if not indices:
        raise ValueError(
            f"no arm joint column in observation.state names {list(names)}: "
            "expected `<arm>_joint_<n>.pos` entries"
        )
    return indices


def pick_start_pose(frames: FirstFrames) -> StartPosePick:
    """The real episode whose first frame is nearest the per-joint median.

    Distance is the L2 norm of ``(x - median) / std`` over the arm joints, std
    and median taken across episodes. Ties go to the lowest episode_index, so the
    same dataset always yields the same pose.
    """
    values = np.asarray(frames.values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("no usable first frame: the dataset has no episodes left")
    if values.shape[1] != len(frames.joint_names):
        raise ValueError(
            f"{values.shape[1]} columns for {len(frames.joint_names)} joint names"
        )
    std = values.std(axis=0)
    # Zero (or numerically zero) spread: every episode shares the value, so the
    # numerator is 0 for all of them and any positive divisor gives the same
    # ranking. 1.0 keeps that column out of the distance without a division by 0.
    safe_std = np.where(std > 1e-12, std, 1.0)
    median = np.median(values, axis=0)
    distances = np.sqrt((((values - median) / safe_std) ** 2).sum(axis=1))
    order = np.lexsort((np.asarray(frames.episode_indices), distances))
    best = int(order[0])
    return StartPosePick(
        positions={
            name: float(values[best, column])
            for column, name in enumerate(frames.joint_names)
        },
        episode_index=int(frames.episode_indices[best]),
        n_episodes=int(values.shape[0]),
        distance=float(distances[best]),
    )


def _state_names(info: Mapping[str, Any]) -> list[str]:
    feature = (info.get("features") or {}).get(OBS_STATE_KEY)
    if feature is None:
        raise ValueError(f"{INFO_PATH} declares no {OBS_STATE_KEY!r} feature")
    names = feature.get("names")
    # v2.x datasets sometimes nest the list one level down ({"motors": [...]}).
    if isinstance(names, Mapping) and len(names) == 1:
        names = next(iter(names.values()))
    if not isinstance(names, list) or not names:
        raise ValueError(
            f"{INFO_PATH} features[{OBS_STATE_KEY!r}]['names'] is {names!r}; the "
            "arm joints can only be selected by name"
        )
    return [str(name) for name in names]


def read_first_frames(dataset_root: Path) -> FirstFrames:
    """Read frame 0 of every episode from a LeRobot dataset directory.

    Parquet only, and only three columns of it (episode_index, frame_index,
    observation.state), filtered to frame_index == 0 -- the first frames are a
    few hundred rows out of a dataset's hundreds of thousands, and nothing else
    is needed. Every file under data/ is scanned rather than resolved through
    meta/episodes, because the data_path template changed across codebase
    versions and the glob does not care.
    """
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    root = Path(dataset_root)
    info = json.loads((root / INFO_PATH).read_text(encoding="utf-8"))
    names = _state_names(info)
    columns = arm_joint_columns(names)
    files = sorted((root / "data").glob("**/*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files under {root / 'data'}")

    rows: dict[int, list[float]] = {}
    for path in files:
        table = pq.read_table(
            path, columns=["episode_index", "frame_index", OBS_STATE_KEY]
        )
        table = table.filter(pc.equal(table["frame_index"], 0))
        for episode, state in zip(
            table["episode_index"].to_pylist(),
            table[OBS_STATE_KEY].to_pylist(),
            strict=True,
        ):
            if episode in rows:
                raise ValueError(
                    f"episode {episode} has more than one frame_index 0 ({path})"
                )
            if len(state) != len(names):
                raise ValueError(
                    f"episode {episode}: observation.state has {len(state)} values "
                    f"for {len(names)} names in {INFO_PATH}"
                )
            rows[int(episode)] = [float(state[column]) for column in columns]

    expected = info.get("total_episodes")
    if isinstance(expected, int) and expected != len(rows):
        logger.warning(
            f"{root}: meta/info.json says {expected} episodes, found a first frame "
            f"for {len(rows)}"
        )
    kept = sorted(
        episode for episode, row in rows.items() if all(math.isfinite(v) for v in row)
    )
    dropped = tuple(sorted(set(rows) - set(kept)))
    if dropped:
        logger.warning(
            f"{root}: dropped episodes with a non-finite first frame: {dropped}"
        )
    return FirstFrames(
        joint_names=tuple(names[column] for column in columns),
        episode_indices=tuple(kept),
        values=np.array([rows[episode] for episode in kept], dtype=np.float64).reshape(
            len(kept), len(columns)
        ),
        dropped_episode_indices=dropped,
    )


class StartPoseError(Exception):
    """A refusal the operator can act on; main() prints it and exits 2.

    For the cases where carrying on would emit a block whose provenance points at
    bytes that are not the ones read -- the one thing the block must not do.
    """


@dataclass(frozen=True)
class TrainingDataset:
    """What a checkpoint's train_config.json says it was trained on.

    The fields of lerobot's DatasetConfig (configs/default.py) that decide WHICH
    BYTES were trained on: ``repo_id`` + ``revision`` name a hub dataset, ``root``
    a local directory that may not exist on the hub at all (lerobot then loads
    from there and repo_id is only a label), and ``episodes`` a subset.
    """

    repo_id: str
    revision: str | None = None
    root: str | None = None
    episodes: tuple[int, ...] | None = None


def training_dataset(policy_path: str) -> TrainingDataset:
    """The training dataset out of a checkpoint's train_config.json.

    A local directory is read in place (lerobot writes train_config.json beside
    config.json in pretrained_model/); anything else is a hub model id.
    """
    local = Path(policy_path).expanduser()
    if local.is_dir():
        path = local / TRAIN_CONFIG_FILENAME
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} does not exist; point at the pretrained_model/ directory "
                "or pass --dataset STAGE=REPO"
            )
    else:
        from huggingface_hub import hf_hub_download

        try:
            path = Path(
                hf_hub_download(repo_id=policy_path, filename=TRAIN_CONFIG_FILENAME)
            )
        except Exception as error:
            raise StartPoseError(
                f"could not read {TRAIN_CONFIG_FILENAME} of {policy_path!r} "
                f"({type(error).__name__}: {error}); pass --dataset STAGE=REPO "
                "(and --dataset-root STAGE=DIR for a local copy)"
            ) from error
    train_config = json.loads(path.read_text(encoding="utf-8"))
    dataset = train_config.get("dataset") or {}
    repo_id = dataset.get("repo_id")
    if not isinstance(repo_id, str) or not repo_id:
        raise ValueError(
            f"{path}: dataset.repo_id is {repo_id!r}; pass --dataset STAGE=REPO"
        )
    revision = dataset.get("revision")
    root = dataset.get("root")
    episodes = dataset.get("episodes")
    return TrainingDataset(
        repo_id=repo_id,
        revision=str(revision) if revision else None,
        root=str(root) if root else None,
        episodes=(
            tuple(sorted({int(episode) for episode in episodes}))
            if episodes is not None
            else None
        ),
    )


def hub_cache_location(root: Path) -> tuple[str | None, str | None]:
    """(repo_id, commit) of a hub-cache snapshot directory, else (None, None).

    huggingface_hub lays a dataset out as
    ``datasets--<org>--<name>/snapshots/<commit>/``, replacing "/" by "--"; repo
    names cannot themselves contain "--" (huggingface_hub validate_repo_id), so
    the split is unambiguous. lerobot's own cache (~/.cache/huggingface/lerobot/
    hub/) uses the same layout. Anywhere else both are unknown and recorded as
    null rather than guessed.
    """
    resolved = root.expanduser().resolve()
    if resolved.parent.name != "snapshots":
        return None, None
    parts = resolved.parent.parent.name.split("--")
    repo_id = (
        f"{parts[1]}/{parts[2]}" if len(parts) == 3 and parts[0] == "datasets" else None
    )
    return repo_id, resolved.name


def restrict_to_episodes(frames: FirstFrames, episodes: Sequence[int]) -> FirstFrames:
    """Keep only the training subset (train_config.json dataset.episodes).

    A pose picked from episodes the checkpoint never saw is not "where its
    training demos began", and n_episodes would count the wrong set.
    """
    wanted = {int(episode) for episode in episodes}
    present = set(frames.episode_indices) | set(frames.dropped_episode_indices)
    missing = sorted(wanted - present)
    if missing:
        raise StartPoseError(
            f"train_config.json dataset.episodes lists {missing}, which have no "
            "first frame in this dataset: it is not the one the checkpoint was "
            "trained on"
        )
    keep = [
        row for row, episode in enumerate(frames.episode_indices) if episode in wanted
    ]
    return FirstFrames(
        joint_names=frames.joint_names,
        episode_indices=tuple(frames.episode_indices[row] for row in keep),
        values=frames.values[keep, :],
        dropped_episode_indices=tuple(
            episode for episode in frames.dropped_episode_indices if episode in wanted
        ),
    )


def download_dataset(
    repo_id: str, revision: str | None
) -> tuple[Path, str, str | None]:
    """Fetch meta/info.json and data/ only; return (root, revision, commit sha)."""
    from huggingface_hub import HfApi, snapshot_download
    from huggingface_hub.errors import RepositoryNotFoundError, RevisionNotFoundError

    revision = revision or DEFAULT_DATASET_REVISION
    # Resolve the commit FIRST and download that commit, so the recorded
    # provenance is the bytes that were read and not whatever a moving tag
    # points at by the time the download finishes.
    try:
        commit = HfApi().dataset_info(repo_id, revision=revision).sha
    except (RepositoryNotFoundError, RevisionNotFoundError) as error:
        raise StartPoseError(
            f"dataset {repo_id!r} at revision {revision!r} is not on the hub "
            f"({type(error).__name__}). A checkpoint trained from a local copy "
            "needs --dataset-root STAGE=DIR; a different hub dataset --dataset "
            "STAGE=REPO (and --revision STAGE=REV)"
        ) from error
    root = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=commit or revision,
        allow_patterns=list(DOWNLOAD_PATTERNS),
    )
    return Path(root), revision, commit


def _yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        # repr: the shortest string that round-trips to the same float, so the
        # pasted value is bit-for-bit the one the pick chose.
        return repr(value)
    if isinstance(value, int):
        return str(value)
    return json.dumps(str(value), ensure_ascii=False)


def format_start_pose_block(
    stage_id: str,
    pick: StartPosePick,
    *,
    dataset_repo_id: str,
    dataset_revision: str | None,
    dataset_commit: str | None,
    policy_path: str,
    min_goal_time_s: float,
    position_tolerance_rad: float,
    dropped_episode_indices: Sequence[int] = (),
    training_episodes: Sequence[int] | None = None,
    dataset_root: str | None = None,
    generated_at: datetime | None = None,
) -> str:
    """The ``start_pose:`` block for one stage, indented to paste under it.

    Hand-built rather than yaml.dump, for two reasons: the comment lines (which a
    dumper drops) are the instructions the operator reads at paste time, and the
    key order should be the order of StartPoseConfig. test_start_pose parses the
    block back through draccus to keep the two in step.
    """
    stamp = (
        (generated_at or datetime.now(timezone.utc))
        .astimezone()
        .isoformat(timespec="seconds")
    )
    indent = STAGE_KEY_INDENT
    lines = [
        (
            f"# stage {stage_id}: paste under `- id: {stage_id}`, at the same "
            "indent as `terminator:`"
        ),
        f"{indent}start_pose:",
        f"{indent}  # generated by `python -m stage_runner.start_pose` at {stamp}.",
        (
            f"{indent}  # Regenerate instead of hand-editing: the values are "
            f"episode {pick.episode_index}'s first frame."
        ),
        (
            f"{indent}  # Arm joints only -- the grippers and the base are never "
            "commanded by a start pose."
        ),
        f"{indent}  positions:",
    ]
    lines += [
        f"{indent}    {name}: {_yaml_scalar(value)}"
        for name, value in pick.positions.items()
    ]
    lines += [
        f"{indent}  min_goal_time_s: {_yaml_scalar(float(min_goal_time_s))}",
        f"{indent}  position_tolerance_rad: {_yaml_scalar(float(position_tolerance_rad))}",
        f"{indent}  source:",
        f"{indent}    dataset_repo_id: {_yaml_scalar(dataset_repo_id)}",
        f"{indent}    dataset_revision: {_yaml_scalar(dataset_revision)}",
        f"{indent}    dataset_commit: {_yaml_scalar(dataset_commit)}",
        f"{indent}    policy_path: {_yaml_scalar(policy_path)}",
        f"{indent}    episode_index: {pick.episode_index}",
        f"{indent}    n_episodes: {pick.n_episodes}",
        f"{indent}    method: {_yaml_scalar(pick.method)}",
        f"{indent}    distance: {_yaml_scalar(pick.distance)}",
    ]
    if dropped_episode_indices:
        lines.append(
            f"{indent}    # excluded from n_episodes (non-finite first frame): "
            f"{list(dropped_episode_indices)}"
        )
    # Comments, not schema fields: StartPoseSourceConfig stays as it is, and the
    # block still records how to reproduce the pick.
    if training_episodes is not None:
        lines.append(
            f"{indent}    # restricted to the {len(training_episodes)} episodes "
            f"train_config.json dataset.episodes lists: {list(training_episodes)}"
        )
    if dataset_root is not None:
        lines.append(f"{indent}    # read from the local copy {dataset_root}")
    return "\n".join(lines) + "\n"


def _parse_assignments(values: Sequence[str], flag: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        stage_id, separator, target = value.partition("=")
        if not separator or not stage_id or not target:
            raise SystemExit(f"{flag} expects STAGE_ID=VALUE, got {value!r}")
        parsed[stage_id] = target
    return parsed


def _stages_from_config(config_path: Path) -> list[tuple[str, str]]:
    """(id, policy_path) of every stage, read with plain YAML.

    Not through config.parse_config: that needs the robot plugin registered and
    the robot block resolved, none of which this tool uses, and it should run on
    a laptop without the Trossen drivers.
    """
    import yaml

    document = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    stages = document.get("stages") or []
    return [
        (str(stage.get("id", "")), str(stage.get("policy_path", "") or ""))
        for stage in stages
    ]


def _stage_block(
    stage_id: str,
    policy_path: str,
    *,
    dataset: str | None,
    revision: str | None,
    root: str | None,
    min_goal_time_s: float,
    position_tolerance_rad: float,
) -> str | None:
    """One stage's block, None for a stage with no dataset to read (skipped).

    WHICH DATASET, in order: ``--dataset`` (the operator's explicit override, so
    train_config.json is not consulted), else the checkpoint's training dataset.
    ``--dataset-root`` only says where the bytes are; it never names the dataset
    by itself -- a filesystem path in dataset_repo_id would lose the provenance
    the block exists to carry. Every way the bytes read could differ from the
    bytes named is refused rather than recorded:

    * the checkpoint trained from a LOCAL root (train_config dataset.root) and no
      --dataset-root was given -> the hub copy, if one exists at all, may not be
      those bytes;
    * --dataset-root is a hub-cache snapshot of a DIFFERENT repo than the one
      named;
    * train_config's dataset.episodes lists an episode the dataset lacks.

    A plain directory cannot be checked against the name; that is a warning.
    """
    trained: TrainingDataset | None = None
    if dataset is None and policy_path and not policy_path.startswith("mock://"):
        trained = training_dataset(policy_path)
    repo_id = dataset or (trained.repo_id if trained else None)
    if repo_id is None:
        if root is not None:
            cached_repo, _ = hub_cache_location(Path(root))
            if cached_repo is None:
                raise StartPoseError(
                    f"--dataset-root {root} names no dataset and policy_path "
                    f"{policy_path!r} has no train_config.json to name one; add "
                    f"--dataset {stage_id}=<repo_id> so the block records it"
                )
            repo_id = cached_repo
        else:
            logger.warning(
                f"stage {stage_id!r}: policy_path {policy_path!r} has no training "
                "dataset; skipped (give --dataset or --dataset-root)"
            )
            return None

    if root is not None:
        local = Path(root).expanduser()
        cached_repo, commit = hub_cache_location(local)
        if cached_repo is not None and cached_repo != repo_id:
            raise StartPoseError(
                f"--dataset-root {root} is the hub-cache copy of {cached_repo!r}, "
                f"but the block would name {repo_id!r}"
                + (f" (the training dataset of {policy_path})" if trained else "")
                + "; point --dataset-root at the right copy"
            )
        if cached_repo is None:
            logger.warning(
                f"stage {stage_id!r}: {local} is not a hub-cache snapshot, so it "
                f"cannot be checked to be {repo_id!r}; the block records that name"
            )
        frames_root = local
        dataset_root: str | None = str(local)
        # The bytes' revision is only known through the snapshot's commit.
        recorded_revision = revision
    else:
        if trained is not None and trained.root is not None:
            raise StartPoseError(
                f"{policy_path} was trained from the local directory "
                f"{trained.root} (train_config.json dataset.root), so the hub "
                f"copy of {repo_id!r}, if there is one, may not be those bytes. "
                f"Pass --dataset-root {stage_id}=<a copy of that directory>, or "
                f"--dataset {stage_id}={repo_id} to read the hub copy deliberately"
            )
        wanted = revision or (trained.revision if trained else None)
        frames_root, recorded_revision, commit = download_dataset(repo_id, wanted)
        dataset_root = None

    frames = read_first_frames(frames_root)
    training_episodes = trained.episodes if trained is not None else None
    if training_episodes is not None:
        frames = restrict_to_episodes(frames, training_episodes)
    pick = pick_start_pose(frames)
    logger.info(
        f"stage {stage_id!r}: episode {pick.episode_index} of {pick.n_episodes} "
        f"in {repo_id} (distance {pick.distance:.3f})"
    )
    return format_start_pose_block(
        stage_id,
        pick,
        dataset_repo_id=repo_id,
        dataset_revision=recorded_revision,
        dataset_commit=commit,
        policy_path=policy_path,
        min_goal_time_s=min_goal_time_s,
        position_tolerance_rad=position_tolerance_rad,
        dropped_episode_indices=frames.dropped_episode_indices,
        training_episodes=training_episodes,
        dataset_root=dataset_root,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m stage_runner.start_pose",
        description=(
            "Print a start_pose block per stage: the first-frame arm pose of the "
            "training episode nearest the per-joint median."
        ),
    )
    parser.add_argument(
        "--config_path", type=Path, help="stage_runner YAML to read stages from"
    )
    parser.add_argument(
        "--stage",
        action="append",
        default=[],
        metavar="ID[=POLICY_PATH]",
        help="only these stages; with =POLICY_PATH, no --config_path is needed",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        metavar="ID=REPO_ID",
        help="use this dataset instead of the checkpoint's train_config.json one",
    )
    parser.add_argument(
        "--revision",
        action="append",
        default=[],
        metavar="ID=REV",
        help="dataset revision for that stage (default: the training config's, else v3.0)",
    )
    parser.add_argument(
        "--dataset-root",
        action="append",
        default=[],
        metavar="ID=DIR",
        help="read a local dataset directory instead of downloading",
    )
    parser.add_argument("--min-goal-time-s", type=float, default=3.0)
    parser.add_argument("--position-tolerance-rad", type=float, default=0.05)
    parser.add_argument(
        "--output", type=Path, help="also write the blocks to this file"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    datasets = _parse_assignments(args.dataset, "--dataset")
    revisions = _parse_assignments(args.revision, "--revision")
    roots = _parse_assignments(args.dataset_root, "--dataset-root")

    stages: list[tuple[str, str]] = (
        _stages_from_config(args.config_path) if args.config_path else []
    )
    explicit: dict[str, str | None] = {}
    for value in args.stage:
        stage_id, _, policy_path = value.partition("=")
        explicit[stage_id] = policy_path or None
    if explicit:
        known = dict(stages)
        stages = [
            (stage_id, policy_path or known.get(stage_id, ""))
            for stage_id, policy_path in explicit.items()
        ]
    if not stages:
        print("no stages: pass --config_path and/or --stage", file=sys.stderr)
        return 2
    if not math.isfinite(args.min_goal_time_s) or args.min_goal_time_s <= 0:
        print("--min-goal-time-s must be a positive number of seconds", file=sys.stderr)
        return 2
    if (
        not math.isfinite(args.position_tolerance_rad)
        or args.position_tolerance_rad <= 0
    ):
        print("--position-tolerance-rad must be a positive number", file=sys.stderr)
        return 2

    blocks: list[str] = []
    for stage_id, policy_path in stages:
        try:
            block = _stage_block(
                stage_id,
                policy_path,
                dataset=datasets.get(stage_id),
                revision=revisions.get(stage_id),
                root=roots.get(stage_id),
                min_goal_time_s=args.min_goal_time_s,
                position_tolerance_rad=args.position_tolerance_rad,
            )
        except StartPoseError as error:
            print(f"stage {stage_id!r}: {error}", file=sys.stderr)
            return 2
        if block is not None:
            blocks.append(block)
    text = "\n".join(blocks)
    sys.stdout.write(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    return 0 if blocks else 2


if __name__ == "__main__":
    sys.exit(main())
