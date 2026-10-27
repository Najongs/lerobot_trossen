"""The OFFLINE half of the start pose: which demonstration, and the YAML it emits.

Everything here runs on synthetic data -- a hand-built LeRobot-shaped directory
(meta/info.json + data/*.parquet) -- so it needs neither the network nor a
dataset download. What it pins is the user-approved formula
(stage_runner/start_pose.py docstring): first frame of every episode, arm joint
columns BY NAME, per-joint median of the std-standardised values, and the REAL
episode nearest to it -- never the median itself.

    uv run --no-sync --with pytest python -m pytest packages/stage_runner/tests/test_start_pose.py
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from stage_runner import config, start_pose
from stage_runner.start_pose import FirstFrames, pick_start_pose

# The Mobile AI kit's observation.state names WITH the base, in recording order
# (the cached kiroaiseoul/task04 dataset's meta/info.json reads exactly this).
# Grippers sit INSIDE the list, after each arm's six joints, and the base at the
# end -- so a selection by position ("the first 12") would take the left gripper
# and drop right_joint_5. That is the bug the by-name rule exists to rule out.
KIT_STATE_NAMES = [
    *(f"left_joint_{index}.pos" for index in range(6)),
    "left_left_carriage_joint.pos",
    *(f"right_joint_{index}.pos" for index in range(6)),
    "right_left_carriage_joint.pos",
    "x.vel",
    "theta.vel",
]
KIT_ARM_NAMES = [
    *(f"left_joint_{index}.pos" for index in range(6)),
    *(f"right_joint_{index}.pos" for index in range(6)),
]


def _frames(rows: list[list[float]], names: list[str] | None = None) -> FirstFrames:
    names = names or [f"left_joint_{index}.pos" for index in range(len(rows[0]))]
    return FirstFrames(
        joint_names=tuple(names),
        episode_indices=tuple(range(len(rows))),
        values=np.array(rows, dtype=np.float64),
    )


def test_the_pick_is_the_nearest_real_episode_not_the_median() -> None:
    """The per-joint median here is (2, 2), which NO episode has.

    Both columns are permutations of 0..4, so their std is the same (sqrt 2) and
    the standardised distances are the raw ones over sqrt 2: e2=(2,3) is 1 away
    from the median, every other episode further. A pick that returned the median
    itself would return (2, 2) -- a pose combining joint_0 of e2/e4 with joint_1
    of e4, which no demonstration started from.
    """
    rows = [[0.0, 4.0], [1.0, 0.0], [2.0, 3.0], [3.0, 1.0], [4.0, 2.0]]
    frames = _frames(rows)
    median = np.median(frames.values, axis=0)
    assert median.tolist() == [2.0, 2.0]
    assert median.tolist() not in rows, "fixture must have a median no episode has"

    pick = pick_start_pose(frames)

    assert pick.episode_index == 2
    assert pick.positions == {"left_joint_0.pos": 2.0, "left_joint_1.pos": 3.0}
    assert pick.positions != dict(zip(frames.joint_names, median.tolist(), strict=True))
    assert pick.n_episodes == 5
    assert math.isclose(pick.distance, 1.0 / math.sqrt(2.0))
    assert pick.method == start_pose.METHOD


def test_standardising_makes_the_pick_independent_of_joint_scale() -> None:
    """A joint that varies 1000x more must not decide "typical" on its own.

    Scaling one column changes every raw L2 distance but no standardised one, so
    the pick must not move -- and the fixture is built so that the RAW nearest
    episode (dominated by the scaled column) is a different one, which is what
    makes this a test of the standardisation and not of nothing.
    """
    rows = [[0.0, 4.0], [1.0, 0.0], [2.0, 3.0], [3.0, 1.0], [4.0, 2.0]]
    scaled = [[a, b * 1000.0] for a, b in rows]
    raw = np.array(scaled)
    raw_nearest = int(np.argmin(np.linalg.norm(raw - np.median(raw, axis=0), axis=1)))

    pick = pick_start_pose(_frames(scaled))

    assert pick.episode_index == 2
    assert raw_nearest != pick.episode_index, (
        "fixture must make raw L2 disagree, or this proves nothing about std"
    )


def test_a_zero_spread_joint_is_guarded_and_contributes_nothing() -> None:
    """Every episode has joint_2 = 0.7: std 0 must not become a division by zero."""
    rows = [[0.0, 4.0, 0.7], [1.0, 0.0, 0.7], [2.0, 3.0, 0.7], [3.0, 1.0, 0.7]]
    pick = pick_start_pose(_frames(rows))
    without = pick_start_pose(_frames([row[:2] for row in rows]))
    assert math.isfinite(pick.distance)
    assert pick.episode_index == without.episode_index
    assert math.isclose(pick.distance, without.distance)
    assert pick.positions["left_joint_2.pos"] == 0.7


def test_a_single_episode_is_its_own_pick() -> None:
    pick = pick_start_pose(_frames([[0.3, -0.2]]))
    assert pick.episode_index == 0
    assert pick.distance == 0.0


def test_ties_go_to_the_lowest_episode_index() -> None:
    """Two equidistant episodes: the same dataset must always give the same pose."""
    frames = FirstFrames(
        joint_names=("left_joint_0.pos",),
        episode_indices=(7, 3, 5),
        values=np.array([[1.0], [-1.0], [0.0]]),
    )
    # median 0.0 -> episode 5 (distance 0). Drop it and 7 and 3 tie.
    assert pick_start_pose(frames).episode_index == 5
    tied = FirstFrames(
        joint_names=("left_joint_0.pos",),
        episode_indices=(7, 3),
        values=np.array([[1.0], [-1.0]]),
    )
    assert pick_start_pose(tied).episode_index == 3


def test_arm_columns_are_chosen_by_name_not_by_position() -> None:
    """Grippers, the base and per-joint velocities are all excluded, wherever they sit."""
    names = [
        "left_joint_0.vel",  # include_velocity puts these in observation.state
        *KIT_STATE_NAMES,
    ]
    selected = [names[index] for index in start_pose.arm_joint_columns(names)]
    assert selected == KIT_ARM_NAMES
    assert not any("carriage" in name for name in selected)
    assert "x.vel" not in selected and "theta.vel" not in selected
    with pytest.raises(ValueError, match="no arm joint column"):
        start_pose.arm_joint_columns(
            ["x.vel", "theta.vel", "left_left_carriage_joint.pos"]
        )


def _write_dataset(
    root: Path,
    episodes: dict[int, list[list[float]]],
    names: list[str],
    *,
    files: int = 2,
) -> Path:
    """A LeRobot-shaped directory: meta/info.json + episodes split over parquet files.

    Episodes are spread over several files on purpose -- the reader must scan all
    of them, and a first frame that lives in file-001 must still be found.
    """
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v3.0",
                "total_episodes": len(episodes),
                "features": {
                    "observation.state": {
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
        episode_column, frame_column, state_column = [], [], []
        for episode, frames in items[file_index::files]:
            for frame_index, state in enumerate(frames):
                episode_column.append(episode)
                frame_column.append(frame_index)
                state_column.append(state)
        table = pa.table(
            {
                "episode_index": pa.array(episode_column, pa.int64()),
                "frame_index": pa.array(frame_column, pa.int64()),
                "observation.state": pa.array(state_column, pa.list_(pa.float32())),
            }
        )
        pq.write_table(table, chunk / f"file-{file_index:03d}.parquet")
    return root


def _kit_state(arm: list[float], *, gripper: float, base: float = 0.0) -> list[float]:
    """Assemble one 16-dim kit state from 12 arm values plus gripper/base fillers."""
    return [*arm[:6], gripper, *arm[6:], gripper, base, base]


def test_read_first_frames_takes_frame_zero_arm_columns_only(tmp_path: Path) -> None:
    """Only frame 0 counts, only arm columns, and a gripper outlier changes nothing.

    Episode 1's gripper is wildly different from the others and its LATER frames
    are far away; with the gripper column or the later frames included it would
    not be picked. Its first-frame ARM pose is the median-nearest, so it must be.
    A NaN first frame drops that episode from n, with the drop reported.
    """
    base_arm = [0.1 * index for index in range(12)]

    def shifted(delta: float) -> list[float]:
        return [value + delta for value in base_arm]

    episodes = {
        0: [
            _kit_state(shifted(-0.2), gripper=0.02),
            _kit_state(shifted(5.0), gripper=0.02),
        ],
        1: [
            _kit_state(shifted(0.0), gripper=0.09),
            _kit_state(shifted(9.0), gripper=0.09),
        ],
        2: [_kit_state(shifted(0.3), gripper=0.02)],
        3: [
            _kit_state(shifted(-0.1), gripper=0.02),
            _kit_state(shifted(-0.1), gripper=0.02),
        ],
        4: [_kit_state(shifted(0.2), gripper=0.02)],
        5: [_kit_state([float("nan")] + base_arm[1:], gripper=0.02)],
    }
    root = _write_dataset(tmp_path / "ds", episodes, KIT_STATE_NAMES, files=3)

    frames = start_pose.read_first_frames(root)

    assert frames.joint_names == tuple(KIT_ARM_NAMES)
    assert frames.episode_indices == (0, 1, 2, 3, 4)
    assert frames.dropped_episode_indices == (5,)
    assert frames.values.shape == (5, 12)
    pick = pick_start_pose(frames)
    assert pick.episode_index == 1
    assert pick.n_episodes == 5
    assert set(pick.positions) == set(KIT_ARM_NAMES)
    for name, expected in zip(KIT_ARM_NAMES, base_arm, strict=True):
        # float32 storage: compare at float32 precision, not exactly.
        assert math.isclose(pick.positions[name], expected, abs_tol=1e-6), name


def test_names_must_come_from_meta_info(tmp_path: Path) -> None:
    """No names -> refuse; the arm joints are never guessed from column positions."""
    root = _write_dataset(tmp_path / "ds", {0: [[0.0, 0.0]]}, ["a", "b"])
    info = json.loads((root / "meta" / "info.json").read_text())
    info["features"]["observation.state"]["names"] = None
    (root / "meta" / "info.json").write_text(json.dumps(info))
    with pytest.raises(ValueError, match="selected by name"):
        start_pose.read_first_frames(root)


def _pick() -> start_pose.StartPosePick:
    return start_pose.StartPosePick(
        positions={
            name: 0.1 * (index + 1) + 1e-9 for index, name in enumerate(KIT_ARM_NAMES)
        },
        episode_index=17,
        n_episodes=50,
        distance=0.8312345678901234,
    )


def test_the_emitted_block_carries_its_provenance_and_round_trips() -> None:
    """Paste-ready: every provenance field present, every float bit-for-bit back."""
    pick = _pick()
    block = start_pose.format_start_pose_block(
        "s4",
        pick,
        dataset_repo_id="kiroaiseoul/task04_pour_liquid_from_tubes_to_beaker",
        dataset_revision="v3.0",
        dataset_commit="6178e4e51f3ed5a089100b5c9b85611ab3ec2dd2",
        policy_path="kiroaiseoul/act_task04_pour_liquid_from_tubes_to_beaker_60000",
        min_goal_time_s=3.0,
        position_tolerance_rad=0.05,
        dropped_episode_indices=[5],
        generated_at=datetime(2026, 10, 19, 12, 0, tzinfo=timezone.utc),
    )
    # The first line is a comment telling the operator where to paste, and the
    # block itself sits at a stage key's indent.
    assert block.startswith("# stage s4: paste under `- id: s4`")
    assert "\n    start_pose:\n" in block
    parsed = yaml.safe_load(block)["start_pose"]
    assert parsed["positions"] == pick.positions, "floats must round-trip exactly"
    assert list(parsed["positions"]) == KIT_ARM_NAMES, "key order = name order"
    assert parsed["min_goal_time_s"] == 3.0
    assert parsed["position_tolerance_rad"] == 0.05
    assert parsed["source"] == {
        "dataset_repo_id": "kiroaiseoul/task04_pour_liquid_from_tubes_to_beaker",
        "dataset_revision": "v3.0",
        "dataset_commit": "6178e4e51f3ed5a089100b5c9b85611ab3ec2dd2",
        "policy_path": "kiroaiseoul/act_task04_pour_liquid_from_tubes_to_beaker_60000",
        "episode_index": 17,
        "n_episodes": 50,
        "method": start_pose.METHOD,
        "distance": 0.8312345678901234,
    }
    assert "[5]" in block, "dropped episodes must be named in the block"


def test_the_emitted_block_parses_as_a_stage_config(tmp_path: Path) -> None:
    """The real consumer is draccus, not yaml.safe_load: paste it and parse it.

    Spliced under the s4 entry of the hardware-free smoke config, exactly as the
    README tells the operator to, and read back through config.parse_config.
    Every field of StartPoseConfig / StartPoseSourceConfig must land, which is
    also what keeps the hand-built emitter and the dataclasses in step.
    """
    pick = _pick()
    block = start_pose.format_start_pose_block(
        "s4",
        pick,
        dataset_repo_id="kiroaiseoul/task04",
        dataset_revision=None,
        dataset_commit=None,
        policy_path="mock://sine",
        min_goal_time_s=4.0,
        position_tolerance_rad=0.02,
    )
    source = (Path(__file__).parent / "configs" / "smoke_mock.yaml").read_text()
    anchor = "    precondition: null\n"
    first = source.index(anchor) + len(anchor)
    spliced = source[:first] + block + source[first:]
    path = tmp_path / "posed.yaml"
    path.write_text(spliced)

    parsed = config.parse_config(["--config_path", str(path)])

    posed, untouched = parsed.stages
    assert untouched.start_pose is None
    assert posed.start_pose is not None
    assert posed.start_pose.positions == pick.positions
    assert posed.start_pose.min_goal_time_s == 4.0
    assert posed.start_pose.position_tolerance_rad == 0.02
    assert posed.start_pose.source == config.StartPoseSourceConfig(
        dataset_repo_id="kiroaiseoul/task04",
        dataset_revision=None,
        dataset_commit=None,
        policy_path="mock://sine",
        episode_index=17,
        n_episodes=50,
        method=start_pose.METHOD,
        distance=0.8312345678901234,
    )


def test_training_dataset_is_read_from_the_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "pretrained_model"
    checkpoint.mkdir()
    (checkpoint / "train_config.json").write_text(
        json.dumps({"dataset": {"repo_id": "kiroaiseoul/task04", "revision": "abc"}})
    )
    assert start_pose.training_dataset(str(checkpoint)) == start_pose.TrainingDataset(
        repo_id="kiroaiseoul/task04", revision="abc"
    )
    (checkpoint / "train_config.json").write_text(
        json.dumps(
            {
                "dataset": {
                    "repo_id": "kiroaiseoul/task04",
                    "revision": None,
                    "root": "/home/dgx-2/data/task04",
                    "episodes": [3, 1, 1],
                }
            }
        )
    )
    assert start_pose.training_dataset(str(checkpoint)) == start_pose.TrainingDataset(
        repo_id="kiroaiseoul/task04",
        revision=None,
        root="/home/dgx-2/data/task04",
        episodes=(1, 3),
    )
    empty = tmp_path / "no_train_config"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="--dataset"):
        start_pose.training_dataset(str(empty))


def test_the_cli_writes_one_block_per_stage_from_local_datasets(tmp_path: Path) -> None:
    """End to end without the network: --stage + --dataset-root -> pasteable blocks."""
    arm = [0.05 * index for index in range(12)]
    episodes = {
        episode: [_kit_state([value + 0.01 * episode for value in arm], gripper=0.03)]
        for episode in range(5)
    }
    root = _write_dataset(tmp_path / "ds", episodes, KIT_STATE_NAMES)
    output = tmp_path / "poses.yaml"

    code = start_pose.main(
        [
            "--stage",
            "s4=kiroaiseoul/act_task04_60000",
            "--dataset",
            "s4=kiroaiseoul/task04",
            "--dataset-root",
            f"s4={root}",
            "--output",
            str(output),
        ]
    )

    assert code == 0
    parsed = yaml.safe_load(output.read_text())["start_pose"]
    # Five equally spaced episodes: the median is episode 2's pose exactly.
    assert parsed["source"]["episode_index"] == 2
    assert parsed["source"]["n_episodes"] == 5
    assert parsed["source"]["dataset_repo_id"] == "kiroaiseoul/task04"
    assert parsed["source"]["policy_path"] == "kiroaiseoul/act_task04_60000"
    # A plain directory is not a hub snapshot, so the commit is unknown -- null,
    # not a guess.
    assert parsed["source"]["dataset_commit"] is None
    assert set(parsed["positions"]) == set(KIT_ARM_NAMES)


def _checkpoint(tmp_path: Path, **dataset: Any) -> Path:
    checkpoint = tmp_path / "pretrained_model"
    checkpoint.mkdir()
    (checkpoint / "train_config.json").write_text(json.dumps({"dataset": dataset}))
    return checkpoint


def _five_episodes(root: Path) -> Path:
    arm = [0.05 * index for index in range(12)]
    episodes = {
        episode: [_kit_state([value + 0.01 * episode for value in arm], gripper=0.03)]
        for episode in range(5)
    }
    return _write_dataset(root, episodes, KIT_STATE_NAMES)


def _hub_cache_copy(tmp_path: Path, repo_id: str, commit: str) -> Path:
    org, name = repo_id.split("/")
    return _five_episodes(
        tmp_path / "hub" / f"datasets--{org}--{name}" / "snapshots" / commit
    )


def test_a_checkpoint_trained_from_a_local_root_needs_that_root(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """train_config dataset.root set -> no silent read of the hub copy."""
    checkpoint = _checkpoint(
        tmp_path,
        repo_id="kiroaiseoul/task10_14D_basepose",
        root="/home/dgx-2/jwyoon/data/task10_14D_basepose",
    )
    with mock.patch.object(start_pose, "download_dataset") as download:
        code = start_pose.main(["--stage", f"s10={checkpoint}"])
    assert code == 2
    download.assert_not_called()
    error = capsys.readouterr().err
    assert "/home/dgx-2/jwyoon/data/task10_14D_basepose" in error
    assert "--dataset-root s10=" in error

    # Given the local copy, the block names the TRAINING dataset, not the path.
    root = _five_episodes(tmp_path / "copy")
    output = tmp_path / "poses.yaml"
    code = start_pose.main(
        [
            "--stage",
            f"s10={checkpoint}",
            "--dataset-root",
            f"s10={root}",
            "--output",
            str(output),
        ]
    )
    assert code == 0
    text = output.read_text()
    source = yaml.safe_load(text)["start_pose"]["source"]
    assert source["dataset_repo_id"] == "kiroaiseoul/task10_14D_basepose"
    assert f"# read from the local copy {root}" in text


def test_dataset_root_alone_never_records_a_path_as_the_repo_id(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    plain = _five_episodes(tmp_path / "plain")
    # No checkpoint to name the dataset and no --dataset: refused.
    assert (
        start_pose.main(["--stage", "s4=mock://sine", "--dataset-root", f"s4={plain}"])
        == 2
    )
    assert "--dataset s4=<repo_id>" in capsys.readouterr().err

    # A hub-cache snapshot names itself: repo id and commit both recovered.
    cached = _hub_cache_copy(tmp_path, "kiroaiseoul/task04", "6178e4e5")
    output = tmp_path / "out" / "poses.yaml"  # parent does not exist yet
    code = start_pose.main(
        [
            "--stage",
            "s4=mock://sine",
            "--dataset-root",
            f"s4={cached}",
            "--output",
            str(output),
        ]
    )
    assert code == 0
    source = yaml.safe_load(output.read_text())["start_pose"]["source"]
    assert source["dataset_repo_id"] == "kiroaiseoul/task04"
    assert source["dataset_commit"] == "6178e4e5"


def test_a_local_root_of_a_different_dataset_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The checkpoint trained on _14D; the root is a copy of the plain dataset."""
    checkpoint = _checkpoint(tmp_path, repo_id="kiroaiseoul/task10_14D")
    wrong = _hub_cache_copy(tmp_path, "kiroaiseoul/task10", "142820d2")
    code = start_pose.main(
        ["--stage", f"s10={checkpoint}", "--dataset-root", f"s10={wrong}"]
    )
    assert code == 2
    error = capsys.readouterr().err
    assert "'kiroaiseoul/task10'" in error and "'kiroaiseoul/task10_14D'" in error

    right = _hub_cache_copy(tmp_path, "kiroaiseoul/task10_14D", "aaaa")
    assert (
        start_pose.main(
            ["--stage", f"s10={checkpoint}", "--dataset-root", f"s10={right}"]
        )
        == 0
    )


def test_the_training_episode_subset_is_the_set_picked_from(tmp_path: Path) -> None:
    root = _five_episodes(tmp_path / "ds")
    # Without the subset the pick is episode 2 (the median of 0..4).
    checkpoint = _checkpoint(tmp_path, repo_id="kiroaiseoul/task04", episodes=[3, 4])
    output = tmp_path / "poses.yaml"
    code = start_pose.main(
        [
            "--stage",
            f"s4={checkpoint}",
            "--dataset-root",
            f"s4={root}",
            "--output",
            str(output),
        ]
    )
    assert code == 0
    text = output.read_text()
    source = yaml.safe_load(text)["start_pose"]["source"]
    assert source["n_episodes"] == 2
    assert source["episode_index"] == 3
    assert "dataset.episodes lists: [3, 4]" in text

    frames = start_pose.read_first_frames(root)
    with pytest.raises(start_pose.StartPoseError, match=r"\[7\]"):
        start_pose.restrict_to_episodes(frames, [3, 7])


def test_a_dataset_missing_from_the_hub_is_an_actionable_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from huggingface_hub.errors import RepositoryNotFoundError

    def missing(*args: Any, **kwargs: Any) -> Any:
        raise RepositoryNotFoundError("404 (injected)")

    with mock.patch("huggingface_hub.HfApi.dataset_info", missing):
        code = start_pose.main(["--stage", "s4=mock://sine", "--dataset", "s4=local/x"])
    assert code == 2
    error = capsys.readouterr().err
    assert "--dataset-root" in error and "'local/x'" in error


def test_a_mock_stage_without_a_dataset_is_skipped() -> None:
    assert start_pose.main(["--stage", "s4=mock://sine"]) == 2


def test_the_copied_lerobot_constants_still_match_lerobot() -> None:
    from lerobot.configs.train import TRAIN_CONFIG_NAME
    from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION
    from lerobot.utils.constants import OBS_STATE

    assert start_pose.OBS_STATE_KEY == OBS_STATE
    assert start_pose.TRAIN_CONFIG_FILENAME == TRAIN_CONFIG_NAME
    assert start_pose.DEFAULT_DATASET_REVISION == CODEBASE_VERSION
