"""record_adapter.discard_episode -- the `dataset.save: false` cleanup.

The smoke chain's mock robot has no cameras, so it cannot show the case this
exists for: camera frames are VIDEO keys, written as PNG until encoded, and
neither VideoEncodingManager (only on an exception) nor clear_episode_buffer
(image keys only) removes them after a clean, unsaved trial.
"""

from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from stage_runner import record_adapter

CAMERA_KEY = "observation.images.cam"


def test_discard_episode_removes_the_camera_frames(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    dataset = LeRobotDataset.create(
        "local/eval_discard",
        30,
        root=root,
        features={
            CAMERA_KEY: {
                "dtype": "video",
                "shape": (8, 8, 3),
                "names": ["height", "width", "channels"],
            },
            "action": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]},
        },
        use_videos=True,
        image_writer_threads=2,
    )
    for _ in range(3):
        dataset.add_frame(
            {
                CAMERA_KEY: np.zeros((8, 8, 3), dtype=np.uint8),
                "action": np.zeros(2, dtype=np.float32),
                "task": "discard",
            }
        )
    dataset._wait_image_writer()
    assert list((root / "images").rglob("*.png")), "the PNGs this test removes"

    record_adapter.discard_episode(dataset)

    assert not list((root / "images").rglob("*.png"))
    assert dataset.episode_buffer["size"] == 0
    dataset.finalize()
    assert dataset.meta.total_episodes == 0
