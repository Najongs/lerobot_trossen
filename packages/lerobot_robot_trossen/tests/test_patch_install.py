"""What gets installed for which environment, checked in fresh interpreters.

The switches are frozen at import, so every case runs ``record_loop_probe.py`` in
its own process with its own environment.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

PROBE = Path(__file__).resolve().parent / "record_loop_probe.py"
SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
SWITCH_VARIABLES = (
    "LEROBOT_CHUNK_PREFETCH_TICKS",
    "LEROBOT_CHUNK_PREFETCH_INLINE",
    "LEROBOT_BASE_LEAD_TICKS",
    "LEROBOT_REPLAY_DATASET",
    "LEROBOT_REPLAY_EPISODES",
    "LEROBOT_REPLAY_RUN_POLICY",
    "LEROBOT_CHUNK_EXECUTION_LOG",
    "LEROBOT_LOOP_HZ_LOG",
)


def probe(extra_environment: dict, *arguments: str) -> tuple[dict, str]:
    environment = {
        key: value for key, value in os.environ.items() if key not in SWITCH_VARIABLES
    }
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(SOURCE_ROOT), environment.get("PYTHONPATH", "")]
    )
    environment.update(extra_environment)
    completed = subprocess.run(
        [sys.executable, str(PROBE), *arguments],
        env=environment,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stderr[-3000:]
    report = json.loads(completed.stdout.strip().splitlines()[-1])
    # The code under test must be this checkout, not an editable install elsewhere.
    assert report["package_file"].startswith(str(SOURCE_ROOT)), report["package_file"]
    return report, completed.stderr


LOOP_RATE = "loop_rate_log.py"
OURS = "chunk_execution_patch.py*"


@pytest.mark.parametrize(
    ("environment", "expect_ours", "expect_loop_rate"),
    [
        ({}, False, True),
        ({"LEROBOT_LOOP_HZ_LOG": "0"}, False, False),
        ({"LEROBOT_CHUNK_PREFETCH_TICKS": "3"}, True, True),
        ({"LEROBOT_BASE_LEAD_TICKS": "2", "LEROBOT_LOOP_HZ_LOG": "0"}, True, False),
        ({"LEROBOT_CHUNK_EXECUTION_LOG": "/dev/null"}, True, True),
        # Set but ineffective: installed so the banner can say it did nothing.
        ({"LEROBOT_CHUNK_PREFETCH_INLINE": "1"}, True, True),
        ({"LEROBOT_CHUNK_PREFETCH_TICKS": "0"}, True, True),
    ],
)
def test_installed_wrappers(environment, expect_ours, expect_loop_rate):
    report, _ = probe(environment)
    chain = report["chain"]
    assert (OURS in chain) == expect_ours, chain
    assert (LOOP_RATE in chain) == expect_loop_rate, chain
    if expect_ours:
        assert chain[0] == OURS, "ours must wrap the loop rate wrapper, not under it"


def test_real_record_loop_without_switches_leaves_the_policy_alone():
    report, stderr = probe({}, "run", "40")
    episode = report["episode"]
    assert episode["instance_select_action"] is False
    assert episode["instance_reset"] is False
    assert episode["alignment_errors"] == 0
    assert "Base latency switches" not in stderr


def test_real_record_loop_with_prefetch_and_lead(tmp_path):
    log_path = tmp_path / "execution.csv"
    report, stderr = probe(
        {
            "LEROBOT_CHUNK_PREFETCH_TICKS": "3",
            "LEROBOT_BASE_LEAD_TICKS": "1",
            "LEROBOT_CHUNK_EXECUTION_LOG": str(log_path),
        },
        "run",
        "40",
    )
    episode = report["episode"]
    assert episode["instance_select_action"] is True
    assert episode["alignment_errors"] == 0
    assert episode["frames"] == episode["sent"] > 20
    assert episode["observation_ticks"][:3] == [0, 7, 14]
    assert "Base latency switches: LEROBOT_CHUNK_PREFETCH_TICKS=3" in stderr
    assert "prefetch 3 ticks, background" in stderr
    assert "Chunk execution summary" in stderr
    lines = log_path.read_text().splitlines()
    data_rows = [line for line in lines[1:] if not line.startswith("#")]
    assert len(data_rows) == episode["sent"]
    assert any(line.startswith("# loop start") for line in lines)


def test_real_record_loop_replays_a_local_demo(tmp_path):
    import numpy
    from fake_policies import ACTION_NAMES
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    home = tmp_path / "lerobot_home"
    features = {
        "action": {"dtype": "float32", "shape": (4,), "names": list(ACTION_NAMES)},
        "observation.state": {"dtype": "float32", "shape": (1,), "names": ["s"]},
    }
    demo = LeRobotDataset.create(
        "local/demo",
        fps=20,
        features=features,
        root=home / "local/demo",
        use_videos=False,
    )
    for frame in range(60):
        demo.add_frame(
            {
                "action": numpy.array(
                    [frame * 10, 1, frame * 10 + 2, frame * 10 + 3], dtype=numpy.float32
                ),
                "observation.state": numpy.zeros(1, numpy.float32),
                "task": "demo",
            }
        )
    demo.save_episode()
    demo.finalize()

    report, stderr = probe(
        {
            "HF_LEROBOT_HOME": str(home),
            "HF_HUB_OFFLINE": "1",
            "LEROBOT_REPLAY_DATASET": "local/demo",
            "LEROBOT_REPLAY_EPISODES": "0",
            "LEROBOT_CHUNK_PREFETCH_TICKS": "3",
            "LEROBOT_BASE_LEAD_TICKS": "2",
            "PROBE_EVAL_REPOSITORY": "local/eval_replay_probe",
        },
        "run",
        "40",
    )
    episode = report["episode"]
    assert "replay: episode 0 of local/demo" in stderr
    assert episode["sent"] > 20
    for tick, (arm, theta) in enumerate(
        zip(episode["sent_arm"], episode["sent_theta"])
    ):
        assert arm == tick * 10  # row t at tick t
        assert theta == (tick + 2) * 10 + 3  # base two rows ahead


def test_real_record_loop_refuses_a_replay_that_cannot_load(tmp_path):
    environment = {
        "HF_LEROBOT_HOME": str(tmp_path),
        "HF_HUB_OFFLINE": "1",
        "LEROBOT_REPLAY_DATASET": "local/missing",
        "PROBE_EVAL_REPOSITORY": "local/eval_replay_probe",
    }
    completed = subprocess.run(
        [sys.executable, str(PROBE), "run", "10"],
        env={
            **{k: v for k, v in os.environ.items() if k not in SWITCH_VARIABLES},
            "PYTHONPATH": str(SOURCE_ROOT),
            **environment,
        },
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.returncode != 0
    assert "ReplayPreparationError" in completed.stderr


def test_an_ineffective_variable_is_named_in_the_banner():
    _, stderr = probe({"LEROBOT_CHUNK_PREFETCH_INLINE": "1"}, "run", "10")
    assert "does nothing without LEROBOT_CHUNK_PREFETCH_TICKS" in stderr
