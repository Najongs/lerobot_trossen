"""A base_replay stage that carries the next policy stage's start_pose.

The arms are posed before the base moves and held there through the replay
(robot PC 2026-10-25, used from t03 to the full 1-11 chain t06).
"""

from pathlib import Path

import test_base_smoke as smoke
from stage_runner import cli

POSE: dict[str, float] = {
    "left_joint_0.pos": -0.25,
    "left_joint_1.pos": 1.25,
    "left_joint_2.pos": 0.45,
    "left_joint_3.pos": 0.5,
    "left_joint_4.pos": 0.35,
    "left_joint_5.pos": -0.3,
    "right_joint_0.pos": 0.55,
    "right_joint_1.pos": 1.3,
    "right_joint_2.pos": 0.5,
    "right_joint_3.pos": 0.4,
    "right_joint_4.pos": -0.25,
    "right_joint_5.pos": -0.2,
}


def test_move_stage_poses_arms_first_and_holds_them(tmp_path: Path) -> None:
    path = smoke._write_smoke_config(tmp_path)
    block = (
        "    start_pose:\n      positions:\n"
        + "".join(f"        {k}: {v}\n" for k, v in POSE.items())
        + "      min_goal_time_s: 3.0\n      position_tolerance_rad: 0.05\n"
        "      source:\n"
        '        dataset_repo_id: "local/hand_written_smoke_pose"\n'
        '        policy_path: "mock://hold"\n'
        "        episode_index: 0\n        n_episodes: 1\n"
    )
    text = path.read_text()
    anchor = "    executor: base_replay\n"
    first = text.index(anchor)
    second = text.index(anchor, first + 1)  # s3, the second replay
    text = text[: second + len(anchor)] + block + text[second + len(anchor) :]
    path.write_text(text)

    result = smoke._cli(tmp_path, path)

    assert result["exit_code"] == cli.EXIT_OK
    repositions = {e["stage_id"]: e for e in smoke._of(result, "stage_reposition")}
    assert repositions["s3"]["outcome"] == "reached", repositions["s3"]
    replay = {e["stage_id"]: e for e in smoke._of(result, "stage_base_replay")}["s3"]
    held = replay["held_arm_positions"]
    assert max(abs(held[k] - v) for k, v in POSE.items()) < 0.05
    assert replay["outcome"] == "settled"
