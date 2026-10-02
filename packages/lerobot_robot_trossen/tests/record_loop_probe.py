"""Run in a fresh process by test_patch_install.py; prints one JSON line.

The switches are read from the environment once, at import, so each environment
combination needs its own interpreter. This imports the plugin the way lerobot
does, reports which record_loop wrappers are installed, and optionally drives
lerobot's real record_loop for a short episode with a fake robot and policy.
"""

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lerobot_robot_trossen  # noqa: E402,F401  (installs the patches)
from fake_policies import (  # noqa: E402
    FakeDataset,
    FakeRobot,
    IdentityPostprocessor,
    QueuePolicy,
    TickPreprocessor,
    decode,
)
from lerobot.scripts import lerobot_record  # noqa: E402

from lerobot_robot_trossen import chunk_execution_patch  # noqa: E402


def wrapper_chain(function) -> list[str]:
    chain = []
    while function is not None:
        source = Path(function.__code__.co_filename).name
        if getattr(function, chunk_execution_patch.WRAPPER_MARKER, False):
            source += "*"
        chain.append(source)
        function = getattr(function, "__wrapped__", None)
    return chain


def run_episode(ticks: int) -> dict:
    import os

    from lerobot.processor import make_default_processors

    teleop_processor, robot_processor, observation_processor = make_default_processors()
    fps = 200
    robot = FakeRobot()
    dataset = FakeDataset(
        fps=fps,
        repository_id=os.environ.get("PROBE_EVAL_REPOSITORY", "local/eval_fake"),
    )
    policy = QueuePolicy(action_steps=10)
    lerobot_record.record_loop(
        robot=robot,
        events={"exit_early": False},
        fps=fps,
        teleop_action_processor=teleop_processor,
        robot_action_processor=robot_processor,
        robot_observation_processor=observation_processor,
        dataset=dataset,
        policy=policy,
        preprocessor=TickPreprocessor(),
        postprocessor=IdentityPostprocessor(),
        control_time_s=ticks / fps,
        single_task="probe",
    )
    alignment_errors = 0
    for tick, sent in enumerate(robot.sent_actions):
        observation_tick, step, _ = decode(sent["left_joint_0.pos"])
        alignment_errors += observation_tick + step != tick
    return {
        "frames": len(dataset.frames),
        "sent": len(robot.sent_actions),
        "alignment_errors": alignment_errors,
        "observation_ticks": policy.observation_ticks,
        "instance_select_action": "select_action" in policy.__dict__,
        "instance_reset": "reset" in policy.__dict__,
        "sent_arm": [sent["left_joint_0.pos"] for sent in robot.sent_actions],
        "sent_theta": [sent["theta.vel"] for sent in robot.sent_actions],
    }


def main() -> None:
    # force=True: importing lerobot with transformers installed calls
    # logging.debug() at module level, which already configured the root
    # logger at WARNING; without force this call would do nothing.
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, force=True)
    report = {
        "package_file": lerobot_robot_trossen.__file__,
        "chain": wrapper_chain(lerobot_record.record_loop),
    }
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        report["episode"] = run_episode(int(sys.argv[2]))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
