# LeRobot Trossen Integration

## Overview

This package contains LeRobot integrations for the Trossen AI series of robots.

> **This is the KIRO fork** of [`TrossenRobotics/lerobot_trossen`](https://github.com/TrossenRobotics/lerobot_trossen) with a fix for the Mobile AI **mobile-base velocity NaN** bug — `mobileai.py` refreshes and sanitizes garbage base velocity values so policies trained on the data don't fault with `Joint 0 ... contains NaN` at inference. Use this fork for data acquisition. The examples below target the **Mobile AI** dual-arm platform.

See the [LeRobot documentation](https://huggingface.co/docs/lerobot) for details on more advanced usage like using the HuggingFace Hub, model training, and using different teleoperation methods.
See the [Trossen AI documentation](https://docs.trossenrobotics.com/trossen_arm/main/tutorials/lerobot_plugin.html) for details on configuration and usage of Trossen AI robots with LeRobot.

## Installation

We use `uv` to manage our dependencies.
Follow the instructions [here](https://docs.astral.sh/uv/getting-started/installation/) to install `uv`.

Run the following command to install this package and its dependencies:

```shell
# Clone this repository
git clone https://github.com/kiro-ai-division/lerobot_trossen.git

# Install the trossen lerobot packages and their dependencies
uv sync

# Verify installation
uv pip list | grep trossen
# lerobot-robot-trossen
# lerobot-teleoperator-trossen
# trossen-arm
# trossen-slate
```

## Usage

> Camera serial numbers are **platform-specific** — replace the `<…_serial>` placeholders with your own. See the [Trossen AI configuration docs](https://docs.trossenrobotics.com/trossen_arm/main/tutorials/lerobot_plugin/configuration.html) for how to find them, and the [Mobile AI Quickstart Guide](https://github.com/kiro-ai-division/mobile-ai-quickstart-guide) for the serials of the KIRO/GIST platforms.

### Teleoperation Script

Teleoperate a Mobile AI robot (dual-arm leader → dual-arm follower).

```shell
uv run lerobot-teleoperate \
  --robot.type=mobileai_robot \
  --robot.left_arm_ip_address=192.168.1.5 \
  --robot.right_arm_ip_address=192.168.1.4 \
  --robot.id=follower \
  --teleop.type=mobileai_leader_teleop \
  --teleop.left_arm_ip_address=192.168.1.3 \
  --teleop.right_arm_ip_address=192.168.1.2 \
  --teleop.id=leader \
  --display_data=false
```

### Record Script

Record 10 episodes with duration 45s of a cube pickup task with a Mobile AI robot using the RealSense camera interface.
This dataset will not be pushed to the Hugging Face Hub after recording.

```shell
uv run lerobot-record \
  --robot.type=mobileai_robot \
  --robot.left_arm_ip_address=192.168.1.5 \
  --robot.right_arm_ip_address=192.168.1.4 \
  --robot.id=follower \
  --robot.cameras="{
    cam_high: {type: intelrealsense, serial_number_or_name: "<cam_high_serial>", width: 640, height: 480, fps: 30},
    cam_left_wrist: {type: intelrealsense, serial_number_or_name: "<cam_left_wrist_serial>", width: 640, height: 480, fps: 30},
    cam_right_wrist: {type: intelrealsense, serial_number_or_name: "<cam_right_wrist_serial>", width: 640, height: 480, fps: 30}
  }" \
  --teleop.type=mobileai_leader_teleop \
  --teleop.left_arm_ip_address=192.168.1.3 \
  --teleop.right_arm_ip_address=192.168.1.2 \
  --teleop.id=leader \
  --display_data=true \
  --dataset.push_to_hub=false \
  --dataset.repo_id=${HF_USER}/mobileai-cube-pickup \
  --dataset.episode_time_s=45 \
  --dataset.reset_time_s=15 \
  --dataset.num_episodes=10 \
  --dataset.single_task="Grab the cube"
```

Record 25 episodes with duration 60s of a handover task with a Mobile AI robot.
Datasets are pushed to the Hugging Face Hub after recording by default - make sure to set the `HF_USER` environment variable and be logged in with the `huggingface-cli login` command before running this script.

```shell
uv run lerobot-record \
  --robot.type=mobileai_robot \
  --robot.left_arm_ip_address=192.168.1.5 \
  --robot.right_arm_ip_address=192.168.1.4 \
  --robot.id=follower \
  --robot.cameras="{
    cam_high: {type: intelrealsense, serial_number_or_name: "<cam_high_serial>", width: 640, height: 480, fps: 30},
    cam_left_wrist: {type: intelrealsense, serial_number_or_name: "<cam_left_wrist_serial>", width: 640, height: 480, fps: 30},
    cam_right_wrist: {type: intelrealsense, serial_number_or_name: "<cam_right_wrist_serial>", width: 640, height: 480, fps: 30}
  }" \
  --teleop.type=mobileai_leader_teleop \
  --teleop.left_arm_ip_address=192.168.1.3 \
  --teleop.right_arm_ip_address=192.168.1.2 \
  --teleop.id=leader \
  --display_data=true \
  --dataset.repo_id=${HF_USER}/mobileai-handover-cube \
  --dataset.num_episodes=25 \
  --dataset.episode_time_s=60 \
  --dataset.reset_time_s=15 \
  --dataset.single_task="Grab and handover the red cube to the other arm"
```

### Optional Observation Features

By default, Mobile AI followers only observe joint positions (`<joint>.pos`).
You can optionally record additional per-joint signals by enabling the following flags.
All are disabled by default.

| Flag | Observation key | Description |
| ---- | --------------- | ----------- |
| `include_velocity` | `<joint>.vel` | Joint velocity. Measured in rad/s for the arm joints and m/s for the gripper carriage. |
| `include_effort` | `<joint>.eff` | Total motor effort, combining gravity, friction, and any external load. Measured in Nm for the arm joints and N for the gripper carriage. Nonzero even when the arm is holding still against gravity. |
| `include_external_effort` | `<joint>.ext_eff` | Estimated externally applied effort, after gravity and friction compensation. Measured in Nm for the arm joints and N for the gripper carriage. Useful for contact and force sensing; an unloaded arm reports values near zero. |

Pass them as `--robot.<flag>=true` when running any command that constructs the robot (for example `lerobot-record` or `lerobot-teleoperate`). The flags are shared across both arms, and the resulting observation keys are prefixed per arm, e.g. `left_<joint>.eff` and `right_<joint>.eff`. For example, to record with all three enabled on a Mobile AI follower:

```shell
uv run lerobot-record \
  --robot.type=mobileai_robot \
  --robot.left_arm_ip_address=192.168.1.5 \
  --robot.right_arm_ip_address=192.168.1.4 \
  --robot.id=follower \
  --robot.include_velocity=true \
  --robot.include_effort=true \
  --robot.include_external_effort=true \
  --dataset.repo_id=${HF_USER}/mobileai-cube-pickup \
  --dataset.single_task="Grab the cube" \
  --teleop.type=mobileai_leader_teleop \
  --teleop.left_arm_ip_address=192.168.1.3 \
  --teleop.right_arm_ip_address=192.168.1.2 \
  --teleop.id=leader
```

### Dataset Visualization

If you uploaded your dataset to the Hugging Face Hub using ``--control.push_to_hub=true``, you can [visualize your dataset online](https://huggingface.co/spaces/lerobot/visualize_dataset).
To do so, copy and paste your repository ID into the provided field.
Your repository ID follows the format:

```
<huggingface-username>/<dataset-id>
```

### Model Eval (Record with Policy) Script

Evaluate a trained policy by recording 2 episodes of a cube pickup task with a Mobile AI robot using the RealSense camera interface.

```shell
uv run lerobot-record \
  --robot.type=mobileai_robot \
  --robot.left_arm_ip_address=192.168.1.5 \
  --robot.right_arm_ip_address=192.168.1.4 \
  --robot.id=follower \
  --robot.cameras="{
    cam_high: {type: intelrealsense, serial_number_or_name: "<cam_high_serial>", width: 640, height: 480, fps: 30},
    cam_left_wrist: {type: intelrealsense, serial_number_or_name: "<cam_left_wrist_serial>", width: 640, height: 480, fps: 30},
    cam_right_wrist: {type: intelrealsense, serial_number_or_name: "<cam_right_wrist_serial>", width: 640, height: 480, fps: 30}
  }" \
  --dataset.repo_id=${HF_USER}/mobileai-cube-pickup \
  --dataset.num_episodes=2 \
  --dataset.single_task="Grab the cube" \
  --policy.path=${HF_USER}/act-mobileai-cube-pickup
```

### Replay Script

Replay episode 3 of a cube pickup task with a Mobile AI robot.

```shell
uv run lerobot-replay \
  --robot.type=mobileai_robot \
  --robot.left_arm_ip_address=192.168.1.5 \
  --robot.right_arm_ip_address=192.168.1.4 \
  --robot.id=follower \
  --robot.enable_base_motor_torque=true \
  --dataset.repo_id=${HF_USER}/mobileai-cube-pickup \
  --dataset.episode=3
```
