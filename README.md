# LeRobot Trossen Integration

## Overview

This package contains LeRobot integrations for the Trossen AI series of robots.

> **This is the KIRO fork** of [`TrossenRobotics/lerobot_trossen`](https://github.com/TrossenRobotics/lerobot_trossen). Use it for Mobile AI data acquisition and policy eval; the examples below target the **Mobile AI** dual-arm platform.

### Changes in this fork

| Change | What it does | Where |
| ------ | ------------ | ----- |
| **Mobile-base velocity sanitisation** | `mobileai.py` refreshes the base state before reading it and zeroes out garbage velocity readings left in stale serial buffers. Without it, policies trained on the recorded data fault with `Joint 0 ... contains NaN` at inference. Always on. | [#2](https://github.com/kiro-ai-division/lerobot_trossen/pull/2) |
| **`include_base_in_state` flag** | Drops the base velocity from `observation.state` so 14-dim policies can be evaluated. | [below](#base-velocity-in-the-observation-state) · [#4](https://github.com/kiro-ai-division/lerobot_trossen/pull/4) |
| **`LEROBOT_FAST_OBS`** | Moves eval-time image preprocessing to the GPU. On by default; roughly doubles the control-loop rate on the Mobile AI 3-camera setup. | [below](#environment-variables) · [#8](https://github.com/kiro-ai-division/lerobot_trossen/pull/8), [#14](https://github.com/kiro-ai-division/lerobot_trossen/pull/14) |
| **`LEROBOT_LOOP_HZ_LOG`** | Opt-in control-loop rate and per-section timing meter. | [below](#environment-variables) · [#6](https://github.com/kiro-ai-division/lerobot_trossen/pull/6) |
| **cu128 torch wheels** | `torch`/`torchvision` pinned to the CUDA 12.8 index so inference runs on Blackwell (sm_120) GPUs. | [below](#installation) · [#12](https://github.com/kiro-ai-division/lerobot_trossen/pull/12) |

For the KIRO/GIST platform command lines (camera serials, checkpoints, recording conventions), see the [Mobile AI Quickstart Guide](https://github.com/kiro-ai-division/mobile-ai-quickstart-guide).

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

> **CUDA wheels.** `torch` and `torchvision` are pinned to the [cu128 index](https://download.pytorch.org/whl/cu128) so that policy inference works on Blackwell GPUs (sm_120, e.g. the RTX 5090 in the robot PC) — the default PyPI wheels are built against CUDA 12.6 and ship kernels only up to sm_90, which makes eval die on the first frame with `no kernel image is available for execution on the device`. Note that cu128 wheels drop sm_50/sm_60/sm_70 (Maxwell, Pascal, Volta); Turing and newer are unaffected.

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

### Base Velocity in the Observation State

By default a Mobile AI follower appends the mobile base velocity (`x.vel`, `theta.vel`) to
`observation.state`, giving a **16-dim** state (6 arm joints + 1 gripper carriage, per arm,
plus the two base channels). Set `--robot.include_base_in_state=false` to drop them and emit
a **14-dim** state (arms only).

| Flag | Default | `observation.state` |
| ---- | ------- | ------------------- |
| `include_base_in_state` | `true` | 16-dim — both arms + base `x.vel`, `theta.vel` |
| | `false` | 14-dim — both arms only |

**Match this flag to the checkpoint you evaluate.** LeRobot does not reshape robot
observations to the policy's `input_features`: the slicing rule that produced a base-free
training set exists only in the dataset, so a 14-dim policy fed a 16-dim state breaks on the
normalisation buffers. Train on a base-in-state dataset → leave it `true`; train on a dataset
with the base channels sliced out → pass `false` at eval time.

`action_features` are untouched, so the base is still commanded either way — the flag only
gates what the policy *observes*.

```shell
uv run lerobot-record \
  --robot.type=mobileai_robot \
  --robot.include_base_in_state=false \
  ...
  --policy.path=${HF_USER}/act-mobileai-nobasestate
```

### Environment Variables

| Variable | Default | Effect |
| -------- | ------- | ------ |
| `LEROBOT_FAST_OBS` | `1` (on) | Converts camera frames to float32 and permutes HWC→CHW **on the GPU** instead of the CPU. Set `0` to fall back to the stock lerobot path. |
| `LEROBOT_LOOP_HZ_LOG` | unset (off) | Set `1` to log the achieved control-loop rate and a per-frame section breakdown. |

**`LEROBOT_FAST_OBS`** — lerobot's `prepare_observation_for_inference` converts and permutes
camera frames CPU-side and only then copies them to the GPU, shipping 4× the bytes over PCIe
and paying for an elementwise divide plus a full `.contiguous()` copy per camera. On the
Mobile AI 3-camera setup that measured **54.7 ms → 1.6 ms per frame** (p50 under CPU load),
and on hardware the eval loop went from **9.7 Hz to 20.7 Hz** against a 21.5 Hz teleop
recording baseline. This matters beyond throughput: the SLATE base holds a velocity command
until the next `send_action`, so a loop running at half the recording rate integrates every
rotation roughly twice as far. The patch no-ops if upstream ships the same fix
([huggingface/lerobot#4339](https://github.com/huggingface/lerobot/pull/4339), still open) and
swallows its own errors so plugin discovery cannot fail because of it. To check which path a
run took, `grep LEROBOT_FAST_OBS <run log>`.

**`LEROBOT_LOOP_HZ_LOG`** — off by default so normal operation carries no logging overhead.
When enabled, `send_action` measures every loop iteration and emits a summary every 30 frames:

```
Control loop rate over last 30 frames: mean=20.7 Hz, min=18.9 Hz
(target_fps / mean_hz = base over-rotation multiplier)
 | per-frame: arms_read=3ms  arms_write=2ms  base_read=21ms  base_write=21ms
   cam:cam_high=1ms  ...  other=4ms
```

`target_fps / mean_hz` is the base over-rotation multiplier — a mean near the target fps rules
the loop-slowdown hypothesis out, a mean near half confirms it. The section breakdown says
which I/O is responsible; `other` is loop time outside any instrumented section (policy
`select_action`, preprocessing, `dataset.add_frame`, processors, `busy_wait`). Episode-reset
gaps longer than 1 s are dropped so an idle pause cannot masquerade as a slow loop. Combine
with `LEROBOT_FAST_OBS=0` for an A/B comparison.

### Dataset Visualization

If you uploaded your dataset to the Hugging Face Hub using ``--control.push_to_hub=true``, you can [visualize your dataset online](https://huggingface.co/spaces/lerobot/visualize_dataset).
To do so, copy and paste your repository ID into the provided field.
Your repository ID follows the format:

```
<huggingface-username>/<dataset-id>
```

### Model Eval (Record with Policy) Script

Evaluate a trained policy by recording 2 episodes of a cube pickup task with a Mobile AI robot using the RealSense camera interface.

**The policy architecture is not a CLI argument.** `lerobot-record` reads the policy type
(`act`, `pi0`, `smolvla`, …) and the input/output feature shapes from the checkpoint's
`config.json`, so ACT and pi0 eval are the same command with a different `--policy.path`.
Two failure modes are worth knowing:

- Passing `--policy.type` alongside `--policy.path` aborts with
  `Cannot specify both --policy.path and --policy.type`.
- Passing `--policy.type` *without* `--policy.path` silently builds a **randomly initialised**
  policy — `lerobot-eval` warns about this, `lerobot-record` does not.

`--dataset.single_task` is required regardless of policy type, but what it does differs:
language-conditioned policies (pi0, SmolVLA) tokenise it, so it must match the wording used
during training, while ACT never reads it — for ACT it is only the dataset label.

Also check `--robot.include_base_in_state` against the checkpoint's state dimension — see
[Base Velocity in the Observation State](#base-velocity-in-the-observation-state).

Verified against lerobot 0.4.0–0.4.4 (this fork pins 0.4.0). From 0.6.0 policy deployment
moves to `lerobot-rollout` and `lerobot-record` refuses `--policy.path`, but the
type-from-checkpoint rule still holds there.

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
  --robot.enable_base_motor_torque=true \
  --dataset.repo_id=${HF_USER}/mobileai-cube-pickup \
  --dataset.num_episodes=2 \
  --dataset.single_task="Grab the cube" \
  --policy.path=${HF_USER}/act-mobileai-cube-pickup
```

> **`--robot.enable_base_motor_torque=true` is required whenever the policy has to drive the
> mobile base, and it is not the default.** During teleoperated recording the base is moved by
> its own controller and `MobileAILeaderTeleop` merely echoes the measured velocity back as an
> action, so recording works with the torque disabled. At eval the roles reverse: the policy's
> `x.vel`/`theta.vel` reach `base.set_cmd_vel()` on every `send_action`, but with motor torque
> off the base ignores them. **Nothing errors** — the arms behave and the base silently stays
> put, which is easy to misread as the policy having learned no base motion. The flag is applied
> once in `connect()`, so it has to be on the command line from the start.

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
