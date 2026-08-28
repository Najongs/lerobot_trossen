# LeRobot Trossen Integration

[`TrossenRobotics/lerobot_trossen`](https://github.com/TrossenRobotics/lerobot_trossen)의 **KIRO fork.** Mobile AI 양팔 플랫폼 **데이터 취득 → 학습 → eval** 명령 정본.

- 로봇 조작·취득 단계 시작·종료 기준 → [Mobile AI Quickstart Guide](https://github.com/kiro-ai-division/mobile-ai-quickstart-guide)
- 인자 설명·함정 → [인자 레퍼런스](#인자-레퍼런스) · fork 변경점 → [Fork reference](#fork-reference)

---

## 0. 설치

로봇 PC·`sandia`·DGX-1에 이미 세팅 완료 = `~/lerobot_trossen`. **`cd ~/lerobot_trossen`부터.**

새 기기만:

```shell
git clone https://github.com/kiro-ai-division/lerobot_trossen.git
cd lerobot_trossen
uv sync
uv run hf auth login          # kiroaiseoul org write 토큰
```

- 서버 반입·인터프리터 고정 → [설치 함정](#설치-함정)
- 카메라 시리얼 — `cam_high` `230422273501` / `cam_left_wrist` `230422271234` / `cam_right_wrist` `230322274369`

---

## 1. 데이터 취득

### 1-1. Teleoperation (작동 확인)

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
  --display_data=true \
  --robot.cameras='{
    cam_high: {type: intelrealsense, serial_number_or_name: "230422273501", width: 640, height: 480, fps: 30},
    cam_left_wrist: {type: intelrealsense, serial_number_or_name: "230422271234", width: 640, height: 480, fps: 30},
    cam_right_wrist: {type: intelrealsense, serial_number_or_name: "230322274369", width: 640, height: 480, fps: 30}
}'
```

### 1-2. Record

`<본인_단계_repo>`·`<본인 단계 지시문>` → [`데이터 취득 현황` 탭](https://docs.google.com/spreadsheets/d/1pTFT3Cg3L735v0ujUAgG2XvwRv0q5obI8FIK4B0OZjw/edit#gid=1380290557)

```shell
uv run lerobot-record \
  --robot.type=mobileai_robot \
  --robot.left_arm_ip_address=192.168.1.5 \
  --robot.right_arm_ip_address=192.168.1.4 \
  --robot.id=follower \
  --robot.cameras='{
    cam_high: {type: intelrealsense, serial_number_or_name: "230422273501", width: 640, height: 480, fps: 30},
    cam_left_wrist: {type: intelrealsense, serial_number_or_name: "230422271234", width: 640, height: 480, fps: 30},
    cam_right_wrist: {type: intelrealsense, serial_number_or_name: "230322274369", width: 640, height: 480, fps: 30}
}' \
  --teleop.type=mobileai_leader_teleop \
  --teleop.left_arm_ip_address=192.168.1.3 \
  --teleop.right_arm_ip_address=192.168.1.2 \
  --teleop.id=leader \
  --display_data=true \
  --dataset.repo_id=kiroaiseoul/<본인_단계_repo> \
  --dataset.num_episodes=3 \
  --dataset.episode_time_s=90 \
  --dataset.reset_time_s=30 \
  --dataset.single_task="<본인 단계 지시문>"
```

- 이어 찍기 = 마지막 줄 끝에 `\` + `--resume=true` → [Record](#record)
- 녹화 키 — `→` 구간 조기 종료 · `←` 현재 에피소드 재취득 · `ESC` 중단·저장
- 취득 직후 QC → [4. 데이터셋 확인·편집](#4-데이터셋-확인편집) · 오염 자가점검 → [취득 후 점검](#취득-후-점검)

---

## 2. 학습

`sandia`·DGX-1 동일 — 잡을 GPU만 기기, 상황별로 다르게. 

### 2-0. GPU 고르기

```shell
tmux new -s acttrain-<본인이름>
cd ~/lerobot_trossen

nvidia-smi                            # 하단 Processes 표가 빈 GPU를 고른다
export CUDA_VISIBLE_DEVICES=0,1,2,3   # 고른 GPU 번호로
```

2-1 ~ 2-3은 **이 셸에서** 이어 실행. 새 셸을 열면 2-0부터 다시.

### 2-1. 스모크런 (본 학습 전 필수)

```shell
rm -rf outputs/_smoke_<본인이름>

uv run accelerate launch \
  --num_processes=1 \
  -m lerobot.scripts.lerobot_train \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.repo_id=kiroaiseoul/<본인_단계_repo> \
  --output_dir=outputs/_smoke_<본인이름> \
  --batch_size=16 --steps=10 --save_freq=10 \
  --wandb.enable=false
```

통과 기준 — `outputs/_smoke_<본인이름>/checkpoints/000010/pretrained_model/model.safetensors` 수백 MB.

### 2-2. 본 학습

```shell
uv run accelerate launch \
  --multi_gpu \
  --num_processes=4 \
  -m lerobot.scripts.lerobot_train \
  --policy.type=act \
  --policy.device=cuda \
  --dataset.repo_id=kiroaiseoul/<본인_단계_repo> \
  --policy.repo_id=kiroaiseoul/act_<본인_단계>_<스텝> \
  --output_dir=outputs/<본인_이름>/<본인_단계> \
  --batch_size=16 --steps=60000 --save_freq=20000 \
  --wandb.enable=false
```

- `<본인_단계_repo>` → [`데이터 취득 현황` 탭](https://docs.google.com/spreadsheets/d/1pTFT3Cg3L735v0ujUAgG2XvwRv0q5obI8FIK4B0OZjw/edit#gid=1380290557)
- 유효 배치 **64** = `batch_size` × 장 수 → [유효 배치](#유효-배치)
- `--steps` → [스텝 수](#스텝-수) · 스모크런과 달라지는 인자 → [학습](#학습) · 공유 계정 수칙 → [공유 서버](#공유-서버)
- tmux — 붙기 `tmux attach -t acttrain-<본인이름>` · 떼기 `Ctrl+b` `d`

### 2-3. 중단·재개

```shell
uv run accelerate launch \
  --multi_gpu \
  --num_processes=4 \
  -m lerobot.scripts.lerobot_train \
  --config_path=outputs/<본인_이름>/<본인_단계>/checkpoints/last/pretrained_model/train_config.json \
  --resume=true
```

### 2-4. 학습이 끝나면

1. Hub에 `--policy.repo_id` 이름으로 올라갔는지 확인
2. [`모델 체크포인트` 탭](https://docs.google.com/spreadsheets/d/1pTFT3Cg3L735v0ujUAgG2XvwRv0q5obI8FIK4B0OZjw/edit#gid=259237510)에 한 줄 — 레포 이름·보유상황·학습 로그·담당·파라미터(`batch_size` × 장 수 · `fp32`)
3. → [3. Eval](#3-eval)

---

## 3. Eval

> 🚨 베이스 진행 방향을 비우고, 비상정지에 손이 닿는 위치에서.

- `<체크포인트>` → [`모델 체크포인트` 탭](https://docs.google.com/spreadsheets/d/1pTFT3Cg3L735v0ujUAgG2XvwRv0q5obI8FIK4B0OZjw/edit#gid=259237510) · `<본인 단계 지시문>` → [`데이터 취득 현황` 탭](https://docs.google.com/spreadsheets/d/1pTFT3Cg3L735v0ujUAgG2XvwRv0q5obI8FIK4B0OZjw/edit#gid=1380290557)
- `--dataset.repo_id` = `eval_` + 회차 식별자 → [eval 함정](#eval-함정)
- 실행 중 뜨는 ERROR·경고 → [무해한 로그](#무해한-로그)

### 3-1. 연결 점검

팔 4개와 카메라 3개가 붙어 있는지 먼저 본다 — 안 붙은 채로 돌리면 에피소드 중간에 죽는다.

```shell
for ip in 192.168.1.5 192.168.1.4 192.168.1.3 192.168.1.2; do
  ping -c1 -W1 $ip >/dev/null 2>&1 && echo "OK $ip" || echo "NG $ip"; done
uv run python -c "import pyrealsense2 as rs; [print(d.get_info(rs.camera_info.serial_number)) for d in rs.context().devices]"
```

`OK` 4줄 + 아래 명령의 시리얼 3개와 같은 값이면 통과. **베이스 비상정지 버튼도 돌려 빼 둔다** — 걸린 채면 `connect()`에서 `RuntimeError: Robot is in emergency stop state. …`로 즉사 → [Base Emergency Stop Detection](#base-emergency-stop-detection).

### 3-2. 체크포인트 경로 잡기

`--policy.path`는 **`config.json`이 있는 디렉터리**를 가리켜야 하는데, 그 위치가 repo마다 다르다(허브 26개 중 루트 11 : `pretrained_model/` 하위 15). 아래 한 블록이 둘 다 처리한다.

```shell
POLICY=$(uv run python -c "
from huggingface_hub import snapshot_download
from pathlib import Path
d = Path(snapshot_download('kiroaiseoul/<체크포인트>'))
print(d if (d/'config.json').exists() else d/'pretrained_model')
" | tail -1)
echo "$POLICY"
```

로컬 학습 산출물이면 받을 것 없이 `POLICY=outputs/<본인_이름>/<본인_단계>/checkpoints/<스텝>/pretrained_model`.

### 3-3. ACT + 리더암

```shell
uv run lerobot-record \
  --robot.type=mobileai_robot \
  --robot.left_arm_ip_address=192.168.1.5 \
  --robot.right_arm_ip_address=192.168.1.4 \
  --robot.id=follower \
  --robot.cameras='{
    cam_high: {type: intelrealsense, serial_number_or_name: "230422273501", width: 640, height: 480, fps: 30},
    cam_left_wrist: {type: intelrealsense, serial_number_or_name: "230422271234", width: 640, height: 480, fps: 30},
    cam_right_wrist: {type: intelrealsense, serial_number_or_name: "230322274369", width: 640, height: 480, fps: 30}
}' \
  --robot.enable_base_motor_torque=true \
  --teleop.type=mobileai_leader_teleop \
  --teleop.left_arm_ip_address=192.168.1.3 \
  --teleop.right_arm_ip_address=192.168.1.2 \
  --teleop.id=leader \
  --display_data=true \
  --dataset.repo_id=kiroaiseoul/eval_act_<단계>_<회차> \
  --dataset.single_task="<본인 단계 지시문>" \
  --policy.path="$POLICY" \
  --dataset.episode_time_s=120 \
  --dataset.reset_time_s=90 \
  --dataset.num_episodes=10
```

- 리셋 구간 = 리더암으로 시작 자세·파지, 끝나면 `→`. 에피소드 구간은 정책 구동.

---

## 4. 데이터셋 확인·편집

**로컬 재생(QC)** — 온라인은 [뷰어](https://huggingface.co/spaces/lerobot/visualize_dataset)에 `repo_id` 붙여넣기.

```shell
uv run lerobot-dataset-viz --repo-id kiroaiseoul/<dataset> --episode-index 0
```

**손상 에피소드 삭제** → [데이터셋 편집](#데이터셋-편집)

```shell
uv run lerobot-edit-dataset \
  --repo_id kiroaiseoul/<dataset> \
  --new_repo_id kiroaiseoul/<dataset>_clean \
  --operation.type delete_episodes \
  --operation.episode_indices "[9, 43, 78]" \
  --push_to_hub true
```

**Replay** — 🚨 기록된 궤적대로 팔·베이스 실제 구동.

```shell
uv run lerobot-replay \
  --robot.type=mobileai_robot \
  --robot.left_arm_ip_address=192.168.1.5 \
  --robot.right_arm_ip_address=192.168.1.4 \
  --robot.id=follower \
  --robot.enable_base_motor_torque=true \
  --dataset.repo_id=kiroaiseoul/<dataset> \
  --dataset.episode=3
```

---

# 인자 레퍼런스

## 설치 함정

- **`sandia`·DGX-1엔 GitHub 자격증명이 없다** — 비공개 repo라 `git clone`이 죽는다. 로컬에서 `rsync -az --exclude=.venv --exclude=outputs <로컬repo>/ <서버>:~/lerobot_trossen/`로 넣는다.
- 🛑 **`.python-version`(3.11)을 지우거나 3.12로 올리지 말 것** — 락이 3.12를 경계로 갈려 있어 3.12에서는 lerobot 0.6.1이 잡히고, [§3 Eval](#3-eval)의 `--policy.path`가 거부된다.

## Record

- `--robot.type` / `--robot.left_arm_ip_address` / `--robot.right_arm_ip_address` / `--robot.id` — follower 플랫폼 종류·좌우 팔 IP·명칭
- `--robot.cameras` — 카메라 종류·시리얼·해상도·FPS. **바깥은 홑따옴표** — 겹따옴표면 안쪽 `"`가 셸에서 벗겨져 시리얼이 정수로 파싱된다.
- `--teleop.*` — 위와 같은 항목의 leader 쪽
- `--display_data` — 취득 영상·관절각도 실시간 표시
- `--dataset.repo_id` — 저장할 Hugging Face dataset (`<username>/<name>`)
- `--dataset.num_episodes` — **이번에 추가로 찍을 개수**(누적 목표가 아니다). `--resume=true`면 기존 repo의 마지막 다음부터 붙는다.
- `--dataset.episode_time_s` / `--dataset.reset_time_s` — 1회 취득 시간(초) · 에피소드 간 초기화 대기(초)
- `--dataset.single_task` — 언어 지시문 레이블
- `--dataset.push_to_hub` — 기본 `true`. 로컬에만 두려면 `false`

### 취득 후 점검

`meta/stats.json`의 base 차원(state/action dim 14·15) `std`가 유한한지 본다. NaN·거대값이면 base 속도 garbage가 섞인 것이고 **사후 복구가 안 된다.** 차단 메커니즘 → [Changes in this fork](#changes-in-this-fork)

- ⚠️ **빨간 `[MOBILE AI BASE]` 경고가 뜬 구간은 베이스가 안 움직인 구간이다** — 경고는 실행을 막지 않으므로 그 구간의 base 채널이 정지값으로 기록된다. **리셋 구간 발화는 정상**(손으로 밀려고 누르는 것), **녹화 구간 발화는 재취득 대상** → [Base Emergency Stop Detection](#base-emergency-stop-detection)

## Eval

eval도 `lerobot-record`로 돌린다. **`--policy.path` 유무가 데이터 취득(teleop 시연)과 eval(정책 구동)을 가른다.** Record와 달라지는 것만 적는다.

- `--policy.path` — **`config.json`이 있는 디렉터리**. `lerobot-train --policy.repo_id=…`가 올린 체크포인트는 repo 루트에 그 파일들이 있어 bare repo id가 그대로 먹지만, 디렉터리째 업로드된 것은 `pretrained_model/` 아래에 중첩된다. **이름으로는 구별되지 않으므로** 레이아웃을 따지지 말고 [3-2](#3-2-체크포인트-경로-잡기)를 쓴다. bare repo id를 그냥 주면 중첩형에서 로드에 실패한다.
- **정책 종류는 명령에 안 쓴다** — `lerobot-record`가 체크포인트의 `config.json` `type`에서 정책 종류와 입출력 차원을 읽는다. **`--policy.type`은 주지 말 것** — `--policy.path`와 같이 주면 `Cannot specify both …`로 죽고, 혼자 주면 **경고 없이 랜덤 가중치 정책이 로봇을 구동한다**(`lerobot-eval`엔 있는 경고가 `lerobot-record`엔 없다). `--policy.path`가 안 먹으면 위 중첩 레이아웃부터 의심할 것.
- `--robot.enable_base_motor_torque` — **eval에선 `true`가 필수이고 기본값이 아니다.** 끄면 정책의 `x.vel`·`theta.vel`이 base에 도달해도 무시되는데 **아무 에러도 안 난다** — 팔만 움직이고 base가 가만있는 것이 "base 동작을 못 배웠다"로 오독된다. `connect()`에서 한 번 적용되므로 처음부터 명령줄에 있어야 한다.
- `--dataset.repo_id` — **반드시 `eval_`로 시작**한다(정책을 주면서 아니면 즉시 `ValueError`). 반대로 **취득용 repo는 `eval_`로 시작하면 안 된다.**
- `--dataset.single_task` — 정책 종류와 무관하게 **필수**지만 쓰임이 갈린다. **pi0는 언어조건부**라 학습 때와 같은 문구를 줘야 하고, **ACT는 이 문자열을 정책 입력으로 쓰지 않아**(토크나이저 단계가 없다) 데이터셋 라벨로만 기록된다.
- `--teleop.*` — 리셋 구간에서 리더암으로 시작 자세·파지를 만들기 위한 것. 에피소드 구간은 `--policy.path`가 있으므로 정책이 구동한다. **리더암이 필요한 이유** = staged 자세 이동이 `connect()` 때 한 번뿐이라 **2번째 에피소드부터는 아무것도 자세를 되돌려 주지 않는다.** 리더암을 붙인 채로 돌 수 있게 된 근거 → [Joint Velocity Pacing](#joint-velocity-pacing)
- `--robot.velocity_safety_factor` — **기본값 `0.4`를 올리지 말 것.** `0.8`·`0.5`는 둘 다 실기에서 트립했다(조건은 `sf ≤ 1/2.07 = 0.483`) → [Joint Velocity Pacing](#joint-velocity-pacing)
- `--dataset.reset_time_s` — 리셋 구간이 자세 잡기까지 맡으므로 upstream 기본값 60이 아니라 **90**.
- `--robot.include_base_in_state` — 체크포인트의 state 차원과 짝을 맞춘다 → [Base Velocity in the Observation State](#base-velocity-in-the-observation-state)

**확인 범위** — lerobot 0.4.0~0.4.4에서 동일. 0.6.0부터는 정책 배포가 `lerobot-rollout`으로 분리되고 `lerobot-record`가 `--policy.path`를 거부하나, 체크포인트로 타입을 판별하는 원칙은 유지된다.

### eval 함정

- 🛑 **eval repo 이름이 겹치면 즉사한다** — `LeRobotDataset.create()`가 `root.mkdir(exist_ok=False)`라 같은 이름이 있으면 덮지 않고 `FileExistsError`로 죽는다. 중단된 빈 런이 자리를 잡고 있는 경우가 흔하고, 로봇 PC 캐시엔 `eval_*`이 이미 수십 개 있어 단계별 기본 이름은 대부분 선점됐다.
- ⚠️ **`| tail`을 붙여 돌리지 말 것** — 파이프의 마지막 명령 종료코드가 잡혀 **실패가 `exit 0`으로 보고된다.** 위 `FileExistsError`가 「정상 종료」로 올라온 적이 있다.
- 🛑 **`Joint 0 position input contains NaN`은 그 관절 문제가 아니다** — 정책 normalizer stats가 손상된 것(학습 데이터에 base 속도 garbage가 섞여 `std=NaN`이 되고 전 출력으로 전파된다). **코드로 못 고친다** — 손상 에피소드를 지우고 **재학습**해야 한다. 취득 쪽 차단은 [취득 후 점검](#취득-후-점검).
- ⚠️ **`Feature mismatch … Missing/Extra features`** — 체크포인트가 학습 때 쓴 카메라 키·해상도가 지금 로봇과 다르다. `--rename_map`으로 맞춘다. ⚠ **이름만 검사한다** — 해상도·채널 순서·fps가 달라도 **조용히 통과**하므로 「에러가 안 났다」가 「맞게 돌고 있다」의 증거가 못 된다.
- ⚠️ **관절이 `idle`로 떨어져 죽으면 세션 안에서 회복이 안 된다** — 프로세스를 다시 띄워야 `configure(clear_error=True)`가 걸린다.
- ⚠️ **베이스 e-stop 경고 읽는 법** → [취득 후 점검](#취득-후-점검)

### 무해한 로그

아래 셋은 **정상**이다. 실패로 읽지 말 것.

| 뜨는 것 | 정체 |
| --- | --- |
| 종료 시 Rerun `transport error`·`gracefully disconnected`·`channel closed` (ERROR 3~4줄) | 뷰어 gRPC 스트림이 닫히는 한 사건을 세 층에서 본 것. 저장은 정상(로컬·Hub 프레임 수 일치 확인) |
| 리셋 구간의 `No policy or teleoperator provided, skipping action generation…` 대량 출력 (`--teleop.type` 없이 돌린 실행 한정) | 리셋 구간이 정책 없이 도는 것은 정상. 다만 `record()`의 리셋 호출은 `teleop`을 그대로 넘기므로 **`--teleop.type`을 준 실행(취득·§3-3 eval)에서는 안 뜨고, 리더암 없는 실행에서만** 뜬다 |
| `Record loop is running slower (21.4 Hz) than the target FPS (30 Hz)` | 취득 파이프라인의 천장(카메라 USB 대역·base 시리얼). **기존 환경도 같은 값** |

- 🛑 반대로 **`→`·`←`·`ESC`가 아무 반응이 없는데 에러도 없으면** 세션이 Wayland다. `echo $XDG_SESSION_TYPE`으로 확인하고 "GNOME on Xorg"로 재로그인 → [Quickstart Guide](https://github.com/kiro-ai-division/mobile-ai-quickstart-guide#작업-pc-준비)

## 학습

- `--dataset.repo_id` — 학습에 쓸 데이터셋. 단계별 취득이므로 단계 repo 이름을 그대로 준다.
- `--policy.type` — 정책 종류(`act`·`pi0`·`smolvla`). **학습에서만 쓰는 인자다** — eval에선 체크포인트가 스스로 밝히므로 주지 않는다.
- `--policy.repo_id` — 학습 결과를 올릴 Hub repo. `push_to_hub` 기본값이 `true`라 **빼면 `ValueError: 'policy.repo_id' argument missing`으로 죽는다.**
- 🛑 **스모크런의 `--policy.push_to_hub=false`를 본 학습에 옮기지 말 것** — 붙으면 60k를 완주하고도 허브엔 껍데기 repo만 남고 대장 자동 열이 빈다(실제 발생: `task10_move_to_beaker_shelf_kiro`).
- `--output_dir` — 로컬 체크포인트 경로. 실제 로드 가능한 디렉터리는 `<output_dir>/checkpoints/<스텝>/pretrained_model`이다.
- `--batch_size` — **GPU 1장당** 배치.
- `--save_freq` — 체크포인트 저장 간격(스텝).
- `--wandb.enable` — `false`. 켜려면 그 기기에 `wandb login`이 선행돼야 하고, 끄더라도 loss·grad_norm·lr은 `log_freq`마다 콘솔에 찍힌다. 남기려면 `2>&1 | tee <로그파일>`.
- **lr은 GPU 수에 맞춰 자동 스케일되지 않는다.** 바꾸려면 **`--policy.optimizer_lr`**로 준다 — `--optimizer.lr`은 **경고 없이 무시되고** 정책 기본값 `1e-5`로 되돌아간다.
- ACT는 사전학습 체크포인트가 없어 scratch부터 학습한다(비전 백본만 ImageNet ResNet18로 자동 초기화). 그래서 `--policy.pretrained_path`가 없다.
- `tmux` — 학습이 몇 시간 걸려 SSH가 끊기면 프로세스가 같이 죽는다. tmux 안에서 돌리면 살아남는다.

### 유효 배치

**유효 배치 = `--batch_size` × 장 수 = 64.** 이 곱이 실제 학습 배치라, 바뀌면 기존 체크포인트와 **비교가 안 되는 다른 실험**이 된다. 장 수를 줄이면 `batch_size`를 그만큼 올린다(2장이면 32, 4장이면 16).

⚠️ **`batch_size` 16은 아직 안 재봤다** — 우리가 직접 돌린 조합은 `batch_size` 8뿐이다(sandia 4장 · DGX-1 8장 · 단일 GPU). 위임 7건도 `batch_size` 8이었고 **장 수는 회수되지 않아 미확인**이다. 스모크런이 OOM을 잡아 준다.

### 스텝 수

⚠️ **`60000`은 관행이지 결정이 아니다** — 위임 런 7건이 그 값으로 돌아왔을 뿐, 근거 기록이 없다. 우리 조사 결론은 **스텝을 고정하지 말고 데이터셋 크기에서 계산하라**는 것이다.

```
프레임당 학습 횟수(≈epoch) = steps × batch_size × 장 수 ÷ total_frames
```

- 목표 밴드는 **5~10 epoch**. 단계를 합쳐 에피소드가 길어지면 프레임 수가 배로 뛰는데 `--steps`는 그대로라 같은 스텝이 얇게 퍼진다.
- 스텝을 늘려 해결되는 문제가 아니라는 쪽 증거가 더 강하다 — 커뮤니티 실측에서 100k steps(40 epoch)로도 66%에서 정체한 사례와, 10k steps(~4 epoch)로 성공한 사례가 함께 있다.

### 정밀도 — fp32

**`--mixed_precision`을 주지 않는다.**

- ACT·Diffusion Policy 공식 구현 어느 쪽도 `autocast`·`GradScaler`를 쓰지 않는다. PyTorch AMP 안내도 회귀·생성 계열은 fp32가 필요할 수 있다고 적고 있고 ACT는 L1 회귀다.
- **V100(DGX-1)에선 bf16이 해롭다** — bf16 연산기가 없어 **경고 없이 에뮬레이션으로 떨어져 fp32보다 느려진다.** `torch.cuda.is_bf16_supported()`가 `True`를 주기 때문에 아무 신호도 없다.
- ⚠️ 속도 이득은 재지 않았다. 되돌리려면 §2-2 명령의 `accelerate launch` 뒤에 `--mixed_precision=bf16 \`을 넣는다.
- ⚠️ 기존 체크포인트는 대부분 bf16으로 학습됐고 `모델 체크포인트` 탭에 정밀도가 기록돼 있지 않다. **앞으로의 런부터** `학습 파라미터`에 `· fp32`를 적는다.

### 공유 서버

- 실행 전 `nvidia-smi` 하단 **Processes 표**로 점유를 확인하고 **프로세스가 없는 장만** 잡는다. 빈 메모리 수치가 아니라 프로세스 유무로 판단할 것(`--query-compute-apps=pid,used_memory --format=csv`는 스크립트가 파싱할 때 쓰는 형태). `sandia`는 RTX 3090 ×4가 전부이고 DGX-1은 V100 ×8이다.
- `--output_dir`·tmux 세션명에 **본인 이름**을 넣는다 — 공유 계정이라 그게 누구 런인지 남는 유일한 기록이다.
- 남의 프로세스·tmux 세션은 건드리지 않는다.

### DeepSpeed·FSDP는 쓰지 않는다

ACT는 51.6M 파라미터라 가중치·그래디언트·옵티마이저를 합쳐도 0.77 GiB다. 24GB에서 쪼개 봐야 아낄 것이 없다. 게다가 DGX-1 ZeRO-2 실측에서 **학습은 도는데 체크포인트에 `config.json`만 남았다** — DeepSpeed가 파라미터를 평탄 버퍼에 담아 각 텐서가 *부분 뷰*가 되는데 safetensors가 그것을 거부하기 때문이다. 메모리가 모자라면 `--batch_size`를 줄인다.

## 데이터셋 편집

- **항상 `--new_repo_id`를 줄 것** — 안 주면 원본이 `<경로>_old`로 밀리고 원래 자리에 결과가 덮인다.
- `episode_indices`는 **삭제 전 인덱스 기준**. 삭제 후 `0..N-1`로 자동 재번호된다.
- ⚠️ **같은 `repo_id`에 덮어쓰지 말 것** — 재패킹으로 파일 구성이 바뀌는데 push가 원격의 옛 파일을 지우지 않아 orphan이 남아 데이터셋 일관성이 깨진다.
- 다른 연산(`split`·`merge`·`info`·`remove_feature` 등)은 `uv run lerobot-edit-dataset --help`.
- `lerobot-dataset-viz`는 한 번에 한 에피소드만 연다. 데이터셋 기본 위치는 `~/.cache/huggingface/lerobot/<repo-id>`, 캐시 밖이면 `--root <경로>`.

## 부록 — pi0 (담당자 전용)

> 🛑 **이 repo 환경에서는 pi0·smolVLA가 돌지 않는다** — `transformers`가 의존성에 없다. **pi0는 조건부 import라 에러 없이 반쪽으로 돌고**(더 위험), smolVLA는 `ModuleNotFoundError`로 즉사한다. pi0는 별도 conda 환경(`fix/lerobot_openpi` fork)이 필요하니 윤준원에게 문의할 것.
>
> **팀 학습은 전부 ACT다** — 이 절은 건너뛴다.

---

# Fork reference

여기부터는 **이 fork가 upstream과 무엇이 다른가**의 레퍼런스다. 위 명령을 그대로 쓰는 데는 읽지 않아도 된다.

## Changes in this fork

| Change | What it does | Where |
| ------ | ------------ | ----- |
| **Mobile-base velocity sanitisation** | `mobileai.py` refreshes the base state before reading it and zeroes out garbage velocity readings left in stale serial buffers. Without it, policies trained on the recorded data fault with `Joint 0 ... contains NaN` at inference. Read path only; always on. | [#2](https://github.com/kiro-ai-division/lerobot_trossen/pull/2) |
| **Base command guard** | Non-finite base velocity *commands* are zeroed and clamped to +/-1.0 (m/s linear, rad/s angular) before they reach the base. Without it a NaN from the policy arrives as full-speed reverse. Always on. | [below](#base-command-guard) - [#20](https://github.com/kiro-ai-division/lerobot_trossen/pull/20) |
| **Base emergency stop detection** | `connect()` refuses to start while the base is in emergency stop, and `get_observation()` logs one red warning each time the base enters or leaves an abnormal state (emergency stop, controller fault, charging). The mid-run check rides the chassis block `update_state()` already fetches, so it costs no extra serial transaction. On by default; `--robot.estop_check=false` disables the connect-time error only. | [below](#base-emergency-stop-detection) - [#38](https://github.com/kiro-ai-division/lerobot_trossen/pull/38) |
| **Joint velocity pacing** | Stretches `goal_time` so no joint is commanded past its hard velocity limit, which is what used to kill the process at the policy/teleop handoff. `velocity_safety_factor` defaults to `0.4`; `LEROBOT_PACING_LOG` logs the decision per frame. | [below](#joint-velocity-pacing) - [#16](https://github.com/kiro-ai-division/lerobot_trossen/pull/16) |
| **`include_base_in_state` flag** | Drops the base velocity from `observation.state` so 14-dim policies can be evaluated. | [below](#base-velocity-in-the-observation-state) · [#4](https://github.com/kiro-ai-division/lerobot_trossen/pull/4) |
| **`LEROBOT_FAST_OBS`** | Moves eval-time image preprocessing to the GPU. On by default; roughly doubles the control-loop rate on the Mobile AI 3-camera setup. | [below](#environment-variables) · [#8](https://github.com/kiro-ai-division/lerobot_trossen/pull/8), [#14](https://github.com/kiro-ai-division/lerobot_trossen/pull/14) |
| **`LEROBOT_LOOP_HZ_LOG`** | Opt-in control-loop rate and per-section timing meter. | [below](#environment-variables) · [#6](https://github.com/kiro-ai-division/lerobot_trossen/pull/6) |
| **Single-wheel torch pin** | `torch` 2.8–2.10 on the cu128 index with `torchcodec` left on PyPI, so one lockfile covers Volta (V100), Ampere (RTX 3090/A6000) and Blackwell (RTX 5090). `.python-version` pins the interpreter so every clone resolves alike. | [#28](https://github.com/kiro-ai-division/lerobot_trossen/pull/28), [#30](https://github.com/kiro-ai-division/lerobot_trossen/pull/30) |

See the [LeRobot documentation](https://huggingface.co/docs/lerobot) and the [Trossen AI documentation](https://docs.trossenrobotics.com/trossen_arm/main/tutorials/lerobot_plugin.html) for anything beyond this fork.

## Base Velocity in the Observation State

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

## Joint Velocity Pacing

A single `set_all_positions` moves the arm over a fixed window
(`min_time_to_move_multiplier / loop_rate`, 0.1 s by default). A large position jump squeezed
into that window asks for a velocity the controller refuses:

```
[ERROR] Joint 3 velocity limit exceeded: expected [-9.424778, 9.424778], reported 9.633699. Setting to idle.
[ERROR] Joint 0 mode mismatch: 1 != 0
trossen_arm.RuntimeError: Robot input with modes different than configured modes received
```

The controller drops the offending joint to `idle` (mode 0) while the driver keeps sending
position commands (mode 1), so the next write is rejected and the process dies with no
in-session recovery. Two things produce such a jump: the **policy/teleop handoff** at episode
reset (observed in both directions) and a discontinuous **policy chunk boundary**.

`send_action` therefore paces every move:

```
goal_time = max( min_time_to_move,  max_j( |delta_j| / (velocity_safety_factor * velocity_max_j) ) )
```

`velocity_max` is cached once in `configure()` from `get_joint_limits()` (joints 3-5: 3*pi ~
9.4248 rad/s, joints 0-2: 2*pi, gripper 0.25 m/s). `blocking=False` is unchanged, so the
following loop iterations simply catch up. This supersedes `max_relative_target`, which caps
the delta but guarantees nothing about velocity.

| Config field | Default | Effect |
| ------------ | ------- | ------ |
| `velocity_safety_factor` | `0.4` | Fraction of each joint's hard velocity limit the pacing model may command. Lower is slower and safer. Pass as `--robot.velocity_safety_factor=<x>`. |

**Do not raise the default without re-measuring.** The controller enforces its limit on the
*peak* of the trajectory it generates, while this factor scales the *average* we command. On
hardware the peak measured **2.05-2.07x** the commanded average (joint_3, both arms, ~20 Hz
loop), so `0.8` and `0.5` both tripped and only `0.4` survived across 12 phase transitions.
The condition is `sf <= 1 / 2.07 = 0.483`. The tracking cost is negligible: pacing engaged on
**12 of 2165** policy-driven frames (0.6%). That 2.07 figure depends on the ratio of loop
period to `goal_time`, so re-measure it if the loop rate changes.

**Bigger jumps are not the dangerous ones.** Handoffs of 0.92 and 1.53 rad all passed; the
crash happened at 0.46-0.48 rad, right at the pacing threshold. And pacing does not remove the
risk entirely - frames where pacing never fires still carry a hard limit around
**delta_crit ~ 0.455 rad**, which no choice of `velocity_safety_factor` moves.

Raising `min_time_to_move_multiplier` instead is the worse trade: it stretches *every* frame
and blurs the whole trajectory, whereas pacing is a selective brake that only fires on the jump.

With this in place a leader arm can stay connected during eval, which is what makes staged
evaluation possible - the operator sets the next episode's start pose and grasp by hand during
the reset window, and the policy drives the episode itself.

## Base Command Guard

Base velocity *commands* are checked for non-finite values and clamped to +/-1.0 (m/s for the
linear channel, rad/s for the angular one) before they reach `set_cmd_vel()`. The read path has been sanitised since the base velocity NaN
incident ([#2](https://github.com/kiro-ai-division/lerobot_trossen/pull/2)); the command path was not, and the asymmetry was backwards - a
corrupted reading poisons a dataset, a corrupted command drives the robot.

The hardware clamp inside `TrossenSlate::set_cmd_vel` is `min(MAX, max(-MAX, v))`. NaN compares
false against everything, so `max(-MAX, NaN)` returns `-MAX`: a NaN action reached the base as
**full-speed reverse**, and eval runs with `enable_base_motor_torque=True`, so that command was
actually executed. Verified in the installed `trossen_slate` 0.0.3 binary. Failed writes are
now reported rather than swallowed, throttled to one warning per second per channel.

## Base Emergency Stop Detection

The base reports its own system state inside the chassis block `update_state()` already reads
once per control-loop iteration, so the mid-run check is free: `read()` only copies that struct.
Nothing calls `update_state()` an extra time inside the loop - that call is a 21 ms Modbus
transaction and a de-rated loop is a base over-rotation multiplier. `connect()` does issue one of
its own, because nothing has filled the chassis buffer at that point and judging an
uninitialised buffer would make the check silently useless; once, at startup, that cost is
irrelevant.

| Where | Level | Catches |
| ----- | ----- | ------- |
| `connect()`, after `init_base()` and before `enable_motor_torque()` | `RuntimeError` | Starting a run with the button already pressed |
| `get_observation()`, on the existing `update_state()` success path | Red log warning, no sound | The button being pressed mid-run |

The two levels are deliberately different. A run that *starts* in emergency stop records nothing
usable, so it is stopped at `connect()` before a single frame is written, and the arms are
released first because nothing else would (`is_connected` is still `False` at that point, so
`record()`'s `finally` does not fire). Note that `LeRobotDataset.create()` runs *before*
`robot.connect()`: an aborted start still leaves an empty dataset directory behind, so reusing
that `--dataset.repo_id` afterwards fails with `FileExistsError` - see [eval 함정](#eval-함정).

A run that *enters* emergency stop is only warned about. `lerobot-record` drives the record phase
and the reset phase through the same `record_loop()` and the robot cannot tell them apart, while
pressing the button during reset to push the base by hand is normal operation. Raising there
would abort before `dataset.save_episode()`, which runs *after* the reset window, so the episode
just recorded would be lost. **A warning during the reset window is expected; a warning during
the recording window is not** - the base did not move for that stretch.

Warnings are edge-triggered: one line when the base leaves the normal state and one when it
returns, never one per frame. The same path also reports the controller fault codes and the
charging state; those are warnings only and never block a run. A value that is not one of the
known state codes is ignored rather than warned about, because the chassis buffer is
uninitialised until the first successful `update_state()` and an unrecognised code is more likely
garbage than a state. The message carries a `[MOBILE AI BASE]` prefix and is printed bold red
through `termcolor`, because lerobot's log format has no logger name field and a fork warning is
otherwise indistinguishable from an upstream one. Colour is dropped automatically on a non-TTY -
`termcolor` tests `stdout` while the log handler writes to `stderr`, so redirecting only `stderr`
keeps the escape codes; `NO_COLOR=1` disables colour everywhere.

| Config field | Default | Effect |
| ------------ | ------- | ------ |
| `estop_check` | `true` | Gates the `connect()` hard error only. Pass `--robot.estop_check=false` to start a run with the base in emergency stop, e.g. an eval that deliberately keeps the base immobilised. The `get_observation()` warnings are unaffected. |

## Upstream flags (not fork changes)

### Optional Observation Features

By default, Mobile AI followers only observe joint positions (`<joint>.pos`).
You can optionally record additional per-joint signals by enabling the following flags.
All are disabled by default.

| Flag | Observation key | Description |
| ---- | --------------- | ----------- |
| `include_velocity` | `<joint>.vel` | Joint velocity. Measured in rad/s for the arm joints and m/s for the gripper carriage. |
| `include_effort` | `<joint>.eff` | Total motor effort, combining gravity, friction, and any external load. Measured in Nm for the arm joints and N for the gripper carriage. Nonzero even when the arm is holding still against gravity. |
| `include_external_effort` | `<joint>.ext_eff` | Estimated externally applied effort, after gravity and friction compensation. Measured in Nm for the arm joints and N for the gripper carriage. Useful for contact and force sensing; an unloaded arm reports values near zero. |

Pass them as `--robot.<flag>=true` when running any command that constructs the robot. The
flags are shared across both arms, and the resulting observation keys are prefixed per arm,
e.g. `left_<joint>.eff` and `right_<joint>.eff`.

## Environment Variables

| Variable | Default | Effect |
| -------- | ------- | ------ |
| `LEROBOT_FAST_OBS` | `1` (on) | Converts camera frames to float32 and permutes HWC→CHW **on the GPU** instead of the CPU. Set `0` to fall back to the stock lerobot path. |
| `LEROBOT_LOOP_HZ_LOG` | unset (off) | Set `1` to log the achieved control-loop rate and a per-frame section breakdown. |
| `LEROBOT_PACING_LOG` | unset (off) | Set `1` to log the joint velocity pacing decision on *every* frame. Frames where pacing actually fired are logged either way. |

**`LEROBOT_FAST_OBS`** — lerobot's `prepare_observation_for_inference` converts and permutes
camera frames CPU-side and only then copies them to the GPU, shipping 4× the bytes over PCIe
and paying for an elementwise divide plus a full `.contiguous()` copy per camera. An offline bench (RTX 5090 laptop, real ACT
checkpoint, synthetic frames of the production 3-camera shape) measured **54.7 ms → 1.6 ms per
frame** (p50 under CPU load); on the robot the eval loop went from **9.73 Hz to 20.85 Hz**
(140-window mean) against a 21.5 Hz teleop recording baseline, i.e. an over-rotation
multiplier of 2.2 → 1.03. This matters beyond throughput: the SLATE base holds a velocity command
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

**`LEROBOT_PACING_LOG`** - the controller log tells you *that* a velocity limit was tripped but
never what was commanded, so `send_action` logs its own pacing decision:

```
pacing[192.168.1.4] FIRED goal_time=101.8ms max_delta=0.4797@joint_3 avg_v=4.712 deltas=[...]
```

`FIRED` lines - the ones where `goal_time` was stretched past `min_time_to_move` - are always
emitted, because the event is rare and is usually what you are chasing. Setting this variable
adds an `idle` line for every other frame, which is what separates "pacing never fired" from
"pacing fired and was not enough". `avg_v` is the average velocity the pacing model believes it
commanded; compare it against that joint's `velocity_max`, and remember the measured peak runs
about twice the average.
