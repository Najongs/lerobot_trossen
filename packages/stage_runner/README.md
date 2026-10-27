# Trossen-LeRobot Stage Chaining Runner

실행 명령·종료 코드·함정의 정본은 루트 README — [3-4. 단계 체이닝 실행](../../README.md#3-4-단계-체이닝-실행-stage_runner) · [단계 체이닝 실행](../../README.md#단계-체이닝-실행). 이 문서는 **단계별 팔 시작 자세(`start_pose`)**, **리더암 리셋 단계(`leader_reset`)**, **베이스 이동 단계(`base_replay`·`base_align`)** 만 다룬다.

## 단계 시작 자세 (start_pose)

단계를 시작하기 전에 두 팔의 관절을 **그 단계 학습 데이터의 대표 시작 자세**로 옮긴다. 첫 단계 포함, 이후 단계는 경계의 베이스 정지 뒤. 선택 사항 — `start_pose:`가 없는 단계는 지금처럼 앞 단계가 남긴 자세에서 시작한다.

- 🛑 **그리퍼는 명령하지 않는다** — 이동 중 그리퍼 목표 = 앞 단계 정책의 마지막 명령(쥔 힘 유지). 쥔 물체(비커·튜브)가 이동 중에 안 미끄러지는지는 **실기 미확인** — 첫 실행은 비상정지 옆에서.

### 1. 자세 계산 — 네트워크 되는 PC에서

```shell
uv run python -m stage_runner.start_pose --config_path configs/chain_task45.yaml --output outputs/start_poses.yaml
```

- 단계별 데이터셋 = 그 단계 체크포인트 `train_config.json`의 `dataset.repo_id`(`dataset.episodes`가 있으면 그 에피소드만) — 다른 데이터셋을 쓸 땐 `--dataset s4=<repo_id>`
- 로컬 사본으로 계산 → `--dataset-root s4=<데이터셋 디렉터리>`(내려받기 없음). 블록의 `dataset_repo_id`는 경로가 아니라 체크포인트의 학습 데이터셋 이름(또는 `--dataset`)
	- 허브 캐시 사본(`datasets--<org>--<이름>/snapshots/<커밋>`)이 학습 데이터셋과 다른 데이터셋이면 종료 코드 `2`
	- 체크포인트가 로컬 디렉터리로 학습됐으면(`train_config.json`의 `dataset.root`) `--dataset-root` 없이는 종료 코드 `2` — 허브 사본이 없거나 학습한 바이트와 다를 수 있어서
- 출력은 `outputs/`(git 무시 폴더) 아래 — 저장소 루트에 두면 추적 안 된 파일로 남는다
- 내려받는 것 = `meta/info.json`·`data/**/*.parquet`만(영상 제외)
- 단계 하나만 → `--stage s4`
- 통과 기준 = 단계마다 `# stage <id>: paste under …` 블록 1개, 블록의 `source.n_episodes`가 데이터셋 에피소드 수와 같음 — 적으면 첫 프레임이 NaN인 에피소드가 빠진 것이고 블록 끝 주석에 번호가 찍힌다

### 2. 설정 파일에 붙여넣기

블록을 해당 단계 항목(`- id: s4`) 아래, `terminator:`와 같은 들여쓰기에 붙인다.

```yaml
  - id: s4
    policy_path: kiroaiseoul/act_task04_pour_liquid_from_tubes_to_beaker_60000
    terminator:
      type: timeout
      timeout_s: 12.9
    precondition: null
    start_pose:
      positions:
        left_joint_0.pos: -0.290112167596817
        # … 두 팔 관절 12개
      min_goal_time_s: 3.0
      position_tolerance_rad: 0.05
      source:
        dataset_repo_id: "kiroaiseoul/task04_pour_liquid_from_tubes_to_beaker"
        episode_index: 39
        n_episodes: 130
        # …
```

- **값은 손으로 고치지 말고 다시 생성할 것** — `source`(데이터셋·커밋·에피소드 번호)가 그 값의 출처라 손댄 값은 출처를 잃는다
- `min_goal_time_s` = 이동 최소 시간(초). 관절이 안전 속도를 넘을 만큼 멀면 그보다 길어진다
- `position_tolerance_rad` = 도착 판정 허용 오차(rad)
- 체크포인트를 바꾸면 다시 생성 — 다른 체크포인트를 가리키는 자세는 preflight가 경고만 하고 돌린다

### 3. 실행

```shell
uv run python -m stage_runner --config_path configs/chain_task45.yaml
```

- 통과 기준 = `events.jsonl`에서 단계마다 `stage_start` **바로 앞**에 `stage_reposition` 한 줄, `outcome: "reached"`
- `reached`가 아니면 그 단계 정책은 **돌지 않고** 시행이 중단된다(종료 코드 `1`) — `reason`에 어느 관절이 얼마나 어긋났는지 → [stage_reposition 이벤트](#stage_reposition-이벤트)
- 자세가 잘못돼 있으면 connect 전에 종료 코드 `2` → [preflight 거부](#preflight-거부)

## 리더암 리셋 단계 (leader_reset)

`stages:` 목록 **아무 자리에나** 넣는 단계 종류. 정책 없이 리더암으로 팔을 움직여 튜브·비커를 쥐여 주고 자세를 잡은 뒤 `→`로 끝낸다 — `lerobot-record`의 리셋 구간과 같다. 기록하지 않는다.

- 🛑 **리셋이 시작되면 리더암이 먼저 저절로 움직인다**(팔로워의 현재 자세로, 2초 이상) — 멈출 때까지 리더암에서 손을 뗀다
- 🛑 **베이스 토크가 꺼진다**(손으로 밀 수 있음) — 리셋이 끝나면 다시 켠다. 연결 때 토크를 켜는 설정(`enable_base_motor_torque: true`)에서만

### 1. 설정 파일에 넣기

```yaml
teleop:
  type: mobileai_leader_teleop
  left_arm_ip_address: 192.168.1.3
  right_arm_ip_address: 192.168.1.2
  id: leader

stages:
  - id: reset_tube
    name: 튜브 쥐여 주기
    executor: leader_reset
    terminator:
      type: manual
      timeout_s: 120.0
  - id: s4
    # … 정책 단계 그대로(start_pose 포함)
```

- `teleop:` = `lerobot-record`의 `--teleop.*`와 같은 값, 최상위에 한 번
- `start_pose`는 리셋 **다음** 정책 단계에 둘 것 — 리셋 단계에 두면(`policy_path`도) connect 전에 종료 코드 `2` → [리셋 단계 preflight 거부](#리셋-단계-preflight-거부)
- `timeout_s` = 상한(초). 상한에 닿으면 `→`를 누른 것처럼 다음 단계로 넘어간다

### 2. 실행

```shell
uv run python -m stage_runner --config_path <설정 파일>
```

리셋 단계마다:

1. 리더암이 팔로워 자세로 움직였다가 멈춤 — 콘솔 `aligning the leader arm …`
2. 콘솔 `leader arm in control` — 리더암으로 물체를 쥐여 주고 자세를 잡는다(베이스는 손으로)
3. `→` — 다음 단계로. 다음 정책 단계에 `start_pose`가 있으면 팔이 그 자세로 옮겨지고 그리퍼는 방금 쥔 명령 그대로
4. `ESC`·`←` — 시행 중단(정책 안 돎, 종료 코드 `1`)

- 통과 기준 = `events.jsonl`에서 리셋 단계의 `stage_start`·`stage_end` 사이에 `stage_reset` 한 줄, `base_torque`가 `released`(토크 켠 설정) 또는 `not_released`
- 리셋 다음 `stage_reposition`의 `gripper_goal_source`가 `previous_command`이고 `gripper_goal`이 `stage_reset.gripper_goals`와 같음

## 정책 단계 개입 취득 (선택)

정책 단계에서 **스페이스바** → 리더암이 넘겨받아 개입 데이터셋에 녹화. 키·동작은 `lerobot-record`와 같다 → [루트 README 3-5](../../README.md#3-5-실패-지점에서-넘겨받아-녹화-개입-취득)

```yaml
teleop:
  type: mobileai_leader_teleop
  left_arm_ip_address: 192.168.1.3
  right_arm_ip_address: 192.168.1.2
  id: leader
```

```shell
LEROBOT_INTERVENTION_REPO_ID=kiroaiseoul/<개입_데이터셋> uv run python -m stage_runner --config_path <설정 파일>
```

- 켜고 끄기 = 변수 한 줄. 없으면 리더암을 연결하지 않음 — `teleop:` 블록이 있어도 시연은 그대로
- 변수가 있는데 `teleop:` 블록이 없으면 연결 전에 종료 코드 `2`
- 스페이스바는 정책 단계에서만 — 베이스 이동·정렬 단계에선 아무 일 없음
- 개입 녹화 상한·개입 리셋 상한 각 60초(`lerobot-record` 기본값) — 단계 타이머와 무관
- `→`로 저장하면 체인이 다음 단계로 이어짐, `ESC`로 저장하면 시행 종료
- 개입 데이터셋은 로컬에만 — 올릴 땐 `LeRobotDataset(<repo>).push_to_hub()`
	- ⚠ 로컬에서 에피소드를 지운 뒤면 `push_to_hub()`만으로는 허브에 옛 파일이 남음 — 먼저 `HfApi().upload_folder(..., delete_patterns=["data/**", "meta/**", "videos/**"])`
- 개입 데이터셋만 필요한 회차 = 체인 녹화 저장 끔 — 설정 `dataset:` 블록에 `save: false`
	- 마지막 저장·인코딩 없이 마지막 단계 직후 종료(카메라 3대 순차 인코딩이면 약 4분 단축), 체인 데이터셋은 0 에피소드
	- 프레임은 실행 중 계속 버퍼에 쌓임 — 베이스 단계가 그리퍼 목표를 녹화 버퍼의 마지막 명령에서 가져오기 때문. 끝에 카메라 PNG까지 삭제
- 개입 저장을 fork 없이 순차 인코딩 — `LEROBOT_INTERVENTION_SERIAL_ENCODING=1`(카메라 끊김 진단용). 개입 하나 저장 약 6초 → 15~18초

## 정책 단계 동안 베이스 토크 끄기 (release_base_torque)

정책이 내는 베이스 명령을 그 단계 동안만 무시하게 함 — 단계 6처럼 pi0가 단계가 끝나기 전에 베이스를 몰기 시작하는 단계용

```yaml
  - id: s6
    executor: lerobot_policy
    release_base_torque: true
```

- 단계 시작 전 속도 0 → 토크 끔, 끝나면(타이머·판별기·키·예외 모두) 속도 0 → 토크 켬
- 이동 단계가 든 1~11 전체 체인용. 이동 단계가 없는 체인은 `robot.enable_base_motor_torque: false` 한 줄로 실행 전체가 같아짐
- 토크가 안 꺼지면 그 단계는 정책을 돌리지 않고 시행 종료, 다시 안 켜지면 다음 단계를 돌리지 않고 시행 종료
- 토크를 끈 채 연결한 로봇(`enable_base_motor_torque: false`)이면 아무것도 안 바꿈
- 단계 안 기록 = `events.jsonl`의 `stage_base_torque` 한 줄, `base_torque` 값은 `released`·`not_released`·`release_failed`·`restore_failed`
- preflight 거부 — 정책 단계가 아닌 단계에 붙음, 토크를 켜고 연결했는데 `robot.base.enable_motor_torque`가 없음
- ⚠ 토크가 꺼진 동안 베이스 바퀴가 풀려 손으로 밀림 — 그 단계 동안 로봇을 밀지 말 것

## 베이스 이동 단계 (base_replay · base_align)

이동 단계를 정책 대신 두 단계 한 쌍으로 돈다 — **데모 경로 재생**(`base_replay`) → **도착 화면 정렬**(`base_align`, cam_high를 데모 도착 화면과 비교해 헤딩 먼저, 거리는 고정물 크기로). 체인 아무 자리에, 이동마다 한 쌍씩. 둘 다 기록한다(프레임의 task 문자열 = 단계 `name`).

- 🛑 **베이스가 스스로 움직인다** — 첫 실행은 비상정지 옆에서. 이상 시 멈추는 문턱값은 실기에서 안 잰 추정값 → [실기에서 아직 안 잰 것 — 베이스 단계](#실기에서-아직-안-잰-것--베이스-단계)

### 1. 로봇PC 로컬 데이터셋 확인

```shell
grep -o '"x.pos"\|"theta.pos"' ~/.cache/huggingface/lerobot/kiroaiseoul/test_replay_timing_1024_basepose/meta/info.json
ls ~/.cache/huggingface/lerobot/kiroaiseoul/test_replay_timing_1024/videos
```

- 통과 기준 = 첫 줄 `"x.pos"`·`"theta.pos"` 두 줄, 둘째 줄 `observation.images.cam_high`가 찍힘
- 첫 줄이 빈 출력·`No such file` → 위치 라벨 사본부터 → [위치 라벨 데이터셋](#위치-라벨-데이터셋)
- 실행기는 데이터셋을 **내려받지 않는다** — 두 데이터셋 모두 이 경로(또는 YAML의 `root`)에 있어야 한다
- 다른 이동 단계 → 그 단계 데모 데이터셋 이름으로

### 2. 설정 파일에 넣기

로봇 예 = [`configs/chain_stage12_align.yaml`](../../configs/chain_stage12_align.yaml)(단계 1 → 정렬 → pi0 단계 2). 이동 하나 = 아래 두 항목.

```yaml
stages:
  - id: s1
    name: move_to_tube_rack          # = task 문자열, 다른 단계와 겹치면 거부
    executor: base_replay
    terminator:
      type: timeout
      timeout_s: 30                  # 폭주 상한
    replay:
      dataset: kiroaiseoul/test_replay_timing_1024_basepose
      episode: 0
      from_row: 0
      until_row: 297                 # 이 행까지(포함)
  - id: s1_align
    name: align_at_tube_rack
    executor: base_align
    terminator:
      type: timeout
      timeout_s: 60                  # max_moves 6이면 57 이상
    align:
      reference_dataset: kiroaiseoul/test_replay_timing_1024
      episode: 0
      frame: 302                     # 데모 도착 화면
      landmark_roi: [285, 197, 93, 61]   # 기준 화면 속 고정물 상자 [x, y, w, h]
      landmark_distance_m: 0.60      # 고정물까지 거리
  - id: s2
    # … 정책 단계(start_pose는 여기에)
```

- `timeout_s` 하한 — 재생 = `0.5 + (행 수 - 1) × row_period_s + 6`초, 정렬 = `max_moves` 6일 때 57초. 모자라면 connect 전에 종료 코드 `2`(필요한 초가 메시지에)
- 재생 목표 속도 상한 — 행 간 최대 변화 ÷ `row_period_s`가 0.6 m/s 또는 0.7 rad/s를 넘으면 connect 전에 종료 코드 `2`(맞는 `row_period_s`가 메시지에). 10/24 실측 최대 = 0.49 m/s·0.57 rad/s
- `start_pose`는 베이스 단계 **다음** 정책 단계에. 예외로 `base_replay`에는 둘 수 있다 — 다음 조작 단계의 시작 자세를 두면 이동 전에 팔을 옮기고 재생 내내 그 자세로 잡아 둔다. `base_align`에 두면 거부
- 이동이 여럿 → 이동마다 한 쌍. 쌍마다 자기 `episode`·행 범위·기준 화면·`landmark_roi`·`landmark_distance_m`
- 기준 화면을 데이터셋 프레임 대신 그 자리에서 찍은 사진으로 → `reference_image: <PNG 경로>`(`reference_dataset` 대신, 둘 중 하나만)
- 나머지 필드·기본값 → [베이스 단계 설정 필드](#베이스-단계-설정-필드)

### 3. 실행

```shell
env LEROBOT_CHUNK_PREFETCH_TICKS=12 LEROBOT_CHUNK_RTC=1 LEROBOT_CHUNK_SEAM_BLEND_TICKS=4 HF_HUB_OFFLINE=1 uv run python -m stage_runner --config_path configs/chain_stage12_align.yaml
```

- `env …` 스위치 셋 = 10/24 pi0 단계 2 eval 조건(정책 단계에만 걸린다) — ACT 체인이면 빼도 된다 → [Chunk Seam Smoothing](../../README.md#chunk-seam-smoothing)
- 통과 기준 = `events.jsonl`에서
	- 재생 단계 `stage_base_replay` 한 줄, `outcome: "settled"`
	- 정렬 단계 `stage_base_align` 한 줄, `outcome: "aligned"`(또는 `"heading_only"` — 헤딩만 맞고 거리를 못 읽음, 체인은 계속)
	- 그다음 정책 단계의 `stage_start`
- 정렬 중 화면 = `outputs/stage_runner/<run_id>/base/<단계 id>/s<번호>_overlay.png` — 기준·현재 화면 겹침
- `→` — 베이스를 그 자리로 인정하고 다음 단계로(`manual`). 재생 중이면 정렬이 베이스가 멈추길 기다렸다 잰다
- `ESC`·`←` — 시행 중단(종료 코드 `1`)
- 그 밖에 멈추면 종료 코드 `1`, 다음 정책은 안 돈다 — `stage_base_*` 줄의 `outcome`·`reason` → [베이스 단계 결과](#베이스-단계-결과)

---

## 레퍼런스

### 자세를 고르는 방식

사용자 승인 공식(2026-10-19). 구현 = `src/stage_runner/start_pose.py`.

1. 데이터셋 = 단계 체크포인트의 학습 데이터셋(`train_config.json`). `--dataset`으로 덮어쓸 수 있다.
2. 모든 에피소드의 **첫 프레임(`frame_index` 0)** `observation.state`에서 **두 팔의 관절 열만** 쓴다 — `meta/info.json`의 `features["observation.state"]["names"]`로 **이름으로** 고른다(`.pos`이면서 그리퍼·베이스가 아닌 열). 위치로 고르지 않는 이유: 그리퍼 열이 두 팔 관절 사이에 끼어 있고(`left_left_carriage_joint.pos`), `include_base_in_state`·`include_velocity`에 따라 같은 위치가 다른 관절이 된다.
3. 관절마다 에피소드 간 표준편차로 표준화(표준편차 0이면 1로 — 모든 에피소드가 같은 값이라 거리에 기여 0), 관절별 중앙값을 구하고, 표준화 L2 거리가 가장 가까운 **실제 에피소드**를 고른다. 동점이면 에피소드 번호가 작은 쪽.
4. 자세 = 그 에피소드 첫 프레임의 팔 관절 값.

- **중앙값 자체를 쓰지 않는 이유** — 관절별 중앙값은 관절마다 다른 시연에서 온 값을 섞어, 어떤 시연에도 없던 자세가 될 수 있다. 정책은 그 자세의 카메라 영상·state를 본 적이 없다. 실제 에피소드는 정책이 그 자세에서 시작하도록 학습된 자세다.
- **표준화하는 이유** — 시연 간 흔들림이 관절마다 크게 다르다. 표준화하지 않으면 가장 흔들리는 관절 하나가 「대표」를 정한다.
- **물체를 쥔 채 시작하는 단계** — 그리퍼 열은 계산에서도 이동에서도 빠진다. 시연 첫 프레임의 그리퍼 폭(그날 물체의 폭)으로 그리퍼를 옮기면 오늘 물체 폭과 다를 때 놓치거나 더 조이므로, 그리퍼는 앞 단계가 남긴 목표를 그대로 둔다 → [무엇을 움직이고 무엇을 안 움직이나](#무엇을-움직이고-무엇을-안-움직이나)
- `source.distance` = 고른 에피소드의 표준화 거리, `source.n_episodes` = 계산에 들어간 에피소드 수.

### 무엇을 움직이고 무엇을 안 움직이나

구현 = `src/stage_runner/reposition.py`.

- **팔 관절만** — `TrossenArmDriver.set_arm_positions`(팔 관절 벡터만 받고 그리퍼는 그대로 둔다). 값은 이름으로 드라이버 관절 순서에 매핑한다.
- **그리퍼 안 움직임** — 이동 중 그리퍼 목표 = 앞 단계 정책의 마지막 명령.
	- 경계의 베이스 정지(`stop_base`)는 팔 관절·그리퍼 목표를 **측정 위치**로 다시 보낸다. 물체를 쥔 그리퍼의 측정 위치는 물체 폭이라, 그대로면 위치 오차가 0이 돼 쥔 힘이 거의 사라진 채 몇 초 이동한다(위치 제어 원리로 본 추정 — 실측 없음).
	- 그래서 **다음 단계에 `start_pose`가 있는 경계만** 정지 명령의 그리퍼 값을 앞 단계 마지막 명령(에피소드 버퍼의 마지막 `action`, 앞 단계가 리셋 단계면 리셋의 마지막 명령)으로 둔다. `start_pose` 없는 경계는 이전과 같다.
	- 어느 값이 걸렸는지 = `stage_reposition.gripper_goal_source`(아래 표).
- **베이스 안 움직임** — 속도 제어라 「자세」가 없고, 위치는 앞 단계의 결과다.
- 이동 시간 = `max(min_goal_time_s, 관절별 이동량 / (velocity_max × velocity_safety_factor))` — 정책 구동 중 `WidowXAIFollower.send_action`과 같은 페이싱 규칙(`velocity_safety_factor` 0.4).
- **이동 전에 직전 궤적이 끝나기를 기다린다** — 직전 팔 명령(경계의 베이스 정지 명령, 보통 0.1초짜리 궤적)의 보낸 시각 + `goal_time` + 여유 0.05초까지 잠깐 쉰 뒤 이동을 보낸다. 팔·그리퍼에 따로 명령하지 않는다.
	- 이유 — trossen_arm 1.9.0 드라이버는 새 궤적을 **실행 중인 궤적의 명령 위치·속도·가속도**에서 시작한다(측정값이 아니라). 0.1초 궤적 도중에 3초 이동을 보내면 그 도중 가속도를 이어받아(영향이 이동 시간의 제곱으로 커진다) 두 팔이 반대 방향으로 최대 160° 휘두른 뒤 도착했다(2026-10-20 실기 로그). 궤적이 끝난 뒤면 시작 상태가 (정지 목표, 0, 0)이다.
	- 보낸 시각·`goal_time`은 `WidowXAIFollower`가 매 명령에 남기는 기록(`last_command_sent_s`·`last_command_goal_time_s`)에서 읽는다 — 적응형 `goal_time`(페이싱으로 늘어난 값)도 그대로 들어 있다. 두 팔 중 **늦게 끝나는 쪽**까지 기다린다.
	- 기록이 아예 없는 팔(옛 `lerobot_robot_trossen`)이면 고정 1초를 기다리고 경고 로그를 남긴다 — 실기 임시 패치가 1초 쉬어 휘두름을 없앤 값이고, 페이싱으로 1초보다 긴 프레임은 못 덮는다.
	- 기다리는 시간 = 직전 궤적의 남은 시간 + 여유 — 경계에선 보통 0.1초 안팎(경계 처리에 이미 쓴 시간만큼 줄어든다). 직전 명령이 이미 끝났으면(실기 첫 단계 — connect의 블로킹 이동 뒤) 또는 명령이 없으면(mock 첫 단계) 0.
- 두 팔 동시에 — 왼팔 비블로킹, 오른팔 블로킹(같은 이동 시간), 끝나면 두 팔 위치를 읽어 `position_tolerance_rad` 안인지 1초까지 확인.
- 이동 중에는 **프레임을 기록하지 않는다**(`record_loop` 밖) — `stage_reposition.frame_idx`가 앞 `stage_end`와 같다.
- 건너뛰거나 거부하는 경우:
	- 앞 경계에서 `ESC`·`←`를 이미 눌렀음 → `skipped`(팔 안 움직임, 시행은 평소처럼 중단)
	- 앞 경계의 베이스 정지가 실패(`stop_base_path: "failed"`) → `refused`(베이스가 움직이는 중일 수 있어 몇 초짜리 이동을 하지 않는다)
	- 관절 한계 밖 목표·드라이버 예외 → `error`
	- 허용 오차 밖 → `not_reached`
	- 이동 중 `Ctrl+C` → `interrupted` 기록 후 예외 그대로 전파(베이스는 이미 정지 상태, 기존 예외 경로대로 에피소드는 저장 안 함)
- `reached`가 아닌 결과(`skipped`·`interrupted` 제외)에선 그 단계에 `stage_start`·`stage_end`(`terminator: "error"`, 0프레임, `elapsed_s` 0.0)를 남기고 시행을 **중단**으로 끝낸다 — 앞 단계 프레임은 부분 에피소드로 **저장**된다(이동 중 버퍼에 아무것도 안 들어가 버퍼 상태가 확실하므로).

### preflight 거부

connect 전, 체크포인트를 내려받기 전에 종료 코드 `2`로 막는 것.

- `positions` 키가 이 로봇 팔 관절과 정확히 같지 않음(빠진 관절·모르는 키)
- 그리퍼 키(`*_carriage_joint.pos`)·베이스 키(`x.vel`·`theta.vel`)가 들어 있음
- NaN·inf 값
- `positions`가 비어 있음, `min_goal_time_s`·`position_tolerance_rad`가 0 이하
- 로봇에 팔별 드라이버(`robot.arms.left_arm.driver`)가 없음(예: 평면 mock)
- 경고만 — `source.policy_path`가 단계의 `policy_path`와 다름, `source.dataset_repo_id`가 비어 있음

### stage_reposition 이벤트

`start_pose`가 있는 단계마다 그 단계 `stage_start` 바로 앞에 한 줄.

| 필드 | 뜻 |
|---|---|
| `outcome` | `reached`·`not_reached`·`error`·`refused`·`skipped`·`interrupted` |
| `reason` | 판정 문장(어긋난 관절·오차 포함) |
| `target` | 목표 자세(설정의 `positions`) |
| `start_positions`·`end_positions` | 이동 전·후 실측 팔 관절 위치(`reached`·`not_reached`만) |
| `max_abs_error_rad` | 이동 후 최대 관절 오차 |
| `goal_time_s` | 페이싱이 정한 이동 시간 |
| `expiry_wait_s` | 이동 전에 직전 궤적이 끝나기를 기다린 시간(`reached`·`not_reached`만, 나머지는 `null`) |
| `expiry_wait_source` | 그 대기의 근거 — `recorded`(두 팔의 기록에서 계산, 이미 끝났으면 0)·`no_command`(기록은 있으나 아직 명령이 없음 — 대기 없음)·`unrecorded`(기록 없는 팔이 있어 고정 1초, 경고 로그) |
| `previous_goal_time_s` | 팔별(`left_arm`·`right_arm`) 직전 명령의 `goal_time`, 아직 명령이 없거나 기록이 없는 팔(`expiry_wait_source: unrecorded`)이면 `null` |
| `reposition_s` | 실제로 걸린 시간(직전 궤적 대기·확인 대기 포함) |
| `source` | 설정의 출처 블록 그대로 |
| `boundary_stop_base_path` | 직전 경계의 베이스 정지 경로, 첫 단계는 `null` |
| `gripper_commanded` | 항상 `false` — 이동 자체는 그리퍼에 명령하지 않는다 |
| `gripper_goal_source` | 이동 중 그리퍼 목표의 출처 — `previous_command`(앞 단계 마지막 명령, 정상)·`measured`(마지막 명령을 못 읽어 측정 위치 — 쥔 힘이 약할 수 있음, 경고 로그)·`unchanged`(첫 단계, 또는 베이스 정지가 팔 명령 없는 대체 경로로 감) |
| `gripper_goal` | 그 그리퍼 목표 값, `unchanged`면 `null` |

### 집계에서의 이동 시간

- `stage_end_to_stage_start_residual_s`·`boundary_cost_s`는 **이동 시간을 빼고** 잰다 — 이동이 그 창 안에서 일어나므로 `reposition_s`를 빼서, 자세가 없던 회차와 같은 뜻을 유지한다.
- 뺀 결과가 0보다 1 ms 이내로 작으면 0으로 본다 — `wall_clock_iso`가 ms 단위로 잘려 생기는 오차라 시계 경고를 내지 않는다(그보다 작으면 경고)
- 이동 시간은 따로 `stage_reposition_s`(도착한 이동만, 직전 궤적 대기 포함), 미도달은 `stage_reposition_failed`로 집계한다. 경계 전체 정지 시간이 필요하면 `boundary_cost_s + stage_reposition_s`.

### 설정 판 번호

- `version: 1` 그대로 — `start_pose`는 선택 필드이고 기본값 `null`이 이전 동작이라 옛 YAML의 뜻이 바뀌지 않는다(`config.py` `LATEST_CONFIG_VERSION` 주석). 거꾸로 `start_pose`가 든 YAML을 옛 실행기에 주면 draccus가 모르는 필드로 파싱 단계에서 죽는다.
- `start_pose`가 없는 단계는 `config.resolved.yaml`에도 그 키를 쓰지 않는다 — 실로봇 YAML(`chain_task45.yaml` 등)의 스냅샷은 이 기능 전과 같은 바이트다. mock 로봇 YAML은 `arm_layout: flat` 한 줄이 더 찍힌다.

### 리셋 단계가 하는 일

구현 = `src/stage_runner/executors.py` `run_leader_reset_stage` · `src/stage_runner/leader_reset.py`.

1. **리더암 정렬** — 리더암을 위치 제어로 팔로워의 **측정 자세**(그리퍼 포함)로 옮긴 뒤 중력 보상으로 되돌린다(`BiWidowXAILeaderRobot.align_to`). 이동 시간 = `max(2초, 팔 관절 최대 이동량 / 0.5 rad/s)`.
	- 왜 — `record_loop`은 첫 틱에 리더 자세를 그대로 팔로워에 보낸다. 연결 직후엔 둘이 같은 대기 자세지만, 체인 중간에선 팔로워가 정책이 끝난 자리에 있어 첫 명령이 큰 점프가 된다. 9월 27일 크래시 5회 중 3회가 이 「정책 → 리더암」 전환 순간이었다(`Joint 3 velocity limit exceeded` — 점프가 원인이라는 건 미검증).
2. **베이스 토크 끔** — 속도 0 명령 뒤 `enable_motor_torque(False)`. 연결 때 토크를 켠 설정에서만 — 꺼 둔 설정은 손대지 않는다.
3. **리더암 조종** — `record_loop`에 리더암만 주고 데이터셋 없이(`record_adapter.call_reset_loop`). 리셋 중 보내는 **베이스 속도는 항상 0** — 리더암이 내는 `x.vel`·`theta.vel`은 베이스의 *측정* 속도라(`mobileai_leader.py`), 토크가 켜진 베이스에 되돌려 보내면 밀린 만큼 계속 움직이는 되먹임이 된다.
4. **베이스 토크 복구** — 속도 0 명령 뒤 `enable_motor_torque(True)`, 어떻게 끝났든(예외 포함). **복구가 실패하면 시행 중단**(다음 정책을 돌리지 않는다).
5. 끝난 이유 = 정책 단계와 같은 판정 — `→` `manual`, 상한 `timeout`(둘 다 다음 단계로), `ESC` `stop_recording`·`←` `rerecord_requested`(중단). 정렬 실패는 `error`(중단).

- 리셋의 **마지막 그리퍼 명령**이 다음 경계의 정지 명령·시작 자세 이동 동안 그대로 유지된다 — 쥐여 준 물체를 놓지 않게.
- 리더암은 리셋 단계가 하나라도 있을 때만 만들고 연결한다 — **팔로워보다 먼저** 연결해, 리더암이 꺼져 있으면 팔로워에 전원이 들어가기 전에 실패한다. 종료 땐 팔로워(베이스 정지) 다음.

### stage_reset 이벤트

리셋 단계마다 그 단계 `stage_start`·`stage_end` 사이에 한 줄.

| 필드 | 뜻 |
|---|---|
| `outcome` | 끝난 이유(`manual`·`timeout`·`stop_recording`·`rerecord_requested`·`error`), 정렬 중 `ESC`면 `aborted`, 루프 예외면 `interrupted` |
| `align_goal_time_s`·`align_max_arm_error_rad` | 리더암 정렬 이동 시간·정렬 후 최대 팔 관절 오차(기록만, 판정 안 함) |
| `align_target`·`leader_before` | 정렬 목표(팔로워 측정 자세)·정렬 전 리더 자세 |
| `base_torque` | `released`(껐다 켬)·`not_released`(안 건드림)·`restore_failed`(복구 실패 — 시행 중단) |
| `base_torque_error` | 복구 실패 사유, 정상이면 `null` |
| `sent_actions` | 리셋 중 보낸 명령 수 |
| `gripper_goals` | 마지막 그리퍼 명령 — 다음 시작 자세 이동, 또는 바로 다음 베이스 단계 동안 유지되는 값 |
| `loop_s` | 리더암 조종 시간 |

### 집계에서의 리셋 단계

- 성공률 축(`stage_ids`)·`per_stage`에서 **빠진다** — 라벨을 달 정책 결과가 없고, 축의 첫 단계 라벨로 채점 대상을 고르므로 리셋이 첫 단계면 모든 시행이 채점에서 빠지게 된다.
- 리셋 시간은 따로 `timing.leader_reset_s`(사람이 조종한 시간 — 로봇 시간 아님), 끝난 이유 분포 `terminators`.
- 옛 로그(`stage_start`에 `executor`가 없는 것)는 전부 정책 단계로 읽는다.

### 리셋 단계 preflight 거부

- `policy_path`·`start_pose`가 있음 / 최상위 `teleop:`이 없음 / 모든 단계가 리셋
- 리더암 키와 로봇 행동 키(베이스 제외)가 정확히 같지 않음, 리더암에 `align_to`가 없음

### 실기에서 아직 안 잰 것

mock 드라이버는 즉시·정확하게 도착하므로 아래는 첫 실기 회차의 `stage_reposition`·`stage_reset`에서 읽는다.

- 리더암 정렬 이동이 손에 위험하지 않은 속도인지(0.5 rad/s 평균은 실측 근거 없는 보수값), 정렬 후 중력 보상으로 넘어갈 때 리더가 처지는 정도(`align_max_arm_error_rad`)
- 베이스 토크를 껐다 켤 때 베이스가 튀지 않는지, `enable_motor_torque`가 실패를 제대로 돌려주는지
- 리셋 → 정책 전환에서 크래시가 나는지(반대 방향 전환은 9월 27일 기록이 있고, 이 방향은 기록 없음)
- `position_tolerance_rad` 0.05·확인 대기 1초가 실제 정착 오차·시간에 맞는지(`max_abs_error_rad`)
- 물체를 쥔 채 움직일 때의 정착 오차
- 쥔 물체가 이동 중에 미끄러지지 않는지(`gripper_goal_source: "previous_command"`일 때) — mock 그리퍼엔 쥐는 힘이 없어 시험에서 못 본다
- 실제 이동 시간(`reposition_s`)
- 직전 궤적 대기의 실제 길이(`expiry_wait_s`)와, 대기 뒤 이동 초반 휘두름이 사라졌는지 — 관절 궤적을 기록하는 로그는 아직 없다
- 여유 0.05초(`TRAJECTORY_EXPIRY_MARGIN_S`)가 드라이버의 궤적 시작 지연을 덮는지

### 위치 라벨 데이터셋

`base_replay`가 따라가는 목표 = 데모 에피소드의 **베이스 위치 라벨**(`action`의 `x.pos` 누적 전진 거리 m · `theta.pos` 누적 헤딩 rad). 취득 데이터셋은 속도 라벨(`x.vel`·`theta.vel`)이라 그대로는 거부된다.

- 만드는 법 = 10/24 현장 스크립트 `derive_pose.py` — 속도 라벨을 **취득 때 실측한 틱 간격**(취득의 `*_record.basevel.csv`)으로 적분해 `…_basepose` 사본을 로컬에 쓴다. 이 저장소엔 없다(로봇PC 작업지능 킷의 `_inbox/20261024-heading-align/`에서 편입된 사본, 경로 상수가 그 데이터셋 전용).
- 데이터셋의 타임스탬프(`fps` 30)로 적분하면 안 된다 — 실제 취득 루프는 그보다 느려, 단계 1에서 85.8°가 59.5°로 나온다.
- 재생 단계가 읽는 것 = `meta/info.json`·`data/**/*.parquet`(영상·통계 안 읽음). 지정 에피소드의 `frame_index`가 0부터 빠짐없이 이어져야 한다. 기준 화면 데이터셋은 `meta/`(`info.json`·`tasks.parquet`·`episodes/`)와 그 카메라 영상까지 — 하나라도 없으면 lerobot이 허브에서 받으려 하므로 실행기가 먼저 거부한다.
- `from_row`가 0보다 크면 목표를 `from_row - 1`행 기준으로 다시 잰다 — 데모 중간부터 재생해도 「지금 자리에서 그만큼 더」가 된다.

### 베이스 단계 설정 필드

`replay:`

| 필드 | 기본값 | 뜻 |
|---|---|---|
| `dataset` | (필수) | 위치 라벨 데이터셋 repo id |
| `root` | `null` | 로컬 경로. `null` = `~/.cache/huggingface/lerobot/<dataset>` |
| `episode` | `0` | 에피소드 번호 |
| `from_row`·`until_row` | `0`·(필수) | 재생할 행 범위(양끝 포함) |
| `row_period_s` | `1/21` | 목표 한 행의 벽시계 시간 — 10/24 재생 주기. 루프 주기와 무관(시간으로 보간). 줄이면 목표가 빨라져 속도 상한에 걸린다 |

`align:`

| 필드 | 기본값 | 뜻 |
|---|---|---|
| `reference_dataset`·`reference_root`·`episode`·`frame` | — | 기준 화면 = 그 데이터셋 에피소드의 프레임(`frame`은 필수) |
| `reference_image` | `null` | 또는 기준 화면 사진 파일(PNG·JPG). 둘 중 하나만 |
| `camera` | `cam_high` | 비교할 카메라 |
| `landmark_roi` | (필수) | 기준 화면 속 고정물 상자 `[x, y, w, h]` — 거리 추정이 이 상자의 크기 변화로 잰다 |
| `landmark_distance_m` | (필수) | 기준 화면에서 고정물까지 거리(m) |
| `yaw_tol_deg`·`dist_tol_cm` | `2.0`·`1.0` | 정렬 판정 허용 오차 |
| `max_moves` | `6` | 보정 이동 최대 횟수(측정은 최대 `max_moves + 1`번) |
| `coarse_heading` | `true` | 정밀 추정이 안 될 때 고정물 덩어리 위치로 대략 돌린 뒤 다시 잼 → [정렬 단계가 하는 일](#정렬-단계가-하는-일) |
| `camera_up` | 10/24 추정값 | 카메라 좌표의 로봇 세로축(제자리 회전으로 산출) |
| `camera_matrix`·`dist_coeffs` | `null` | 렌즈값. `null` = RealSense 장치에서 읽음. RealSense가 아닌 카메라면 둘 다 필수 |
| `turn_kp`·`min_turn_rate_rad_s` | `2.0`·`0.0` | 작은 회전이 덜 도는 것을 보완하는 손잡이 — 기본값이 10/24 동작 |
| `distance_width_fallback` | `false` | SIFT 크기 일치점이 8개 미만일 때 현재 화면 상자 폭 ÷ ROI 폭으로 거리. `landmark_roi`가 기준 화면의 랙 상자(`dark_blob_box`)일 때만 — 다른 ROI와 함께 켜면 connect 전에 종료 코드 `2` |

### 재생 단계가 하는 일

구현 = `src/stage_runner/base_stages.py` `replay` · 추종기 `lerobot_robot_trossen/base_pose_control.py`(PR 69 추종기 사본).

1. 목표·기준 화면은 **connect 전에** 전부 읽는다 — 단계 사이에 디스크를 읽지 않는다.
2. 0.5초 정지 프레임(기록함) → 그 마지막 오도메트리가 이 단계의 원점 — 재생마다 그 자리에서 다시 잰다.
3. 매 틱 = 관측 하나 · 명령 하나 · 프레임 하나. 팔·그리퍼는 잡아 둔다(그리퍼는 바로 앞이 리더암 리셋이면 그 리셋의 마지막 그리퍼 명령, 아니고 앞 단계 프레임이 있으면 마지막 기록 명령, 둘 다 없으면 측정 위치), 베이스는 추종기의 `x.vel`·`theta.vel`. 로봇은 속도 모드 그대로.
4. 목표 = 경과 시간 ÷ `row_period_s` 행을 선형 보간 — 루프가 21 Hz보다 느려도 데모 속도 그대로.
5. 마지막 행 뒤 정착 — 0.5초 정지면 `settled`, 4초 지나도 안 서면 `not_settled`(실패 아님, 정렬이 실제 자리를 잰다).
6. 끝나면 베이스에 직접 0 명령 → `base/<단계 id>_replay.csv` → `stage_base_replay` 한 줄.

### 정렬 단계가 하는 일

구현 = `base_stages.py` `align` · 추정 `landmark_align.py`(10/24 `heading_align.py` 이식).

1. 베이스가 0.3초 연속 정지하면 cam_high 프레임을 복사해 **작업 스레드**에서 잰다 — 그동안 본 루프는 베이스 0 명령으로 정지 프레임을 계속 기록. 3초 안에 안 서면 `not_still`.
2. 헤딩 먼저 — `|헤딩| > yaw_tol_deg`면 `-헤딩`만큼 회전(0.6 rad/s 램프 + 1초 유지).
3. 헤딩이 맞으면 거리 — 고정물 상자 크기 비율로 `|거리| > dist_tol_cm`면 직진(0.1 m/s 램프 + 1초 유지, 헤딩 유지).
4. 둘 다 맞으면 `aligned`. 헤딩은 맞는데 거리를 못 읽으면 `heading_only`(10/24 `RESULT=HEADING_ONLY`와 같음, 체인 계속).
5. **대략 헤딩**(`coarse_heading: true`) — 정밀 추정이 안 되고(맞는 점 15개 미만 또는 20° 초과) 고정물 덩어리가 보이면 그 위치로 대략 회전하고 다시 잰다. 대략 추정은 오차 안이어도 정렬로 끝내지 않는다. 대략 회전도 `max_moves`에 들어가고, 3번째 대략 회전이 필요하면 `estimate_unusable`. 랙 덩어리가 안 보이는 곳(비커 쪽·냉장고 앞·선반)에선 정밀 추정이 맞는 점 8개 이상·20° 이내면 그 값을 대략 헤딩으로 쓴다. 정밀·대략 둘 다 안 되면 그 자리에서 `estimate_unusable`. 측정마다 `heading_mode`(`fine`·`coarse`)가 기록된다.
6. 같은 종류 보정의 오차가 직전보다 커지면(헤딩 +0.5°·거리 +5 mm 초과) `diverging`, `max_moves`를 다 쓰면 `not_converged`. 직진은 헤딩 기준을 초기화하지 않는다(10/24 규칙).
7. 측정 실패 셋(`estimate_unusable`·`diverging`·`not_converged`)은 체인을 멈추지 않는다 — 그 자리에서 베이스를 세우고 다음 단계로 간다. 사유는 `stage_base_align`에 남는다(2026-10-25 실기 결정, 옆에서 보는 사람이 `ESC`로 끊을 수 있다). 안전 실패는 아래 표대로 시행 중단.
8. 끝나면 베이스에 직접 0 명령, 작업 스레드는 기다리지 않고 종료 → `base/<단계 id>_align.csv` → `stage_base_align` 한 줄.

### 베이스 단계 결과

`terminated_by` = `converged`(다음 단계로) · `timeout`(다음 단계로) · `manual`(다음 단계로) · `base_failed`(**시행 중단**, 종료 코드 `1`) · `stop_recording`·`rerecord_requested`(키 중단).

| `outcome` | 단계 | `terminated_by` | 뜻 |
|---|---|---|---|
| `settled` | 재생 | `converged` | 마지막 목표에서 정지 |
| `not_settled` | 재생 | `timeout` | 4초 안에 정지 판정 못 함 — 정렬이 이어서 잰다 |
| `aligned` | 정렬 | `converged` | 헤딩·거리 모두 허용 오차 안 |
| `heading_only` | 정렬 | `converged` | 헤딩만 맞음, 거리 추정 불가 |
| `manual` | 둘 다 | `manual` | `→` — 그 자리 인정 |
| `missed_target` | 재생 | `base_failed` | 정지 뒤 목표와 3° 또는 5 cm 넘게 어긋남 |
| `estimate_unusable` | 정렬 | `converged` | 헤딩을 못 읽음(위 5번) — 그 자리에서 다음 단계로 |
| `diverging`·`not_converged` | 정렬 | `converged` | 위 6번 — 그 자리에서 다음 단계로 |
| `move_failed` | 정렬 | `base_failed` | 보정 이동이 반대로 감, 또는 명령의 20% 미만만 움직임(비상정지·토크 꺼짐 의심) |
| `not_still` | 정렬 | `base_failed` | 측정 전 3초 안에 정지 안 함 |
| `intrinsics_unavailable` | 정렬 | `base_failed` | 렌즈값을 못 읽음·크기가 기준 화면과 다름(움직이기 전) |
| `no_odometry` | 둘 다 | `base_failed` | 오도메트리 갱신 실패가 10틱 넘게 연속 |
| `base_refused` | 둘 다 | `base_failed` | 베이스가 명령을 10번 넘게 연속 거부(10/24 현장 규칙) |
| `base_estop` | 둘 다 | `base_failed` | 베이스 상태 `SYS_ESTOP`·`SYS_ERR*` |
| `tracking_lost` | 둘 다 | `base_failed` | 목표와 10 cm 또는 각도 한계(재생 10°·정렬 이동 15°) 넘게 0.5초 이상 벌어짐(멈춘 베이스에 명령이 쌓였다 풀리며 튀는 것 방지). 10/24 실측 최대 = 재생 6.4°·정렬 회전 9.1° — 각도 한계를 이보다 좁히지 말 것 |
| `stage_ceiling` | 둘 다 | `base_failed` | `timeout_s` 도달 |
| `aborted` | 둘 다 | 키 | `ESC`·`←` |
| `interrupted` | 둘 다 | `error` | 예외 — 베이스 직접 0 뒤 예외 그대로 전파 |

- 문턱값은 `base_stages.py` 상수(YAML 아님) — 각 상수 옆 주석이 출처(현장값·추정)

### 베이스 단계 이벤트

단계마다 `stage_start`·`stage_end` 사이에 한 줄 — `stage_base_replay` 또는 `stage_base_align`. 정렬은 측정·이동마다 `stage_base_align_step`도.

| 이벤트 | 주요 필드 |
|---|---|
| `stage_base_replay` | `outcome`·`reason` · 데이터셋·행 범위(`dataset`·`episode`·`from_row`·`until_row`·`rows`) · `anchor_odom` · `final_target_*`·`final_*`·`final_error_m`·`final_error_deg` · `max_tracking_error_m`·`_deg` · 오도메트리·명령 실패 수 · `tick_mean_s`·`tick_p95_s` · `gripper_hold_source`(`reset`·`recorded`·`measured`) · `csv` |
| `stage_base_align` | `outcome`·`reason` · `reference`(출처·타임스탬프) · `intrinsics`(K·D·출처)·`field_intrinsics_delta`(10/24 상수와 차이) · `landmark_roi`·`auto_roi` · 허용 오차·`max_moves`·`coarse_heading` · `measurements`·`moves`·`move_count`·`coarse_turns` · `final_yaw_deg`·`final_move_cm` · `measure_s_total`·`tick_p95_s_while_measuring` · `gripper_hold_source`(`reset`·`recorded`·`measured`) · `csv` |
| `stage_base_align_step` (`kind: measure`) | `step` · `capture_frame_idx` · `still_wait_s` · `yaw_deg`·`inliers`·`heading_mode`·`heading_yaw_deg`·`coarse_yaw_deg` · `scale`·`scale_inliers`·`scale_source`·`move_m`·`distance_valid` · `measure_s` · `live_png`·`overlay_png` · `decision` |
| `stage_base_align_step` (`kind: move`) | `move`(`turn`·`drive`) · `commanded`·`achieved`·`ratio` · `ramp_s`·`hold_s`·`still_at_exit` · `max_tracking_error_*` · `frame_idx_start`·`frame_idx_end` |

남는 파일 — `outputs/stage_runner/<run_id>/base/`:

- `<단계 id>_replay.csv` · `<단계 id>_align.csv` — 기록 프레임마다 한 줄(목표·명령·측정 속도·오도메트리·갱신 성공 여부)
- `<단계 id>/reference.png` · `s<번호>_live.png` · `s<번호>_overlay.png` — 기준 화면, 측정마다 현재 화면·겹침

### 베이스 단계 preflight 거부

connect 전 종료 코드 `2`.

- 베이스 단계에 `policy_path`·`instruction`이 있음, `base_align`에 `start_pose`가 있음, `name`이 비었음, `terminator.type`이 `timeout`이 아님
- `base_replay`에 `replay:`가 없거나 `align:`이 있음(정렬은 반대), 정책·리셋 단계에 `replay:`·`align:`이 있음
- 베이스 단계 `name`이 다른 단계의 task 문자열과 같음
- 행 범위·허용 오차·ROI·거리·`camera_up`·렌즈값 형식이 틀림, `timeout_s`가 하한 미만
- 데이터셋이 로컬에 없음·속도 라벨임·에피소드·행·프레임이 없음, 기준 화면 크기가 카메라와 다름, ROI가 화면 밖
- 로봇에 베이스 속도 키·`robot.base`(`get_pose`·`get_vel`·`update_state`·`set_cmd_vel`)가 없음, `enable_base_motor_torque: false`
- 정렬 카메라가 로봇에 없음, RealSense가 아닌데 `camera_matrix`가 없음, 카메라 설정에 회전·RGB 아닌 색 모드

### 집계에서의 베이스 단계

- 성공률 축(`stage_ids`)·`per_stage`에서 **빠진다** — 리셋 단계와 같은 이유(라벨 대상이 아님, 체인 첫 단계가 재생이면 모든 시행이 채점에서 빠짐).
- 따로 `timing.base_replay_s`·`timing.base_align_s` — 시간·프레임·`hertz` 분포, `terminators`, 그리고 단계 이벤트의 `outcomes`(`aligned`와 `heading_only`를 따로 센다).
- `timing.base_stage_failed` — `base_failed`로 끝난 베이스 단계(시행 id·단계·`outcome`·`reason`). 그 시행은 중단이라 채점 대상에서 빠지고(`excluded_run_ids`) 여기가 이유를 남긴다.

### 실기에서 아직 안 잰 것 — 베이스 단계

mock 베이스는 명령을 정확히 따르고 mock 화면은 순수 회전이라 아래는 첫 실기 회차의 이벤트·CSV에서 읽는다.

- 실제 루프 주기(`tick_mean_s`·`tick_p95_s`)와 측정 중 주기(`tick_p95_s_while_measuring`) — 측정은 순수 파이썬 RANSAC이라 본 루프와 GIL을 나눈다
- 측정 시간(`measure_s`) — 사무실 약 0.5초, 로봇PC 미측정
- 새 중단 조건(`no_odometry`·`tracking_lost`·`base_estop`·`not_still`·`move_failed`·`missed_target`)의 문턱값이 정상 회차를 끊지 않는지 — 끊으면 `reason`의 값으로 상수를 조정
- 재생 중 비상정지 → `base_estop`, 해제 때 튀지 않는지
- 장치 렌즈값과 10/24 상수의 차이(`field_intrinsics_delta`), `auto_roi`가 `[285, 197, 93, 61]`과 같은지
- 3° 미만 작은 회전이 덜 도는 정도(10/24 50~70%) — 허용 오차 2°로 버티는지
- 정렬 뒤 pi0 단계 2 성공률이 10/24(헤딩·거리 보정 뒤 4/4, 헤딩만 2/3)와 비슷한지
