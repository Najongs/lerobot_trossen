# Trossen-LeRobot Stage Chaining Runner

실행 명령·종료 코드·함정의 정본은 루트 README — [3-4. 단계 체이닝 실행](../../README.md#3-4-단계-체이닝-실행-stage_runner) · [단계 체이닝 실행](../../README.md#단계-체이닝-실행). 이 문서는 **단계별 팔 시작 자세(`start_pose`)** 와 **리더암 리셋 단계(`leader_reset`)** 만 다룬다.

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
| `gripper_goals` | 마지막 그리퍼 명령 — 다음 시작 자세 이동 동안 유지되는 값 |
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
