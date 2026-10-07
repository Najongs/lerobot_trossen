# `configs/chain/` — 11단계 자율 체인 러너 설정

여기 있는 것:

| 파일 | 무엇 | 누가 만드나 |
|---|---|---|
| `stage_params.json` | 단계별 **시작 자세 12관절 · 길이 분위 p10/p50/p90 · 끝 자세 후보** | sim 레포 `scripts/export_chain_params.py` 가 생성 → 여기로 복사 |
| `chain_m1_all11.yaml` | M1(16D, 진행도 없음) 체인 설정 — 참조군 | 손으로 |
| `chain_tph_all11.yaml` | 새 모델(17D, 진행도 칸) 체인 설정 — 본선 | 손으로 |

실행은 `scripts/eval_chain.sh <M1|TPH|<허브 repo id>> [회차태그]`.
로더·검증기는 `packages/stage_runner/src/stage_runner/chain_params.py`.

> **`stage_params.json` 은 이 레포에서 손으로 고치지 마라.** 시연 데이터에서 뽑는
> 값이고, sim 레포의 생성 스크립트가 정본이다. 손으로 고치면 `source` 의 커밋
> SHA 가 거짓이 되고, 어떤 시연에서 나온 자세인지 추적이 끊긴다.

---

## `stage_params.json` 스키마 (schema_version 1)

```json
{
  "schema_version": 1,
  "fps": 21,
  "source": {
    "sim_commit": "<trossen-ai-simulation 커밋 SHA>",
    "lerobot_trossen_commit": "<이 레포 커밋 SHA>",
    "generated_at": "2026-10-07T11:22:33+09:00"
  },
  "arm_joint_order": [
    "left_joint_0", "left_joint_1", "left_joint_2",
    "left_joint_3", "left_joint_4", "left_joint_5",
    "right_joint_0", "right_joint_1", "right_joint_2",
    "right_joint_3", "right_joint_4", "right_joint_5"
  ],
  "stages": {
    "1": {
      "name": "task01",
      "instruction": "<record 의 single_task 로 쓰일 문장>",
      "p10_s": 11.9,
      "p50_s": 17.4,
      "p90_s": 24.8,
      "start_pose_rad_arm12": [12개 실수, arm_joint_order 순서],
      "start_pose_source": "reset_poses_candidate.json#task01 (봉우리1, 이웃 7@0.3)",
      "end_poses_rad_arm12": [[12개], [12개], ...],
      "end_pose_tol_rad": 0.15
    },
    "2": { ... },
    "...": "11 까지 전부. 빠지면 로더가 거부한다"
  }
}
```

### 필드별로 무엇이고 누가 읽나

| 필드 | 단위 | 읽는 쪽 | 없거나 틀리면 |
|---|---|---|---|
| `schema_version` | — | 로더 | 1 이 아니면 **즉시 거부**. 다음 세대가 같은 키에 다른 뜻을 줄 수 있고, 이 파일의 값은 팔 위치가 된다 |
| `fps` | Hz | 로더 (`dataset.fps` 와 대조) | 다르면 거부. 분위는 초라서 fps 변화에 견디지만 **리셋 램프의 틱 수와 감시자 창 길이는 루프 주기에서 나온다** |
| `source` | — | 보고서 | 자유 형식. 생성 스크립트가 두 레포 SHA 를 적는다 — 이 자세가 어느 데이터에서 나왔는지 유일한 단서 |
| `arm_joint_order` | — | 로더 | 12개 **이름** 집합이 맞아야 한다. **순서는 자유** — 모든 자세를 이 순서에 zip 해서 이름→라디안 dict 로 만든다. 인덱스 가정 금지 (16D 실데이터는 `[left7, right7, x, θ]`, sim 텔레옵은 베이스가 앞 — 섞으면 차원만 맞고 조용히 틀린다) |
| `p10_s` | 초 | 완료 감시자 (하한) | 경과 < p10 이면 완료 신호를 **무시**한다. 시연 중 23~35% 가 작업 중 2초 이상 멈추므로(§93) 정지만으로는 완료가 아니다 |
| `p50_s` | 초 | 보고서 표 | 판정에 쓰지 않는다 |
| `p90_s` | 초 | 타임아웃 (`× timeout_factor`) | 도달 = **체인 실패**, 재시도 없음 |
| `start_pose_rad_arm12` | rad | 리셋 실행기 (목표) · 완료 감시자 (**출발 래치의 기준점**) · `pose_guide` | 12개 **유한** 실수. NaN 은 조용히 통과한다 — 팔 pacing 이 `>` 비교라서 NaN 과의 비교는 전부 False 가 되고 클램프가 발동하지 않는다 |
| `start_pose_source` | — | 보고서 | 「어느 봉우리인가」를 사람이 읽는 곳 |
| `end_poses_rad_arm12` | rad | 완료 감시자 (**16D 모델만**) | 비어 있으면 거부. 16D 모델은 「알려진 끝 자세 근처에서 멈췄나」가 유일한 완료 신호다 — 단, 그것만으로는 **안 된다**: 아래 「출발 래치」 |
| `end_pose_tol_rad` | rad | 위 NN 판정의 문턱 (∞-노름, 관절별 최대 오차) | 양수 유한 |

### 출발 래치 — 끝 자세 NN 만으로는 안 되는 이유

이 파일에서 **t03·t04·t10 의 지정 시작 자세는 자기 끝 자세 20개 중 하나와
0.0008 rad 이내**다(실측: 0.00077 · 0.00077 · 0.00038). t01·t05·t07 도 자기
`end_pose_tol_rad` 안에 들어온다(0.121/0.15 · 0.187/0.374 · 0.285/0.32).
**왜** 그렇게 작은지는 [추정] 이다 — 팔을 거의 안 움직이고 돌거나 가는 단계면
시작≡끝이 되는 것이 자연스럽지만, 그 해석은 작업 **이름**에서 읽은 것이고
t04 는 `task04_pour_liquid_from_...` 로 이동 단계가 아니다. 판정에 쓰는 것은
해석이 아니라 실측 거리다. 그래서
「정지 ∧ 끝 자세 NN ∧ 경과 ≥ p10」 은 **시작 자세에 가만히 서 있는 팔에서도
전부 참**이다 — M1 이 첫 청크에서 정지를 예측하는 비율이 홀드아웃 20~40%(P2)이고,
그러면 11단계가 아무것도 안 하고 「완주」로 기록된다.

그래서 완료에는 전제가 하나 더 있다. 단계가 **시작 장면을 떠났어야** 한다
(`completion.py`, 래치는 한 번 걸리면 안 풀린다). 16D 의 NN 경로와 17D 의
진행도 경로 **둘 다** 이 전제를 쓴다 — 시작 장면에서 p=1.0 을 읽는 머리는
센서만 다른 같은 실패다.

| 조건 | 기본값 | 왜 그 값 |
|---|---|---|
| 팔: **그 단계의 첫 실측 자세**에서 ∞-노름 이탈, **연속 3틱** | `departure_arm_rad` 0.10 rad | 리셋 도달 허용치(`tol_rad` 0.05)의 **2배**. 「겨우 도달한 리셋」이 래치를 걸 수 없다. 연속 틱 수는 `completion.DEPARTURE_ARM_TICKS` (상수, YAML 노브 아님) |
| 베이스 회전: 명령 누적 `|Σ θ·dt|` | `departure_base_rot_rad` 0.17 rad (≈10°) | 이름상 t01·t03·t10 은 팔을 거의 안 움직이고 돌거나 간다 [추정] — 팔 조건만 두면 그 단계는 영원히 래치가 안 걸린다. **|합|** 이라서 제자리에서 흔들리는 베이스는 출발이 아니다 |
| 베이스 전진: 명령 누적 `|Σ x·dt|` | `departure_base_fwd_m` 0.10 m | 위와 같은 이유 |

래치가 안 걸린 채 타임아웃이면 종료 사유가 `never_departed` 다. 그 단계는
**어려워서 실패한 것이 아니라 시작하지 않았다** — 측정된 것이 없으므로 볼 곳은
정책이 아니라 시작 장면(바로 앞 리셋의 도달 오차 · 원핫 인덱스 · 카메라)이다.
보고서(`scripts/eval_chain_report.py`)의 정책 표 `출발 s` 열에 「언제 떠났나」가,
떠나지 않았으면 `✗` 가 찍힌다. **`?` 는 「안 떠났음」이 아니라 「모름」**이다 —
감시자가 눈이 멀어(`progress` 칸 없음 · `.pos` 키 없음 · 트랜지션 파싱 실패)
팔을 못 읽은 경우로, 그때 `0.000 rad` 는 측정값이 아니다. 그 단계는
「출발하지 않은 단계」 집계에서도 빠지고, 종료 사유도 `never_departed` 가 아니라
「감시자가 BLIND 였다」 로 적힌다. 볼 곳이 시작 장면이 아니라 감시자의 입력이다.
(`progress` 칸만 없는 17D 런은 눈이 멀어도 팔·베이스는 읽히므로 출발은
**측정된다** — 감시만 못 한다.)

#### 10/06 교차검토로 바뀐 세 가지 (래치 쪽)

- **기준점은 「지정 시작 자세」가 아니라 「그 단계의 첫 실측 팔 자세」다.** 둘이 같은 것은
  바로 앞 리셋이 팔을 거기로 끌고 간 보통의 체인에서만이다. `chain.reset.initial: false` +
  `from_stage > 1`(중간부터 이어 돌리기)에서는 팔을 지정 자세로 끌고 간 것이 **아무것도
  없고**, 그러면 지정 자세 기준 이탈은 첫 틱에 **이미 참**(= 래치가 꺼진 것과 같다)이거나
  **영원히 거짓**(팔이 지정 자세 쪽으로 움직이는 것이 「집에 가까워지는」 것으로 읽힌다)이
  된다. 「이 단계가 움직였나」를 뜻하는 읽기는 **출발한 자리에서 재는 것** 하나뿐이다.
  지정 자세와 첫 실측이 0.3 rad(`completion.START_POSE_WARN_RAD`) 이상 떨어져 있으면
  **WARNING 한 줄**이 나오고 판정은 바뀌지 않는다. 값은
  `stage_end.reason_detail.start_pose_offset_rad` 에 남는다.
- **팔 조건은 연속 3틱**이다. 래치는 한 번 걸리면 안 풀리고 모든 완료의 전제라서, 한 틱짜리
  인코더 글리치가 「가만히 서 있는 팔」에게 나머지 단계를 통째로 넘겨 버렸다 — 래치가 막으려는
  그 실패 그대로다. 21 Hz 에서 3틱 = 143 ms 이고, 실제 출발이 커버하는 0.25~1.28 rad 보다
  훨씬 작다. **베이스 조건은 연속 틱을 요구하지 않는다**: 적분이라 한 틱이 1틱치 라디안만
  더하고 혼자서는 0.17 rad 을 못 넘는다 — 적분이 그 자체로 필터다.
- **NaN·inf 가 섞인 샘플은 「증거 없음」**이고, 양방향으로 그렇다. `max(a, nan)` 은 `a` 를,
  `nan >= x`·`nan < x` 는 둘 다 False 를 내므로, 신호가 끊긴 것이 전에는 **「완벽하게 멈춘
  팔이 끝 자세에서 0.00 rad」**(= 완료 조건 그 자체)로 읽혔다. 이제 NaN 이 하나라도 든 창은
  정지 판정·진행도 판정 모두 거부되고, 그 틱은 출발 적분·기준점 설정에서도 빠지고,
  **연속 3틱**(`completion.NON_FINITE_BLIND_TICKS`)이면 감시자가 BLIND 가 된다.

### 그리퍼는 왜 없나

`arm_joint_order` 는 **12관절뿐**이고 그리퍼(`left_left_carriage_joint` ·
`right_left_carriage_joint`, 각 팔의 7번째 값)는 들어가지 않는다.
경계 리셋은 그리퍼를 **관측값 그대로 유지**한다 — 물체를 들고 있을 수 있다.
베이스도 리셋하지 않는다(`x.vel = theta.vel = 0.0`).

> ⚠️ 유지란 「매 틱 관측된 위치를 다시 명령」이다. 물체를 쥔 상태에서 이것이 쥐는
> 힘을 유지하는지는 **실기에서 확인해야 한다** (bring-up ③ 전 관문).

### 시작 자세는 평균이 아니다

t02·03·04·05·10 은 시연 시작 자세가 **쌍봉**이라 관절별 평균·중앙값이
시연에 없는 자세다(오프라인 §92.1). 지정 규칙(계획 §설계결정 2):

1. 실존·밀집 (이웃 ≥5 @0.3 rad)
2. 모델이 출발하는 봉우리 (출발 지도)
3. 전이 최소

`pose_guide.py` 의 `TARGETS_DEG` 는 **중앙값 표**다 — 체인 러너는
`pose_guide.set_stage(stage, target_deg=...)` 로 이 파일의 지정 자세를 넘겨
리셋 중 1 Hz `POSE` 줄이 **실제 목표까지의 거리**를 보여 주게 한다.

---

## 거부되는 경우 (전부 로봇이 움직이기 전)

로더는 **문제를 전부 모아서 한 번에** 던진다 (오타 하나당 한 번 실패하지 않게):

- `schema_version != 1` → 즉시
- `fps` 가 양의 정수가 아님 / `dataset.fps` 와 불일치
- `arm_joint_order` 가 12개가 아니거나 이름 집합이 다름
- `stages` 에 `"1"`~`"11"` 중 하나라도 없음 / 그 밖의 키가 있음 (키는 **문자열**)
- 필수 키 누락 (`name` `instruction` `p10_s` `p50_s` `p90_s`
  `start_pose_rad_arm12` `end_poses_rad_arm12` `end_pose_tol_rad`)
- `p10_s <= p50_s <= p90_s` 위반, 또는 `p10_s == p90_s` (완료 하한과 타임아웃이
  같아지면 완료가 성립할 창이 없다)
- 자세 길이가 12 가 아님 / 값이 유한하지 않음 / `true`·`false` (JSON 의 bool —
  `float(True)` 는 1.0 이라 1 라디안 명령이 된다)
- `end_poses_rad_arm12` 가 빈 리스트
- `end_pose_tol_rad <= 0`

### 체인 YAML 쪽 거부 (`preflight.check_chain_definitions`)

**임계값은 전부 범위 검사한다** (`preflight.threshold_problems`, 10/06). 네 개만 보던
옛 판은 나머지를 손 편집 그대로 통과시켰고, 그 실패는 둘 다 **조용했다**: 0 이하면 영원히
성립하지 않는 조건(`stall_s: 0` → 모든 단계가 타임아웃 → 끝낸 11단계가 「실패」로 기록)이
되거나, 영원히 성립하는 조건(`departure_arm_rad: 0` → 래치가 첫 틱에 걸린다 = **래치가 꺼진다**)
이 된다. 검사는 `not (x > 0)` 꼴로 쓴다 — `x <= 0` 은 **NaN 에 False** 라서 NaN 이 그대로
관절 임계값이 된다.

- `completion`: `p_done` 이 (0, 1] 밖 · `p_hold_s`·`stall_s`·`stall_arm_rad`·`stall_track_rad`·
  `stall_base`·`departure_arm_rad`·`departure_base_rot_rad`·`departure_base_fwd_m` 이 0 이하
  (또는 NaN) · `timeout_factor <= 1.0`
- **`stall_track_rad < stall_arm_rad`** — 10/06 에 쪼갠 두 노브의 **순서가 뒤집힌 것**이다.
  추종 허용치가 **더 느슨한** 쪽이어야 한다(짐을 들고 중력에 버티는 팔은 0.06 rad 뒤처져
  있어도 멈춘 팔이다). 뒤집히면 정지한 팔이 「아직 움직인다」로 읽혀 단계가 타임아웃까지 간다 —
  쪼개기로 고친 바로 그 회귀다
- `reset`: `t_min_s`·`v_des_rad_s`·`max_jump_rad`·`initial_max_jump_rad`·`tol_rad`·`settle_s`
  가 0 이하 (또는 NaN) · `ceiling_factor <= 1.0` (램프가 틱 기반이라 상한이 램프보다 짧아진다)
- **`initial_max_jump_rad > max_jump_rad`** — 최초 램프는 검증된 분포가 없는
  자세에서 출발하므로 둘 중 **더 좁은** 쪽이어야 한다
- **`chain.model.has_progress` 가 null** (10/06). 체크포인트의 action 폭이 사실이고 키와
  **둘 다** 맞아야 통과한다. 폭에서 조용히 유도하던 옛 판은 **녹화 데이터셋이 17번째 action
  칸을 선언하는지**, 따라서 정규화기가 어느 폭으로 로드되는지를 정하는 단 하나의 사실을
  설정 파일에 안 남겼다 — 어느 모델로 돌린 회차인지 YAML 로 알 수 없어진다.
  `scripts/_chain_preflight.py` 도 같은 것을 (더 앞에서) 막는다
- **체크포인트 action 폭 ≠ 녹화 데이터셋 action 칸 수** (`policies.check_action_width`).
  두 사실은 독립이다 — 앞은 학습된 가중치, 뒤는 `has_progress` + 로봇의 action features.
  어긋나면 정규화기가 **틀린 폭으로** 로드되고 관절 명령이 「말이 안 되지는 않게」 틀린다
- 체크포인트 `config.json` 에 **`temporal_ensemble_coeff` 가 있는데 실행
  `n_action_steps > 1`** — ACT 가 자기 `__post_init__` 에서 막는 조합인데, 체인은
  `from_pretrained` **뒤에** `n_action_steps` 를 대입해서(= `--policy.n_action_steps=`
  와 같은 방식) 그 검사를 **우회한다**. 실측 비용: 루프 20% 느려지고 과회전
  1.29배(aa30006). `scripts/_chain_preflight.py` 도 같은 것을 막지만 그쪽은
  `eval_chain.sh` 경유 런만 본다 — `python -m stage_runner` 직접 실행에는 게이트가
  없었다

### 리셋 램프 거부 (움직이기 전, `plan_reset`)

| 경우 | 문턱 | 왜 |
|---|---|---|
| 경계 리셋의 최대 관절 간격 | `max_jump_rad` 1.5 rad | 측정된 10 경계가 0.25~1.28 rad. 넘으면 팔이 이전 단계가 남길 자리에 없다 |
| **최초 리셋**의 최대 관절 간격 | `initial_max_jump_rad` **0.6 rad** | 출발점이 **사람이 둔 자리**라 분포가 없다. 거부 메시지가 「task01 시작 자세 근처(0.3 rad 이내)에 손으로 두고 시작하라」 고 말한다 — 문턱을 올리는 것이 아니라 팔을 옮기는 것이 답이다 |
| 관절 누락 · 유한하지 않은 값 | — | NaN 은 pacing 의 `>` 비교를 전부 False 로 만들어 클램프 없이 통과한다 |

**도달 판정은 두 조건이다** (10/06): 마지막 `settle_s` 창의 **90% 이상이 tol 안** ∧
**가장 최근 샘플이 tol 안**. 비율만 보면 「19/21 인데 못 들어온 둘이 **끝에** 있다」 =
자리 잡았다가 **지금 벗어나는 중**인 팔을 도달로 읽는다. `reached` 는 팔을 다음 단계의 정책에
넘기는 신호이고, 비율이 버리는 「순서」가 바로 「자리 잡았다」와 「떠나는 중이다」를 가르는
부분이다. 창 중간의 글리치 하나는 전과 같은 틱에 도달하고, 최신 샘플이 나쁜 경우에만
**한 틱** 늦어진다.

리셋이 `not_reached` 로 끝나는 경우는 **세 가지**이고 보고서의 「사유」 열이
구분한다: `거부(움직이기 전)` · `상한 초과` · `사람이 끊음(→)`.
`→` 는 정책 단계 전용이다 — 리셋 중에 누르면 램프가 끊기고, 팔이 다음 단계 시작
자세에 없으므로 그것도 체인 실패다.

---

## 텔레옵 구간 키 (StageRunnerConfig 최상위, 10/07)

| 키 | 기본 | 뜻 |
|---|---|---|
| `teleop` | 없음 | lerobot `TeleoperatorConfig`(draccus choice, `--teleop.type=mobileai_leader_teleop …`). 주면 첫 단계 전에 리더암 텔레옵 구간을 한 번 돈다. 없으면 옛 동작 |
| `teleop_time_s` | 300 | 그 구간 한 번의 상한(초). `→` 가 먼저 오면 거기서 끝. 넘기면 **중단**(첫 단계가 시작 전 중단) |
| `teleop_base_from_leader` | false | 리더 action 의 x.vel/theta.vel 을 베이스에 그대로 보낼지. 기본은 0 으로 덮는다(되먹임 구조, 리뷰 10/07) |

두 체인 YAML 은 키를 두지 않는다 — `scripts/eval_chain.sh` 가 CLI 로 얹는다(`TELEOP=0` 으로 끔). 테스트는 `stage_runner_mock_teleop`(`mock_teleop.py`, `x_vel`/`theta_vel` 로 0 아닌 리더 베이스 속도를 흉내낼 수 있다).

`teleop_time_s` 는 양수·유한이어야 한다 — 아니면 preflight 가 exit 2 로 거부한다(0 이면 창이 한 틱도 안 돌고 `timeout` 중단이 되던 것, 리뷰 10/07).
events 의 `teleop_end.ended_by` 어휘: `arrow`(→ 로 정상 종료) · `left_arrow_restart` · `esc` · `timeout`(**중단**) · `esc_before_start` · `left_arrow_before_start` · `nan_gate`(창 안에서 NaN/Inf 게이트가 끊음, exit 4) · `signal`(SIGTERM/SIGHUP) · `exception`. 창 안에서 예외·시그널로 풀릴 때는 단계 경계가 없으므로 비상 stop_base 결과가 `trial_end.emergency_stop_base_{path,is_stopped,error}` 에 남는다.

## 테스트용 가짜 파일

`packages/stage_runner/tests/data/stage_params_mock.json` 은 **합성 파일**이다.
스키마는 같지만 자세는 시연에서 나온 것이 아니고, p10/p90 이 0.3/0.8 초로 짧다
(하드웨어 없는 체인 테스트가 몇 초 안에 끝나도록). **실기에 쓰지 마라.**
`source.note` 에 그렇게 적혀 있다.
