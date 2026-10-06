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
| `start_pose_rad_arm12` | rad | 리셋 실행기 (목표) · `pose_guide` | 12개 **유한** 실수. NaN 은 조용히 통과한다 — 팔 pacing 이 `>` 비교라서 NaN 과의 비교는 전부 False 가 되고 클램프가 발동하지 않는다 |
| `start_pose_source` | — | 보고서 | 「어느 봉우리인가」를 사람이 읽는 곳 |
| `end_poses_rad_arm12` | rad | 완료 감시자 (**16D 모델만**) | 비어 있으면 거부. 16D 모델은 「알려진 끝 자세 근처에서 멈췄나」가 유일한 완료 신호다 |
| `end_pose_tol_rad` | rad | 위 NN 판정의 문턱 (∞-노름, 관절별 최대 오차) | 양수 유한 |

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

---

## 테스트용 가짜 파일

`packages/stage_runner/tests/data/stage_params_mock.json` 은 **합성 파일**이다.
스키마는 같지만 자세는 시연에서 나온 것이 아니고, p10/p90 이 0.3/0.8 초로 짧다
(하드웨어 없는 체인 테스트가 몇 초 안에 끝나도록). **실기에 쓰지 마라.**
`source.note` 에 그렇게 적혀 있다.
