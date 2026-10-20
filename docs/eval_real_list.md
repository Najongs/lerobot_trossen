# 실기 Eval 전체 목록 — 학습한 것은 전부 로봇에서 돌린다 (2026-10-10, 표 하나)

> **사용자(10/10)**: 「오프라인 평가를 너무 신뢰하지 말고 Eval 실제로 다 돌려보게 리스트에 표로 정리해둬」. 오프라인 열은 **순서를 정하는 참고**일 뿐이고 판정은 이 표의 실기 결과다. 회차 상세·명령 정본은 [`eval_queue_1008.md`](eval_queue_1008.md)(O·A·R·S·V·N·C·E 번호)와 [`model_selection_1009.md`](model_selection_1009.md) §3, 누적 성공률은 [`eval_scoreboard.md`](eval_scoreboard.md). 학습 쪽 종합 표는 sim `docs/loop_board.md`.
> 서버: 1호기 = DGX1 · 2호기 = sandia · 실기 = Trossen PC1(사람이 실행, Claude 는 명령·기록).

## ▶ 10/20 — 단계 전문가 먼저 (이 절부터 본다, DGX1 → Trossen PC1, 2026-10-19 22:25 KST)
> **사용자(10/19)**: 「Task 1, 2, 3 개별 모델 하나씩 제작하자 … 성능체크 해보자」 → 「2호기도 써서 만들어둬. **내일 Eval 진행 이거 먼저 진행하게끔 집중**해두고」. 그래서 10/20 은 **t01·t02·t03 단계 전문가(#76~#79) 비교가 먼저**다. 아래 10/19 절(2c 자리 보정 시험 등)은 이 절이 끝난 뒤에 이어 간다.
> 모델 4개 — 전부 #51(J47) 레시피(청크 100 · GIST 30% · 원핫 없음 16D), **우리 데이터 전 에피소드(홀드아웃 없음) · 120K 학습 중 60K 체크포인트**(사용자 10/19 23시 「우리 데이터셋 다 써서 … 홀드아웃없이」 「학습량도」). 120K 는 10/20 ≈22~23 시에 `…_120k_s1000` 으로 올라온다 — 오늘 60K 결과가 좋은 모델은 다음 날 120K 와 짝으로 돈다. 상세는 표 맨 끝 #76~#79 행. (22시대에 적은 #72~#75 는 홀드아웃을 남긴 판이라 학습을 멈추고 이것으로 바꿨다.)
>
> | # | 태스크 | 학습 | 60K 허브에 올라오는 때 [추정] |
> |---|---|---|---|
> | #79 | t03 — 우리 117 ep 전부 + GIST task3 | 1호기 J74 | **public 10/20 08:59** (허브 85aa9bb9) |
> | #77 | t02 — 우리 133 ep 전부(투명 33) + GIST task2 | 1호기 J72 | 10/20 ≈10:50 [추정, 09:44 실측 54K · 0.705 s/step] |
> | #76 | t01 — 우리 100 ep 전부 + GIST task1 | 2호기 J71 | **public 10/20 09:53** (허브 f06fe51b) |
> | #78 | t02 상대 EEF 변형 | 2호기 J73 | **public 10/20 09:21** (허브 068842c7) |
>
> **정정(10/20 09:50 KST)**: #76·#78·#79 는 이미 public, #77 만 ≈10:50. 그래서 순서는 **§0 → §3 t01 → §2 t03(튜브는 손에 쥐여 10/16 §2 3행 자리) → §1 t02** 로 돈다. 120K 끝 [추정]: #79 ≈18:40 · #78 ≈20:30 · #76 ≈20:50 · #77 ≈22:35 KST.
>
> **넷 다 10:30~10:45 쯤 나온다 — 오전은 §0(진단 2 회)과 같은 날 재생 기준 회차(t01 재생 ep30 · t02 재생 ep102)를 먼저 하고 기다린다.** 먼저 나온 것부터 돈다(태스크 순서 §1 → §2 → §3 은 바꿔도 된다). **허브에 60K 가 없으면 그 행은 건너뛰고 다음 태스크로 간다**(다른 체크포인트로 대신 돌리지 않는다). 확인: `uv run python -c "from huggingface_hub import list_repo_files as l; print(l('<repo>'))"` 에 `model.safetensors`·`multi_manifest.json`.

### 0. 시작 (≈15 분)
1. `git pull origin main`.
2. **출발 이상 진단 2 회** — 아래 10/19 절 0′-2 그대로(`BASE_TORQUE=0`, 재생 ep30 자리, 각 20 s). 이 둘이 10/19 의 「첫 1~2 청크가 기록 영상으로 재현 안 됨」 을 가린다. 결과를 기다리지 말고 1 로 넘어간다. DGX 가 오늘 전문가 회차도 첫 청크를 다시 예측해 그 회차가 판정에 쓸 수 있는지 표시한다.
3. 전문가 4개는 전부 16D · 원핫 없음이다 — **로그에 `zero-pad only` 가 없으면 그 회차를 중단**한다. #78 는 추가로 `[rel_eef] 상대 EEF 체크포인트 -- … (state_free=False …)` 줄.

### 1. t02 튜브 집기 — 2a 자리(시연 튜브·시연 구멍) · 9 회 (+ 조건부 4)
- 자리: 재생 ep102 로 A✓ 먼저(10/16 §2 의 2a 재생 명령 그대로).
- 순서: **(#77 → #51 → #2) × 3** — 같은 자리·같은 구멍에서 번갈아.
  - #77 `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_task02_c100_gist30_soloall_16D_60k_s1000 2 60 1`
  - #51 `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_task02_c100_gist30_solo_16D_60k_s1000 2 60 1`
  - #2 `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 LEROBOT_CHUNK_SMOOTH_TICKS=3 HOT=2/11 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 2 60 1`
- 조건부: #78 가 허브에 있으면 **#63 1 회(상대 EEF 어댑터 첫 실기 — 10/19 절 §0-3 줄을 다 본다)** → 정상이면 **#78 × 3**(`…task02_c100_gist30_soloall_reef_16D_60k_s1000`, 명령은 #77 과 같고 repo 만). #63 에서 로그 줄이 하나라도 없거나 팔이 이상하면 상대 EEF 회차는 모두 멈춘다.
- 판정: 양손으로 쥐고 들면 ✓, 왼손만 +p. **#77 이 #51 보다 2 회 이상 많으면 우리 데이터 전부(106 → 133 ep, 투명 6 → 33) + 120K 학습 중 60K 가 레버**, #51 이 #2 보다 2 회 이상 많으면 「같은 레시피의 단계 전문가 > 11단계 모델」(O11 뒤집힘).

### 2. t03 제자리 회전 — **튜브를 양손에 쥔 채 출발** · 9 회
- 자리: t02 를 끝낸 자세 그대로(또는 10/16 §2 3행 자리). 비커 정면까지 ≈ −81°(21 Hz 환산).
- 순서: **(#79 → #2 → 팀 전문가) × 3**
  - #79 `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_task03_c100_gist30_soloall_16D_60k_s1000 3 60 1`
  - #2 `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 LEROBOT_CHUNK_SMOOTH_TICKS=3 HOT=3/11 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 3 60 1`
  - 팀 전문가 `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/task03_turn_to_face_beaker_14D 3 60 1`
- 기록: 실측 회전(°) · 회전 시작 시각 · 튜브를 놓쳤는지. 판정: 비커 정면에서 멈추면 ✓ (#2 의 10/13 성공 실측 −84°). 덜 돌면(−44~−67° 류) ✗ 로 적고 실측값을 남긴다.

### 3. t01 랙으로 이동 — 6 회 (+ 재생 1)
- 자리: 재생 ep30 자리. **먼저 같은 날 재생 1 회**로 도착 기준(멈춘 곳 실측 θ·x)을 잰다(10/19 재생 명령 그대로).
- 순서: **(#76 → #27) × 3**, 둘 다 회전 예산 켬(실측 기준 84°):
  - #76 `BASE_BUDGET_DEG=84 BASE_BUDGET_SRC=meas LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_task01_c100_gist30_soloall_16D_60k_s1000 1 60 1`
  - #27 `BASE_BUDGET_DEG=84 BASE_BUDGET_SRC=meas LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 LEROBOT_CHUNK_SMOOTH_TICKS=3 HOT=1/11 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_120k_s1000 1 60 1`
- 판정(10/19 새 기준): 재생 도착 대비 |Δθ| ≤ 8° · |Δx| ≤ 0.15 m · 충돌 없음. 회차 꼬리에 `_bb84m` 이 붙었는지 확인.

### 4. 기록
- 회차마다 맨 아래 「결과 기록」 에 한 줄 + `eval_najy_results_1020.md`. 회차 이름·모델 #·성공/실패·실패 꼴(출발 못함·그리퍼·덜 돎/더 돎·충돌)·루프 Hz.
- 끝나면 10/19 절 §2(2c 자리 보정 시험 등)로 이어 간다.

## ▶ 10/19 정비 — 이 절부터 본다 (DGX1 → Trossen PC1, 2026-10-19 KST)
> **정정(10/19 22:25 KST)**: 10/20 은 위 「10/20 — 단계 전문가 먼저」 절이 먼저다. 이 절은 그 뒤에 이어 간다.
> **사용자(10/19)**: 「실기 돌려볼 리스트 정리해서 Trossen PC에 보내줘.」 마지막 실기는 10/15 다(10/16 절은 아직 한 회차도 돌지 않았다). 그래서 **10/16 절의 순서표·명령을 그대로 쓰고**, 그 뒤에 나온 **상대 EEF 모델(#62~#69)** 을 이 절에서 끼워 넣는다. 옛 모델 명령은 10/16 §3 그대로다. 이 절의 시각은 전부 KST 다.
> 바뀐 것: ① **t02 에 2c 「자리 보정 시험」 신설(1순위)** — 랙을 3~5 cm 옮긴 자리에서 #63 ↔ #2 3 대 3. ② t02 2a(시연 구멍 그대로)의 반복에 #63 을 넣었다. ③ #69(state 제거 120K)는 2순위 — #62 를 대신한다. ④ 시드 짝(#66·#67)과 120K(#68)는 2c 결과에 따라 도는 조건부 회차. ⑤ #64·#65·#62 는 보류, #70·#71(120K public 10/19 21:40 KST)은 #69 의 시드 짝으로 보류.

### 0′. 10/19 오후 — 다음 세션 첫 순서 (DGX 답은 [`eval_najy_results_1019.md`](eval_najy_results_1019.md) 「DGX 답」)
1. `git pull origin main` — **ade0a78** 이 있어야 한다(`grep -c BASE_TORQUE scripts/eval_najy.sh` ≥ 1).
2. **출발 이상 진단 2 회 (먼저, 각 1 ep, 베이스 토크 끔 — 바퀴가 안 돌아 안전하다. 팔은 움직인다)**. 재생 ep30 자리에서:
   ```bash
   BASE_TORQUE=0 HOT=1/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_120k_s1000 1 60 1
   BASE_TORQUE=0 LEROBOT_FAST_OBS=0 HOT=1/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_120k_s1000 1 60 1
   ```
   (10/19 t01 #27 명령과 같은 스위치인지 그날 명령과 대조하고 다르면 그날 것을 따른다.) 각 회차 20 s 쯤 돌고 ESC. 둘째 회차 로그에 `GPU-side observation preprocessing is active` 줄이 **없어야** 한다(루프가 10 Hz 대로 떨어지는 것은 정상 — 바퀴가 안 돌아 과회전 위험 없음). 데이터셋이 허브에 올라가면 DGX 가 첫 두 청크를 다시 예측해 기록과 맞는지 본다. **결과를 기다리지 말고 3 으로 넘어간다.**
3. t02 2a(#63 첫 1 회) → **2c 자리 보정 시험**(아래 §2 그대로). 출발 이상이 t02 에도 있는지는 DGX 가 회차마다 재예측으로 표시한다 — 회차는 그대로 기록한다.
4. t01 은 진단 결과 뒤로 미룬다. **새 t01 성공 기준**: 같은 날 재생 도착(멈춘 곳 실측)과 |Δθ| ≤ 8° · |Δx| ≤ 0.15 m · 충돌 없음. 다음 t01 비교는 #27 e60 ↔ `BASE_BUDGET_DEG=84 BASE_BUDGET_SRC=meas`(회차 꼬리 `_bb84m`) 3 대 3. **t01 전문가는 접는다**(시연 범위 밖 과회전). 10/19 의 #27 e60 · bb105 · e30 비교는 출발 이상이 섞여 판정에 쓰지 않는다.

> **추가(10/19 21:50 KST, 사용자 「Task 1, 2, 3 개별 모델 하나씩 … 성능체크」)** — 단계 전문가 #72(t01) · #73(t02) · #74(t02 상대 EEF 변형) · #75(t03) 를 학습 중 — **23:10 정정: #72~#75 는 멈추고 우리 데이터 전부·120K 판 #76~#79 로 바꿨다(맨 위 10/20 절). 22:25 갱신: #73·#75 는 1호기(#75 ≈07:45 · #73 ≈10:15 [추정]), #72·#74 는 2호기로 옮김(1호기 CPU 병목)**, 순서는 맨 위 10/20 절. 아래 표 맨 끝 행. **실기는 진단 2 회로 10/19 출발 이상이 풀린 뒤** — 그 전 결과는 10/19 #27 회차처럼 판정에 못 쓴다. 허브에 없으면 건너뛴다.

### 0. 시작 전 — 10/16 §0 에 아래를 더한다
1. `git pull origin main` 뒤 **상대 EEF 어댑터 커밋 8f927a2 · 0a1c5d3** 이 있는지 본다:
   `ls packages/lerobot_robot_trossen/src/lerobot_robot_trossen/rel_eef_patch.py` (없으면 `eval_najy.sh` 가 상대 EEF 체크포인트를 스스로 거부한다 — 관절 목표로 변위가 나가는 사고를 막는다).
   어댑터 import 는 numpy·safetensors·huggingface_hub·lerobot 뿐이라 추가 설치는 없다. 단 테스트는 DGX 에서만 돌았다 — 로봇 PC 에서는 처음이다.
2. 처음 한 번 `DRY_RUN=1` 로 #63 명령을 돈다. stderr 에 `행동 rel_eef` 와 `상대 EEF 체크포인트 — 실행 로그에 '[rel_eef] 상대 EEF 체크포인트' 줄이 없으면 중단` 안내가 찍혀야 한다.
3. **로그 확인 줄 (상대 EEF 회차) — 없으면 그 회차를 중단한다**

   | 줄 | 확인할 것 |
   |---|---|
   | `[rel_eef] 상대 EEF 체크포인트 -- 청크를 관절로 바꿔 보낸다 (state_free=…, 앞 60 스텝 IK)` | **state_free 값이 모델과 맞는가** — #63·#66·#67·#68 = `False`, #62·#64·#65·#69 = `True`. `앞 60 스텝` 이 exec 와 같은가 |
   | `[rel_eef] 전처리 맨 앞에 측정 관절 복사 단계(observation.rel_eef_q)를 넣었다` | 있어야 한다 |
   | `installed` · `stage i/11 active` | 다른 27D ACT 와 같다 (state 제거 모델도 HOT 가 필수 — 없으면 eval_najy.sh 가 거부) |
   | `[rel_eef] 청크 N: IK 잔차 최대 … mm / … rad · 유지 누적 K` | 첫 청크와 20 청크마다. **유지 누적 K 를 회차마다 적는다** |
   | `[rel_eef] IK 실패/튐 -- 팔을 그 스텝부터 마지막 성공 자세로 유지: {…}` | 찍히면 그 청크에서 **그 팔·그리퍼가 청크 끝까지 멈춘다**(안전 쪽 동작). 시각·어느 팔인지 적는다. 오프라인에서는 우리 장면 홀드아웃 0/712 였다 |

4. **루프 속도**: 청크를 관절로 바꾸는 데 DGX 에서 71 ms 걸렸다(로봇 PC 미측정). 첫 회차 `.summary.txt` 의 루프 Hz 를 같은 단계의 #2 회차와 비교해 적는다 — 눈에 띄게 낮으면 그 자리에서 멈추고 알린다.
5. **짝 회차의 스위치를 같게**: 상대 EEF 회차도 #2 와 같이 `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10` · `LEROBOT_CHUNK_SMOOTH_TICKS=3` · exec 60 이다(블렌딩·평활은 관절로 바꾼 뒤에 붙는다 — `rel_eef_patch.py` 머리말). **큰 표 #62·#63·#66 행의 명령(블렌딩 없음)은 쓰지 말고 아래 §3 명령을 쓴다.**

### 1. 지금까지 실기 — 10/16 §1 과 같다 (10/16 이후 회차 없음). 상대 EEF 모델은 아직 실기 0 회.

### 2. 순서 — 10/16 §2 표 그대로, 아래 행만 바뀐다

| 순서 | 태스크 | 검증하려는 것 | 자리 | 1순위 | 2순위 | 결과 → 다음 |
|---|---|---|---|---|---|---|
| 2a | task02 (시연 튜브, 시연 구멍) | 10/16 2a 그대로 + **상대 EEF #63 이 시연 자리에서 잡나** (어댑터 첫 실기) | 재생 ep102 로 A✓ | **#63 1 회 먼저**(어댑터 확인 — §0-3·4 줄을 다 본다) → (#40 → 전문가 → #36 → #2 → #63 → #39 → #51) ×3 → #1 → #1 (24) | 10/16 그대로 (4) | 첫 #63 에서 로그 줄이 하나라도 없거나 팔이 이상하게 가면 **상대 EEF 회차 전부 멈추고 알린다**. 그 밖은 10/16 판정 그대로 |
| **2c** | **task02 자리 보정 시험 (새)** | **상대 EEF 행동이 잡는 자리를 따라가나** — 오프라인: 홀드아웃 ep 마다 다른 튜브 자리를 20틱 전 R² .92 로 따라감(#2 .71·.77) | 재생 ep102 로 A✓ 를 맞춘 뒤 **랙을 로봇 기준 좌 또는 우로 3~5 cm 옮긴다**(시연 잡는 자리 퍼짐 4.5~4.9 cm 안). 바닥에 표시하고 **방향·거리를 적는다**. 6 회 내내 그 자리, **#2 도 같은 자리에서** | (#63 → #2) ×3 (6) | 같은 자리에서 #69 ×3 → (조건부) #66 → #67 → #68 → #68 (7) | 양손 쥠 ✓ · 한 손 +p. **#63 이 #2 보다 2 회 이상 많으면** 「상대 EEF 가 자리 어긋남을 보정한다」를 실기로 확정 → 다음 레시피 기본값. 이때 #66·#67(시드 짝)을 1 회씩 돌아 시드 운인지 보고, #68(120K)을 2 회 돌아 학습량 효과를 본다. **같거나 적으면** 2a 결과와 함께 보고(조건부 회차는 건너뛴다). #69 가 #63 과 2 회 이상 다르면 state 효과 |
| 3 | task03 | 10/16 그대로 + 상대 EEF 가 회전 중 팔을 흘리나 | 10/16 그대로 | 10/16 그대로 (9) | 10/16 끝에 #63 (1) | 회전 중 팔이 쥔 튜브를 놓치거나 흘러가면 적는다 |
| 5 | task05 | 10/16 그대로 + 상대 EEF 의 IK 유지(오프라인: GIST 장면 t05 에서 몰림, 우리 장면 0) | 10/16 그대로 | 10/16 그대로 (14) | 10/16 끝에 #63 → #69 (2) | 버리기 ✓/+p 와 「IK 실패/튐」 줄 수를 같이 적는다 |
| 11 | task11 | 10/16 그대로 + 상대 EEF 의 출발(오프라인 t11 출발이 약함: #63 .68 · #69 .43, #2 .55) | 10/16 그대로 | 10/16 그대로 (2) | 10/16 끝에 #63 → #69 (2) | 출발하는지(첫 3 s 팔 움직임) 따로 적는다 |

**남은 회차**: 1순위 **88 회**(10/16 의 79 + 2a 4 + 2c 6, t02 2a 가 20 → 24), 2순위 **126 회**(10/16 의 114 + 2c 7 + t03 1 + t05 2 + t11 2). 1순위 ≈9.4 h [추정 — 10/16 절과 같은 비율].
**급하면** 2a 의 #63 첫 1 회 → 2c 6 회만 먼저 돌아도 된다(≈1 h [추정]) — 이번 학습 라운드가 묻는 것이 이것이다.

### 3. 명령 — 상대 EEF 모델 (그 단계 재생으로 A✓ 를 먼저 — 재생 명령은 10/16 §3)
```bash
# #63 (상대 EEF + state 유지 60K) — <s> = 단계 번호 (t02 = 2)
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=<s>/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_reef_27D_60k_s1000 <s> 60 1
# #69 (상대 EEF + state 제거 120K)
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=<s>/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_reef_sf_27D_120k_s1000 <s> 60 1
# #68 (#63 을 120K 까지) — 2c 조건부
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=<s>/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_reef_27D_120k_s1000 <s> 60 1
# #66 · #67 (#63 의 시드 2000 · 3000) — 2c 조건부
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=<s>/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_reef_27D_60k_s2000 <s> 60 1
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=<s>/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_reef_27D_60k_s3000 <s> 60 1
# #2 (짝) — 10/16 §3 와 같다
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=<s>/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 <s> 60 1
```
- `MAX_REL=none` 을 붙이지 않는다(상대 EEF 체크포인트는 거부된다). 팔 0.1 rad · sf 0.4 · 베이스 ±1.0 · e-stop 은 그대로다.

### 4. 보류 (이 절 기준)
| 모델 | 이유 |
|---|---|
| #62 (state 제거 60K) | #69(같은 런을 120K 까지)가 오프라인에서 모두 낫다(IK 유지 123 → 70/2760, 잡기 40틱 전 오른손 .62 → .76). #69 와 #63 의 실기 차이가 크면 그때 학습량 효과를 보려고 돈다 |
| #64 · #65 (#62 의 시드 짝) | #62 가 보류라 같이 보류 |
| #70 · #71 (#64·#65 를 120K 까지) | #69 의 시드 짝 — 오프라인에서 #69 와 같은 띠(GIST IK 유지 48·64 vs 70, t11 .29·.41 vs .43). #69 결과가 극단적일 때 시드 운인지 가를 때만 돈다 |

### 5. 기록
- 회차마다 맨 아래 「결과 기록」 에 한 줄 + 그날 날짜 문서 `eval_najy_results_<MMDD>.md` 에 상세. 상대 EEF 회차는 **state_free 값 · 유지 누적 K · 루프 Hz** 를 같이 적는다. 2c 는 **랙을 옮긴 방향·거리**를 회차마다 적는다.
- 그다음 `git push origin main` (로봇 PC 의 origin = Najongs fork).

## ▶ 10/16 정비 — 이 절만 본다 (DGX1 → Trossen PC1, 2026-10-16 KST)
> **10/19 정정**: 위 「▶ 10/19 정비」 절을 먼저 본다. 이 절의 §0·§2·§3·§4 는 그대로 쓰고, 상대 EEF 모델(#62~#69)과 t02 2c 는 10/19 절에 있다.
> **사용자(10/16)**: 「그래 실기 검토 한번 해보자. 리스트 다시 잘 정비해서 Trossen PC에 보내둬」. 아래에 쌓여 있던 「▶ 지금 돌릴 것」 절 8개(10/11~10/15)를 **이 절 하나로 합쳤다**. 옛 절은 비교 정의(C·O 번호)와 지난 명령의 기록으로만 남긴다 — **거기 명령은 돌리지 마라**. 이 절의 시각은 전부 **KST** 다(옛 절은 UTC 가 섞여 있다).
> 바뀐 것 요약: ① t01 에 **베이스 회전 예산(bb105) 켬 ↔ 끔 3 대 3** 을 새로 넣었다(과회전을 실행 층에서 자르는 시험). 그리고 기존 「전문가 ↔ #27」·「#27 e30 ↔ e60」 과 **한 줄로 엮었다** — #27 기준 회차(e60·예산 끔) 3 회를 세 비교가 같이 쓴다. ② 10/16 단계 헤드 #58~#61 의 회차와 명령을 t04·t05·t08·t11 에 넣었다. ③ 결과가 이미 나왔거나 사용자가 건너뛴 것을 「안 하는 것 · 보류」 로 옮겼다(하이브리드 1→2, 흔든 시작 B 등). ④ 남은 회차 수를 행에서 다시 셌다.

### 0. 시작 전 — 매 세션
1. `cd ~/NAJY/lerobot_trossen && git pull origin main` (로봇 PC 의 origin = Najongs fork). 필요한 커밋: **08a0631**(단계 헤드 로드 훅 — #59·#60) · **a76ce00**(베이스 회전 예산 — `ls packages/lerobot_robot_trossen/src/lerobot_robot_trossen/base_budget.py` 로 확인; 커밋 제목에는 `base_budget` 이 없어 `git log --oneline | grep` 은 안 걸린다).
   - 예산 커밋은 codex 교차 검토 뒤 a76ce00 으로 올라갔다(10/16 16:13 KST). 위 파일이 없으면 **bb105 회차만 건너뛰고** 나머지는 그대로 돈다. 그 커밋이 없는 체크아웃에서는 `BASE_BUDGET_DEG=105` 가 **조용히 무시된다**(예산 없이 돈다). 그래서 아래 확인 줄이 필수다.
2. 처음 한 번 `DRY_RUN=1` 로 명령을 확인한다. e-stop 을 손에 두고, 회차는 1 ep 씩 돈다.
3. **로그 확인 줄 — 없으면 그 회차를 중단한다**

   | 모델 | 로그에 있어야 하는 줄 |
   |---|---|
   | 27D ACT 전부 (HOT 를 준 회차) | `installed` · `stage i/11 active` |
   | #50 · #51 · #52 (원핫 없음 16D) | `zero-pad only` |
   | #49 | `base slots = measured velocity` |
   | #40 | `[action_query] model.action_queries (64) 를 붙여 싣는다` |
   | #59 · #60 | `[stage_heads] 단계 헤드 K=11 를 붙여 싣는다` (없으면 shape 오류로 바로 죽는다) |
   | **bb105 회차** | **`[base_budget] 켜짐`** (시작 직후). 발동하면 `[base_budget] 발동 — … tick …` 이 찍히고, 끝나면 `[base_budget] policy 구간 1 끝: …` 이 찍힌다. `켜졌지만 policy 구간을 한 번도 보지 못했다` 가 찍히면 그 회차는 **끔** 으로 기록한다. **예산을 끈 회차에는 `[base_budget]` 줄이 하나도 없어야 한다** |
4. **판정**: 같은 자리에서 번갈아(ABAB) 각 3 회씩 돌고, **성공 수가 2 이상 차이 날 때만** 판정한다. 짝 회차끼리는 진단 스위치(`CHUNK_LOG`·`POSE_GUIDE`)를 같게 둔다 — 기본값 그대로(`CHUNK_LOG` 끔). `CHUNK_LOG=1` 은 청크 실행기를 설치해 실행 경로가 바뀌므로 한쪽에만 켜면 안 된다.
5. **함정**
   - 같은 분 안에 다시 돌리면 FileExistsError 로 무효가 된다(10/15 0058·0103). 1분 띄운다.
   - 재생 명령에서 `LEROBOT_REPLAY_DATASET` 가 빠지면 정책이 돈다(10/15 1059).
   - 명령에 `| tail` 을 붙이지 않는다(실패해도 exit 0 이 된다).
   - **`BASE_BUDGET_DEG` 는 `export` 하지 말고 명령 앞에만 붙인다.** 끔 회차는 이 변수가 없는 명령 그대로 돈다.
6. **기록**: 회차마다 맨 아래 「결과 기록」 에 한 줄을 적고, 상세는 `eval_najy_results_1016.md`(새로 만든다)에 적는다. 그다음 `git push origin main`. 끝나면 `scripts/eval_najy_post.sh <회차>` 의 `.summary.txt` 에 `-- 베이스 회전 예산` 줄이 생긴다(bb105 회차만 내용이 있다).

### 1. 지금까지 실기 (요약 — 상세는 결과 기록·날짜 문서)

| 태스크 | 결과 | 비고 |
|---|---|---|
| t01 | **#27 3/3**(10/15 0104 · 1045 · 1053). #2 1/3(10/11 2143 랙 빈 성공 · 0054 +54° · 1047 전진 충돌). #39 0/2 · #49 0/1 · #29 0/1 · #25 0/2 · #1 0/2(출발 못 함) · SmolVLA 0/5 | 10/15 오전부터 재생 ep30 자리 + 바닥 표시. 그 전 1023~1031 은 자리가 어긋나 무효 |
| t02 | **#40 1/1**(10/15 013856, 양손 쥐고 정지). 전문가 0/1+p(왼손). #2 는 10/11~10/15 전부 실패(A✓ 에서 왼손만 1 회) · #39 · #29 · #27 · #25 0/1 · #32 0/3 · #43·#44 전부 실패 | 재생 ep102 로 A✓ (10/14 1647) |
| t03 | **#2 1/1**(10/13 1139, 빈손 출발). #32 · #3 · #36 0/1 | 튜브를 쥐고 출발한 청크 100 회차는 아직 없다 |
| t04 · t05 | 청크 100 실기 없음(청크 30 세대만: t04 M1 1/1 · t05 전문가 e30 1/3+p · e100 1/1) | |
| t06 | #40 0/1(10/15 110222, 후진 0.40 m·−51°, 출발 자세 미확인 → 판정에서 뺀다). 1059 는 재생이 아니었다(무효) | |
| t07 ~ t11 | 실기 없음 | |

### 2. 태스크 순서 표 — 행마다 1순위를 먼저, 시간이 남으면 2순위

| 순서 | 태스크 | 검증하려는 것 | 자리 | 1순위 (이 순서로) | 2순위 | 결과 → 다음 |
|---|---|---|---|---|---|---|
| **1** | **task01 랙으로 이동** | ⓐ **베이스 회전 예산(bb105)이 과회전만 자르고 도착·정지를 해치지 않나** (새로) · ⓑ 단계 전문가가 도착도 더 정확한가 (전문가 ↔ #27) · ⓒ exec 30 이 도착 뒤 꼬리(묵은 청크 최대 2.9 s)를 줄이나, 출발은 막지 않나 (#27 e30 ↔ e60) · #2 를 3 회로 채움 · C4 #49 · C3 #39 | 재생 ep30 + 바닥 표시. 랙에 튜브 두 개(시연과 같게) | **(전문가 → #27 → #27 bb105 → #27 e30) ×3 → #2 → #2 → #39 → #49 (16)**. 예산 커밋이 아직 없으면 bb105 3 회만 빼고 돈다 | #36 → #36 → #4 → #9 → #4 → #9 → #32 → #50 → #2 랙 빈 → #2 랙 빈 → #31 → #60 (12) | **ⓐ** 회차마다 적는다: 발동 여부 · 발동 시각 · 막은 tick(`.summary.txt`) · 82° 도달 시각 · 끝 각도 · 랙 앞에서 멈췄나. bb105 성공 수가 끔과 같거나 많고 넘친 회차의 꼬리가 사라지면 **t01 기본값으로 켠다**(t03·t10 예산은 그 단계 시연으로 따로 정한다). bb105 가 성공을 2 회 이상 잃으면 폐기한다. **ⓑ** 전문가가 #27 보다 2 회 이상 많으면 t01 = 단계 전문가. **ⓒ** e30 이 이기면 팔·베이스 exec 분리를 구현한다(그 전에 실기 t02 프레임으로 베이스 누출을 확인). e10·e20 은 넣지 않는다(10/16 오프라인: 출발이 6 s 넘게 늦어진다). **#60** 은 끝에서 멈추나만 본다 |
| 2a | task02 튜브 집기 (시연 튜브) | C1′ #40 성공이 우연인가 · 쿼리 효과인가(#36) · C5 전문가 · C5′ 같은 레시피 단계 전문가 #51 · 왼손 높이(#1 ×2 · #26, O15) · C3 #39 | 재생 ep102 로 A✓. 튜브는 매번 같은 시연 구멍 | (#40 → 전문가 → #36 → #2 → #39 → #51) ×3 → #1 → #1 (20) | #26 → #50 → #49 → #31 (4) | 양손 ✓, 왼손만 +p. #40 이 #2 보다 2 회 이상 많으면 t02 = SmolVLA. #36 도 #40 만큼 쥐면 쿼리는 이득이 없다. #51 이 #2 보다 2 회 이상 많으면 O11(11단계 한 모델 ≥ 단계 전문가)이 t02 에서 뒤집힌다 |
| 2b | task02 (투명 튜브) | 투명 튜브를 더 배운 #41 이 #2 보다 잘 잡나 | 같은 A✓, 투명 튜브 | — | #2 → #41 → #2 → #41 → #40 (5) | #41 만 쥐면 투명 튜브 녹화를 추가한다 |
| **3** | **task03 제자리 회전** | 이동 후보 #27 · **C4 베이스 속도 입력 #49 ↔ #2** · 블렌딩 켬/끔(O16) · #50 회전 출발 · 튜브 쥐고 출발(O14) | **양손에 튜브를 쥐어 준 채**, 재생 ep79 로 A✓ | (#2 → #27 → #49) ×3 (9) | (#2 → #2 블렌딩 끔) ×3 → #50 → #50 → #39 → #29 → #32 → #36 → 전문가 → #1 (14) | 비커 정면에서 멈추면 ✓, 회전각을 적는다(시연 ≈ −90°). **C4 의 판정은 이 행에서 한다** — t01 의 #49 는 2 회뿐이라 닫히지 않는다. #49 가 #2 보다 2 회 이상 많으면 베이스 속도 입력을 채택한다. 블렌딩 끔이 2 회 이상 적으면 블렌딩 켬을 확정한다. **#1 은 2순위 끝으로 내렸다**(t01 에서 2/2 출발하지 못했다) |
| 4 | task04 붓기 | 청크 100 첫 실기 · #40 조작 · **단계 헤드 #59 ↔ #58**(t03 끝 자리 — 오프라인은 나빠진 경계) | 튜브를 쥐어 준 채, 재생 ep96 로 A✓ | #2 → #40 (2) | #39 → #29 → 전문가 → (#58 → #59) ×3 → #60 → #61 (11) | 붓는 동작까지 가면 ✓. #59 가 #58 보다 2 회 이상 많으면 단계 헤드가 경계 단계를 가른다 |
| **5** | **task05 튜브 버리기** | C2 원핫이 단계를 고르나(HOT 4/11 vs 5/11) · C2′ 원핫 없는 #50 ↔ #2 · C5 전문가 · #40 · 조작 exec 길이(전문가 e30 · e60 · e100) · **단계 헤드 #59 ↔ #58**(t04 끝 자리) | 양손에 튜브, 재생 ep69 로 A✓. 원핫 시험은 장면·자리 그대로 HOT 만 4/11 | (#2 → 전문가 → #50) ×3 → #40 → #2 HOT=4/11 → #2 HOT=4/11 → #25 → #25 HOT=4/11 (14) | (전문가 e30 → e60 → e100) ×3 → #39 → #29 → #27 → #36 → #4 → #9 → #52 ×3 → (#58 → #59) ×3 → #60 → #61 (26) | 둘 다 트레이에 떨어뜨리면 ✓, 하나면 +p. HOT=4/11 에서도 버리면 「원핫은 단계를 못 고른다」 를 실기로 확정한다. 전문가 e100 이 e30 보다 2 회 이상 많으면 조작은 길게 실행한다 |
| 6 | task06 비커 잡고 냉장고로 | 이동 후보 #27 · 잡기와 이동 중 어디서 실패하나 · #40 재시도 | 재생 ep93 으로 A✓ — **재생이 끝난 자세에서 정책을 시작한다**(10/15 #40 은 출발 자세를 확인하지 못했다) | #2 → #27 → #40 (3) | #39 → #29 → 전문가 (3) | 잡기 ✓/✗ 와 도착 ✓/✗ 를 **나눠** 적는다 |
| 7 | task07 냉장고 열기 | 첫 실기 · #40 · #53 · #60 끝 정지 | 재생 ep184 로 A✓ | #2 → #40 (2) | #39 → #29 → 전문가 → #53 → #60 (5) | 문이 열리면 ✓. #60 은 끝에서 멈추나 |
| **8** | **task08 비커 꺼내 놓기** | C3 확정(#39) · #40 · #53 ↔ #54(t08 새 녹화) · **단계 헤드 #59 ↔ #58**(t07 끝 자리) | 재생 ep157 로 A✓ | (#2 → #39) ×3 → #40 (7) | #29 → 전문가 → (#53 → #54) ×3 → (#58 → #59) ×3 → #60 (15) | 놓으면 ✓. #39 가 #2 보다 2 회 이상 많으면 lr cosine 을 채택한다. #54 가 2 회 이상 많으면 t08 새 녹화를 다음 레시피에 넣는다. **#58~#61 은 t08 출발이 약해졌으므로**(오프라인) #27 대체 후보로 쓰지 말고 #58 과의 비교로만 본다 |
| 9 | task09 냉장고 닫기 | 첫 실기 · #40 · 닫기와 물러나기를 나눠 보기 | 재생 ep174 로 A✓ | #2 → #40 (2) | #27 → #39 → #29 → 전문가 (4) | 닫기 ✓/✗ 와 물러남 ✓/✗ 를 나눠 적는다 |
| **10** | **task10 선반으로 이동** | 이동 후보 #27 · #32 | 비커를 쥐어 준 채, 재생 ep88 로 A✓ | #2 → #27 (2) | #32 → #39 → #29 → 전문가 → #1 (5) | 선반 앞에서 멈추면 ✓, 더 돈 각도를 적는다. **#1 은 2순위로 내렸다**(t03 과 같은 이유) |
| 11 | task11 비커 선반에 놓기 | 첫 실기 · #40 · **단계 헤드 #59 ↔ #58**(t10 끝 자리) · #60 끝 정지 | 재생 ep98 로 A✓ | #2 → #40 (2) | #39 → #29 → (#58 → #59) ×3 → #60 → #61 (10) | 놓으면 ✓ |

**남은 회차**: 1순위 79 회(t01 16 · t02 20 · t03 9 · t04 2 · t05 14 · t06 3 · t07 2 · t08 7 · t09 2 · t10 2 · t11 2), 2순위 114 회(t01 12 · t02 2a 4 + 2b 5 · t03 14 · t04 11 · t05 26 · t06 3 · t07 5 · t08 15 · t09 4 · t10 5 · t11 10). 옛 표의 추정 비율(85 회 ≈ 9 h, 자리 맞추기 포함)로 1순위 ≈8.4 h, 2순위 ≈12 h 다.

**t01 베이스 회전 예산(bb105)이 무엇을 하나** — `packages/lerobot_robot_trossen/src/lerobot_robot_trossen/base_budget.py`
- 에피소드(policy 구간)마다 **보낸 `theta.vel` 을 실측 시간으로 적분한 순회전**을 잰다. 0 에서 시작한다.
- |순회전| 이 105° 에 닿으면 **같은 방향 `theta.vel` 만 0** 으로 바꾼다. 되돌리는 방향, `x.vel`, 팔은 그대로다.
- 명령을 0 으로 줄이기만 한다. 안전 상수(sf 0.4 · 팔 0.1 rad · 베이스 ±1.0 · e-stop)는 그대로다.
- 리셋(리더 팔) 구간에서는 작동하지 않는다.
- 105° 의 근거: 시연 순회전 76~89°(중앙 82°) + 출발 각 어긋남 10° + 여유. **명령 단위**라 실측 회전은 이보다 작다.
- `basevel.csv` 에는 예산이 적용된 명령이 남고, **데이터셋 action 칸과 `chunks.csv` 에는 정책 원출력(예산 전)** 이 남는다. **켬 ↔ 끔 비교의 명령 누적·끝 각도는 `basevel.csv` 의 `cmd_theta_vel`(와 실측 열)로만 낸다** — 데이터셋 action 으로 내면 켬 회차도 끔처럼 163° 로 나온다.
- 0.5 s 넘게 멈춘 틱은 0.5 s 로만 세고, 0 을 보낸 뒤 베이스가 미끄러지는 회전은 세지 않는다 — 실측 끝 각도는 105° 를 몇 도 넘을 수 있다.
- 예산 바로 아래에서 통과한 명령은 다음 틱까지 돈다(21 Hz·1 rad/s 면 ≈2.7°, 0.5 s 멈춤이면 최대 ≈29°). 마지막 policy 틱 뒤 저장 중 유지분과, 베이스 쓰기 실패(`Mobile AI base transaction failed` 경고) 때 이전 명령이 남은 회전도 요약에 안 잡힌다 — 그 경고가 찍힌 bb 회차는 결과 옆에 적어라.
- **발동 뒤에도 전진(`x.vel`)은 계속된다**(방위만 고정). 발동 뒤 어디로 가서 멈추는지 적는다 — 충돌하면 그 회차에서 멈추고 기록한다.
- 오프라인 미리보기(실기 t01 기록 명령에 다시 걸어 봄, `/raid/kiro-ai/eval/real/BUDGET_1016.log`, 폐루프 아님):
  - 넘친 회차는 9.5~14 s 에 끊긴다: 0054 163° · 0101 140° · 1029 150° · 0100 109° → 전부 105°.
  - #27 성공 회차 중 1053(최대 99°)·0104(100°)는 건드리지 않는다. 1045(107°)는 82° 를 6.4 s 에 지났고, 19.3 s 이후의 마지막 2°(105 → 107°)만 막힌다.
  - 재생 ep30(82°)은 건드리지 않는다.

### 3. 명령 — 태스크별 (복사해서 쓴다. 모든 정책 회차 앞에 그 단계 재생으로 A✓ 를 맞춘다)

#### t01
```bash
# 자리 맞추기(A✓) — 정책 회차마다 먼저
LEROBOT_REPLAY_DATASET=kiroaiseoul/task01_move_to_tube_rack LEROBOT_REPLAY_EPISODES=30 LEROBOT_REPLAY_RUN_POLICY=0 RUN_SUFFIX=replay_ep30 HOT=1/11 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 1 60 1
# #27 (e60, 예산 끔 — 기준)
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=1/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_120k_s1000 1 60 1
# #27 bb105 (예산 켬, e60) — 로그에 [base_budget] 켜짐 이 없으면 중단
BASE_BUDGET_DEG=105 LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=1/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_120k_s1000 1 60 1
# #27 e30 (예산 끔)
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=1/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_120k_s1000 1 30 1
# 전문가
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/task01_move_to_tube_rack_14D 1 60 1
# #2
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=1/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 1 60 1
# #39
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=1/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_cos_27D_60k_s1000 1 60 1
# #49
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=1/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_vel_27D_60k_s1000 1 60 1
# #36
POLICY=kiroaiseoul/NAJY_smolvla_all11_gist30_fp32_60k_s1000 scripts/eval_smolvla.sh 1 rec 1
# #4
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=1/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_m1_27D_60k_s1000 1 60 1
# #9
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=1/11 LEROBOT_CHUNK_SMOOTH_TICKS=5 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c150_m1_27D_60k_s1000 1 60 1
# #32
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=1/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30mv_sw_27D_60k_s1000 1 60 1
# #50
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_nohot_16D_60k_s1000 1 60 1
# #31
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=1/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist50_27D_60k_s1000 1 60 1
# #60
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=1/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_headspad_27D_150k_s1000 1 60 1
```

#### t02 (2a · 2b)
```bash
# 자리 맞추기(A✓) — 정책 회차마다 먼저
LEROBOT_REPLAY_DATASET=kiroaiseoul/task02_pickup_tubes LEROBOT_REPLAY_EPISODES=102 LEROBOT_REPLAY_RUN_POLICY=0 RUN_SUFFIX=replay_ep102 HOT=2/11 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 2 60 1
# #40
POLICY=kiroaiseoul/NAJY_smolvla_all11_gist30_fp32_aq64_60k_s1000 scripts/eval_smolvla.sh 2 rec 1
# #2
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=2/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 2 60 1
# #39
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=2/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_cos_27D_60k_s1000 2 60 1
# #51
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_task02_c100_gist30_solo_16D_60k_s1000 2 60 1
# #1
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=2/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_m1_27D_60k_s2000 2 60 1
# #36
POLICY=kiroaiseoul/NAJY_smolvla_all11_gist30_fp32_60k_s1000 scripts/eval_smolvla.sh 2 rec 1
# #26
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=2/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s3000 2 60 1
# #50
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_nohot_16D_60k_s1000 2 60 1
# #49
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=2/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_vel_27D_60k_s1000 2 60 1
# #31
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=2/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist50_27D_60k_s1000 2 60 1
# #41
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=2/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_t02il_27D_60k_s1000 2 60 1
# 전문가
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/task02_pickup_tubes_14D 2 60 1
```

#### t03
```bash
# 자리 맞추기(A✓) — 정책 회차마다 먼저
LEROBOT_REPLAY_DATASET=kiroaiseoul/task03_turn_to_face_beaker LEROBOT_REPLAY_EPISODES=79 LEROBOT_REPLAY_RUN_POLICY=0 RUN_SUFFIX=replay_ep79 HOT=3/11 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 3 60 1
# #2
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=3/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 3 60 1
# #27
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=3/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_120k_s1000 3 60 1
# #49
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=3/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_vel_27D_60k_s1000 3 60 1
# #50
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_nohot_16D_60k_s1000 3 60 1
# #39
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=3/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_cos_27D_60k_s1000 3 60 1
# #29
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=3/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_dq_27D_60k_s1000 3 60 1
# #32
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=3/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30mv_sw_27D_60k_s1000 3 60 1
# #36
POLICY=kiroaiseoul/NAJY_smolvla_all11_gist30_fp32_60k_s1000 scripts/eval_smolvla.sh 3 rec 1
# #1
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=3/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_m1_27D_60k_s2000 3 60 1
# #2 블렌딩 끔 (끔을 명시 — 셸에 켠 값이 남아도 끈다)
LEROBOT_CHUNK_SEAM_BLEND_TICKS=0 HOT=3/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 3 60 1
# 전문가
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/task03_turn_to_face_beaker_14D 3 60 1
```

#### t04
```bash
# 자리 맞추기(A✓) — 정책 회차마다 먼저
LEROBOT_REPLAY_DATASET=kiroaiseoul/task04_pour_liquid_from_tubes_to_beaker LEROBOT_REPLAY_EPISODES=96 LEROBOT_REPLAY_RUN_POLICY=0 RUN_SUFFIX=replay_ep96 HOT=4/11 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 4 60 1
# #2
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=4/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 4 60 1
# #40
POLICY=kiroaiseoul/NAJY_smolvla_all11_gist30_fp32_aq64_60k_s1000 scripts/eval_smolvla.sh 4 rec 1
# #39
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=4/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_cos_27D_60k_s1000 4 60 1
# #29
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=4/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_dq_27D_60k_s1000 4 60 1
# #58
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=4/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_ctrl_27D_150k_s1000 4 60 1
# #59
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=4/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_heads_27D_150k_s1000 4 60 1
# #60
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=4/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_headspad_27D_150k_s1000 4 60 1
# #61
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=4/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_padhold_27D_150k_s1000 4 60 1
# 전문가
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/act_task04_pour_liquid_from_tubes_to_beaker_60000 4 60 1
```

#### t05
```bash
# 자리 맞추기(A✓) — 정책 회차마다 먼저
LEROBOT_REPLAY_DATASET=kiroaiseoul/task05_tube_disposal LEROBOT_REPLAY_EPISODES=69 LEROBOT_REPLAY_RUN_POLICY=0 RUN_SUFFIX=replay_ep69 HOT=5/11 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 5 60 1
# #2
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=5/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 5 60 1
# #50
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_nohot_16D_60k_s1000 5 60 1
# #40
POLICY=kiroaiseoul/NAJY_smolvla_all11_gist30_fp32_aq64_60k_s1000 scripts/eval_smolvla.sh 5 rec 1
# #25
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=5/11 LEROBOT_CHUNK_SMOOTH_TICKS=5 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_drop_27D_60k_s1000 5 60 1
# #39
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=5/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_cos_27D_60k_s1000 5 60 1
# #29
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=5/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_dq_27D_60k_s1000 5 60 1
# #27
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=5/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_120k_s1000 5 60 1
# #36
POLICY=kiroaiseoul/NAJY_smolvla_all11_gist30_fp32_60k_s1000 scripts/eval_smolvla.sh 5 rec 1
# #4
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=5/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_m1_27D_60k_s1000 5 60 1
# #9
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=5/11 LEROBOT_CHUNK_SMOOTH_TICKS=5 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c150_m1_27D_60k_s1000 5 60 1
# #52
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_task05_c100_gist30_solo_16D_60k_s1000 5 60 1
# #58
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=5/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_ctrl_27D_150k_s1000 5 60 1
# #59
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=5/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_heads_27D_150k_s1000 5 60 1
# #60
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=5/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_headspad_27D_150k_s1000 5 60 1
# #61
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=5/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_padhold_27D_150k_s1000 5 60 1
# 전문가 (e60)
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/act_task05_tube_disposal_60000 5 60 1
# 전문가 e30
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/act_task05_tube_disposal_60000 5 30 1
# 전문가 e100
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/act_task05_tube_disposal_60000 5 100 1
# #2 원핫 시험 HOT=4/11 (장면·자리 그대로)
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=4/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 5 60 1
# #25 원핫 시험 HOT=4/11
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=4/11 LEROBOT_CHUNK_SMOOTH_TICKS=5 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_drop_27D_60k_s1000 5 60 1
```

#### t06
```bash
# 자리 맞추기(A✓) — 정책 회차마다 먼저
LEROBOT_REPLAY_DATASET=kiroaiseoul/task06_task06-2_pickup_beaker_and_move_to_refrigerator_14D_new_2 LEROBOT_REPLAY_EPISODES=93 LEROBOT_REPLAY_RUN_POLICY=0 RUN_SUFFIX=replay_ep93 HOT=6/11 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 6 60 1
# #2
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=6/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 6 60 1
# #27
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=6/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_120k_s1000 6 60 1
# #40
POLICY=kiroaiseoul/NAJY_smolvla_all11_gist30_fp32_aq64_60k_s1000 scripts/eval_smolvla.sh 6 rec 1
# #39
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=6/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_cos_27D_60k_s1000 6 60 1
# #29
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=6/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_dq_27D_60k_s1000 6 60 1
# 전문가
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/act_task06_task06-2_pickup_beaker_and_move_to_refrigerator_14D_new_2 6 60 1
```

#### t07
```bash
# 자리 맞추기(A✓) — 정책 회차마다 먼저
LEROBOT_REPLAY_DATASET=kiroaiseoul/task07_open_refrigerator LEROBOT_REPLAY_EPISODES=184 LEROBOT_REPLAY_RUN_POLICY=0 RUN_SUFFIX=replay_ep184 HOT=7/11 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 7 60 1
# #2
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=7/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 7 60 1
# #40
POLICY=kiroaiseoul/NAJY_smolvla_all11_gist30_fp32_aq64_60k_s1000 scripts/eval_smolvla.sh 7 rec 1
# #39
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=7/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_cos_27D_60k_s1000 7 60 1
# #29
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=7/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_dq_27D_60k_s1000 7 60 1
# #53
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=7/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_basefix_27D_60k_s1000 7 60 1
# #60
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=7/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_headspad_27D_150k_s1000 7 60 1
# 전문가
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/act_task07_open_refrigerator_14D_260916 7 60 1
```

#### t08
```bash
# 자리 맞추기(A✓) — 정책 회차마다 먼저
LEROBOT_REPLAY_DATASET=kiroaiseoul/task08_takeout_and_put_beaker LEROBOT_REPLAY_EPISODES=157 LEROBOT_REPLAY_RUN_POLICY=0 RUN_SUFFIX=replay_ep157 HOT=8/11 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 8 60 1
# #2
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=8/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 8 60 1
# #39
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=8/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_cos_27D_60k_s1000 8 60 1
# #40
POLICY=kiroaiseoul/NAJY_smolvla_all11_gist30_fp32_aq64_60k_s1000 scripts/eval_smolvla.sh 8 rec 1
# #29
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=8/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_dq_27D_60k_s1000 8 60 1
# #53
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=8/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_basefix_27D_60k_s1000 8 60 1
# #54
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=8/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_t08add_27D_60k_s1000 8 60 1
# #58
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=8/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_ctrl_27D_150k_s1000 8 60 1
# #59
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=8/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_heads_27D_150k_s1000 8 60 1
# #60
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=8/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_headspad_27D_150k_s1000 8 60 1
# 전문가
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/act_task08_takeout_and_put_beaker_100k_251data 8 60 1
```

#### t09
```bash
# 자리 맞추기(A✓) — 정책 회차마다 먼저
LEROBOT_REPLAY_DATASET=kiroaiseoul/task09_close_refrigerator LEROBOT_REPLAY_EPISODES=174 LEROBOT_REPLAY_RUN_POLICY=0 RUN_SUFFIX=replay_ep174 HOT=9/11 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 9 60 1
# #2
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=9/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 9 60 1
# #40
POLICY=kiroaiseoul/NAJY_smolvla_all11_gist30_fp32_aq64_60k_s1000 scripts/eval_smolvla.sh 9 rec 1
# #27
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=9/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_120k_s1000 9 60 1
# #39
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=9/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_cos_27D_60k_s1000 9 60 1
# #29
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=9/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_dq_27D_60k_s1000 9 60 1
# 전문가
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/act_task09_close_refrigerator_100k_250data 9 60 1
```

#### t10
```bash
# 자리 맞추기(A✓) — 정책 회차마다 먼저
LEROBOT_REPLAY_DATASET=kiroaiseoul/task10_move_to_beaker_shelf LEROBOT_REPLAY_EPISODES=88 LEROBOT_REPLAY_RUN_POLICY=0 RUN_SUFFIX=replay_ep88 HOT=10/11 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 10 60 1
# #2
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=10/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 10 60 1
# #27
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=10/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_120k_s1000 10 60 1
# #32
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=10/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30mv_sw_27D_60k_s1000 10 60 1
# #39
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=10/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_cos_27D_60k_s1000 10 60 1
# #29
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=10/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_dq_27D_60k_s1000 10 60 1
# #1
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=10/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_m1_27D_60k_s2000 10 60 1
# 전문가
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/act_task10_move_to_beaker_shelf_14D_100k_fp32 10 60 1
```

#### t11 (단계 전문가 없음)
```bash
# 자리 맞추기(A✓) — 정책 회차마다 먼저
LEROBOT_REPLAY_DATASET=kiroaiseoul/task11_place_beaker_on_the_shelf LEROBOT_REPLAY_EPISODES=98 LEROBOT_REPLAY_RUN_POLICY=0 RUN_SUFFIX=replay_ep98 HOT=11/11 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 11 60 1
# #2
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=11/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 11 60 1
# #40
POLICY=kiroaiseoul/NAJY_smolvla_all11_gist30_fp32_aq64_60k_s1000 scripts/eval_smolvla.sh 11 rec 1
# #39
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=11/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_cos_27D_60k_s1000 11 60 1
# #29
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=11/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_dq_27D_60k_s1000 11 60 1
# #58
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=11/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_ctrl_27D_150k_s1000 11 60 1
# #59
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=11/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_heads_27D_150k_s1000 11 60 1
# #60
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=11/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_headspad_27D_150k_s1000 11 60 1
# #61
LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=11/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_padhold_27D_150k_s1000 11 60 1
```

### 4. 안 하는 것 · 보류 (이유와 함께 — 학습한 모델은 전부 여기나 위 표에 있다)
- **보류(사용자 10/15 건너뜀)**: 하이브리드 1→2(#27 t01 → 재생 ep102 → #40 t02). 이동·조작 모델이 각각 정해진 뒤 다시 올린다.
- **스킵(사용자 10/15)**: 흔든 시작 B(O17 은 실기 미확인으로 남는다).
- **안 함(10/16 오프라인)**: 팔·베이스 exec 분리 스위치와 t01 exec 10·20. t01 e30 ↔ e60 이 먼저 판정한다.
- **#56 · #57**(#27 → 240K, 이동 가중 1.65 / 그대로): 2호기(sandia)에서 학습 중이다. 끝나면 DGX 가 채점하고 행을 따로 올린다. **지금은 명령이 없다.**
- **#55**(#53 시드 짝): public 이지만 시드 짝 규칙(짝 모델이 실기에서 이긴 뒤)에 따라 #53 결과를 본 뒤에 돈다.
- state 드롭아웃 #43~#48 · #19 · #33 · #7 · #14 · #10 · #11 · #20 · #3 · #8 · #24 · #37 · #21~#23 · 시드 짝 #5 · #6 · #30 · #34 · #38 · #42 · #46 · #47 · 옛 SmolVLA #12 · #13 · #35 · odom 입력: 이유는 옛 「▶ 10/15 한 번에」 절의 「안 하는 것」 그대로다.
- **넘어오지 않은 옛 항목 — 이번 라운드에서 하지 않는다**
  - 빨간 테이프, 자리 ±5 cm·±10° 다양화 녹화 → 이동 결과(t01·t03 판정) 뒤에 다시 정한다.
  - t04 전문가 exec A1·A2 → t05 전문가 exec 비교로 대체했다.
  - exec 60·다듬기 w=3 의 실기 3 대 3 → 이번엔 블렌딩(t03)과 exec(t01·t05)만 본다.
- **아직 유효한 부탁(로봇 PC → DGX)**: t03 회차의 `basevel.csv` 업로드, `eval_najy_1011_2141` 재업로드. 시간이 날 때 한다.

## ▶ 10/15 한 번에 — 남은 것 · 오프라인으로만 닫은 결론 · 새 방향(하이브리드) 전부, 태스크 순서 (DGX1 → Trossen PC1, 2026-10-14 18:40 UTC) · 이 표만 본다
> **⛔ 10/16 정비로 대체됨 — 이 절의 명령은 돌리지 마라. 맨 위 「▶ 10/16 정비」 절만 본다.** (비교 정의·지난 기록으로만 남긴다)
> 시각 정정: 이 절은 2026-10-14 **17:34 UTC** 커밋(ea32470)이다.
> **추가(10/16 05:00 KST, 1호기(DGX1) 단계 헤드 4런 끝)** — **#58~#61 을 표에 올렸다(전부 public).** #58 = #27 에서 그대로 +30K(대조), #59 = + 단계별 헤드, #60 = 헤드 + pad_hold, #61 = pad_hold 만. **#59·#60 은 로봇 플러그인의 새 로드 훅이 있어야 실린다** — 로봇 PC 에서 `git pull` 뒤 로그에 `[stage_heads] 단계 헤드 K=11 를 붙여 싣는다` 가 찍혀야 한다(없으면 shape 오류로 즉시 죽는다 — 조용히 틀리게 도는 경로는 없다). 회차: 같은 장면 경계 단계(t05 · t11 · t08 · t04)에서 **#58 과 번갈아** 각 3 회, #60 은 t01·t07·t11 끝에서 멈추나도 본다. **주의 — 4런 공통으로 t08 출발이 약해졌다**(오프라인 h100 비 #27 .58 → .20~.35, 이어 학습 자체 효과 + pad_hold 가 더함): t08 은 #27 대체 후보로 쓰지 말고 #58 과의 비교로만. 우선 2(1순위 회차 뒤). #53·#54·#55 는 public·채점 끝(t08 출발: #53 .58 → #54 1.05).
> **추가·정정(10/15 01:40 UTC, 사용자 「오프라인 결론으로 닫힌 것 중 한 번에 돌려야 하는 것 없는지 다시 확인」)** — 빠져 있던 것을 넣었다. ① **실행 층 기본값(블렌딩 켬)도 실기 3 대 3 이 없었다**(10/13 1052 끔 ↔ 1053 켬 각 1 회, 둘 다 실패, 자리 A?, 동작 지표로만 정함) → t03(#2 가 10/13 1139 에 성공한 단계)에서 **#2 블렌딩 켬 ↔ 끔 번갈아 각 3 회**를 2순위에(같은 날·같은 자리에서 짝이 되도록 1순위 회차와 섞지 않는다, O16). exec 60·다듬기 w=3 도 실기 비교 없이 오프라인으로 정한 값이다 — 이번엔 블렌딩만. ② **흔든 시작(B) 시험이 어디에도 없었다** — 목표가 「흔든 시작에서 70%」 인데 「자리 어긋남을 보정하는 모델은 없다」 는 [추정]이고(model_selection §1), 오프라인 프록시는 변별력이 없다(sim §94.42). t01 #27 · t02 #40 · t03 #2 에 A✓ 에서 베이스만 5 cm / 10° 옮긴 회차를 2순위로(O17). ③ **t01 랙 튜브 장면** — 10/11 #2 성공 회차는 랙이 비어 있었고, 튜브 있는 장면에서 82° 지점 남은 회전 예측이 더 크다(10/15 재예측) → t01 「랙 빈」 #2 2 회(O18). ④ **#51(t02 단계 전문가, 같은 레시피)을 1순위 3 회로** — 모델 행은 「각 3 회」 인데 표는 2순위 2 회였다(불일치). 오프라인 t02 출발 GIST ho .98 vs #2 .70 으로 가장 강한 신호다. #52 는 2순위 3 회. ⑤ #50 을 t03 2순위에 2 회(위 23:30 줄에서 말만 하고 행에 없었다). ⑥ #53·#54(1호기 ≈10/15 09:20 UTC 끝)를 **허브에 있으면** t08 3 대 3 · t07 1 회로. ⑦ GIST 50% #31 을 t01·t02 스크리닝 1 회씩(O3 의 혼합 비율). ⑧ 2a 행의 「O16」 은 O15 의 오기 — 고쳤다. ⑨ **SmolVLA(#36·#40) 는 실행 층이 다르다** — `eval_smolvla.sh rec` 는 자체 commit 30 + fade 5 + 4 샘플 앙상블이고 블렌딩·다듬기 변수를 쓰지 않는다. t02 C1′(#40 ↔ 전문가 ↔ #2)은 「모델 + 그 모델의 실행 층」 비교로 읽는다. 블렌딩 끔 회차는 `LEROBOT_CHUNK_SEAM_BLEND_TICKS=0` 을 명령에 명시했다(셸에 켜 둔 값이 있어도 끈다).
> **추가(10/15 06:00 UTC, 사용자 「Eval 리스트에 올려두고」)** — **t05 전문가 exec 30 ↔ 60 ↔ 100 번갈아 각 3 회를 t05 2순위 맨 앞에 넣었다.** 묻는 것: 조작은 길게 실행하는 쪽이 유리한가. 지금 근거는 같은 모델의 e30 1/3 · e100 1/1(10/02 · 10/06, 블렌딩·다듬기 전, 1~3 회)뿐이고, 60 은 실기 비교 없이 정했다. 이동(t01 #27 e30 ↔ e60)과 짝을 이뤄 「이동은 출발 길게 · 도착 짧게, 조작은 길게」 가설을 확인한다. 기록할 것: 버리기 성공(둘/하나), 청크 경계 움찔, 중간 멈춤에 갇혔는지. 2순위 77 → 86 회 ≈ 9.1 h.
> **추가(10/15 05:40 UTC, 사용자 「exec 30 ↔ 60 t01 2순위로 올리고」)** — **#27 exec 30 ↔ exec 60 번갈아 3 대 3 을 t01 2순위 맨 앞에 넣었다.** 같은 ep30 자리 + 바닥 표시에서 한다. 묻는 것: exec 60 은 82° 에 도달한 뒤에도 최대 2.9 s 동안 묵은 청크를 그대로 실행한다(꼬리). 실측 꼬리는 +2~+20°, 대개 +11° 이고, 새 청크로 더 돈 큰 과회전과는 따로 갈랐다(DGX 분석). exec 30 에서 꼬리와 실측 도착각이 줄어드나, 출발은 막히지 않나(exec 5 는 거의 정지했다). 기록할 것: 82° 도달 시각 · 도달 뒤 회전(명령/실측) · 끝 각도 · 처음 3 s 회전·전진 · 정지 여부. 근거: [1015 DGX 분석](eval_najy_results_1015.md). 2순위 71 → 77 회 ≈ 8.1 h. (t01 2순위 칸의 「(12)」 는 전문가를 1순위로 옮긴 뒤 11 이었다 — 이번에 17 로 고쳤다.)
> **추가(10/15 04:30 UTC, 사용자 「일단 올려두고」)** — **t01 전문가 `task01_move_to_tube_rack_14D` ↔ #27 번갈아 3 대 3 을 t01 1순위로 올렸다**(2순위에서 뺐다). 근거는 실기 출발 프레임 재예측이다: 전문가는 첫 3 s 회전 +45~59°·전진 0 이고 어긋난 자리에서도 유지했다. 11단계 #27 은 5~21° 다([1015 DGX 분석](eval_najy_results_1015.md)). 묻는 것: 단일 단계가 도착(각도·거리·정지)도 더 정확한가. 기록할 것: 도착각·거리·정지 여부·다음 단계로 넘어가는지. **#1 은 남은 회차에서 뺐다**(출발 못 함 2/2, 재예측에서도 출발 경계에 걸림). t01 1순위 남은 회차 10 회. 모두 재생 ep30 자리 + 바닥 표시에서 한다.
> **정정(10/15, 사용자 「흔든 시작 일단 스킵하자」)** — ② 의 흔든 시작(B) 7 회를 표에서 뺐다. O17 은 실기 미확인으로 남는다. 2순위 78 → 71 회.
> **사용자(10/14)**: 「오프라인으로 결론 닫고 왔는데 실험해 보면 좋을 것들도 정리해서 Eval 한 번에 진행하자. Trossen PC 에 전달해 줘」. 아래 두 절(「먼저 돌릴 것」 의 남은 2b·3·5·8 과 「10/15~ 비교 라운드」)을 **이 표 하나로 합쳤다**. 그 두 절은 비교 정의(C1~C10)·지난 명령의 기록으로만 둔다.
> **방식**: 태스크 순서(t01 → t11)를 지킨다. 행마다 **1순위**를 먼저 돌고, 시간이 남으면 **2순위**. 1순위 85 회 ≈ 9 h, 2순위 53 회 ≈ 5.5 h (**10/15 01:40 UTC 갱신 → 1순위 88 회 ≈ 9.3 h · 2순위 78 회 ≈ 8.3 h**, 아래 정정 줄 · **같은 날 다시: 흔든 시작 스킵 → 2순위 71 회 ≈ 7.5 h**) — 하루에 1순위, 다음 날 2순위가 현실적이다. **(10/14 18:10 UTC 바꿈: 사용자 「속도 입력은 실기 비교해야 하는 것 아님?」 → #49(C4)를 t01·t03 에서 #2 와 각 3 회, #50(C2′)을 t05 에서 각 3 회로 1순위에 올림)** 「확정」 비교는 같은 자리에서 번갈아 각 3 회(판정: 성공 수 2 이상 차이).
> **공통**: 1 ep 씩 · 처음 `DRY_RUN=1` 한 번 · e-stop 대기 · 먼저 `git pull origin main`. ACT 는 블렌딩 켬·exec 60·w=3(**#25·#9 만 w=5** — 명령에 이미 들어 있다). 단계마다 재생으로 A✓ 를 먼저 맞춘다(t01 은 재생 자가 없어 바닥 테이프).
> **새 모델 확인 줄**: #40 은 로드 로그 `[action_query] model.action_queries (64) 를 붙여 싣는다` 가 없으면 중단. **#49**(베이스 속도 입력)는 `base slots = measured velocity`, **#50·#51·#52**(원핫 없음 16D)는 `zero-pad only` 가 로그에 없으면 중단. #49·#50 은 1호기 학습이 ≈10/14 22:40 UTC 에 끝나 허브에 올라간 뒤(아침에 허브 확인), #51·#52 는 2호기 학습 뒤 — **허브에 없으면 건너뛴다**.
> **10/14 23:30 UTC 추가 — #49·#50 허브 공개 확인(예정보다 일찍), 아침에 바로 돌릴 수 있다.** 출발 첫 청크 채점(DGX1, 시연 첫 프레임에서 100 스텝 예측의 베이스 θ 합, 예측/시연 rad; #2 는 허브와 sha 일치한 60K 가중치로 같은 조건 재채점): 우리 홀드아웃 t01 #2 +0.31 · #49 +0.31 · #50 +0.35 (시연 +0.30) / t03 #2 −0.13 · #49 −0.11 · #50 −0.22 (−0.11) · GIST 홀드아웃 t01 #2 +0.19 · #49 +0.20 · #50 +0.27 (+0.17) / t03 #2 −0.20 · #49 −0.21 · #50 −0.06 (−0.33) / t10 #2 −0.34 · #49 −0.31 · #50 −0.40 (−0.35). → **#49 는 출발에서 #2 와 구분되지 않는다** — 이 채점은 시연 첫 프레임(베이스 속도 0)만 보므로, 속도 입력의 걱정(움직이는 중에 state 속도 = 직전 명령을 베끼는 것)은 **실기에서만 보인다**. C4 3 대 3 이 그 시험이다. **#50 은 회전 출발이 #2 와 다르다**(t03 GIST 에서 덜 돌고, t01 에서 더 돈다) — 원핫이 이동 단계 출발에 쓰인다는 신호(오프라인, 1회 채점). 실기 C2′ 는 t05 에서 하지만, 시간이 남으면 #50 을 t03 에도 1~2 회.

**오프라인으로만 닫고 온 결론 — 이번에 실기로 확인하는 곳**

| # | 오프라인 결론 (어디서) | 실기 확인 | 행 |
|---|---|---|---|
| O1 | 원핫은 단계를 고르지 않는다 — 모델은 장면으로 고른다 (sim §89, 10/06) · 원핫은 성능 레버가 아니다 | t05 같은 장면에서 HOT 4/11 vs 5/11 (#2·#25) · **원핫 없는 #50 ↔ #2 t05 각 3 — 1순위** | 5 · 1 · 2a |
| O2 | 베이스 state 는 14D(0) — 속도를 넣으면 베끼고, odom 은 0 과 같다 (S2-15·16, 9/28) | **#49(속도 입력) ↔ #2 t01·t03 각 3 — 1순위**. **odom 은 학습된 모델이 없어 이번엔 못 한다** | 1 · 3 |
| O3 | GIST 30% 고정 혼합이 득 (§94.38) | #2 ↔ #1(GIST 없음) t01·t03·t10 | 1 · 3 · 10 |
| O4 | GIST 를 이동 단계에만 섞는 레시피 (J36) | #32 | 1 · 3 · 10 |
| O5 | 레시피 차이보다 시드 폭이 크다 (§94.37) | GIST 셋째 시드 #26 (t02 낮게 겨냥 쪽) · M1 두 시드 #4 ↔ #1 | 2a · 1 |
| O6 | 청크 150 은 t01 기다림을 담지만 떨림 1.2~1.5× (§94.31~37) | #4 ↔ #9 | 1 · 5 |
| O7 | lr cosine 이 조작에 득, t08 .60 → .90 (J39) | #39 ↔ #2 확정 | 8 · 2a |
| O8 | GIST 저품질 제외로 떨림 ↓ (J35) | #29 | 2순위 전 단계 |
| O9 | 120K 가 예측 떨림 30~50% ↓ (J30) | #27 — **이동 1순위 후보로** | 1 · 3 · 6 · 10 |
| O10 | 영상 드롭아웃 이득 없음, 원핫 경로는 열림 (J28) | #25 원핫 시험 | 5 |
| O11 | 11단계 한 모델이 단계 전문가와 같거나 낫다 (§83·§85) | 팀 전문가 · 같은 설정 전문가 **#51 ↔ #2 t02 각 3 — 1순위** · #52 t05 3(2순위) | 2a · 5 · 전 단계 2순위 |
| O12 | SmolVLA 는 ACT 와 같거나 진다 (§74, 9월) — **10/15 t02 에서 뒤집힘** | #40 을 조작 단계 전부(t02·04~09·11)에 · t02 확정 | 2a · 4~9 · 11 |
| O13 | 쿼리 토큰은 오프라인 같은 띠 · 10/15 재예측 #36 = #40 | #36 ↔ #40 t02 각 3 | 2a |
| O14 | 빈손으로 출발하면 t03 회전이 느려진다 (10/13 재예측) | t03 튜브 쥐고 출발 | 3 |
| O15 | 11단계 ACT 는 t02 왼손을 1~4 cm 높게 겨냥, #1 만 예외 (10/15 재예측) | #1 ×2 · #26 | 2a |
| O16 | 경계 블렌딩이 득 — 경계 튐 18× → 0.8×, 방향 바뀜 0.46 → 0.07/s (10/13 동작 지표, 성공 비교 없음) | **t03 #2 블렌딩 켬 ↔ 끔 번갈아 각 3 (2순위)** | 3 |
| O17 | 지금 모델은 자리 어긋남을 카메라로 보정하지 못한다 [추정] (model_selection §1 · 오프라인 프록시 변별력 없음 §94.42) | **이번엔 스킵**(사용자 10/15 「흔든 시작 일단 스킵하자」) — 미확인으로 남긴다 | — |
| O18 | t01 도착 뒤 과회전의 일부는 장면(랙 튜브) 탓 (10/15 재예측 — 10/11 성공은 랙 빈 장면) | t01 #2 「랙 빈」 2 회 | 1 |
| 새 | 이동은 ACT, 조작은 SmolVLA 로 단계를 나누면 되나 (10/14 논의) | #27 t01 → (베이스 그대로) 재생 ep102 → #40 t02 | 1→2 |

**태스크 순서 표**

| 순서 | 태스크 | 검증하려는 것 | 장면·자리 | 1순위 회차 (이 순서로) | 2순위 (시간 남으면) | 명령 | 결과 → 다음 |
|---|---|---|---|---|---|---|---|
| **1** | **task01 랙으로 이동** | **이동 1순위 #27 재현**(10/15 1/1) · **C4 베이스 속도 입력(#49 ↔ #2 각 3, O2)** · C6 GIST(#2 ↔ #1, O3) · C3(#39) · ○ C1(#36) · C7 청크 150(#4 ↔ #9, O6) · C6′ 이동만 GIST(#32, O4) · C2′ 원핫 없음(#50, O1) · C5 전문가 · **랙 튜브 장면(O18)** · GIST 50%(#31) | t01 출발 자리(재생 자 없음 — **바닥에 테이프로 출발 자리를 표시**해 매 회차 같게). 랙에 튜브 두 개(시연과 같게). **「랙 빈」 회차만 튜브를 빼고** ~~#27 → #2 → #49 → #1 → #39 → #27 → #2 → #49 → #1 → #39 → #27 → #2 → #49 → #1 → #39 (15)~~ **10/15 오전 6 회 끝(1045~1053). 남은 것(10/15 04:30 UTC 바꿈): 전문가 → #27 → #2 → 전문가 → #27 → #2 → 전문가 → #27 → #39 → #49 (10)** — 전문가 ↔ #27 번갈아 3 대 3 · #2 는 오늘 1 회 + 2 회 = 3 · #1 은 뺌 | **#27 e30 → #27 e60 → #27 e30 → #27 e60 → #27 e30 → #27 e60 (10/15 추가, 꼬리 과회전)** → #36 → #36 → #4 → #9 → #4 → #9 → #32 → #50 → #2 랙 빈 → #2 랙 빈 → #31 (17) | #27: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=1/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_120k_s1000 1 60 1` · #27 e30: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=1/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_120k_s1000 1 30 1` · #2: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=1/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 1 60 1` · #49: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=1/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_vel_27D_60k_s1000 1 60 1` · #1: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=1/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_m1_27D_60k_s2000 1 60 1` · #39: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=1/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_cos_27D_60k_s1000 1 60 1` · #36: `POLICY=kiroaiseoul/NAJY_smolvla_all11_gist30_fp32_60k_s1000 scripts/eval_smolvla.sh 1 rec 1` · #4: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=1/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_m1_27D_60k_s1000 1 60 1` · #9: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=1/11 LEROBOT_CHUNK_SMOOTH_TICKS=5 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c150_m1_27D_60k_s1000 1 60 1` · #32: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=1/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30mv_sw_27D_60k_s1000 1 60 1` · 전문가: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/task01_move_to_tube_rack_14D 1 60 1` · #50: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_nohot_16D_60k_s1000 1 60 1` · #31: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=1/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist50_27D_60k_s1000 1 60 1` | 도착·정지(시연 +82° / 1.64 m 근처에서 멈춤)면 ✓, 도착 뒤 더 돈 각도 기록. **#49 가 #2 보다 2 회 이상 많으면 베이스 속도 입력 채택**(오프라인 「베낀다」 를 뒤집음), 적으면 0 유지 확정. #49 가 아침에 아직 허브에 없으면 #49 3 회만 t01 끝에 몰아 돈다. **#27 이 3 회 중 2 회 이상 ✓ 이면 이동 모델 = #27 로 정하고 t03·t06·t10 도 #27 우선**. #1 이 #2 보다 2 회 이상 적으면 GIST 혼합 유지 확정. **「랙 빈」 #2 가 덜 돌고 멈추면 10/15 의 +54° 는 장면(랙 튜브) 탓이 크다**(10/11 성공 회차는 랙이 비어 있었다) |
| **1→2** | **연결 — 이동은 ACT, 조작은 SmolVLA** (새로) | **하이브리드가 되나** — #27 이 t01 에서 멈춘 자리에서 SmolVLA #40 이 t02 를 쥐나. 그 자리가 A✓ 인지는 재생으로 따로 잰다(자리 문제와 모델 문제를 가른다) | 위 t01 의 **#27 회차 3 개 각각 바로 뒤에** 한다. **베이스는 손대지 않는다**(리셋 동안 리더 팔만 움직이고 베이스는 안 움직이는지 첫 회에 확인). 튜브는 랙 시연 구멍 | (#27 t01 끝난 자리) 재생 ep102 → 튜브 다시 꽂기·팔 리셋 → #40 t02 (×3 = 9, 그중 #27 t01 3 회는 위 행에서 셈 → 추가 6) | — | 재생: `LEROBOT_REPLAY_DATASET=kiroaiseoul/task02_pickup_tubes LEROBOT_REPLAY_EPISODES=102 LEROBOT_REPLAY_RUN_POLICY=0 RUN_SUFFIX=replay_ep102 HOT=2/11 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 2 60 1` · #40: `POLICY=kiroaiseoul/NAJY_smolvla_all11_gist30_fp32_aq64_60k_s1000 scripts/eval_smolvla.sh 2 rec 1` | 재생이 양손 쥐면 그 자리는 A✓ — 그때 #40 이 쥐면 **하이브리드 가능**(이동 = #27, 조작 = SmolVLA 로 단계 나누기 설계 시작). 재생이 못 쥐면 **이동이 자리를 못 맞춘 것** → 이동 쪽은 자리 다양화 녹화·마커 도킹으로 간다. 3 회 반복 = 9 회차 |
| **2a** | **task02 튜브 집기 (시연 튜브)** | **C1′ #40 성공이 우연인가 · 쿼리 효과인가(#36, O13)** · C5 전문가 · C3(#39) · **C5′ 같은 레시피 단계 전문가 #51(오프라인 t02 출발 GIST ho .98 vs #2 .70)** · **왼손 높이(#1 ×2 · #26, O15)** · ○ C2′(#50) · C4(#49) · GIST 50%(#31) | 재생으로 A✓ 를 먼저: `LEROBOT_REPLAY_DATASET=kiroaiseoul/task02_pickup_tubes LEROBOT_REPLAY_EPISODES=102 LEROBOT_REPLAY_RUN_POLICY=0 RUN_SUFFIX=replay_ep102 HOT=2/11 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 2 60 1`. 튜브는 시연 구멍, 매 회차 같은 구멍 | #40 → 전문가 → #36 → #2 → #39 → #51 → #40 → 전문가 → #36 → #2 → #39 → #51 → #40 → 전문가 → #36 → #2 → #39 → #51 → #1 → #1 (20) | #26 → #50 → #49 → #31 (4) | #40: `POLICY=kiroaiseoul/NAJY_smolvla_all11_gist30_fp32_aq64_60k_s1000 scripts/eval_smolvla.sh 2 rec 1` · 전문가: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/task02_pickup_tubes_14D 2 60 1` · #36: `POLICY=kiroaiseoul/NAJY_smolvla_all11_gist30_fp32_60k_s1000 scripts/eval_smolvla.sh 2 rec 1` · #2: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=2/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 2 60 1` · #39: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=2/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_cos_27D_60k_s1000 2 60 1` · #51: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_task02_c100_gist30_solo_16D_60k_s1000 2 60 1` · #1: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=2/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_m1_27D_60k_s2000 2 60 1` · #26: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=2/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s3000 2 60 1` · #50: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_nohot_16D_60k_s1000 2 60 1` · #49: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=2/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_vel_27D_60k_s1000 2 60 1` · #31: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=2/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist50_27D_60k_s1000 2 60 1` | 양손 쥐면 ✓, 왼손만 +p. **#40 이 #2 보다 2 회 이상 많으면 t02 = SmolVLA**. #36 도 #40 만큼 쥐면 쿼리는 이득 없음(SmolVLA 자체와 자리 덕). #1 이 왼손을 쥐면 「높게 겨냥」 이 11단계 ACT 실패 원인으로 굳는다. **#51 이 #2 보다 2 회 이상 많으면 「11단계 한 모델 ≥ 단계 전문가」(O11)가 t02 에서 뒤집힌다** |
| 2b | task02 (투명 튜브) | 투명 튜브를 더 배운 #41 이 #2 보다 잘 잡나 (10/15 에서 못 돈 것) | 같은 A✓, 투명 튜브를 시연 구멍에 | — | #2 → #41 → #2 → #41 → #40 (5) | #2: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=2/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 2 60 1` · #41: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=2/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_t02il_27D_60k_s1000 2 60 1` · #40: `POLICY=kiroaiseoul/NAJY_smolvla_all11_gist30_fp32_aq64_60k_s1000 scripts/eval_smolvla.sh 2 rec 1` | #41 이 쥐고 #2 가 못 쥐면 투명 튜브 녹화 추가. #40 은 투명 튜브에서도 쥐나 |
| **3** | **task03 제자리 회전** | **이동 후보 #27** · **C4 베이스 속도 입력(#49 ↔ #2 각 3)** · C6 GIST(#2 ↔ #1) · 튜브 쥐고 출발(O14) · **블렌딩 켬/끔 3 대 3(O16 — 2순위 안에서 #2 켬 ↔ 끔 번갈아)** · 원핫 없음 #50(오프라인 회전 출발 −0.06 vs #2 −0.20) · ○ C3 · C9 · C6′ · C1(#36) · C5 | **양손에 튜브를 쥐어 준 채** 출발. 재생으로 A✓: `LEROBOT_REPLAY_DATASET=kiroaiseoul/task03_turn_to_face_beaker LEROBOT_REPLAY_EPISODES=79 LEROBOT_REPLAY_RUN_POLICY=0 RUN_SUFFIX=replay_ep79 HOT=3/11 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 3 60 1` | #2 → #27 → #49 → #1 → #2 → #27 → #49 → #1 → #2 → #27 → #49 → #1 (12) | #2 → #2 블렌딩 끔 → #2 → #2 블렌딩 끔 → #2 → #2 블렌딩 끔 → #50 → #50 → #39 → #29 → #32 → #36 → 전문가 (13) | #2: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=3/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 3 60 1` · #27: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=3/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_120k_s1000 3 60 1` · #49: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=3/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_vel_27D_60k_s1000 3 60 1` · #1: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=3/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_m1_27D_60k_s2000 3 60 1` · #50: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_nohot_16D_60k_s1000 3 60 1` · #39: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=3/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_cos_27D_60k_s1000 3 60 1` · #29: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=3/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_dq_27D_60k_s1000 3 60 1` · #32: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=3/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30mv_sw_27D_60k_s1000 3 60 1` · #36: `POLICY=kiroaiseoul/NAJY_smolvla_all11_gist30_fp32_60k_s1000 scripts/eval_smolvla.sh 3 rec 1` · 전문가: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/task03_turn_to_face_beaker_14D 3 60 1` · 블렌딩 끔: #2: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=0 HOT=3/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 3 60 1` | 비커 정면에서 멈추면 ✓, 회전각 기록(시연 ≈ −90°). #27 이 2 회 이상 ✓ 이면 이동 모델 #27 확정 쪽. **블렌딩 끔 3 회가 켬 3 회보다 2 회 이상 적으면 블렌딩 켬 확정**, 같으면 「동작 지표만의 이득」 으로 남긴다. #50 이 덜 돌면 원핫이 이동 출발에 쓰인다(O1 의 예외) |
| 4 | task04 붓기 | 청크 100 첫 실기 · **SmolVLA 조작(#40)** · C3 · C9 · C5 | 양손에 튜브를 쥐어 준다. 재생으로 A✓: `LEROBOT_REPLAY_DATASET=kiroaiseoul/task04_pour_liquid_from_tubes_to_beaker LEROBOT_REPLAY_EPISODES=96 LEROBOT_REPLAY_RUN_POLICY=0 RUN_SUFFIX=replay_ep96 HOT=4/11 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 4 60 1` | #2 → #40 (2) | #39 → #29 → 전문가 (3) | #2: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=4/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 4 60 1` · #40: `POLICY=kiroaiseoul/NAJY_smolvla_all11_gist30_fp32_aq64_60k_s1000 scripts/eval_smolvla.sh 4 rec 1` · #39: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=4/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_cos_27D_60k_s1000 4 60 1` · #29: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=4/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_dq_27D_60k_s1000 4 60 1` · 전문가: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/act_task04_pour_liquid_from_tubes_to_beaker_60000 4 60 1` | 붓는 동작까지 가면 ✓. #40 이 되면 「조작 = SmolVLA」 에 t04 를 넣는다 |
| **5** | **task05 튜브 버리기** | **C2 원핫이 단계를 고르나(O1 — 원핫 4/11 vs 5/11)** · **C2′ 원핫 없는 #50 ↔ #2 각 3** · C5 전문가 · **SmolVLA 조작(#40)** · ○ C3 · C9 · C10(#25) · C7 · C5′(#52) | 시연 자리, 양손에 튜브. 재생으로 A✓: `LEROBOT_REPLAY_DATASET=kiroaiseoul/task05_tube_disposal LEROBOT_REPLAY_EPISODES=69 LEROBOT_REPLAY_RUN_POLICY=0 RUN_SUFFIX=replay_ep69 HOT=5/11 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 5 60 1`. 원핫 시험 회차는 장면·자리 그대로 HOT 만 4/11 | #2 → 전문가 → #50 → #2 → 전문가 → #50 → #2 → 전문가 → #50 → #40 → #2 HOT=4/11 → #2 HOT=4/11 → #25 → #25 HOT=4/11 (14) | **전문가 e30 → e60 → e100 → e30 → e60 → e100 → e30 → e60 → e100 (10/15 추가, 조작 실행 길이)** → #39 → #29 → #27 → #36 → #4 → #9 → #52 → #52 → #52 (18) | #2: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=5/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 5 60 1` · 전문가: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/act_task05_tube_disposal_60000 5 60 1` · 전문가 e30: `… act_task05_tube_disposal_60000 5 30 1` · 전문가 e100: `… act_task05_tube_disposal_60000 5 100 1` (앞부분 환경변수는 e60 과 같다) · #50: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_nohot_16D_60k_s1000 5 60 1` · #40: `POLICY=kiroaiseoul/NAJY_smolvla_all11_gist30_fp32_aq64_60k_s1000 scripts/eval_smolvla.sh 5 rec 1` · #25: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=5/11 LEROBOT_CHUNK_SMOOTH_TICKS=5 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_drop_27D_60k_s1000 5 60 1` · #39: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=5/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_cos_27D_60k_s1000 5 60 1` · #29: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=5/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_dq_27D_60k_s1000 5 60 1` · #27: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=5/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_120k_s1000 5 60 1` · #36: `POLICY=kiroaiseoul/NAJY_smolvla_all11_gist30_fp32_60k_s1000 scripts/eval_smolvla.sh 5 rec 1` · #4: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=5/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_m1_27D_60k_s1000 5 60 1` · #9: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=5/11 LEROBOT_CHUNK_SMOOTH_TICKS=5 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c150_m1_27D_60k_s1000 5 60 1` · #52: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_task05_c100_gist30_solo_16D_60k_s1000 5 60 1` · 원핫 시험: #2: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=4/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 5 60 1` · #25: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=4/11 LEROBOT_CHUNK_SMOOTH_TICKS=5 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_drop_27D_60k_s1000 5 60 1` | 둘 다 트레이에 떨어뜨리면 ✓, 하나면 +p. **HOT=4/11 에서도 버리면 「원핫은 단계를 못 고른다」 실기 확정**(4/11 에서 붓기 동작이면 원핫이 듣는다). #25 는 4 와 5 에서 다르게 움직이나 |
| 6 | task06 비커 잡고 냉장고로 (조작 → 이동) | **이동 후보 #27** · 잡기와 이동이 순서대로 섞인 단계 — 어느 쪽에서 실패하나 · **#40 잡기** · C3 · C9 · C5 | 시연 자리. 재생으로 A✓: `LEROBOT_REPLAY_DATASET=kiroaiseoul/task06_task06-2_pickup_beaker_and_move_to_refrigerator_14D_new_2 LEROBOT_REPLAY_EPISODES=93 LEROBOT_REPLAY_RUN_POLICY=0 RUN_SUFFIX=replay_ep93 HOT=6/11 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 6 60 1` | #2 → #27 → #40 (3) | #39 → #29 → 전문가 (3) | #2: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=6/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 6 60 1` · #27: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=6/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_120k_s1000 6 60 1` · #40: `POLICY=kiroaiseoul/NAJY_smolvla_all11_gist30_fp32_aq64_60k_s1000 scripts/eval_smolvla.sh 6 rec 1` · #39: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=6/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_cos_27D_60k_s1000 6 60 1` · #29: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=6/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_dq_27D_60k_s1000 6 60 1` · 전문가: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/act_task06_task06-2_pickup_beaker_and_move_to_refrigerator_14D_new_2 6 60 1` | 잡기 성공·이동 도착을 **나눠 기록**(잡기 ✓/✗ · 도착 ✓/✗). #40 이 잡고 이동에서 실패하면 t06 을 잡기/이동 둘로 나누는 근거 |
| 7 | task07 냉장고 열기 | 첫 실기 · **#40 조작** · C3 · C9 · C5 | 시연 자리. 재생으로 A✓: `LEROBOT_REPLAY_DATASET=kiroaiseoul/task07_open_refrigerator LEROBOT_REPLAY_EPISODES=184 LEROBOT_REPLAY_RUN_POLICY=0 RUN_SUFFIX=replay_ep184 HOT=7/11 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 7 60 1` | #2 → #40 (2) | #39 → #29 → 전문가 → #53 (4) | #2: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=7/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 7 60 1` · #40: `POLICY=kiroaiseoul/NAJY_smolvla_all11_gist30_fp32_aq64_60k_s1000 scripts/eval_smolvla.sh 7 rec 1` · #39: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=7/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_cos_27D_60k_s1000 7 60 1` · #29: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=7/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_dq_27D_60k_s1000 7 60 1` · 전문가: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/act_task07_open_refrigerator_14D_260916 7 60 1` · #53: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=7/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_basefix_27D_60k_s1000 7 60 1` | 문이 열리면 ✓. #53 은 t07 학습분에서 베이스 미기록 67 ep 를 뺀 판 — **허브에 있으면**(1호기 ≈10/15 09:20 UTC 끝, 18:20 KST 무렵) 1 회 |
| **8** | **task08 비커 꺼내 놓기** | **C3 확정(#39 — 오프라인 t08 .60→.90, O7)** · **#40 조작** · ○ C9 · C5 · 베이스 미기록 제외 #53 ↔ 새 녹화 #54(허브에 있으면) | 시연 자리. 재생으로 A✓: `LEROBOT_REPLAY_DATASET=kiroaiseoul/task08_takeout_and_put_beaker LEROBOT_REPLAY_EPISODES=157 LEROBOT_REPLAY_RUN_POLICY=0 RUN_SUFFIX=replay_ep157 HOT=8/11 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 8 60 1` | #2 → #39 → #2 → #39 → #2 → #39 → #40 (7) | #29 → 전문가 → #53 → #54 → #53 → #54 → #53 → #54 (8) | #2: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=8/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 8 60 1` · #39: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=8/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_cos_27D_60k_s1000 8 60 1` · #40: `POLICY=kiroaiseoul/NAJY_smolvla_all11_gist30_fp32_aq64_60k_s1000 scripts/eval_smolvla.sh 8 rec 1` · #29: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=8/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_dq_27D_60k_s1000 8 60 1` · 전문가: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/act_task08_takeout_and_put_beaker_100k_251data 8 60 1` · #53: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=8/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_basefix_27D_60k_s1000 8 60 1` · #54: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=8/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_t08add_27D_60k_s1000 8 60 1` | 놓으면 ✓. #39 가 #2 보다 2 회 이상 많으면 lr cosine 채택. **#53 ↔ #54 는 허브에 있을 때만**(베이스 미기록을 빼 t08 이 절반이 된 판 vs 새 녹화 72 ep 로 채운 판) — #54 가 2 회 이상 많으면 t08 새 녹화를 다음 레시피에 넣는다 |
| 9 | task09 냉장고 닫기 (닫고 → 물러남) | 첫 실기 · **#40 조작** · 닫기와 물러나기 나눠 보기 · C3 · C9 · C5 | 시연 자리. 재생으로 A✓: `LEROBOT_REPLAY_DATASET=kiroaiseoul/task09_close_refrigerator LEROBOT_REPLAY_EPISODES=174 LEROBOT_REPLAY_RUN_POLICY=0 RUN_SUFFIX=replay_ep174 HOT=9/11 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 9 60 1` | #2 → #40 (2) | #27 → #39 → #29 → 전문가 (4) | #2: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=9/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 9 60 1` · #40: `POLICY=kiroaiseoul/NAJY_smolvla_all11_gist30_fp32_aq64_60k_s1000 scripts/eval_smolvla.sh 9 rec 1` · #27: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=9/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_120k_s1000 9 60 1` · #39: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=9/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_cos_27D_60k_s1000 9 60 1` · #29: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=9/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_dq_27D_60k_s1000 9 60 1` · 전문가: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/act_task09_close_refrigerator_100k_250data 9 60 1` | 닫기 ✓/✗ · 물러남 ✓/✗ 를 나눠 기록 |
| **10** | **task10 선반으로 이동** | **이동 후보 #27** · C6 GIST(#1) · C6′(#32) · ○ C3 · C9 · C5 | 시연 자리, 비커를 쥐어 준다. 재생으로 A✓: `LEROBOT_REPLAY_DATASET=kiroaiseoul/task10_move_to_beaker_shelf LEROBOT_REPLAY_EPISODES=88 LEROBOT_REPLAY_RUN_POLICY=0 RUN_SUFFIX=replay_ep88 HOT=10/11 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 10 60 1` | #2 → #27 → #1 (3) | #32 → #39 → #29 → 전문가 (4) | #2: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=10/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 10 60 1` · #27: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=10/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_120k_s1000 10 60 1` · #1: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=10/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_m1_27D_60k_s2000 10 60 1` · #32: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=10/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30mv_sw_27D_60k_s1000 10 60 1` · #39: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=10/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_cos_27D_60k_s1000 10 60 1` · #29: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=10/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_dq_27D_60k_s1000 10 60 1` · 전문가: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/act_task10_move_to_beaker_shelf_14D_100k_fp32 10 60 1` | 선반 앞에서 멈추면 ✓, 더 돈 각도 기록 |
| 11 | task11 비커 선반에 놓기 | 첫 실기 · **#40 조작** · C3 · C9 (전문가 없음) | 시연 자리. 재생으로 A✓: `LEROBOT_REPLAY_DATASET=kiroaiseoul/task11_place_beaker_on_the_shelf LEROBOT_REPLAY_EPISODES=98 LEROBOT_REPLAY_RUN_POLICY=0 RUN_SUFFIX=replay_ep98 HOT=11/11 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 11 60 1` | #2 → #40 (2) | #39 → #29 (2) | #2: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=11/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 11 60 1` · #40: `POLICY=kiroaiseoul/NAJY_smolvla_all11_gist30_fp32_aq64_60k_s1000 scripts/eval_smolvla.sh 11 rec 1` · #39: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=11/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_cos_27D_60k_s1000 11 60 1` · #29: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=11/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_dq_27D_60k_s1000 11 60 1` | 놓으면 ✓ |

- **기록**: 회차마다 맨 아래 「결과 기록」 에 한 줄(태스크 순서), 상세는 `eval_najy_results_1015.md`(또는 날짜 문서) → `git push origin main`. 6·9 행은 **잡기/이동(닫기/물러남)을 나눠** 적는다 — 섞인 단계를 둘로 나눌지 정하는 근거다.
- **안 하는 것** (10/15 01:40 UTC 다시 씀 — 학습한 모델은 전부 이유를 단다): state 드롭아웃 #43~#48(A✓ 에서 이득 없음, 10/14 선정) · **#19** J24 우리 데이터 품질 필터(M1 레시피, 오프라인 A·J10 과 같은 자리 §94.40 — 기준이 GIST 레시피로 바뀌어 그 정제판 #29 가 대신 돈다) · **#33** 통계만 바꾼 판(오프라인 통계 효과 0, §94.54) · **#7** 기하 증강(증강은 60K 에서 이득 없음 §94.37, 실기 t02 에서도 시작 폭 안 넓힘 §94.28) · **#14** M3 이동 4단계 묶음(청크 30 — 정지 시작에서 못 빠져나온다 §94.31; 이동 후보는 #27) · **#10·#11·#20** 청크 150 의 시드·진행도 판(C7 #4 ↔ #9 결과 뒤) · **#3·#8·#24·#37** 진행도 칸 17D(체인 러너 전용, 「거짓 완료」 §94.53 — 체인 bring-up 라운드에서) · **#21~#23** 진행도 없는 드롭아웃(#3 과 짝, 같은 라운드) · **#5·#6·#30·#34·#38·#42·#46·#47** 시드 짝(짝 모델이 실기에서 이긴 뒤) · **#12·#13·#35** 옛 SmolVLA(#12 bf16 갱신 소실 §94.49, #35 언프리즈 — #36·#40 이 대신) · **#55** #53 시드 짝(2호기 학습 중) · odom 입력(학습 모델 없음) · exec 60·다듬기 w=3 의 실기 3 대 3(이번엔 블렌딩만 — 다음 라운드 후보).

## ▶ 먼저 돌릴 것 — 실기 대기 모델 첫 실기, 태스크 순서 (DGX1 → Trossen PC1, 2026-10-14 15:10 UTC) · 10/15 첫 세션
> **⛔ 10/16 정비로 대체됨 — 이 절의 명령은 돌리지 마라. 맨 위 「▶ 10/16 정비」 절만 본다.** (비교 정의·지난 기록으로만 남긴다)
> **대체됨(10/14 17:34 UTC)** — 맨 위 「10/15 한 번에」 표가 이 절의 남은 회차를 포함한다. 비교 정의·명령 기록으로만 둔다.
> **사용자(10/14)**: 「3. 시도와 성과 부분에 실기 대기 중인 것들이 많은데 그거 먼저 테스트 해보게 Eval 할 거 정리해서 Trossen PC 에 보내」. 진행 현황 덱(`/home/kiro-ai/NAJY/NAJY_진행현황_1014.pdf` 7·8쪽)에서 **학습·오프라인 채점은 끝났는데 실기를 한 번도 안 돈 레시피**를 모았다. 아래 「10/15~ 비교 라운드」 보다 **먼저** 돈다.
> **방식**: 실기 첫 확인이라 **모델마다 태스크당 1 회 스크리닝**이다. 같은 자리 A✓ 에서 기준 #2 를 맨 앞(과 t02 는 맨 뒤)에 돌려 그날 자리·장면이 맞는지 확인한다. 되는 모델(부분 성공 포함)이 보이면 그 모델·태스크를 비교 라운드에서 3 회 대 3 회로 늘린다.
> **공통**: 1 ep 씩 · 처음 `DRY_RUN=1` 한 번 · e-stop 대기 · 회차 ≈ 5~7 분 · 먼저 `git pull origin main`. ACT 는 블렌딩 켬·exec 60·다듬기 w=3(**#25 s1000 만 w=5** — 오프라인 떨림이 M1 폭 밖, 비교 라운드 C10 칸도 w=5 로 읽는다). 16D 전문가는 스크립트가 베이스 실측 속도를 state 에 자동으로 넣는다.
> **이 세션에서 돌지 않는 것**: state 드롭아웃 #45~#48(같은 줄 #43·#44 가 A✓ 에서 이득 없음 → 10/14 우선 내림) · 학습 중 #49~#52(로봇 쪽 0 채움·원핫 패치 수정 뒤) · 베이스 미기록 ep 를 뺀 재학습(큐 준비 중).
> **분량**: 32 회 ≈ 3.5 h. 여기서 돈 #39·#29·#25·전문가 회차는 비교 라운드의 같은 태스크 스크리닝 1 회로 센다(중복해 돌지 않는다).

**대기 모델 — 무엇을 보려는가**

| 모델 | 레시피 (#2 와 다른 점) | 오프라인에서 본 것 | 실기에서 볼 것 | 태스크 |
|---|---|---|---|---|
| **#39** J39 | lr cosine 감쇠 | 조작 8/8 · t08 출발 .60 → .90 · 왼손 자리 1.0 cm | t08 이 실제로 나아졌나, 다른 단계 손해는 없나 | t01·t02·t03·t05·t08 |
| **#29** J35 | GIST 저품질 2.1% 제외 | 떨림 10~30% ↓ · 왼손 자리 0.9 cm | 덜 떨면서 같은 자리로 가나 | t01·t02·t03·t05·t08 |
| **#27** J30 | 같은 레시피 **120K**(두 배 학습) | 출발 같음 · 예측 떨림 30~50% ↓ | 오래 구운 것이 실기 동작을 차분하게 만드나 | t01·t02·t03·t05·t08 |
| **#25** J28 s1000 | + 영상 드롭아웃 30% | 이득 없음 · 원핫 경로 열림 | 원핫을 더 따르는가(t05), t02 는 같은가 | t01·t02·t05 |
| **#41** J41 | task02 학습/홀드아웃 재분할(투명 튜브 학습 6 → 27 ep) | 투명 튜브 장면 영상 기울기 소폭 ↑ | **투명 튜브** 장면에서 #2 보다 잘 잡나(#2 는 10/13 이 조건 0/6) | t02 (투명 튜브) |
| **#40** J38 | SmolVLA + 쿼리 토큰 64 | 우리 ho 조작 6/8 · t09 정지 예측 60% | SmolVLA 마지막 레버 — t01 에서 82° 지나 계속 도는 실패가 남나. **로드 로그에 `[action_query] model.action_queries (64) 를 붙여 싣는다` 가 없으면 중단**(쿼리 없이 도는 것) | t01·t02 |
| 전문가 | 팀 단계별 모델(t02 14D · t05 16D) | — | 단계 하나만 배운 모델이 같은 자리에서 #2 보다 나은가(C5 스크리닝) | t02·t05 |

**태스크 순서 표**

| 순서 | 태스크 | 검증하려는 것 | 장면·자리 | 회차 (이 순서로) | 명령 | 결과 → 다음 |
|---|---|---|---|---|---|---|
| **1** | **task01 랙으로 이동** | #39·#29·#27·#25 첫 실기, SmolVLA #40 이 82° 넘게 도는지 | 시연 출발 자리(재생 자 없음 — 로봇 PC 가 쓰던 t01 출발 자리) | #2 → #40 → #39 → #29 → #27 → #25 (6) | #2: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=1/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 1 60 1` · #40: `POLICY=kiroaiseoul/NAJY_smolvla_all11_gist30_fp32_aq64_60k_s1000 scripts/eval_smolvla.sh 1 rec 1` · #39: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=1/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_cos_27D_60k_s1000 1 60 1` · #29: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=1/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_dq_27D_60k_s1000 1 60 1` · #27: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=1/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_120k_s1000 1 60 1` · #25: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=1/11 LEROBOT_CHUNK_SMOOTH_TICKS=5 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_drop_27D_60k_s1000 1 60 1` | 랙 앞에 멈추면 ✓. #40 이 멈추면 C1(ACT vs SmolVLA)에 #40 을 #36 과 함께 넣는다 |
| **2a** | **task02 튜브 집기 (시연 튜브)** | 대기 모델이 A✓ 에서 #2 처럼 왼손으로 집나, 오른손이 더 뻗는 모델이 있나 | 튜브는 시연과 같은 구멍. 재생 `LEROBOT_REPLAY_DATASET=kiroaiseoul/task02_pickup_tubes LEROBOT_REPLAY_EPISODES=102 LEROBOT_REPLAY_RUN_POLICY=0 RUN_SUFFIX=replay_ep102 HOT=2/11 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 2 60 1` 로 A✓ 를 먼저 | #2 → #39 → #29 → #27 → #25 → 전문가 → #40 → #2 (8) | #2: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=2/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 2 60 1` · #39: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=2/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_cos_27D_60k_s1000 2 60 1` · #29: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=2/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_dq_27D_60k_s1000 2 60 1` · #27: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=2/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_120k_s1000 2 60 1` · #25: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=2/11 LEROBOT_CHUNK_SMOOTH_TICKS=5 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_drop_27D_60k_s1000 2 60 1` · 전문가: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/task02_pickup_tubes_14D 2 60 1` · #40: `POLICY=kiroaiseoul/NAJY_smolvla_all11_gist30_fp32_aq64_60k_s1000 scripts/eval_smolvla.sh 2 rec 1` | 양손 잡으면 그 모델을 비교 라운드 t02 확정(3 대 3)에 올린다. 오른 j1 이 시연(145°)에 가까워지는지 기록 |
| **2b** | **task02 튜브 집기 (투명 튜브)** | 투명 튜브를 더 배운 #41 이 #2 보다 잘 잡나 | 같은 A✓ 자리, **투명 튜브**를 시연 구멍에 | #2 → #41 → #2 → #41 (4) | #2: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=2/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 2 60 1` · #41: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=2/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_t02il_27D_60k_s1000 2 60 1` | #41 이 2 회 중 1 회라도 잡고 #2 가 못 잡으면 「투명 튜브 데이터 부족」 확정 → 투명 튜브 녹화 추가 |
| **3** | **task03 제자리 회전** | 대기 모델 첫 실기 — 튜브 쥐고 출발 시 시연만큼 도나 | **양손에 튜브를 쥐어 준 채** 출발. 재생 `LEROBOT_REPLAY_DATASET=kiroaiseoul/task03_turn_to_face_beaker LEROBOT_REPLAY_EPISODES=79 LEROBOT_REPLAY_RUN_POLICY=0 RUN_SUFFIX=replay_ep79 HOT=3/11 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 3 60 1` 로 A✓ 를 먼저 | #2 → #39 → #29 → #27 (4) | #2: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=3/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 3 60 1` · #39: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=3/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_cos_27D_60k_s1000 3 60 1` · #29: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=3/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_dq_27D_60k_s1000 3 60 1` · #27: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=3/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_120k_s1000 3 60 1` | 비커 정면에서 멈추면 ✓. 회전각 기록(시연 ≈ −90°) |
| **5** | **task05 튜브 버리기** | 청크 100 세대 t05 첫 실기 — 대기 모델 · 전문가 중 버리는 모델이 있나 | 양손에 튜브. 재생 `LEROBOT_REPLAY_DATASET=kiroaiseoul/task05_tube_disposal LEROBOT_REPLAY_EPISODES=69 LEROBOT_REPLAY_RUN_POLICY=0 RUN_SUFFIX=replay_ep69 HOT=5/11 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 5 60 1` 로 A✓ 를 먼저 | #2 → #39 → #29 → #27 → #25 → 전문가(16D) (6) | #2: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=5/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 5 60 1` · #39: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=5/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_cos_27D_60k_s1000 5 60 1` · #29: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=5/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_dq_27D_60k_s1000 5 60 1` · #27: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=5/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_120k_s1000 5 60 1` · #25: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=5/11 LEROBOT_CHUNK_SMOOTH_TICKS=5 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_drop_27D_60k_s1000 5 60 1` · 전문가: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/act_task05_tube_disposal_60000 5 60 1` | 둘 다 트레이에 떨어뜨리면 ✓, 하나면 +p. 되는 모델은 비교 라운드 t05 확정에 올린다 |
| **8** | **task08 비커 꺼내 놓기** | #39 의 오프라인 t08 개선(.60 → .90)이 실기에도 보이나 | 재생 `LEROBOT_REPLAY_DATASET=kiroaiseoul/task08_takeout_and_put_beaker LEROBOT_REPLAY_EPISODES=157 LEROBOT_REPLAY_RUN_POLICY=0 RUN_SUFFIX=replay_ep157 HOT=8/11 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 8 60 1` 로 A✓ 를 먼저 | #2 → #39 → #29 → #27 (4) | #2: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=8/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 8 60 1` · #39: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=8/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_cos_27D_60k_s1000 8 60 1` · #29: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=8/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_dq_27D_60k_s1000 8 60 1` · #27: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=8/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_120k_s1000 8 60 1` | #39 만 출발·진행하면 비교 라운드 C3 t08 확정을 먼저 돈다 |

- **기록**: 회차마다 맨 아래 「결과 기록」 에 한 줄(태스크 순서), 상세는 `eval_najy_results_1015.md` → `git push origin main`. 점수판 [`eval_scoreboard.md`](eval_scoreboard.md) 「청크 100 세대」 표에 #39·#29·#27·#25·#41·#40 열을 더한다.
- t04 는 이 세션에서 뺐다(붓기는 액체·비커 준비가 커서 비교 라운드 세션 2 에서 한꺼번에).

## ▶ 지금 돌릴 것 — 10/15~ 비교 라운드, 태스크 순서 (DGX1 → Trossen PC1, 2026-10-14 11:00 UTC) · 이 표만 본다
> **⛔ 10/16 정비로 대체됨 — 이 절의 명령은 돌리지 마라. 맨 위 「▶ 10/16 정비」 절만 본다.** (비교 정의·지난 기록으로만 남긴다)
> **대체됨(10/14 17:34 UTC)** — 맨 위 「10/15 한 번에」 표가 이 절의 남은 회차를 포함한다. 비교 정의·명령 기록으로만 둔다.
> **(10/14 15:10 UTC 추가)** 이 라운드 **앞에** 위 「먼저 돌릴 것 — 실기 대기 모델 첫 실기」 세션을 돈다. 거기서 돈 #39·#29·#25·전문가 회차는 이 라운드의 같은 태스크 스크리닝 1 회로 센다.
> **사용자(10/14)**: 「다 실기에서 비교 안 된 거라 모르겠다는데 왜 실기에 추가 안 한 거야. 다 추가해 비교를 해야 확정을 하지」. 그래서 지금까지 **오프라인으로만 정한 결정을 전부 실기 비교**로 넣었다. 아래 「비교 목록」 의 C1~C10 이 그것이고, 태스크 표의 「검증하려는 것」 칸에 어느 비교의 회차인지 적었다.
> **확정 규칙**: 「확정」 이 붙은 비교는 **같은 자리 A✓ 에서 두 모델을 번갈아 각 3 회**(ABABAB) 돈다 — 1 회 비교는 스크리닝일 뿐이다. 나머지 칸은 1 회 스크리닝이고, 차이가 보이면 같은 방식으로 3 회로 늘린다.
> **공통**: 단계마다 **재생으로 자리 A✓ 를 먼저 맞춘다**(10/14 발견). 1 ep 씩 · `DRY_RUN=1` 한 번 · e-stop 대기 · 회차 ≈ 5~7 분 · 먼저 `git pull origin main`. ACT 는 블렌딩 켬·exec 60·w=3(청크 150 #9 만 w=5). 16D 전문가는 스크립트가 베이스 실측 속도를 state 에 자동으로 넣는다. 결과는 회차마다 [`eval_scoreboard.md`](eval_scoreboard.md) 「★ 청크 100 세대」 표에 **A✓** 로.
> **분량**: 약 100 회 ≈ 10 h → **세션 1 = t01~t03(45 회 — 10/15 t02 행 17 회로) · 세션 2 = t04~t07(27 회) · 세션 3 = t08~t11(21 회)**. 태스크 순서는 지킨다. 시간이 모자라면 「확정」 회차를 먼저, 스크리닝 회차를 나중에.

**비교 목록 — 지금까지 실기로 확인 안 된 결정들**

| 비교 | 질문 | 모델 짝 | 확정 태스크 (각 3 회) | 스크리닝 태스크 | 오프라인에서 본 것 |
|---|---|---|---|---|---|
| **C1** | ACT 와 SmolVLA 중 무엇이 나은가 | #2 ↔ #36(SmolVLA fp32, 가장 차분했던 것) | t01 | t02·t03·t05 | SmolVLA 가 같거나 짐 · 실기 0/7 은 자리 미확인 |
| **C1′** | (10/15 추가) t02 조작은 SmolVLA+쿼리(#40)가 전문가보다 나은가 · **쿼리가 효과인가(#36)** | #40 ↔ t02 전문가 `task02_pickup_tubes_14D` ↔ #36(쿼리 없음) | t02 | — | 10/15 실기 1 회: #40 양손 쥠·정지, 전문가 왼손만. 재예측: 11단계 ACT 는 1~4 cm 높게 겨냥하고 전문가·#40 은 튜브 높이 쪽(실기 성패가 높이 하나로 갈리지는 않음 — #40 +1.6 cm 성공)([1015 DGX 분석](eval_najy_results_1015.md)). t01 정지는 #40 도 실패 — 이동 단계 비교 아님 |
| **C2** | 원핫이 실제로 단계를 고르나 | 같은 t05 장면에서 원핫 **5/11 vs 4/11** — #2 · #25(영상 드롭아웃, 원핫 경로가 더 열린 모델) | t05 (#2 원핫 4 ×2) | — | 원핫 이득 없음 · 모델은 장면으로 단계를 고른다 |
| C2′ | 원핫 없는 11단계 모델이 단계별 실기에서 나쁜가 (원핫을 쓸 이유가 실기에 있나) | #2 ↔ **#50 J46**(J14 에서 원핫만 뺌, 16D) — **학습 중**(1호기, ≈10/14 22:30 UTC) + 로봇 쪽 0 채움 경로(eval_najy.sh 수정, codex) 뒤 | t05 · t02 | 나머지 | 없음(10/14 투입) |
| **C3** | lr 감쇠(cosine)가 나은가 | #2 ↔ #39 | t02 · t08 | 나머지 전 단계 | 조작 8/8, t08 .60→.90 |
| C4 | 베이스 속도를 state 에 넣으면 나빠지나 | 실기 비교 가능한 짝이 **없다**. 만들려면 둘 다 필요하다: ① #2 레시피 + 베이스 속도 학습(task06 데이터셋은 베이스 칸 없이 녹화돼 그 단계만 0 이 섞인다) ② 로봇 원핫 패치가 지금 베이스 칸을 항상 0 으로 채우므로 실측 속도를 넣게 고치기(로봇 코드 — codex 검토 대상). **DGX 가 준비안을 사용자에게 올린 상태 — 이번 라운드에서는 돌지 않는다.** 참고로 t04·t05 16D 전문가는 베이스 실측 속도를 쓴다(단일 태스크라 교락) | 학습 뒤 | t04·t05(참고) | 속도는 누수로 확실히 나쁨 · 누적 이동값 = 0 과 같음 |
| **C5** | 단계별 전문가가 11단계 모델보다 나은가 | #2 ↔ 각 단계 전문가(청크 100, 팀 학습) | t02 · t05 | t01·t03·t04·t06~t10 (t11 은 전문가 없음) | 11단계 한 모델이 같거나 나음 · t05 전문가는 10/08 실기 1/1 |
| C5′ | 같은 학습 설정의 단계 전문가가 11단계 모델보다 나은가 (허브 전문가는 레시피가 달라 교락) | #2 ↔ **#51 J47**(t02) · **#52 J48**(t05) — 2호기 학습(J14 설정, 그 단계 우리+GIST 만, 원핫 없음 16D, 60K) | t02 · t05 | — | 없음. **해석 범위**: 원핫 제거·통계 재집계·단계당 노출 ≈11배가 같이 바뀐다 → 「같은 60K 예산의 전문가 vs 11단계 모델」 의 답이지 「단일 데이터 자체의 효과」 가 아니다(codex 10/14) |
| **C6** | 이동 단계에 GIST 를 섞으면 나은가 | #2(GIST 30%) ↔ #1(GIST 없음) | t01 · t03 | t10 | GIST 홀드아웃 출발 8/8 vs 3/8 |
| C6' | 「이동만 GIST」 레시피 | #32 | — | t01·t03·t10 | 이동 같음, t03 과회전 |
| C7 | 청크 100 vs 150 | #4(M1 c100 s1000) ↔ #9(M1 c150 s1000, 같은 시드) | — (t01 각 2) | t05 | 150 은 t01 기다림을 담지만 떨림 1.2~1.5× |
| **C9** | GIST 정제가 나은가 | #2 ↔ #29 | — | 전 단계 | 같은 자리, 떨림 ↓ |
| C10 | 영상 드롭아웃 | #2 ↔ #25 | — | t02·t05 | 이득 없음, 원핫 경로 열림 |

> **정정(10/14 13:30 UTC, DGX1)** — 위 C4 행: 「task06 데이터셋은 베이스 칸 없이 녹화돼 그 단계만 0 이 섞인다」 는 틀렸다. #2 레시피 매니페스트는 옛 task06(16D)을 쓰고, 27셋 전부 16D·베이스 속도 칸 분산 있음(meta/stats 실측). 사용자 「다음 학습부터 올려」(10/14)로 **#49 J45 = #2 + 베이스 속도 입력**을 1호기에 띄웠다(13:25 UTC, ≈22:30 UTC 끝). 실기는 학습 끝 + 로봇 원핫 패치가 베이스 칸을 0 으로 덮지 않게 고친 뒤(codex). 확정 태스크 t01·t03(베이스가 움직이는 단계), 스크리닝 t10.

state 드롭아웃(#43·#44)은 10/14 에 A✓ 에서 이미 비교했다(이득 없음). 베이스 칸을 0 으로 두는 것과 아예 빼는 것은 수식상 같아(상수 입력 = 첫 층 편향) 비교하지 않는다.

**태스크 순서 표**

| 순서 | 태스크 | 검증하려는 것 (비교) | 장면·자리 | 회차 (이 순서로, 번갈아) | 명령 |
|---|---|---|---|---|---|
| **1** | **task01 랙으로 이동** | C1 확정(#2·#36 각 3) · C6 확정(#1 각 3) · C7(#4·#9 각 2) · C5 · C3 · C9 · C6'(#32) | 시연 자리(로봇 PC 가 쓰던 t01 출발 자리). t01 은 재생 자 없음. 재생 자 없음 — 시연 출발 자리 | #2 → #36 → #1 → 전문가 → #2 → #36 → #1 → #4 → #9 → #2 → #36 → #1 → #4 → #9 → #39 → #29 → #32 (17) | #2: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=1/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 1 60 1` · #39: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=1/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_cos_27D_60k_s1000 1 60 1` · #29: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=1/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_dq_27D_60k_s1000 1 60 1` · #1: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=1/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_m1_27D_60k_s2000 1 60 1` · #32: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=1/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30mv_sw_27D_60k_s1000 1 60 1` · #4: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=1/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_m1_27D_60k_s1000 1 60 1` · #9: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=1/11 LEROBOT_CHUNK_SMOOTH_TICKS=5 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c150_m1_27D_60k_s1000 1 60 1` · #36: `POLICY=kiroaiseoul/NAJY_smolvla_all11_gist30_fp32_60k_s1000 scripts/eval_smolvla.sh 1 rec 1` · 전문가: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/task01_move_to_tube_rack_14D 1 60 1` (HOT 없음) |
| **2** | **task02 튜브 집기** | **C1′ 확정(#40 ↔ 전문가 각 3 — 10/15 #40 양손 성공이 우연인가)** · C3 확정(#39 각 3) · C5 확정(전문가 각 3) · **왼손 높이(#1 ×2 — 재예측에서 11단계 ACT 10개 중 #1 만 낮게 겨냥, GIST·시드와 무관. 10/15 DGX 분석)** · C1 | 튜브는 시연과 같은 구멍. 재생 `LEROBOT_REPLAY_DATASET=kiroaiseoul/task02_pickup_tubes LEROBOT_REPLAY_EPISODES=102 LEROBOT_REPLAY_RUN_POLICY=0 RUN_SUFFIX=replay_ep102 HOT=2/11 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 2 60 1` 로 A✓ 를 먼저 | **(10/15 바꿈)** **#40 → 전문가 → #36 → #2 → #39** 를 3 번 돈다(15) → #1 → #1 (17). #29·#25 는 10/15 스크리닝으로 끝(빠짐). **#36 을 3 회로 늘린 이유**: 쿼리 없는 #36·#12 도 같은 프레임 재예측이 #40 과 같다 — 10/15 성공이 쿼리 덕인지 재생으로 맞춘 자리 덕인지 가린다 | #2: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=2/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 2 60 1` · #39: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=2/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_cos_27D_60k_s1000 2 60 1` · #1: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=2/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_m1_27D_60k_s2000 2 60 1` · #40: `POLICY=kiroaiseoul/NAJY_smolvla_all11_gist30_fp32_aq64_60k_s1000 scripts/eval_smolvla.sh 2 rec 1` (로드 로그 `[action_query] model.action_queries (64)` 없으면 중단) · #36: `POLICY=kiroaiseoul/NAJY_smolvla_all11_gist30_fp32_60k_s1000 scripts/eval_smolvla.sh 2 rec 1` · 전문가: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/task02_pickup_tubes_14D 2 60 1` (HOT 없음) |
| **3** | **task03 제자리 회전** | C6 확정(#1 각 3) · C6' · C1 · C5 · C3 · C9 | **양손에 튜브를 쥐어 준 채** 출발. 재생 `LEROBOT_REPLAY_DATASET=kiroaiseoul/task03_turn_to_face_beaker LEROBOT_REPLAY_EPISODES=79 LEROBOT_REPLAY_RUN_POLICY=0 RUN_SUFFIX=replay_ep79 HOT=3/11 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 3 60 1` 로 A✓ 를 먼저 | #2 → #1 → #2 → #1 → #2 → #1 → #32 → #36 → 전문가 → #39 → #29 (11) | #2: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=3/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 3 60 1` · #39: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=3/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_cos_27D_60k_s1000 3 60 1` · #29: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=3/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_dq_27D_60k_s1000 3 60 1` · #1: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=3/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_m1_27D_60k_s2000 3 60 1` · #32: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=3/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30mv_sw_27D_60k_s1000 3 60 1` · #36: `POLICY=kiroaiseoul/NAJY_smolvla_all11_gist30_fp32_60k_s1000 scripts/eval_smolvla.sh 3 rec 1` · 전문가: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/task03_turn_to_face_beaker_14D 3 60 1` (HOT 없음) |
| **4** | **task04 붓기** | 첫 실기 · C3 · C9 · C5(·C4 참고: 16D 전문가는 베이스 실측 속도 입력) | 양손에 튜브를 쥐어 준다. 재생 `LEROBOT_REPLAY_DATASET=kiroaiseoul/task04_pour_liquid_from_tubes_to_beaker LEROBOT_REPLAY_EPISODES=96 LEROBOT_REPLAY_RUN_POLICY=0 RUN_SUFFIX=replay_ep96 HOT=4/11 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 4 60 1` 로 A✓ 를 먼저 | #2 → #39 → #29 → 전문가(16D) (4) | #2: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=4/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 4 60 1` · #39: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=4/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_cos_27D_60k_s1000 4 60 1` · #29: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=4/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_dq_27D_60k_s1000 4 60 1` · 전문가: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/act_task04_pour_liquid_from_tubes_to_beaker_60000 4 60 1` (HOT 없음) |
| **5** | **task05 튜브 버리기** | C5 확정(전문가 각 3) · **C2 원핫**(같은 t05 장면에서 원핫 4 와 5 — #2·#25) · C3 · C9 · C1 · C10 · C7 | 시연 자리. 원핫 시험 회차는 장면·자리 그대로, 원핫만 4/11. 재생 `LEROBOT_REPLAY_DATASET=kiroaiseoul/task05_tube_disposal LEROBOT_REPLAY_EPISODES=69 LEROBOT_REPLAY_RUN_POLICY=0 RUN_SUFFIX=replay_ep69 HOT=5/11 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 5 60 1` 로 A✓ 를 먼저 | #2 → 전문가(16D) → #2 → 전문가 → #2 → 전문가 → #39 → #29 → #36 → #25 → #4 → #9 → **원핫 시험**: #2 HOT=4/11 → #2 HOT=4/11 → #25 HOT=4/11 (15) | #2: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=5/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 5 60 1` · #39: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=5/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_cos_27D_60k_s1000 5 60 1` · #29: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=5/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_dq_27D_60k_s1000 5 60 1` · #25: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=5/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_drop_27D_60k_s1000 5 60 1` · #4: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=5/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_m1_27D_60k_s1000 5 60 1` · #9: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=5/11 LEROBOT_CHUNK_SMOOTH_TICKS=5 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c150_m1_27D_60k_s1000 5 60 1` · #36: `POLICY=kiroaiseoul/NAJY_smolvla_all11_gist30_fp32_60k_s1000 scripts/eval_smolvla.sh 5 rec 1` · 전문가: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/act_task05_tube_disposal_60000 5 60 1` (HOT 없음) · 원핫 시험: 위 #2·#25 명령에서 `HOT=5/11` 만 `HOT=4/11` 로 |
| **6** | **task06 비커 들고 냉장고로** | 첫 실기 · C3 · C9 · C5 | 시연 자리. 재생 `LEROBOT_REPLAY_DATASET=kiroaiseoul/task06_task06-2_pickup_beaker_and_move_to_refrigerator_14D_new_2 LEROBOT_REPLAY_EPISODES=93 LEROBOT_REPLAY_RUN_POLICY=0 RUN_SUFFIX=replay_ep93 HOT=6/11 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 6 60 1` 로 A✓ 를 먼저 | #2 → #39 → #29 → 전문가 (4) | #2: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=6/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 6 60 1` · #39: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=6/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_cos_27D_60k_s1000 6 60 1` · #29: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=6/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_dq_27D_60k_s1000 6 60 1` · 전문가: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/act_task06_task06-2_pickup_beaker_and_move_to_refrigerator_14D_new_2 6 60 1` (HOT 없음) |
| **7** | **task07 냉장고 열기** | 첫 실기 · C3 · C9 · C5 | 시연 자리. 재생 `LEROBOT_REPLAY_DATASET=kiroaiseoul/task07_open_refrigerator LEROBOT_REPLAY_EPISODES=184 LEROBOT_REPLAY_RUN_POLICY=0 RUN_SUFFIX=replay_ep184 HOT=7/11 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 7 60 1` 로 A✓ 를 먼저 | #2 → #39 → #29 → 전문가 (4) | #2: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=7/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 7 60 1` · #39: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=7/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_cos_27D_60k_s1000 7 60 1` · #29: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=7/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_dq_27D_60k_s1000 7 60 1` · 전문가: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/act_task07_open_refrigerator_14D_260916 7 60 1` (HOT 없음) |
| **8** | **task08 비커 꺼내 놓기** | C3 확정(#39 각 3 — 오프라인 t08 .60→.90 이 실기에도 있나) · C9 · C5 | 시연 자리. 재생 `LEROBOT_REPLAY_DATASET=kiroaiseoul/task08_takeout_and_put_beaker LEROBOT_REPLAY_EPISODES=157 LEROBOT_REPLAY_RUN_POLICY=0 RUN_SUFFIX=replay_ep157 HOT=8/11 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 8 60 1` 로 A✓ 를 먼저 | #2 → #39 → #2 → #39 → #2 → #39 → #29 → 전문가 (8) | #2: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=8/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 8 60 1` · #39: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=8/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_cos_27D_60k_s1000 8 60 1` · #29: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=8/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_dq_27D_60k_s1000 8 60 1` · 전문가: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/act_task08_takeout_and_put_beaker_100k_251data 8 60 1` (HOT 없음) |
| **9** | **task09 냉장고 닫기** | 첫 실기 · C3 · C9 · C5 | 시연 자리. 재생 `LEROBOT_REPLAY_DATASET=kiroaiseoul/task09_close_refrigerator LEROBOT_REPLAY_EPISODES=174 LEROBOT_REPLAY_RUN_POLICY=0 RUN_SUFFIX=replay_ep174 HOT=9/11 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 9 60 1` 로 A✓ 를 먼저 | #2 → #39 → #29 → 전문가 (4) | #2: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=9/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 9 60 1` · #39: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=9/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_cos_27D_60k_s1000 9 60 1` · #29: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=9/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_dq_27D_60k_s1000 9 60 1` · 전문가: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/act_task09_close_refrigerator_100k_250data 9 60 1` (HOT 없음) |
| **10** | **task10 선반으로 이동** | 첫 실기 · C6(이동 GIST) · C6' · C3 · C9 · C5 | 시연 자리. 재생 `LEROBOT_REPLAY_DATASET=kiroaiseoul/task10_move_to_beaker_shelf LEROBOT_REPLAY_EPISODES=88 LEROBOT_REPLAY_RUN_POLICY=0 RUN_SUFFIX=replay_ep88 HOT=10/11 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 10 60 1` 로 A✓ 를 먼저 | #2 → #1 → #32 → #39 → #29 → 전문가 (6) | #2: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=10/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 10 60 1` · #39: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=10/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_cos_27D_60k_s1000 10 60 1` · #29: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=10/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_dq_27D_60k_s1000 10 60 1` · #1: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=10/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_m1_27D_60k_s2000 10 60 1` · #32: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=10/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30mv_sw_27D_60k_s1000 10 60 1` · 전문가: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/act_task10_move_to_beaker_shelf_14D_100k_fp32 10 60 1` (HOT 없음) |
| **11** | **task11 비커 선반에 놓기** | 첫 실기 · C3 · C9 (전문가 없음) | 시연 자리. 재생 `LEROBOT_REPLAY_DATASET=kiroaiseoul/task11_place_beaker_on_the_shelf LEROBOT_REPLAY_EPISODES=98 LEROBOT_REPLAY_RUN_POLICY=0 RUN_SUFFIX=replay_ep98 HOT=11/11 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 11 60 1` 로 A✓ 를 먼저 | #2 → #39 → #29 (3) | #2: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=11/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 11 60 1` · #39: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=11/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_cos_27D_60k_s1000 11 60 1` · #29: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=11/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_dq_27D_60k_s1000 11 60 1` |
| 녹화 | task02 (그리고 다른 단계) | **로봇 자리가 달라도 잡게 할 데이터** — 지금 모델은 양손 모두 시연에서 손이 가던 자세로 그대로 간다 | 시연 40 ep 이상, **ep 마다 로봇 정지 위치를 ±5 cm·±10° 안에서 바꾼다**, 튜브 구멍도 바꿔 가며 | — | `lerobot-record` 로 새 repo `kiroaiseoul/task02_pickup_tubes_place_<날짜>` (기존 repo 덮어쓰기 금지) · ep 마다 「A 기준 어느 쪽으로 몇 cm·몇 °」 메모 |

- **결과 → 판정**: 확정 비교는 3 회 대 3 회에서 **성공 수가 2 이상 차이**나면 그 쪽으로 정한다(예: 3/3 vs 1/3). 차이가 1 이하면 「차이 없음 — 단순한 쪽(기준 #2)을 유지」. 판정은 DGX 가 결과 문서를 받아 [`model_selection_1014.md`](model_selection_1014.md) 에 적는다.
- **기록**: 회차마다 맨 아래 「결과 기록」 에 한 줄(태스크 순서), 상세는 `eval_najy_results_<MMDD>.md` → `git push origin main`.

## ▶ 지금 돌릴 것 — 10/15 선정 라운드, 태스크 순서 (DGX1 → Trossen PC1, 2026-10-14 10:40 UTC) · 이 표만 본다
> **⛔ 10/16 정비로 대체됨 — 이 절의 명령은 돌리지 마라. 맨 위 「▶ 10/16 정비」 절만 본다.** (비교 정의·지난 기록으로만 남긴다)
> **대체됨(10/14 11:00 UTC)** — 사용자 「다 추가해 비교를 해야 확정을 하지」 로 맨 위 「10/15~ 비교 라운드」 가 이 표를 포함해 넓혔다. 이 표는 기록으로 둔다.
> **사용자(10/14)**: 선정안 승인 — 기준 **#2 J14**, 도전 **#39 J39**(lr cosine) · **#29 J35**(GIST 정제), 대조 **#1 J10**(GIST 없음, t02 오른손만). 근거는 [`model_selection_1014.md`](model_selection_1014.md). state 드롭아웃(#43·#44·#45~#48)과 SmolVLA 는 이번 라운드에서 돌리지 않는다.
> **핵심 규칙(10/14 로봇 PC 발견)**: 단계마다 **시연 ep 재생으로 자리 A✓ 를 먼저 맞춘다** — 재생(모델 없음)이 시연과 같은 타이밍으로 잡을 때까지 로봇을 옮긴 자리가 A✓ 다. 10/14 t02 는 첫 자리에서 재생도 빈손이었다. 재생 ep 는 DGX 가 고른 것이다(그 단계 최근 시연 중 시작 자세가 pose_guide 목표에 가깝고 길이가 중간인 것, t02 는 로봇 PC 가 쓴 ep102). 안 맞으면 다른 ep 로 바꿔도 된다.
> **공통**: 1 ep 씩 · `DRY_RUN=1` 한 번 · e-stop 대기 · 회차 ≈ 5~7 분 · 먼저 `git pull origin main`. ACT 는 블렌딩 켬, `HOT=<s>/11`, exec 60, w=3. `ACT(<s>, #n)` = `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=<s>/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/<repo> <s> 60 1`.
> repo: **#2** `NAJY_act_all11_c100_gist30_27D_60k_s1000` · **#39** `NAJY_act_all11_c100_gist30_cos_27D_60k_s1000` · **#29** `NAJY_act_all11_c100_gist30_dq_27D_60k_s1000` · **#1** `NAJY_act_all11_c100_m1_27D_60k_s2000`
> 분량: t02 6 회 + t03~t11 각 4 회(재생 1 + 모델 3) = 42 회 ≈ 4~5 h — 두 세션으로 나눠도 된다(태스크 순서는 지킨다).

| 순서 | 태스크 | 검증하려는 것 | 장면·자리 | 회차 (이 순서로) | 명령 | 결과 → 다음 |
|---|---|---|---|---|---|---|
| **1** | **task02 튜브 집기** | **A✓ 에서 양손을 잡나** — 10/14 #2 는 왼손만 잡았다. 오른손은 재예측에서 13개 모델 모두 덜 뻗고 그리퍼를 덜 연다. #1(GIST 없음)이 유일하게 오른 그리퍼를 여는 예측을 낸다 → 오른손 문제가 GIST 혼합 탓인지 가른다 | 재생 ep102 로 자리 A✓ 를 먼저 맞춘다 — 재생이 시연처럼 성공할 때까지 로봇 자리를 옮긴다(10/14 t02 는 3 회). 재생이 끝내 안 되면 그 단계는 장면 차이 — 사진을 남기고 다음 단계로. 튜브는 시연과 같은 구멍 | 재생 1 → #2 → #39 → #29 → #1 → #1 (6) | 재생: `LEROBOT_REPLAY_DATASET=kiroaiseoul/task02_pickup_tubes LEROBOT_REPLAY_EPISODES=102 LEROBOT_REPLAY_RUN_POLICY=0 RUN_SUFFIX=replay_ep102 HOT=2/11 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 2 60 1` → 모델: `ACT(2, #2)` · `ACT(2, #39)` · `ACT(2, #29)` · `ACT(2, #1)` | #2·#39·#29 가 양손 → 기준 확정. 셋 다 왼손만, #1 이 오른손까지 → 오른손은 GIST 혼합 탓, 학습 처치로. 넷 다 오른손 못 함 → 그리퍼 진폭 학습 처치(선정안 §5) |
| **2** | **task03 제자리 회전** | **튜브를 쥔 채 출발하면 시연만큼(−81°) 도나** — 10/13 은 빈손 출발이라 팔을 먼저 움직이고 덜 돌았다(J14 만 성공) | 재생 ep79 로 자리 A✓ 를 먼저 맞춘다 — 재생이 시연처럼 성공할 때까지 로봇 자리를 옮긴다(10/14 t02 는 3 회). 재생이 끝내 안 되면 그 단계는 장면 차이 — 사진을 남기고 다음 단계로. 리셋 때 리더암으로 **양손에 튜브를 쥐어 준다** | 재생 1 → #2 → #39 → #29 (4) | 재생: `LEROBOT_REPLAY_DATASET=kiroaiseoul/task03_turn_to_face_beaker LEROBOT_REPLAY_EPISODES=79 LEROBOT_REPLAY_RUN_POLICY=0 RUN_SUFFIX=replay_ep79 HOT=3/11 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 3 60 1` → 모델: `ACT(3, #2)` · `ACT(3, #39)` · `ACT(3, #29)` | −81° 근처 → t03 절차에 「튜브 쥐고 출발」 고정. 여전히 덜 돎 → 베이스 저속 손실, basevel.csv 로 잰다 |
| **3** | **task04 붓기** | 청크 100 모델이 붓기를 하나 (첫 실기) · 기준과 도전 모델의 차이 | 재생 ep96 로 자리 A✓ 를 먼저 맞춘다 — 재생이 시연처럼 성공할 때까지 로봇 자리를 옮긴다(10/14 t02 는 3 회). 재생이 끝내 안 되면 그 단계는 장면 차이 — 사진을 남기고 다음 단계로. 양손에 튜브를 쥐어 준다 | 재생 1 → #2 → #39 → #29 (4) | 재생: `LEROBOT_REPLAY_DATASET=kiroaiseoul/task04_pour_liquid_from_tubes_to_beaker LEROBOT_REPLAY_EPISODES=96 LEROBOT_REPLAY_RUN_POLICY=0 RUN_SUFFIX=replay_ep96 HOT=4/11 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 4 60 1` → 모델: `ACT(4, #2)` · `ACT(4, #39)` · `ACT(4, #29)` | 실패 분류(`출발 못함`·`다른 단계 동작`·`그리퍼`·`중간 정지`·`기타`)를 꼭 적는다. 세 모델 중 성공한 것이 있으면 그 단계 다음 회차는 그 모델로 2 ep 더 |
| **4** | **task05 튜브 버리기** | 같음 (첫 실기) | 재생 ep69 로 자리 A✓ 를 먼저 맞춘다 — 재생이 시연처럼 성공할 때까지 로봇 자리를 옮긴다(10/14 t02 는 3 회). 재생이 끝내 안 되면 그 단계는 장면 차이 — 사진을 남기고 다음 단계로. 시연 자리 | 재생 1 → #2 → #39 → #29 (4) | 재생: `LEROBOT_REPLAY_DATASET=kiroaiseoul/task05_tube_disposal LEROBOT_REPLAY_EPISODES=69 LEROBOT_REPLAY_RUN_POLICY=0 RUN_SUFFIX=replay_ep69 HOT=5/11 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 5 60 1` → 모델: `ACT(5, #2)` · `ACT(5, #39)` · `ACT(5, #29)` | 실패 분류(`출발 못함`·`다른 단계 동작`·`그리퍼`·`중간 정지`·`기타`)를 꼭 적는다. 세 모델 중 성공한 것이 있으면 그 단계 다음 회차는 그 모델로 2 ep 더 |
| **5** | **task06 비커 들고 냉장고로** | 같음 (첫 실기, 조작 + 베이스 이동) | 재생 ep93 로 자리 A✓ 를 먼저 맞춘다 — 재생이 시연처럼 성공할 때까지 로봇 자리를 옮긴다(10/14 t02 는 3 회). 재생이 끝내 안 되면 그 단계는 장면 차이 — 사진을 남기고 다음 단계로. 시연 자리 | 재생 1 → #2 → #39 → #29 (4) | 재생: `LEROBOT_REPLAY_DATASET=kiroaiseoul/task06_task06-2_pickup_beaker_and_move_to_refrigerator_14D_new_2 LEROBOT_REPLAY_EPISODES=93 LEROBOT_REPLAY_RUN_POLICY=0 RUN_SUFFIX=replay_ep93 HOT=6/11 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 6 60 1` → 모델: `ACT(6, #2)` · `ACT(6, #39)` · `ACT(6, #29)` | 실패 분류(`출발 못함`·`다른 단계 동작`·`그리퍼`·`중간 정지`·`기타`)를 꼭 적는다. 세 모델 중 성공한 것이 있으면 그 단계 다음 회차는 그 모델로 2 ep 더 |
| **6** | **task07 냉장고 열기** | 같음 (첫 실기) | 재생 ep184 로 자리 A✓ 를 먼저 맞춘다 — 재생이 시연처럼 성공할 때까지 로봇 자리를 옮긴다(10/14 t02 는 3 회). 재생이 끝내 안 되면 그 단계는 장면 차이 — 사진을 남기고 다음 단계로. 시연 자리 | 재생 1 → #2 → #39 → #29 (4) | 재생: `LEROBOT_REPLAY_DATASET=kiroaiseoul/task07_open_refrigerator LEROBOT_REPLAY_EPISODES=184 LEROBOT_REPLAY_RUN_POLICY=0 RUN_SUFFIX=replay_ep184 HOT=7/11 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 7 60 1` → 모델: `ACT(7, #2)` · `ACT(7, #39)` · `ACT(7, #29)` | 실패 분류(`출발 못함`·`다른 단계 동작`·`그리퍼`·`중간 정지`·`기타`)를 꼭 적는다. 세 모델 중 성공한 것이 있으면 그 단계 다음 회차는 그 모델로 2 ep 더 |
| **7** | **task08 비커 꺼내 놓기** | 같음 (첫 실기) | 재생 ep157 로 자리 A✓ 를 먼저 맞춘다 — 재생이 시연처럼 성공할 때까지 로봇 자리를 옮긴다(10/14 t02 는 3 회). 재생이 끝내 안 되면 그 단계는 장면 차이 — 사진을 남기고 다음 단계로. 시연 자리 | 재생 1 → #2 → #39 → #29 (4) | 재생: `LEROBOT_REPLAY_DATASET=kiroaiseoul/task08_takeout_and_put_beaker LEROBOT_REPLAY_EPISODES=157 LEROBOT_REPLAY_RUN_POLICY=0 RUN_SUFFIX=replay_ep157 HOT=8/11 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 8 60 1` → 모델: `ACT(8, #2)` · `ACT(8, #39)` · `ACT(8, #29)` | 실패 분류(`출발 못함`·`다른 단계 동작`·`그리퍼`·`중간 정지`·`기타`)를 꼭 적는다. 세 모델 중 성공한 것이 있으면 그 단계 다음 회차는 그 모델로 2 ep 더 |
| **8** | **task09 냉장고 닫기** | 같음 (첫 실기) · 오프라인에서 출발이 굼뜬 단계 | 재생 ep174 로 자리 A✓ 를 먼저 맞춘다 — 재생이 시연처럼 성공할 때까지 로봇 자리를 옮긴다(10/14 t02 는 3 회). 재생이 끝내 안 되면 그 단계는 장면 차이 — 사진을 남기고 다음 단계로. 시연 자리 | 재생 1 → #2 → #39 → #29 (4) | 재생: `LEROBOT_REPLAY_DATASET=kiroaiseoul/task09_close_refrigerator LEROBOT_REPLAY_EPISODES=174 LEROBOT_REPLAY_RUN_POLICY=0 RUN_SUFFIX=replay_ep174 HOT=9/11 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 9 60 1` → 모델: `ACT(9, #2)` · `ACT(9, #39)` · `ACT(9, #29)` | 실패 분류(`출발 못함`·`다른 단계 동작`·`그리퍼`·`중간 정지`·`기타`)를 꼭 적는다. 세 모델 중 성공한 것이 있으면 그 단계 다음 회차는 그 모델로 2 ep 더 |
| **9** | **task10 선반으로 이동** | 이동 단계에서 t01·t03 처럼 가나 (첫 실기) | 재생 ep88 로 자리 A✓ 를 먼저 맞춘다 — 재생이 시연처럼 성공할 때까지 로봇 자리를 옮긴다(10/14 t02 는 3 회). 재생이 끝내 안 되면 그 단계는 장면 차이 — 사진을 남기고 다음 단계로. 시연 자리 | 재생 1 → #2 → #39 → #29 (4) | 재생: `LEROBOT_REPLAY_DATASET=kiroaiseoul/task10_move_to_beaker_shelf LEROBOT_REPLAY_EPISODES=88 LEROBOT_REPLAY_RUN_POLICY=0 RUN_SUFFIX=replay_ep88 HOT=10/11 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 10 60 1` → 모델: `ACT(10, #2)` · `ACT(10, #39)` · `ACT(10, #29)` | 실패 분류(`출발 못함`·`다른 단계 동작`·`그리퍼`·`중간 정지`·`기타`)를 꼭 적는다. 세 모델 중 성공한 것이 있으면 그 단계 다음 회차는 그 모델로 2 ep 더 |
| **10** | **task11 비커 선반에 놓기** | 같음 (첫 실기) | 재생 ep98 로 자리 A✓ 를 먼저 맞춘다 — 재생이 시연처럼 성공할 때까지 로봇 자리를 옮긴다(10/14 t02 는 3 회). 재생이 끝내 안 되면 그 단계는 장면 차이 — 사진을 남기고 다음 단계로. 시연 자리 | 재생 1 → #2 → #39 → #29 (4) | 재생: `LEROBOT_REPLAY_DATASET=kiroaiseoul/task11_place_beaker_on_the_shelf LEROBOT_REPLAY_EPISODES=98 LEROBOT_REPLAY_RUN_POLICY=0 RUN_SUFFIX=replay_ep98 HOT=11/11 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 11 60 1` → 모델: `ACT(11, #2)` · `ACT(11, #39)` · `ACT(11, #29)` | 실패 분류(`출발 못함`·`다른 단계 동작`·`그리퍼`·`중간 정지`·`기타`)를 꼭 적는다. 세 모델 중 성공한 것이 있으면 그 단계 다음 회차는 그 모델로 2 ep 더 |
| 녹화 | task02 (그리고 다른 단계) | **로봇 자리가 달라도 잡게 할 데이터** — 10/14 에 같은 모델이 자리만 어긋나면 빈손이었다. 지금 후보는 카메라로 자리를 보정하지 않는다 | 투명 튜브 시연 40 ep 이상. **ep 마다 로봇 정지 위치를 ±5 cm·±10° 안에서 바꾼다**(이것이 본 처치), 튜브 구멍도 바꿔 가며 | — | `lerobot-record` 로 새 repo `kiroaiseoul/task02_pickup_tubes_place_<날짜>`. 기존 repo 를 덮어쓰지 않는다(`push_to_hub` 기본 true) | ep 마다 「자리 A 기준 어느 쪽으로 몇 cm·몇 °」 를 메모 — DGX 가 자리별로 성능을 가른다 |
| 올려 줄 것 | task03 | 베이스 저속 비선형 | 10/13 1139·1141·1143·114526 과 이번 t03 의 `basevel.csv` | — | `docs/run_logs/<날짜>_eval_najy/` 에 사본 | — |

- **기록**: 회차마다 맨 아래 「결과 기록」 과 [`eval_scoreboard.md`](eval_scoreboard.md) 「★ 청크 100 세대」 표(자리 칸에 **A✓** 표시)를 같이 고친다. 상세는 `eval_najy_results_<MMDD>.md` → `git push origin main`.
- 이 라운드가 끝나면 단계별로 「기준이 되는 단계 / 안 되는 단계」 가 나온다 — 그다음 흔든 자리(B) n≥10 은 그 모델로만.

## ▶ 지금 돌릴 것 — 10/14 한 장, 태스크 순서 (DGX1 → Trossen PC1, 2026-10-14 07:10 UTC) · 이 표만 본다
> **⛔ 10/16 정비로 대체됨 — 이 절의 명령은 돌리지 마라. 맨 위 「▶ 10/16 정비」 절만 본다.** (비교 정의·지난 기록으로만 남긴다)
> **끝남(10/14 오후, 로봇 PC)** — 순서 1·2 를 돌았다(결과 [`eval_najy_results_1014.md`](eval_najy_results_1014.md)). 나머지는 맨 위 「▶ 지금 돌릴 것 — 10/15 선정 라운드」 로 옮겼다.
> **사용자(10/14)**: 「Eval 실험은 항상 태스크 순서로 정리해서 표로 만들고 뭘 검증하려고 하는건지도 잘 적어둬. 확인하기 쉽게」. 그래서 아래 표는 **task02 → task11 순서**이고, 행마다 「검증하려는 것」 을 질문 한 줄로 적었다. 위에서부터 돌면 된다.
> **옛 지시 표는 전부 끝났거나 이 표로 대체됐다** — 10/13 D1~D4 · 10/13 P0~P5 · 10/12 T1~T3 · 10/11 S0~S11. 근거는 [`eval_najy_results_1013.md`](eval_najy_results_1013.md) 「DGX1 답」 · sim §94.63.
> **지금까지**: 실기에 올린 모델 7개(#1·#2·#3·#12·#32·#35·#36). 성공은 #2 J14 의 t01(10/11)·t03(10/13) 둘뿐. **task02 는 10/11·10/13 전부 실패** — 모델이 시연에서 튜브가 있던 자리로 가서 닫는데, 10/13 튜브는 그보다 안쪽 구멍에 있었다. **t04~t11 은 청크 100 모델로 아직 한 번도 안 돌았다.**
> **정정(DGX1, 10/14)**: 바로 위 「10/13 튜브는 그보다 안쪽 구멍」 은 틀렸다 — 사람 확인으로 구멍은 시연과 같았고, 어긋난 것은 **로봇 자리**다(10/14 재생 시험, [`eval_najy_results_1014.md`](eval_najy_results_1014.md)). 아래 순서 1~3 의 「구멍」 조건도 「로봇 자리」 로 읽는다.
> **공통**: 1 ep 씩 · 회차마다 `DRY_RUN=1` 한 번 · e-stop 대기 · 회차 ≈ 5~7 분 · 먼저 `git pull origin main`. ACT 는 **블렌딩 켬 기본**, `HOT=<s>/11` 필수, exec 60, w=3. 17D 모델(#3 B)은 `eval_chain.sh` 단일 단계로만(`eval_najy.sh` 는 거부한다).
> **ACT 명령** — 아래 `ACT(<s>, #n)` 은 `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 HOT=<s>/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/<repo> <s> 60 1` 이다.
> repo: **#2** `NAJY_act_all11_c100_gist30_27D_60k_s1000`(J14, 지금 기준 모델) · **#44** `NAJY_act_all11_c100_gist30_sdrop30_27D_60k_s1000`(J42, 팔 state 드롭아웃 .3) · **#43** `NAJY_act_all11_c100_gist30_sdrop30_27D_60k_s2000`(#44 시드 짝) · **#32** `NAJY_act_all11_c100_gist30mv_sw_27D_60k_s1000`(GIST 이동 단계만)

| 순서 | 태스크 | 검증하려는 것 | 장면·조건 | 회차 (이 순서로) | 명령 | 결과 → 다음 |
|---|---|---|---|---|---|---|
| **1** | **task02** 튜브 집기 | **state 드롭아웃(#44·#43)이 「튜브를 보고 따라가기」 를 만드나** — #2 는 이 조건에서 0/6 | **10/13 과 같은 조건**: 투명 튜브 · 10/13 과 같은 안쪽 구멍 · 로봇은 시연 자리 | #2 → #44 → #43 → #44 (4, ≈25 분) | `ACT(2, #2)` · `ACT(2, #44)` · `ACT(2, #43)` | **#44·#43 이 잡고 #2 는 못 잡음** → 드롭아웃이 먹는다. 학습 중인 #45·#48 을 이 조건에 넣고 실기 기본 모델을 #44 로 바꾼다. **셋 다 못 잡음** → 드롭아웃만으론 모자람, 녹화(맨 아래)가 본 처치 |
| **2** | task02 | **실패 원인이 튜브 자리인가** | 투명 튜브를 **시연 ep 100~132 와 같은 구멍**에(`docs/run_logs/2026-10-13_eval_najy/dgx_t02_rack_real_vs_demo.png` 의 demo ep110·115·130 과 대조) | #2 → #44 → #2 → #44 (4, ≈25 분) | `ACT(2, #2)` · `ACT(2, #44)` | **#2 가 잡음** → 원인은 튜브 자리. eval 은 구멍을 고정하고, 학습 과제는 「구멍 따라가기」 로 좁힌다. **못 잡음** → 자리만의 문제가 아니다, 순서 3 과 함께 읽는다 |
| **3** | task02 | **실패 원인이 투명 튜브가 안 보이는 것인가** | 녹화 때와 같은 **빨간 테이프**를 감고 10/13 과 같은 안쪽 구멍에 (ep 0~99 는 테이프 튜브로 찍었다 — 사용자 10/13) | #2 → #44 → #2 → #44 (4, ≈25 분) | `ACT(2, #2)` · `ACT(2, #44)` | **#2 가 잡음** → 원인은 가시성. 실기는 테이프를 감고, 투명 튜브 시연을 늘린다 |
| 4 | task02 (여유 있을 때) | SmolVLA 에 쿼리 토큰을 붙이면 나아지나 | 순서 2 와 같은 장면 | #36 → #40 (2, ≈15 분) | `POLICY=kiroaiseoul/NAJY_smolvla_all11_gist30_fp32_60k_s1000 scripts/eval_smolvla.sh 2 rec 1` · #40 은 `…_fp32_aq64_60k_s1000` | #40 은 로드 로그에 `[action_query] model.action_queries (64) 를 붙여 싣는다` 가 **없으면 중단** |
| **5** | **task03** 제자리 회전 | **10/13 에 덜 돈 원인이 빈손 출발이었나** (10/13 은 빈손 → 팔 먼저 움직이고 회전이 느렸다. J14 만 성공) | 리셋 때 리더암으로 **양손에 튜브를 쥐어 준 채** 출발 | #2 → #32 → #3 B (3, ≈20 분) | `ACT(3, #2)` · `ACT(3, #32)` · B: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 LEROBOT_CHUNK_SMOOTH_TICKS=3 EXEC=60 FROM_STAGE=3 TO_STAGE=3 scripts/eval_chain.sh kiroaiseoul/NAJY_act_all11_c100_drop_prog_27D_60k_s1000 b_t03` | 셋 다 −81° 근처로 돌면 → 원인은 빈손. t03 은 「튜브 쥐고 출발」 을 절차로 고정. 여전히 덜 돌면 → 베이스 저속 손실, `basevel.csv` 로 잰다 |
| **6** | **task04** 붓기 | 청크 100 모델이 붓기를 하나(첫 실기) · 드롭아웃이 조작을 해치지 않나 | 시연 자리 A, 양손에 튜브를 쥐어 준다 | #2 → #44 (2) | `ACT(4, #2)` · `ACT(4, #44)` | 실패 분류를 꼭 적는다(`출발 못함`·`다른 단계 동작`·`그리퍼`·`중간 정지`·`기타`) — 이하 같다 |
| **7** | **task05** 튜브 버리기 | 같음 | 시연 자리 A | #2 → #44 (2) | `ACT(5, #2)` · `ACT(5, #44)` | 같음 |
| **8** | **task06** 비커 들고 냉장고로 | 같음 (조작 + 이동이 섞인 단계) | 시연 자리 A | #2 → #44 (2) | `ACT(6, #2)` · `ACT(6, #44)` | 같음 · 베이스 이동량(∫x·∫θ)도 적는다 |
| **9** | **task07** 냉장고 열기 | 같음 | 시연 자리 A | #2 → #44 (2) | `ACT(7, #2)` · `ACT(7, #44)` | 같음 |
| **10** | **task08** 비커 꺼내 놓기 | 같음 | 시연 자리 A | #2 → #44 (2) | `ACT(8, #2)` · `ACT(8, #44)` | 같음 |
| **11** | **task09** 냉장고 닫기 | 같음 · 오프라인에서 출발이 굼뜬 단계(#44 t09 .35) | 시연 자리 A | #2 → #44 (2) | `ACT(9, #2)` · `ACT(9, #44)` | 같음 · 출발 못함이면 몇 초 기다렸는지 적는다 |
| **12** | **task10** 선반으로 이동 | 이동 단계에서 #2 가 t01·t03 처럼 가나 | 시연 자리 A | #2 (1) | `ACT(10, #2)` | 회전·전진량(∫θ·∫x)을 시연과 대조 |
| **13** | **task11** 비커 선반에 놓기 | 같음 (조작) | 시연 자리 A | #2 → #44 (2) | `ACT(11, #2)` · `ACT(11, #44)` | 같음 |
| 녹화 | task02 | 영상으로 튜브 자리를 찾는 법을 배울 데이터 (사용자 10/13 「투명 튜브도 데이터가 있으면 좋긴 하겠다」) | 투명 튜브 시연 **40 ep 이상**. ep 마다 꽂는 구멍을 바꾼다(안쪽 포함, 좌우 따로). 로봇 정지 위치도 ±5 cm 안에서 조금씩 | — | `lerobot-record` 로 새 repo `kiroaiseoul/task02_pickup_tubes_clear_<날짜>`. 기존 `task02_pickup_tubes` 를 덮어쓰지 않는다(`push_to_hub` 기본 true) | ep 마다 구멍(예: 「L 뒤2·R 뒤4」)을 메모로 — DGX 가 구멍별로 가른다 |
| 올려 줄 것 | task03 | 베이스 저속 비선형을 속도 구간별로 잰다 | 10/13 1139·1141·1143·114526 과 순서 5 의 `basevel.csv` | — | `docs/run_logs/<날짜>_eval_najy/` 에 사본 | — |

- 합계: 순서 1~3 ≈ 1.3 h · 5 ≈ 20 분 · 6~13 ≈ 1.5~2 h (15 회차). 순서 4 는 여유 있을 때.
- **학습 중 — public 되면 순서 1 조건에서 #44 와 번갈아**: #45 J43(드롭 .5, 1호기 DGX1, 60K ≈ 10/14 11:30 UTC → 채점·업로드 뒤) · #48 J44(드롭 .7, 1호기 DGX1, ≈ 12:20 UTC 뒤) · #46 J43-s2000 · #47 J42-s3000(둘 다 2호기 sandia, 60K ≈ 21:20 UTC). 어느 비율을 실기에 올릴지는 오프라인 짝 비교(J4x − J42) 뒤 이 줄에 적는다.
- **아직 실기에 안 올린 public 모델**(#4~#11 · #19~#31 · #33 · #34 · #37~#39 · #41 · #42)은 위 표 뒤에 돈다. 장면은 순서 1 결과로 정한다 — #44 가 잡으면 순서 1 조건에서 1 ep 씩, 아니면 DGX 가 다음 지시를 낸다.
- **기록**: 회차마다 맨 아래 「결과 기록」 에 한 줄을 쓴다(태스크 순서로). 상세는 `eval_najy_results_<MMDD>.md` 에 쓴 뒤 `git push origin main`. 순서 1~3 은 꽂은 구멍 사진(cam_high 한 장)을 같이 남긴다.

## ▶ 지금 돌릴 것 — 한 장 (DGX1 → Trossen PC1, 2026-10-13 00:40 UTC, 사용자 「지금 Eval 돌려볼거 정리해서 Trossen PC 에 넘겨줘」)
> **⛔ 10/16 정비로 대체됨 — 이 절의 명령은 돌리지 마라. 맨 위 「▶ 10/16 정비」 절만 본다.** (비교 정의·지난 기록으로만 남긴다)
> **끝남·대체됨 (10/14)** — 지금 돌릴 것은 맨 위 「▶ 지금 돌릴 것 — 10/14 한 장, 태스크 순서」 이다. 이 절은 기록으로 둔다.
> 10/11 뒤로 실기 결과가 없다. 그래서 **10/12 토너먼트 T1~T3(t02)이 그대로 1순위**다. 그 사이 새로 생긴 것은 둘이다. ⓐ 러너 **그리퍼 쥠 가드**(`b58a686`)가 생겼다 — t02·t03·t04 는 정지 창 3 초의 매 틱에 양손 그리퍼가 ≥ 0.004 여야 자동 완료된다. 10/11 B t02 의 거짓 완료(오른손 0.000)를 막는다. ⓑ **SmolVLA 짝 #35(J37)·#36(J37a)** 이 생겼다.
> 아래 P0 → P5 순서로 돈다. 1 ep 씩, 매 회차 `DRY_RUN=1` 로 한 번 찍고, e-stop 을 대기한다. 자리 A(시연 자리)에서, 같은 장면·같은 시작 짝으로 번갈아 돈다. 회차 1개 ≈ 5~7 분이라 P1~P4 는 ≈ 2 h 다.

| 순서 | 무엇을 가르나 | 명령 (`~/NAJY/lerobot_trossen` 에서) | 회차 | 볼 것 · 기록 |
|---|---|---|---|---|
| **P0** 준비 | 코드 받기 | `git pull origin main` → `git log --oneline \| grep -c b58a686` 이 1 | — | 러너(`eval_chain.sh`) 회차에서 빈손이면 이제 자동 완료가 안 된다 — `→` 키나 timeout 으로 끝난다(정상) |
| **P1 = T1** | 시작 자세 vs 레시피 | ① #2 J14 를 러너로: `LEROBOT_CHUNK_SMOOTH_TICKS=3 EXEC=60 FROM_STAGE=2 TO_STAGE=2 scripts/eval_chain.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 j14run_t02` ② #3 B 를 일반 경로로: `HOT=2/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_drop_prog_27D_60k_s1000 2 60 1` — 17D 라 거부되면 그 사실만 적고 ① 만 | 1 + 1 | 시연 속도·경계 튐. 로그에 `gripper hold guard REFUSED` 가 나오면 그 줄의 `(left, right)` 값을 적는다 |
| **P2 = T2** | 청크 경계 블렌딩 | 끔: `HOT=2/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 2 60 1` · 켬: 같은 명령 앞에 `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10` | 2 + 2 (끔·켬·끔·켬) | 로그에 seam blend 활성 줄이 있는지, 경계 튐, 성공 여부 |
| **P3 = T3** | 조작에서 GIST 를 뺀 효과 | #32: `HOT=2/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30mv_sw_27D_60k_s1000 2 60 1` ↔ #2(P2 의 끔 명령) | t02 2 + 2 (여유 있으면 t05 2 + 2: `HOT=5/11 … 5 60 1`) | 성공·실패 분류 |
| **P4** SmolVLA 셋 | fp32 원본 효과 · VLM 언프리즈 효과 | `POLICY=kiroaiseoul/<repo> scripts/eval_smolvla.sh 2 rec 1` 을 #12 `NAJY_smolvla_all11_gist30_60k_s1000` → #35 `NAJY_smolvla_all11_gist30_fp32_unfreeze_60k_s1000` → #36 `NAJY_smolvla_all11_gist30_fp32_60k_s1000` 순서로 번갈아 (여유 있으면 t01 도 같은 순서) | t02 3 (+ t01 3) | #12 의 실패(끝을 못 잡고 계속 감, 떨림)가 #36(fp32 만)·#35(+언프리즈)에서 줄어드나 |
| **P5** | 허브 업로드 (DGX 재예측용) | `huggingface-cli upload kiroaiseoul/eval_najy_1011_2155_base_t02_e60 ~/.cache/huggingface/lerobot/kiroaiseoul/eval_najy_1011_2155_base_t02_e60 --repo-type dataset --private` · 같은 꼴로 `…_2157_base_t02_e60` | — | **됨 — 10/13 01:15 UTC 둘 다 올라왔다**(public 으로 올라감). 남은 것: `eval_najy_1011_2141_base_t01_e60` 은 허브에 `.gitattributes` 만 있다(업로드 실패) — 같은 명령으로 다시 올려 달라. 2143(J14 t01) 도 있으면 함께 |

- **DGX 재예측 결과(10/13 02:10 UTC, [`eval_najy_results_1011.md`](eval_najy_results_1011.md) 「DGX1 답 10/13」)**: ① 경계 튐(≈0.09 rad)은 **B 에도 있다** — T1 은 「튐」 이 아니라 **속도·움직이는 비율**(B 0.41×·74% vs J14 0.27×·28%)로 가른다. ② 튐은 모델 예측에서 나온다(청크 첫 명령이 현재 자세에서 ≈0.1 rad 떨어져 시작 — 시연 프레임에서도 같다) → **T2 블렌딩이 정확히 이것을 겨눈다**. ③ 같은 J14 모델이 B 회차 프레임에서는 B 만큼 움직인다 → 「시작 자세」 쪽으로 기우는 오프라인 증거, 판정은 T1①.
- **판정 규칙**은 바로 아래 토너먼트 절 「판정 뒤」 그대로다. P1~P4 가 끝나면 그 아래 「로봇 PC 전달」 S0~S11 로 간다.
- **기록**: 회차마다 맨 아래 「결과 기록」 에 한 줄을 쓰고, 상세는 `eval_najy_results_1013.md`(새로)에 쓴 뒤 `git push origin main`. 러너 회차는 `events.jsonl` 의 `stage_end` 에 `grip_refused_ticks`·`grip_meas_last_left/right` 가 남는다 — 지우지 마라.
- **#40 J38 은 아래 표에 있다**(public). 로드 로그에 `[action_query] model.action_queries (64) 를 붙여 싣는다` 가 없으면 **중단** — 쿼리 없이 도는 것이다.

## ⭐ 판정 토너먼트 — t02 를 먼저 가른다 (DGX1 → Trossen PC1, 2026-10-12 11:00 UTC, 사용자 「그렇게 진행해」)
> **⛔ 10/16 정비로 대체됨 — 이 절의 명령은 돌리지 마라. 맨 위 「▶ 10/16 정비」 절만 본다.** (비교 정의·지난 기록으로만 남긴다)
> **끝남·대체됨 (10/14)** — 지금 돌릴 것은 맨 위 「▶ 지금 돌릴 것 — 10/14 한 장, 태스크 순서」 이다. 이 절은 기록으로 둔다.
> 학습 레버(시드·120K·드롭아웃·GIST 정제/비율·통계·이동만 GIST)는 오프라인에서 전부 #2(J14) 자리였고, 오프라인 지표 셋은 실기 t02 를 못 갈랐다(sim §94.43·§94.46.1·§94.52). 그래서 **아래 T1~T3 을 아래 S0~S11 보다 먼저** 돌린다 — 실기 회차를 「무엇이 t02 를 가르나」 에 집중한다. 10/11 실기에서 t02 루프 21 Hz·페이싱 FIRED 0 → 느림·튐은 안전 제한 탓이 아니다.

| 순서 | 무엇을 가르나 | 회차 (t02, 자리 A, 1 ep 씩, 같은 시작 짝으로 번갈아) | 명령 |
|---|---|---|---|
| **T1** 시작 자세 vs 레시피 | 10/11 B(#3) 만 시연 속도·경계 튐 0.6×(J14 는 25×). 그런데 B 는 **러너로 지정 시작 자세에서** 출발했다 — 둘 중 무엇이 원인인가 | ① **#2 J14 를 러너로**(지정 시작 자세) ② **#3 B 를 일반 경로로**(eval_najy, 러너 없이) | ① `LEROBOT_CHUNK_SMOOTH_TICKS=3 EXEC=60 FROM_STAGE=2 TO_STAGE=2 scripts/eval_chain.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 j14run_t02` ② `HOT=2/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_drop_prog_27D_60k_s1000 2 60 1` — **②는 17D 진행도 모델이라 eval_najy.sh 가 거부하면** 그 사실을 적고 ①만 |
| **T2** 청크 경계 블렌딩 | J14 의 경계 점프 25× 가 실행 층(교체 순간) 문제인가 | #2 J14 를 블렌딩 **켬/끔** 짝 2회 (+ 여유 있으면 #29·#32 도 켬 1회씩) | 끔 = 위 T1 이전과 같은 `HOT=2/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_27D_60k_s1000 2 60 1` · 켬 = 앞에 `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10` 추가(교체 순간 이전 청크와의 차이를 5차식으로 10틱 이상에 걸쳐 0 으로) — 로그에 seam blend 활성 줄 확인 |
| **T3** 조작에서 GIST 를 뺀 효과(사용자 가설) | 「이동은 GIST, 조작은 우리만」 | #32(J36) ↔ #2 번갈아 t02 2+2 (가능하면 t05 2+2) | `HOT=2/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30mv_sw_27D_60k_s1000 2 60 1` ↔ #2 같은 명령 |

- **판정 뒤**: T1 이 「시작 자세」 면 이후 전 모델을 러너 단일 단계(지정 시작 자세)로 돌린다 · T2 가 「블렌딩이 튐을 없앤다」 면 이후 기본값에 넣는다 · T1~T3 에서 t02 를 통과한 레시피만 다음 태스크로. 나머지 행(S0~S11)은 그 뒤.
- **허브 업로드 요청 (DGX 재예측용)**: 10/11 의 **`eval_najy_1011_2155_base_t02_e60`(J10)·`eval_najy_1011_2157_base_t02_e60`(J14)** 이 허브에 없다(B 의 `eval_1011_2203_chain_b60_t02_e60` 은 있다 — DGX 가 분석 중). 로봇 PC 캐시 `~/.cache/huggingface/lerobot/kiroaiseoul/` 에 있으면 `huggingface-cli upload kiroaiseoul/<이름> <폴더> --repo-type dataset --private` 로 올려 달라. 이번 T1~T3 회차도 `push_to_hub` 기본값(true) 그대로 둔다.
- 회차마다 맨 아래 「결과 기록」 에 한 줄 + `eval_najy_results_1012.md`(새로) 에 상세 → `origin main` push.

## 로봇 PC 전달 — 돌릴 순서 (DGX1 → Trossen PC1, 2026-10-11 12:30 UTC)
> **⛔ 10/16 정비로 대체됨 — 이 절의 명령은 돌리지 마라. 맨 위 「▶ 10/16 정비」 절만 본다.** (비교 정의·지난 기록으로만 남긴다)
> **끝남·대체됨 (10/14)** — 지금 돌릴 것은 맨 위 「▶ 지금 돌릴 것 — 10/14 한 장, 태스크 순서」 이다. 이 절은 기록으로 둔다.
> 로봇 PC 결과는 10/08 이후 없다. **아래 S0~S11 순서로 돌리고**, 회차마다 맨 아래 「결과 기록」 표에 한 줄 + 상세는 `eval_najy_results_1011.md`(§9 양식) → `origin main` push. 표의 모델 23개는 전부 `kiroaiseoul/…` **public 확인**(10/11 12:20 UTC, HF API). 사용자 지시(10/11): 「Eval 돌릴 거 리스트업 올려서 Trossen PC 에 전달」.

**명령 규칙 — 모델 종류로 고른다** (전부 `DRY_RUN=1` 로 한 번 찍어 보고 실행, 1 에피소드씩, e-stop 대기)
| 종류 (표 #) | 명령 |
|---|---|
| ACT 27D 청크 100 — #1·#2·#4·#5·#6·#7·#19·#21·#22·#23·#25 | `HOT=<s>/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/<repo> <s> 60 1` — **#21·#25(s1000) 은 `…TICKS=5`** |
| ACT 27D 청크 150 — #9·#10 | 같은 명령, `LEROBOT_CHUNK_SMOOTH_TICKS=5` (exec 60 그대로) |
| 17D 진행도(체인 레시피) — #3·#8·#11·#20·#24 | `LEROBOT_CHUNK_SMOOTH_TICKS=<3\|5> EXEC=60 FROM_STAGE=<s> TO_STAGE=<s> scripts/eval_chain.sh kiroaiseoul/<repo> <태그>` — **단일 단계만**(FROM=TO). 체인(1→11)은 S3 의 R1·R2 가 나온 뒤 |
| SmolVLA — #12·#13 | `POLICY=kiroaiseoul/<repo> scripts/eval_smolvla.sh <s> rec 1` |
| M3 묶음 · 전문가 16D — #14·#15 | `scripts/eval_najy.sh M3 <s> 30 1` · `scripts/eval_najy.sh kiroaiseoul/act_task05_tube_disposal_60000 5 <30\|60> 1` |

`HOT` 없이 27D repo id 를 주면 `eval_najy.sh:121` 에서 멈춘다(정상). 자리 A = 시연 자리, B = ±15 cm·±15° 흔든 자리. 번갈아 = 같은 장면·같은 시작에서 두 모델을 교대로.

**세션 순서** (로봇 PC 시간 기준 추정 — 회차 1개 ≈ 5~7분, 10/02~10/08 실측)
| 세션 | 무엇 | 모델 (#) | 회차 | 시간 |
|---|---|---|---|---|
| **S0** | 무접촉 ①(DRY_RUN → 실기 프레임 출발/끝 읽기, `eval_queue_1008.md` O3 방식) — **우선 1·2 의 13개 먼저**, 나머지는 틈날 때 | #1·#2·#3·#4·#5·#6·#9·#12·#13·#14·#15·#19·#25 → #7·#8·#10·#11·#20~#24 | 모델당 10분 | ≈2 h (+2 h) |
| **S1** | exec A/B — t05 에서 exec 30 → 60 → 100 각 1, exec 60 은 다듬기 없이 1회 추가 (A5·A6) | #1 | 4 | 1 h |
| **S2** | **선정 맞대기 ACT vs SmolVLA** M1~M5 — t05 A/B 3+3, t04 A/B 3+3, t03 A 2, 두 모델 같은 시작 짝으로 번갈아 | #2 ↔ #12 | 28 | 3 h |
| **S3** | 체인 가드 R1·R2 — 단일 단계 t04·t05, `arm_travel_rad` 실측 → 체인 금지 해제 조건 | #3 | 3 | 1 h |
| S4 | M1 c100 시드 폭 — t05·t04 A, #1 과 번갈아 | #4·#5 | 4 | 1 h |
| S5 | GIST 계열 — t05·t04 A/B, #2 와 번갈아 (드롭아웃이 실기 장면에서 먹는지) | #6·#25(s1000 w=5 · s2000 w=3) | 12 | 2 h |
| S6 | 데이터 품질 필터 — **t05 A 5 + B 5, t08 A 5**, 짝(#4·#1)과 번갈아, 효과 주장은 n≥5 | #19(s1000↔#4, s2000↔#1) | 30 | 3 h |
| S7 | 청크 150 — **t01 먼저**(∫θ·∫x, w=5) → t05·t04 A | #9·#10 → #11·#20(단일 단계) | 8 | 1.5 h |
| S8 | 드롭아웃만 세 시드(#1 과 번갈아) + 체인 B 시드 짝(단일 단계, #3 과 번갈아) | #21·#22·#23 · #8·#24 | 10 | 2 h |
| S9 | 증강 장면 강건성(①→좋으면 ②~④) · SmolVLA 30K S1~S3 · M3 V1~V4 · 전문가 exec A1·A2 | #7·#13·#14·#15 | ≈10 | 2 h |
| S10 | **미실기 단계** N1~N4 — t06·t07·t08·t09·t11 각 1ep, S2 승자로 | S2 승자 | 5 | 1.5 h |
| S11 | 천장·반경 C1~C3 — 시연 자리 n=10 → ±15 cm·±15° 격자 | S2 승자 | 10+ | 반나절 |

**추가됐다(10/11 16:20 UTC)**: #26 `NAJY_act_all11_c100_gist30_27D_60k_s3000`(J29) · #27 `NAJY_act_all11_c100_gist30_27D_120k_s1000`(J30, 60K→120K 이어받기) — 둘 다 **public, 바로 가능**, S5 에 끼워 #2 와 번갈아(#27 은 t02 에서 #2 와 같은 시작 짝). #28(J31 기하 증강)은 **보류**(사용자 10/11) — 행은 두되 돌리지 않는다. **#29 `NAJY_act_all11_c100_gist30_dq_27D_60k_s1000`(J35, 10/11 23:44 UTC public)** — #2 와 같은 레시피·시드에 GIST 정제만 다른 짝, S5 에서 #2 와 같은 시작 짝으로 번갈아.

**DGX1 이 받고 싶은 것**: 회차마다 「결과 기록」 한 줄(성공/시도·실패 분류) · `~/eval_logs/<회차>` 의 `*.chunks.csv`·basevel 로그 · S2 의 승자(ACT/SmolVLA) · S3 의 `arm_travel_rad` 값. 1006 eval 데이터셋 6개 `/raid/kiro-ai/eval/real/datasets/` 전송 요청은 그대로 유효.

## 공통 절차 (모델마다 같게 — `eval_queue_1008.md` 머리말)
1 에피소드씩 · `DRY_RUN=1` 먼저 · 같은 장면·같은 시작에서 비교 대상과 번갈아 · e-stop 대기. **ACT 27D 는 `HOT=<단계>/11` 필수**(없으면 `eval_najy.sh:121` 에서 멈춤), 청크 100 은 exec **60** + `LEROBOT_CHUNK_SMOOTH_TICKS=3`, 청크 150 은 exec 60 + **w=5**. 체인은 R1·R2(러너 가드 재측정) 전까지 금지 — 단일 단계만.
- **상시 세트(§9)**: ① 무접촉 프레임 시험(DRY_RUN, 출발/끝 읽기) → ② t05 버리기 1ep(시연 자리 A) → ③ t04 붓기 1ep(A) → ④ t05·t04 흔든 자리 B(±15 cm·±15°) → ⑤ 레시피가 바뀐 모델은 이동 t03 1ep(∫θ 명령/실측).
- 기록 한 줄(§9 양식) + 실패 분류(`출발 못함`·`다른 단계 동작`·`그리퍼`·`중간 정지`·`기타`). 결과는 `eval_najy_results_<MMDD>.md`, 누적은 scoreboard.
- 명령 뼈대: ACT `HOT=<s>/11 LEROBOT_CHUNK_SMOOTH_TICKS=<3|5> scripts/eval_najy.sh <repo> <s> 60 1` · SmolVLA `POLICY=<repo> scripts/eval_smolvla.sh <s> rec 1` · 전문가(16D) `scripts/eval_najy.sh <repo> <s> <exec> 1`.

## 표 — 돌릴 것 전부 (public 은 전부 `kiroaiseoul/…`)
| # | 모델 (public 꼬리) | 레시피 · 학습 | 오프라인 (참고만) | 실기 — 돌릴 회차 | 우선 | 상태 | 큐 번호 |
|---|---|---|---|---|---|---|---|
| 1 | `NAJY_act_all11_c100_m1_27D_60k_s2000` | M1 + 청크 100, 60K, sandia(J10) | 조작 8/8 · GIST ho 7/8 | 상시 세트 ①~④ + **exec A/B(30/60/100) t05** + 다듬기 on/off | **1** | t01·t02 1ep (10/11) · 10/14 선정: 대조(t02 오른손) | O3·A5·A6 |
| 2 | `NAJY_act_all11_c100_gist30_27D_60k_s1000` | M1 + 청크 100 + **GIST 30%**, 60K, DGX1(J14) | 조작 8/8 · GIST ho 8/8 | 상시 세트 ①~④ + **선정 맞대기 ACT 팔**(M1~M5, #12 와 번갈아) | **1** | t01·t02 1ep (10/11) | O6'·§8 |
| 3 | `NAJY_act_all11_c100_drop_prog_27D_60k_s1000` | M1 + 영상 드롭아웃 .3 + 진행도 17D, 60K, DGX1(B) | 조작 8/8 · 진행도 80%/0% | 상시 세트 ①~④(러너 단일 단계 `EXEC=60`) + **R1·R2 가드 재측정**(체인 금지 해제 조건) | **1** | t01·t02 1ep (10/11) | O3·R1·R2 |
| 4 | `NAJY_act_all11_c100_m1_27D_60k_s1000` | M1 + 청크 100, 60K, DGX1(A) | 조작 7/8 · GIST ho **3/8** | 상시 세트 ①~③(#1 과 번갈아 — 시드 폭이 실기에도 있나) | 2 | 미실기 | O3 |
| 5 | `NAJY_act_all11_c100_m1_27D_60k_s3000` | M1 + 청크 100, 60K, sandia(J17) | 조작 7/8 | 상시 세트 ①~③(#1 과 번갈아) | 2 | 미실기 | E1 |
| 6 | `NAJY_act_all11_c100_gist30_27D_60k_s2000` | GIST 30%, 60K, DGX1(J14) | 조작 7/8 · GIST ho 8/8 | 상시 세트 ①~④(#2 와 번갈아) | 2 | 미실기 | O6' |
| 7 | `NAJY_act_all11_c100_m1_aff_27D_60k_s1000` | M1 + cam_high 기하 증강, 60K, sandia(J9) | 조작 7/8 | ① 무접촉 프레임(장면 강건성) → 좋으면 ②~④ | 3 | 미실기 | O3 |
| 8 | `NAJY_act_all11_c100_drop_prog_27D_60k_s2000` | drop+prog 둘째 시드, 60K, sandia(J12) | 조작 7/8 | 상시 세트 ①~③(#3 과 번갈아, 단일 단계) | 3 | 미실기 | E4 |
| 9 | `NAJY_act_all11_c150_m1_27D_60k_s1000` | M1 + **청크 150**, 60K, sandia(J19) | 조작 7/8 (t11 .39) · t01 h150 1.08 · 떨림 1.2~1.5× | **t01 먼저**(w=5, ∫θ·∫x) → ②~④ | 2 | 미실기 | E2 |
| 10 | `NAJY_act_all11_c150_m1_27D_60k_s2000` | 청크 150 둘째 시드, sandia(J20) | 조작 6/8 (t08 .27·t11 .41) · w=5 떨림 시연 아래 | #9 와 같은 회차 | 3 | public | E3 |
| 11 | `NAJY_act_all11_c150_drop_prog_27D_60k_s1000` | drop+prog + 청크 150, sandia(J21) | 조작 7/8 (t06 .08 출발 붕괴) · 진행도 59% | t01(w=5) → ②~③, 단일 단계 | 3 | public | E5 |
| 20 | `NAJY_act_all11_c150_drop_prog_27D_60k_s2000` | drop+prog + 청크 150 둘째 시드, sandia(J22) | 조작 7/8 (t06 .85·t11 .38) · 진행도 62% | #11 과 같은 회차 — t06 은 이쪽이 정상 | 3 | public | E6 |
| 21 | `NAJY_act_all11_c100_drop_27D_60k_s1000` | M1 + 영상 드롭아웃 .3, **진행도 없음**, 청크 100, 60K, sandia(J23) | 조작 8/8 (t11 .85·t08 .54) · 원핫 경로 열림 · w=5 권장 | 상시 세트 ①~③(#1 과 번갈아) — 진행도 칸의 득실을 B(#3)와 가름 | 3 | public | E7 |
| 22 | `NAJY_act_all11_c100_drop_27D_60k_s2000` | 드롭아웃만 둘째 시드, sandia(J25) | 조작 6/8 (t08 .37·t11 .35) | #21 과 같은 회차 — 드롭아웃 레시피의 시드 폭 | 3 | public | E8 |
| 23 | `NAJY_act_all11_c100_drop_27D_60k_s3000` | 드롭아웃만 셋째 시드, sandia(J26) | 조작 7/8 (t06 .28) · GIST ho 6/8 · GIST t03/t10 θ 0 | #21·#22 와 같은 회차 — 세 시드 8·6·7 로 M1 과 같은 폭 | 3 | public | E9 |
| 24 | `NAJY_act_all11_c100_drop_prog_27D_60k_s3000` | 체인 B 셋째 시드, sandia(J27) | 조작 7/8 (t11 .43) · 진행도 54%(세 시드 80·52·54) · GIST ho 7/8 | #3(B)·#8(J12)와 같은 회차, 단일 단계 — 체인 레시피의 시드 폭 | 3 | public | E10 |
| 12 | `NAJY_smolvla_all11_gist30_60k_s1000` | **SmolVLA** smolvla_base → GIST 30%, 60K, DGX1(J18) | (h50 만 의미) | **선정 맞대기 VLA 팔** — M1~M5(t05 A/B 3+3, t04 A/B 3+3, t03 A 2), #2 와 같은 시작 짝으로 번갈아 | **1** | **public(11:31 UTC)** · t01·t02 1ep (10/11) | §8 |
| 13 | `NAJY_smolvla_all11_30000` | SmolVLA 11단계 언어 조건, 30K, 우리 데이터만 | 출발 M1 보다 좋음(§94.11) | S1~S3: t05·t04 `rec` 1ep, 떨림 `rec/long/cur` 비교 | 2 | 미실기 | S1~S3 |
| 14 | M3 — 이동 4단계 묶음 ACT (`scripts/eval_najy.sh M3 <단계> 30 1` 별칭, 원핫 {1:1,3:2,6:3,10:4}) | 이동 단계만 묶은 ACT(원핫 1/4) | 미채점 | V1~V4: t01·t03·t06·t10 각 1ep(∫θ·∫x, 과회전 배수) | 2 | 미실기 | V1~V4 |
| 15 | 단계별 전문가(팀 묶음 `act_task06_task07_261006` 등, 16D) | 단계별 ACT, 청크 100 | 전문가 start §94.31 | A1·A2: exec 30 vs 60 t05·t04(출발 실패가 청크 길이 문제인지 실기 확인) | 2 | 미실기 | A1·A2 |
| 16 | `NAJY_act_all11_r4_27D_60k_s1000` (4라운드, 청크 30) | drop+bh, 청크 30 | 실기 t01~t05 전부 실패(10/08) | **다시 안 돈다** — 청크 30 대조군이 필요하면 1ep 만 | — | 실기 끝(1008 문서) | — |
| 17 | 미실기 조작 단계 N1~N4 (t06·t07·t08·t09·t11) | #1 또는 선정 승자 | — | 단계 1ep 씩 — 실기 안 해 본 단계 전부 | 3 | 미실기 | N1~N4 |
| 19 | **데이터 품질 필터 재학습** `NAJY_act_all11_c100_m1dq_27D_60k_s{1000,2000}` (J24) | M1 c100 60K 두 시드, **t05·t08 만** 감사 flag≥2 train 에피소드 제외(t05 7/59 · t08 12/201, 무동작 포함; sim §94.39, `all11_tr_hot_dq.excluded.json`) — 사용자 10/10 「태스크 한두개만」 | 참고만 | **t05·t08 에서 짝과 번갈아**: 1차 짝 s1000 ↔ #4 A(같은 머신·배치), s2000 ↔ #1 J10(sandia 2×16 — 교락 주석). **n=1 은 스크리닝만 — 효과 주장은 조건당 n≥5**(t05 A 자리 5ep + B 자리 5ep, 두 모델 번갈아 ≈1 h; codex 10/10). `HOT=5/11`·`HOT=8/11`, w=3, exec 60. t04 는 대조(제외 안 함). 처치 = 제외 + balance 노출↑ | 2 | **public(18:03 UTC)** — 바로 가능. 오프라인(참고): 조작 7/8·8/8, t05 .94·1.01 / t08 .63·.72 (sim §94.40) | J24 |
| 25 | `NAJY_act_all11_c100_gist30_drop_27D_60k_s{1000,2000}` | GIST 30% + 영상 드롭아웃 .3(27항목 전부), 청크 100, 60K, DGX1(J28) | 조작 7/8·7/8 (t11 .44·.40) · GIST ho 7/8·8/8 — J14 와 같은 자리(sim §94.41) | 상시 세트 ①~④ **#2·#6(J14)와 번갈아** — 드롭아웃이 실기 장면(다른 위치·조명)에서 먹는지; 단일 단계, exec 60, **s1000 은 w=5**(떨림 521/160/95 — M1 폭 밖) · s2000 은 w=3 | 2 | **public(04:30·04:37 UTC)** — 바로 가능 | J28 |
| 18 | 천장·반경 C1~C3 | 선정 승자 | — | 시연 자리 반복 n=10(천장) → 반경 격자(±15 cm·±15°) | 선정 뒤 | — | C1~C3 |
| 26 | `NAJY_act_all11_c100_gist30_27D_60k_s3000` | GIST 30% 셋째 시드, 60K, DGX1(J29) | 조작 7/8 (t11 .24) · GIST ho 8/8 (t09 .78) · 원핫/영상 1/10 · 떨림 t05 283/70 (M1 폭 안) — sim §94.46 | S5 — #2·#6 과 번갈아, 시드 폭; 단일 단계, exec 60, **w=3** | 2 | **public(16:04 UTC)** — 바로 가능 | J29 |
| 27 | `NAJY_act_all11_c100_gist30_27D_120k_s1000` | #2(J14 s1000) 60K → **120K 이어받기**, DGX1(J30) | 조작 8/8 · GIST ho 8/8 — #2 와 같은 자리, **예측 청크 떨림만 30~50% ↓**(t05 w=3 139→66, t02 99→71) — sim §94.46 | S5 — **#2 와 같은 시작 짝으로 t02 에서 번갈아**(더 오래 구운 것이 실기 저속·경계 튐을 줄이나); 단일 단계, exec 60, **w=3** | 2 | **public(16:08 UTC)** — 바로 가능 | J30 |
| 28 | `NAJY_act_all11_c100_gist30_geo_27D_60k_s{1000,2000}` | GIST 30% + **카메라별 기하 증강**(cam_high ±3°·.1 이동·.75~1.1 배율, 손목 ±2°), 60K, DGX1(J31) | 채점 예정 | S5 — **자리 B(흔든 시작) 를 먼저**, #2 와 번갈아 — 증강이 흔들린 시작을 버티게 하나 | 2 | **보류**(사용자 10/11 — 학습 안 함, 재추론·실기 프레임 뒤 재검토) | J31 |
| 29 | `NAJY_act_all11_c100_gist30_dq_27D_60k_s1000` | #2(J14 s1000) 와 **같은 레시피·시드**, GIST 16셋만 동작 기준 정제(무동작·저품질 244 ep·2.1% 제외), 60K, DGX1(J35) | 조작 7/8 (t11 .45) · GIST ho 8/8 (t09 .65) · 원핫/영상 3/10 · 떨림 t05 348/100 (w=5 53) — #2 와 같은 자리, 떨림 w=3 10~30% ↓(시드 폭 안) — sim §94.47 | S5 — **#2 와 번갈아, 같은 시작 짝**(가장 깨끗한 A/B — 데이터 정제만 다름); 단일 단계, exec 60, **w=3** | **1** | **public(23:44 UTC)** — 바로 가능 · 10/14 선정: 도전 모델 | J35 |
| 30 | `NAJY_act_all11_c100_gist30_drop_27D_60k_s3000` | GIST30 + 드롭아웃 셋째 시드, sandia(J32) | 조작 8/8 · GIST ho 8/8 (J28 두 시드 7·7) | J28(#25) 와 같은 회차 — GIST30+드롭아웃의 시드 폭 | 3 | public | — |
| 31 | `NAJY_act_all11_c100_gist50_27D_60k_s1000` | M1 c100 + **GIST 50%**, 60K, sandia(J33, 옛 J15) | 조작 8/8 · GIST ho 8/8 (전부 ≥.74) | #2(J14 gist30 s1000)와 같은 시작 짝 — 혼합 비율 30 vs 50 | 2 | public | — |
| 32 | `NAJY_act_all11_c100_gist30mv_sw_27D_60k_s1000` | #2(J14) 와 같은 레시피·시드, **GIST 는 이동 단계(t01·t03·t10)에만** + 정규화 통계를 샘플링 몫 가중(`stats_by_weight`), 60K, DGX1(J36) | 우리 ho 조작 7/8 (t11 .44·t08 .52) · GIST ho 조작 2/8(예상 — GIST 조작 미학습) · GIST 이동 θ = #2 · 우리 t03 θ 과회전 2.3× — sim §94.52 | **#2 와 같은 시작 짝** — 사용자 가설 「이동은 GIST, 조작은 우리만」 을 가르는 짝: **조작 단계(t02·t04·t05·t08)** 에서 #2 보다 나은가 + 이동(t01·t03)에서 #2 만큼 가나(t03 과회전 주의); 단일 단계, exec 60, w=3 | 2 | **public(10/12 10:02 UTC)** — 바로 가능 | J36 |
| 33 | `NAJY_act_all11_c100_gist30_sw_27D_60k_s1000` | J14 + 통계 가중(stats_by_weight), sandia(J36a) | 조작 7/8 (t11 .29) · GIST ho 8/8 | #2(J14 s1000)와 같은 시작 짝 — 통계만 다름(그리퍼 std .007→.005) | 2 | public | — |
| 34 | `NAJY_act_all11_c100_gist30mv_sw_27D_60k_s2000` | GIST 이동 단계만 + 통계 가중, sandia(J36-s2000) | 조작 7/8 (t07 .39) · GIST ho 1/8(예상) · 원핫 0/10 | J36(DGX1 s1000)과 같은 회차 — 「이동은 GIST, 조작은 우리만」. 원핫 경로가 닫혀 단일 단계 HOT 결과 해석 주의 | 3 | public | — |
| 35 | `NAJY_smolvla_all11_gist30_fp32_unfreeze_60k_s1000` | **SmolVLA** J18 + 학습 파라미터 fp32 원본(bf16 갱신 소실 수정) + **VLM 텍스트 언프리즈**(lr 1e-5), 60K, DGX1(J37) — 배포본은 J18 과 같은 dtype 배치(bf16), VLM 텍스트 61.7% 가 J18 과 다름 | 우리 ho 조작 7/8 (t08 .46) · GIST ho 조작 7/8 (t04 .46) = J18 과 같은 띠 — sim §94.56 | **#12(J18) 와 같은 시작 짝** — VLM 을 풀면 SmolVLA 가 「끝」 을 잡나(#12: 도달 뒤 계속 감·떨림 1.26/s); `scripts/eval_smolvla.sh` 를 #12 와 같은 설정으로, t01·t02 | 2 | **public(10/12 22:01 UTC)** | J37 |
| 36 | `NAJY_smolvla_all11_gist30_fp32_60k_s1000` | **SmolVLA** J18 + 학습 파라미터 fp32 원본만(VLM 동결 — J18 과 같은 학습 범위), 60K, DGX1(J37a) — 배포본은 J18 과 같은 dtype 배치, vlm_text 0% 변경(동결 확인) | 우리 ho 조작 7/8 (t05 .63) · GIST ho 6/8 (t08 .38) — J18·J37 과 같은 띠. J37 의 t04 상승은 fp32 효과, t05·t06 상승은 언프리즈 효과로 읽힌다 — sim §94.57 | **#12·#35 와 같은 시작 짝** — #12 → #35 → #36 을 번갈아: #36 vs #12 = 「fp32 원본 효과」, #35 vs #36 = 「언프리즈 효과」. t02(·t01) | 2 | **public(10/13 00:12 UTC)** | J37a |
| 37 | `NAJY_act_all11_c100_gist30_drop_prog_27D_60k_s1000` | GIST30 + 드롭아웃 + 진행도(17D), sandia(J40) | 조작 7/8 (t09 .34) · GIST ho 8/8 · 진행도 76%(t09 오검출 41%) | **t02 에서 B(#3)·#2 와 짝** — 시연 속도로 가는 B 에 GIST 를 더함. 체인 쓰면 t09 조기 완료 주의 | 2 | public | — |
| 38 | `NAJY_act_all11_c100_gist30_cos_27D_60k_s2000` | J14 + lr cosine 감쇠, sandia(J39-s2000) | 조작 8/8 · GIST ho 8/8 | J39(DGX1 s1000)와 같은 회차 — 감쇠 유무, #2 와 짝 | 3 | public | — |
| 39 | `NAJY_act_all11_c100_gist30_cos_27D_60k_s1000` | #2(J14) 와 같은 레시피·시드 + **lr cosine 감쇠**(3e-5 → 3e-6, 60K), DGX1(J39) — #38 의 시드 짝 | 우리 ho 조작 8/8 (t08 .60→.90 · t11 .55→.65 · t02 .76→.84) · GIST ho 8/8 · 이동 θ 같음 · 떨림 w=3 t02 99→143 · t05 139→183(늘었다) — sim §94.61 | #2 와 같은 시작 짝, **블렌딩 켬**. task02 는 D2·D3 회차에 #2 와 번갈아 1ep 씩(감쇠가 잡기 정밀도에 주는 효과) | **1** | **public(10/13 09:07 UTC)** · 10/14 선정: 도전 모델 | J39 |
| 42 | `NAJY_act_all11_c100_gist30_t02il_27D_60k_s2000` | J14 + task02 재분할(투명 튜브 27 ep), sandia(J41-s2000) | 조작 5/8(ho_il; t02 1.07) · GIST ho 8/8 | **투명 튜브 t02** — J41(#41)·J14 와 번갈아. t09·t11 출발 굼뜸 주의 | 2 | public | — |
| 43 | `NAJY_act_all11_c100_gist30_sdrop30_27D_60k_s2000` | J14 + 팔 state 드롭아웃 p .3, sandia(J42-s2000) | 조작 7/8 (t09 .31) · GIST ho 8/8 · 원핫/영상 0/10 | **t02 잡는 자리** — J42(s1000)·J14 와 번갈아, 영상으로 자리를 정하는지 | 3 | public · 10/14 선정: 우선 내림(A✓ 이득 없음) | — |
| 40 | `NAJY_smolvla_all11_gist30_fp32_aq64_60k_s1000` | **SmolVLA** #36(J37a, fp32 원본·VLM 동결) + **ActionQuery 64**(VLA-Adapter 방식 쿼리 토큰), 60K, DGX1(J38) — 배포본은 J18 과 같은 dtype 배치 | 우리 ho 조작 6/8 (t06 .46 · **t09 .43, 정지 예측 60%**) · GIST ho 7/8 (t04 .41) — #36(7/8·6/8)·#35(7/8·7/8)와 같은 띠. 쿼리 토큰이 오프라인 출발을 바꾸지 않았다 | **#36 과 같은 시작 짝** — 쿼리 토큰 유무. t02(·t01). 로드 로그에 `[action_query] model.action_queries (64) 를 붙여 싣는다` 가 없으면 **중단**(쿼리 없이 도는 것이다). t09 는 출발 정지 주의 | 3 | **public(10/13 13:54 UTC)** | J38 |
| 41 | `NAJY_act_all11_c100_gist30_t02il_27D_60k_s1000` | J14 + **task02 학습/홀드아웃 재분할**(학습 = ep%5≠4 107 ep, 투명 튜브 6 → 27 ep), DGX1(J41) — #42 의 시드 짝 | ho_il 조작 7/8 (t02 1.07 · **t08 .30**) — t02 이외 단계는 재분할과 무관해 t08 은 시드 잡음으로 본다 · task02 영상 바꾸기 기울기(ho_il) tg-40/60/90 **.24**/.34/.50, state .63/.51/.49 — 잡기 직전은 여전히 state 가 정한다(#42 시드는 .39/.48/.67, 시드 차 큼) | **투명 튜브 t02** — D2·D3 회차에서 #2·#44 와 번갈아. 재분할로 투명 튜브 장면이 늘었지만 「튜브를 보고 따라가기」 는 생기지 않았다(오프라인) | 2 | **public(10/13 18:29 UTC)** | J41 |
| 44 | `NAJY_act_all11_c100_gist30_sdrop30_27D_60k_s1000` | #2(J14) 와 같은 레시피·시드 + **팔 state 드롭아웃 p .3**(학습 샘플 30% 에서 팔 12칸을 「모름」 = 정규화 0 으로), DGX1(J42) — #43 의 시드 짝 | 우리 ho 조작 8/8 · GIST ho 7/8 (t09 .35) · **task02 잡기 직전 영상 기울기 .37(#43 .45) vs #2 .16 · J14-s2000 .19** — state 몫 절반. GIST 장면은 ≈0 이하(개선 없음) · **off0(청크 첫 명령 − 현재 자세) 1.3×(#43 1.8×)** · 떨림 t05 최저 — sim §94.63 | **t02 잡는 자리(D2·D3)** — #2 와 같은 시작 짝으로 번갈아. **반드시 블렌딩 켬**(`LEROBOT_CHUNK_SEAM_BLEND_TICKS=10`) — 경계 튐이 #2 보다 크다. 튜브 자리를 바꿨을 때 따라가는지 | 3 | **public(10/13 22:08 UTC)** · 10/14 선정: 우선 내림(A✓ 이득 없음) | J42 |
| 45 | `NAJY_act_all11_c100_gist30_sdrop50_27D_60k_s1000` | #44(J42) 와 같은 레시피·시드, 팔 state 드롭아웃 **p .5**, DGX1(J43) — 드롭 비율 짝 .3/.5/.7 의 가운데 | 1호기 채점·SCENE2·OFF0 끝 — DGX1 `/raid/kiro-ai/eval/j43_status.txt` | 위 「10/14 한 장」 2-1 조건에서 #44 와 번갈아. 블렌딩 켬 필수 | 3 | **public(10/14 11:47 UTC)** · 10/14 선정: 우선 내림 | J43 |
| 46 | `NAJY_act_all11_c100_gist30_sdrop50_27D_60k_s2000` | #45 의 시드 짝, sandia(J43-s2000) | 채점 예정 | #45 와 같은 회차 | 2 | **public(10/14 21:24 KST)** | — |
| 47 | `NAJY_act_all11_c100_gist30_sdrop30_27D_60k_s3000` | #44·#43 의 셋째 시드(p .3), sandia(J42-s3000) | 채점 예정 | #44·#43 과 같은 회차 — p .3 의 시드 폭 | 3 | **public(10/14 21:24 KST)** | — |
| 48 | `NAJY_act_all11_c100_gist30_sdrop70_27D_60k_s1000` | #44(J42) 와 같은 레시피·시드, 팔 state 드롭아웃 **p .7**, DGX1(J44) | 1호기 채점·SCENE2·OFF0 끝 — DGX1 `/raid/kiro-ai/eval/j44_status.txt` | #45 와 같다. off0(경계 튐)이 더 클 수 있다 — 블렌딩 켬 필수 | 3 | **public(10/14 12:20 UTC)** · 10/14 선정: 우선 내림 | J44 |
| 49 | `NAJY_act_all11_c100_gist30_vel_27D_60k_s1000` | #2(J14) 와 같은 설정·시드, **베이스 속도를 state 에**(27항목 base_state vel), DGX1(J45) | 출발 θ 가 #2 와 같음(우리·GIST ho, 10/14 23:30 UTC 채점) | **C4** — t01·t03 에서 #2 와 번갈아 각 3 회. 로봇 원핫 패치 수정(codex) 뒤에만 | 1 | **허브 공개 10/14 23:20 UTC**(public, sha256 3a1b1079…) · 출발 채점 = #2 와 같음(아래 10/15 절) · 로봇 경로 준비됨(원핫 패치 `LEROBOT_TASK_ONEHOT_BASE=vel`, eval_najy.sh 가 매니페스트로 자동 설정 — 로그 `base slots = measured velocity` 확인) | — |
| 50 | `NAJY_act_all11_c100_gist30_nohot_16D_60k_s1000` | #2(J14) 에서 **원핫만 뺌**(16D, 베이스 0), DGX1(J46) | t03 GIST ho 회전 출발 −0.06 (#2 −0.20, 시연 −0.33) · t01 +0.27 (#2 +0.19) | **C2′** — t05·t02 에서 #2 와 번갈아 각 3 회. HOT 없이, 로봇은 14D + 0 채움(eval_najy.sh 수정, codex) | 1 | **허브 공개 10/14 23:24 UTC**(public, sha256 01889428…) · 출발 채점: 회전 출발이 #2 와 다름(아래 10/15 절) · 로봇 경로 준비됨(eval_najy.sh 가 14D + 0 채움 자동 — 로그 `zero-pad only` 확인) | — |
| 51 | `NAJY_act_task02_c100_gist30_solo_16D_60k_s1000` | **task02 단일 단계 전문가** — J14 설정, task02 우리 106 ep + GIST(30%), 원핫 없음 16D, 60K, sandia(J47) | t02 우리 ho .91 (27 ep) · GIST ho .98 · 떨림 271/66/39 (오프라인 참고) | **C5′** — t02 에서 #2 와 번갈아 각 3 회 (#50 과 같은 0 채움 경로) | 1 | 2호기 지시(sim df9bebe) · **public(10/15 08:36 KST)** | — |
| 52 | `NAJY_act_task05_c100_gist30_solo_16D_60k_s1000` | **task05 단일 단계 전문가** — 같은 방식, task05 우리 59 ep + GIST, sandia(J48) | t05 우리 ho .99 (15 ep) · GIST ho .96 · 떨림 329/93/48 (오프라인 참고) | **C5′** — t05 에서 #2 와 번갈아 각 3 회 | 1 | 2호기 지시(sim df9bebe) · **public(10/15 08:05 KST)** | — |
| 53 | `NAJY_act_all11_c100_gist30_basefix_27D_60k_s1000` | #2(J14) 와 같은 설정·시드, **우리 데이터의 베이스 미기록 ep 180 개 제외**(t05 5 · t06 5 · t07 67 · t08 103 — [`data_layout_traps.md`](data_layout_traps.md) §6), DGX1(J49) | 조작 8/8 (t08 .58 · t11 .63) · GIST ho 7/8 (t09 .49) — #2 와 같은 띠 | 다음 기준 레시피 후보(깨끗한 데이터판). t06 은 3% 만 바뀌어 실기 차이가 안 보일 것 — 가설 시험이 아니라 #2 대체 후보로 t01~t11 스크리닝. t08 은 #54 와 짝 | 1 | **public(10/15 KST)** | J49 |
| 54 | `NAJY_act_all11_c100_gist30_t08add_27D_60k_s1000` | #53 + `task08_takeout_and_put_beaker_0930`(새 녹화 72 ep, 베이스 정상)로 t08 보충, DGX1(J50) | 조작 8/8 — **t08 출발 1.05**(#53 .58, 새 녹화가 오프라인 t08 출발을 되살림) · GIST ho 7/8 (t09 .43) | **t08 에서 #53 과 번갈아** — 빼서 절반이 된 t08 데이터를 새 녹화로 채우면 나아지나 | 1 | **public(10/15 KST)** | J50 |
| 55 | `NAJY_act_all11_c100_gist30_basefix_27D_60k_s2000` | #53 의 시드 짝, sandia(J49-s2000) | 조작 우리 ho 7/8 (t11 .42) · GIST ho 8/8 · 원핫/영상 3/10 · 떨림 t05 392/109/68 — J14-s2000 과 같은 띠 (오프라인 참고) | #53 과 같은 회차 — 시드 폭 | 2 | 2호기 지시(J47 뒤) · **public(10/15 17:37 KST)** | — |
| 56 | `NAJY_act_all11_c100_gist30_mv165_27D_240k_s1000` | **#27(120K) 에서 +120K warm start**, 이동 단계 t01·t03·t10 가중 ×1.65(단계 몫 9.1% → 12.7%, 나머지 7.7%, GIST 30% 유지), sandia(J55) | 조작 우리 ho 7/8 (**t08 .31**) · GIST ho 8/8 · 이동 θ #27 과 같은 띠 · 원핫/영상 3/10 · 떨림 t05 220/53/39 (#27 276/67/46) (오프라인 참고) | 이동 노출을 전문가 수준(t01 ≈38 에폭)으로 — **t01·t03·t10 에서 #27 · #57 과 번갈아** | 2 | 2호기 지시(sim ae19647) · **public(10/16 15:30 KST)** | — |
| 57 | `NAJY_act_all11_c100_gist30_27D_240k_s1000` | #56 의 짝 — 같은 시작, 가중 그대로 +120K, sandia(J56) | 조작 우리 ho 7/8 (t11 .47 · t08 .66) · GIST ho 8/8 · 이동 θ #27 과 같음 · 원핫/영상 3/10 · 떨림 t05 209/52/39 (오프라인 참고) | #56 과 같은 회차 — #57 − #27 = 새 옵티마이저 +120K 효과(연속 240K 와 같지 않음) | 2 | 〃 · **public(10/16 15:30 KST)** | — |
| 58 | `NAJY_act_all11_c100_gist30_ctrl_27D_150k_s1000` | #27 에서 **그대로 +30K**(새 옵티마이저, 2장×16, 시드 1000), DGX1(J51) | 조작 7/8 (**t08 .35** — #27 .58) · GIST ho 8/8 · 끝 정지 t07 86% · t10 5% | **#59~#61 의 대조군** — 같은 회차에 같이 돈다. #27 과의 차이는 이어 학습만의 효과(1 시드 — 이 폭보다 작은 차이는 읽지 않는다) | 2 | **public(10/16 KST)** sha256 f27fdf63… | J51 |
| 59 | `NAJY_act_all11_c100_gist30_heads_27D_150k_s1000` | #58 + **단계별 출력 헤드·디코더 쿼리 오프셋**(단계가 파라미터를 고른다, +645,280 params, sim `act_stage_heads.py`), DGX1(J52) | 조작 7/8 (t08 .30) · GIST ho 8/8 · 원핫 영향 ↑(hot/img t07→08 .14→.48) · 직전 원핫 오차 증가: t04→05 **+.16**(#58 +.06) · t10→11 **+.29**(+.23) · t07→08 +.09(+.12, 출발 뒤) · **t03→04 −.22(−.11 — 모든 모델이 음수, 헤드가 키움)** | **같은 장면 경계에서 #58 과 번갈아 각 3 회**: t05(t04 끝 자리) · t11(t10 끝 자리) · t08(t07 끝 자리) · t04(t03 끝 자리 — 오프라인은 나빠짐). 비교 대상은 **#58** (#27 아님 — 이어 학습 효과가 섞인다). **로봇 플러그인에 단계 헤드 로드 훅이 있어야 실린다**(lerobot_trossen `act_stage_heads.py`) — 로그 `[stage_heads] 단계 헤드 K=11 를 붙여 싣는다` 가 없으면 중단 | 2 | **public(10/16 KST)** sha256 09f8c885… | J52 |
| 60 | `NAJY_act_all11_c100_gist30_headspad_27D_150k_s1000` | #59 + **pad_hold**(에피소드 끝 패딩을 「머물기」로 감독), DGX1(J53) | 조작 6/8 (t08 .27 · t11 .46) · GIST ho 8/8 · **끝 정지 t01 95% · t07 93% · t11 80%**(#58 0 · 86 · 35) · 경계는 #59 와 같은 방향 | #59 와 같은 회차 + **끝에서 멈추나**(t01·t07·t11 끝 정지) — 정지 감지 체인의 전제. 로드 줄은 #59 와 같음 | 2 | **public(10/16 KST)** sha256 b27f3c2e… | J53 |
| 61 | `NAJY_act_all11_c100_gist30_padhold_27D_150k_s1000` | #58 + pad_hold 만, DGX1(J54) | 조작 7/8 (**t08 .20**) · GIST ho 8/8 · 끝 정지 t01 100% · t07 93% · t11 55% | #60 의 짝 — 헤드 없이 pad_hold 만으로도 끝 정지가 되나(오프라인: 된다). 출발 비용: t08 | 3 | **public(10/16 KST)** sha256 34442f9c… | J54 |
| 62 | `NAJY_act_all11_c100_gist30_reef_sf_27D_60k_s1000` | #2 레시피(gist30 c100 60K) + **상대 EEF 행동**(팔 12칸 = 청크 시작 손끝 좌표계 변위, sim `scripts/eef_kin.py`) + **state 입력 제거**, 원핫은 ENV 토큰. DGX1(J57). 근거 arXiv 2509.18644 | **잡는 자리 따라감**(t02 ho 27 ep, 예측 닫힘 손끝 vs 그 ep 실제 자리, 20틱 전): R² 왼 .93·오른 .85 / #2 .71·.77, 오차 1.3·1.8 cm / #2 2.3·2.2 cm. 40틱 전 왼 .72 / #2 .49 · 출발 우리 ho 조작 **7/8**(t11 .30 — #2 .55) · GIST ho 8/8 · GIST t03 회전 출발 거의 0(+.06, #2 +.62) · **IK 실패(실행 60스텝 안)**: 우리 ho 2/712 팔-청크 · GIST ho **98/2760, t05 94/400** — 실패하면 그 팔·그리퍼가 그 청크 끝까지 멈춘다 [오프라인, n=27·1 시드] | **자리 보정 시험**: 재생으로 A✓ 를 맞춘 뒤 **랙·로봇 자리를 몇 cm 옮겨** #2 와 번갈아(ABAB). 명령 `HOT=<s>/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_reef_sf_27D_60k_s1000 <s> 60 1`. 로그에 `[rel_eef] 상대 EEF 체크포인트` 줄이 있어야 한다(없으면 중단), `MAX_REL=none` 은 거부된다. 실측: 청크 변환 시간(`[rel_eef] 청크 …` 줄)·IK 유지(`IK 실패/튐` — 팔이 멈추면 이 줄을 본다)·이동 단계에서 팔이 흘러가나 · t11 은 출발이 약할 수 있다 | 2 | **public(10/17 12:30 KST)** sha256 873496ce… | J57 |
| 63 | `NAJY_act_all11_c100_gist30_reef_27D_60k_s1000` | #62 와 같고 **state 유지**(대조), DGX1(J58) | 잡는 자리 따라감 20틱 전 R² .92·.92, 오차 1.2·1.3 cm (#62 와 같은 띠, #2 보다 위) · 40틱 전 .67·.58 · 출발 우리 ho 조작 **8/8**(t05 .59 — 7/15 ep 비<.5, #2 .99) · GIST ho 8/8 · **IK 실패(60스텝 안) 우리 ho 0 · GIST ho 16/2760**(t05) [오프라인] · **10/18 시드 3개(#63·#66·#67) 보정: 잡는 자리 따라감은 셋 다 같은 띠, 「IK 0·8/8」 은 시드 운이 섞였다** — GIST t05 IK 16/64/40, 우리 t11 출발 .68/.57/.44(셋 다 #2 .55 근처), GIST t07 은 #66 만 .50 · **10/19 보정: 로봇 어댑터와 같은 기준(스텝 튐 0.3 rad 포함)으로 다시 세면 GIST ho IK 유지 45/2760**(이전 값은 튐 기준이 빠졌다) | **먼저 돌린다** — #62 와 같은 자리 보정 시험을 #2 와 ABAB. 명령은 #62 와 같고 repo 만 `…_reef_27D_60k_s1000`. #62 와의 차이 = state 효과(오프라인에선 state 가 IK 실패를 줄이고 t11 출발을 지킴) · #2 와의 차이 = 행동공간 + ENV 토큰. t05 출발이 약할 수 있다 | **1** | **public(10/17 12:30 KST)** sha256 4353c1d2… | J58 |
| 64 | `NAJY_act_all11_c100_gist30_reef_sf_27D_60k_s2000` | #62 레시피 시드 2000, sandia(J59, 2장×16) | 우리 ho 조작 8/8 (t11 .57) · GIST ho 7/8 (t09 .48) · GIST t03 출발 회전 없음(θ −.03, 시연 −.33) · IK 잔차 최대 GIST 534 mm (오프라인 참고) | #62 의 시드 짝 | 2 | **public(10/18 08:58 KST)** · 허브 multi_manifest 있음(로봇 어댑터 rel_eef 인식용) | J59 |
| 65 | `NAJY_act_all11_c100_gist30_reef_sf_27D_60k_s3000` | #62 레시피 시드 3000, sandia(J60) | 우리 ho 7/8 (**t11 .25**) · GIST ho 7/8 (t02 .47) · GIST t03 회전 없음(θ −.00) · IK 잔차 최대 GIST 743 mm (오프라인 참고) | 〃 | 3 | **public(10/18 09:05 KST)** · 〃 | J60 |
| 66 | `NAJY_act_all11_c100_gist30_reef_27D_60k_s2000` | #63 레시피(상대 EEF + state 유지) 시드 2000, DGX1(J61) | 잡는 자리 따라감 20틱 전(예측 닫힘) R² .92·.89, 오차 1.2·1.5 cm (#63 과 같은 띠) · 40틱 전 .77·.58 · 출발 우리 ho 조작 8/8(t11 .57 — 9/20 ep 비<.5) · GIST ho **7/8**(t07 .498 — 46/90 ep 비<.5, #63 1.06) · **IK 실패(60스텝 안) 우리 ho 0 · GIST ho 64/2760**(전부 t05) [오프라인, n=27·1 시드] · **10/19 보정: 로봇 어댑터와 같은 기준(스텝 튐 0.3 rad 포함)으로 다시 세면 GIST ho IK 유지 93/2760**(이전 값은 튐 기준이 빠졌다) | #63 의 시드 짝 — #63 이 실기에서 좋거나 나쁠 때 시드 운인지 가르는 용도. 명령 `HOT=<s>/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_gist30_reef_27D_60k_s2000 <s> 60 1`. 확인할 것은 #62 와 같다(`[rel_eef]` 줄·변환 시간·IK 유지). t07(GIST 장면)·t05 IK 유지를 본다 | 3 | **public(10/18 00:40 KST)** sha256 f76ef730… | J61 |
| 67 | `NAJY_act_all11_c100_gist30_reef_27D_60k_s3000` | 〃 시드 3000, DGX1(J62) | 20틱 전 R² .92·.95, 오차 1.2·1.0 cm · 40틱 전 .64·.63 · 출발 우리 ho **7/8**(t11 .44 — 12/20 ep 비<.5) · GIST ho 8/8 · **IK 실패 우리 ho 0 · GIST ho 40/2760**(t05 39·t11 1) [오프라인] · **10/19 보정: 로봇 어댑터와 같은 기준(스텝 튐 0.3 rad 포함)으로 다시 세면 GIST ho IK 유지 68/2760**(이전 값은 튐 기준이 빠졌다) | 〃 명령은 repo 만 `…_s3000`. t11 출발이 약할 수 있다 | 3 | **public(10/18 00:40 KST)** sha256 a1017cf1… | J62 |
| 68 | `NAJY_act_all11_c100_gist30_reef_27D_120k_s1000` | #63(J58) 60K 를 **120K 까지 이어 학습**, DGX1(J63) | 120K: 잡는 자리 따라감(예측 닫힘) 20틱 전 R² .94·.94 · 40틱 전 .78·.53(#63 .67·.58 — 왼손만 나아짐) · 출발 우리 ho 7/8(**t11 .48**, #63 .68) · GIST ho 8/8 · **IK 유지(60스텝 안, 새 기준) 우리 ho 0 · GIST ho 89/2760**(t05 81·t11 6 — #63 45 의 2배) [오프라인, n=27·1 시드]. 중간: 75K t11 .39·IK 58 / 90K 8/8·t11 .62·IK 60 / 105K 우리 8/8·t11 .74·IK 66 이지만 GIST 7/8(t09 .39) — 뚜렷이 나은 중간 체크포인트 없음 | #63 보다 나은 점은 잡기 40틱 전 왼손뿐이고(오른손은 .58→.53) GIST t05 IK 유지는 늘었다 → #63 다음. 명령은 #63 과 같고 repo 만 `…_reef_27D_120k_s1000`. #63 과 ABAB, t05·t11 의 IK 유지 줄을 본다 | 3 | **public(10/19 09:00 KST)** sha256 15d53cb4… (sim 1934ea6, 60K 사본 이어받기) | J63 |
| 69 | `NAJY_act_all11_c100_gist30_reef_sf_27D_120k_s1000` | #62(J57, state 제거) 60K → 120K, DGX1(J64) | 120K: 잡는 자리 따라감(예측 닫힘) 20틱 전 R² .91·.87 · **40틱 전 .75·.76**(#62 .72·.62) · 출발 우리 ho 7/8(t11 .43, #62 .30) · GIST ho 8/8 · GIST t03 회전 없음(θ +.12) · **IK 유지 우리 ho 0 · GIST ho 70/2760**(t05 69 — #62 123) [오프라인]. 중간: 90K 우리 8/8·IK 132 / 105K IK 76 — 120K 가 가장 고름 | **#62 를 대신한다** — state 제거 레시피는 더 구우니 IK 유지·잡기 40틱 전·t11 이 모두 나아졌다. 명령은 #62 와 같고 repo 만 `…_reef_sf_27D_120k_s1000`. #63 과 ABAB 하면 state 효과를 같은 학습량에서 본다 | 2 | **public(10/19 09:01 KST)** sha256 a1ea5594… (sim 1934ea6) | J64 |
| 70 | `NAJY_act_all11_c100_gist30_reef_sf_27D_120k_s2000` | #64(J59) 60K → 120K, sandia(J65) | 120K: 출발 우리 ho 7/8(**t11 .29**, #64 .57) · GIST ho 8/8 · GIST t03 회전 없음(θ −.02) · **IK 유지(60스텝 안, 새 기준) 우리 ho 35/712(t09) · GIST ho 48/2760**(t05 47 — #64 80) [오프라인, 1 시드]. 중간: 90K t11 .69·IK 36 이 가장 좋지만 한 체크포인트뿐 · t11 은 체크포인트마다 .29~.69 | #69 의 시드 짝 — #69 가 실기에서 좋거나 나쁠 때 시드 운인지 가르는 용도. 명령은 #69 와 같고 repo 만 `…_120k_s2000`. t11 출발이 약할 수 있다 | 3 | **public(10/19 21:40 KST)** sha256 a02df08c… · 허브 multi_manifest 있음 | J65 |
| 71 | `NAJY_act_all11_c100_gist30_reef_sf_27D_120k_s3000` | #65(J60) 60K → 120K, sandia(J66) | 120K: 출발 우리 ho 7/8(t11 .41, #65 .25) · GIST ho 8/8 · GIST t03 회전 없음(θ −.00) · **IK 유지 우리 ho 2/712 · GIST ho 64/2760**(t05 62 — #65 99) [오프라인, 1 시드]. **state 제거 120K 시드 3개(#69·#70·#71)**: GIST IK 유지 70·48·64(60K 123·80·99 에서 모두 줄어듦) · t11 출발 .43·.29·.41(셋 다 약함) · GIST t03 회전 출발 셋 다 없음 | 〃 명령은 repo 만 `…_120k_s3000` | 3 | **public(10/19 21:41 KST)** sha256 806b3ab6… · 〃 | J66 |
| 72 | `NAJY_act_task01_c100_gist30_solo_16D_60k_s1000` | **t01 단일 단계 전문가** — #51(J47) 레시피 그대로(c100 · GIST 0.43 · 원핫 없음 16D · base zero · 60K · 2장×16), 데이터 우리 t01 80 ep(0~79) + GIST task1 1,836 ep, **sandia(J67, 1호기에서 옮김)** · 사용자 10/19 「Task 1, 2, 3 개별 모델 하나씩」 | 학습 중 — 15K~60K 마다 홀드아웃 80~99 첫 청크 θ·x 비(#27·팀 전문가와 짝) + 10/15 실기 첫 프레임 재예측 | t01 3 회, 회전 예산 켬: `BASE_BUDGET_DEG=84 BASE_BUDGET_SRC=meas LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_task01_c100_gist30_solo_16D_60k_s1000 1 60 1` — 로그에 `zero-pad only` 없으면 중단. 판정은 10/19 새 기준(같은 날 재생 대비 \|Δθ\| ≤ 8° · \|Δx\| ≤ 0.15 m · 충돌 없음). 짝은 #27 e60 bb84m | **1 (10/20 §3)** | 2호기 투입 10/19 22:27 KST(역검증 OK) · 끝 ≈10/20 09:00 KST [추정] · **중단 10/19 23:08 KST → #76~#79 로 대체**(사용자 「홀드아웃없이」「학습량도」) | J67 |
| 73 | `NAJY_act_task02_c100_gist30_solo_il_16D_60k_s1000` | **t02 단일 단계 전문가** — 같은 레시피, 데이터 우리 t02 재분할 il 107 ep(빨강 80 · **투명 27**) + GIST task2 1,658 ep, DGX1(J68). #51 과 다른 축 = 투명 튜브 학습 장면 6 → 27 (팀 전문가 14D 는 투명 33 을 다 보고 10/15 A✓ 에서 왼손을 쥐었다) | 학습 중. **비교 주의**: il 홀드아웃 26 ep 중 21 ep 를 #51·#2 가 학습했다 → #51·#2 와는 둘 다 안 본 5 ep(109·114·119·124·129)와 실기로만 비교 | 2a 자리(재생 ep102 로 A✓)에서 3 회: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_task02_c100_gist30_solo_il_16D_60k_s1000 2 60 1` — 로그 `zero-pad only` 확인. 짝은 #51 · #2 (번갈아) | **1 (10/20 §1)** | 학습 중 1호기(≈10/20 10:15 [추정]) · **중단 10/19 23:08 KST → #76~#79 로 대체**(사용자 「홀드아웃없이」「학습량도」) | J68 |
| 74 | `NAJY_act_task02_c100_gist30_solo_il_reef_16D_60k_s1000` | #73 + **상대 EEF 행동(state 유지, 원핫·ENV 없음)**, **sandia(J69, 1호기에서 옮김)** — 변형 | 학습 중 — il 홀드아웃 26 ep 에서 #73 과 짝(잡는 자리 따라감 R²·IK 유지) | **#63 첫 실기(2a 어댑터 확인)가 정상일 때만**. 명령은 #73 에서 repo 만 `…_il_reef_16D_60k_s1000`, 로그 `[rel_eef] 상대 EEF 체크포인트 -- … (state_free=False …)`·`zero-pad only` 둘 다 있어야 한다 | 1 조건부 (10/20 §1) | 2호기 투입 10/19 22:27 KST(역검증 OK) · 끝 ≈10/20 09:00 KST [추정] · **중단 10/19 23:08 KST → #76~#79 로 대체**(사용자 「홀드아웃없이」「학습량도」) | J69 |
| 75 | `NAJY_act_task03_c100_gist30_solo_16D_60k_s1000` | **t03 단일 단계 전문가** — 같은 레시피, 데이터 우리 t03 93 ep(0~92) + GIST task3 1,878 ep, DGX1(J70). state 유지(state 제거는 GIST t03 회전 출발을 죽였다) · GIST task3 유지(#2 의 t03 성공과 같은 데이터) | 학습 중 — 홀드아웃 93~116(#2 와 같은 분할)에서 첫 청크 θ 비·max\|θ\|(≥0.3 을 내나 — 느린 회전은 베이스가 44% 만 따라간다) | **튜브를 쥔 채 출발**(10/16 §2 3행 자리) 3 회: `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_task03_c100_gist30_solo_16D_60k_s1000 3 60 1` — 로그 `zero-pad only` 확인. 짝은 #2 · 팀 전문가 `task03_turn_to_face_beaker_14D` | **1 (10/20 §2)** | 학습 중 1호기(≈10/20 07:45 [추정]) · **중단 10/19 23:08 KST → #76~#79 로 대체**(사용자 「홀드아웃없이」「학습량도」) | J70 |
| 76 | `NAJY_act_task01_c100_gist30_soloall_16D_60k_s1000` (120K: `…_120k_s1000`) | **t01 단계 전문가** — #51(J47) 레시피(c100 · GIST 0.43 · 원핫 없음 16D · base zero · lr 3e-5 일정) · **우리 데이터 전 ep(홀드아웃 없음) + GIST 학습 ∪ GIST 홀드아웃(no_motion 제외)** · **120K**(60K 를 먼저 public — 일정 lr 이라 60K 단독 학습과 갱신 경로가 같다). 우리 t01 100 ep + GIST task1 2,040 ep, sandia(J71) | 홀드아웃 없음 — 오프라인 채점 안 함(실기가 판정) | 10/20 §3: `BASE_BUDGET_DEG=84 BASE_BUDGET_SRC=meas LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_task01_c100_gist30_soloall_16D_60k_s1000 1 60 1` ↔ #27 bb84m, 3 대 3. 로그 `zero-pad only` | **1 (10/20 §3)** | 2호기 투입 10/19 23:15 KST(역검증 OK) · **60K public 10/20 09:53 KST**(허브 f06fe51b, multi_manifest 있음) · 120K ≈20:50 KST [추정] | J71 |
| 77 | `NAJY_act_task02_c100_gist30_soloall_16D_60k_s1000` (120K: `…_120k_s1000`) | **t02 단계 전문가** — #51(J47) 레시피(c100 · GIST 0.43 · 원핫 없음 16D · base zero · lr 3e-5 일정) · **우리 데이터 전 ep(홀드아웃 없음) + GIST 학습 ∪ GIST 홀드아웃(no_motion 제외)** · **120K**(60K 를 먼저 public — 일정 lr 이라 60K 단독 학습과 갱신 경로가 같다). 우리 t02 133 ep(빨강 100 · 투명 33) + GIST task2 1,841 ep, DGX1(J72) | 〃 | 10/20 §1: (#77 → #51 → #2) × 3. `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_task02_c100_gist30_soloall_16D_60k_s1000 2 60 1` · 로그 `zero-pad only` | **1 (10/20 §1)** | 학습 중 1호기(투입 23:08 KST, 역검증 OK) · 60K ≈10/20 10:45 [추정] | J72 |
| 78 | `NAJY_act_task02_c100_gist30_soloall_reef_16D_60k_s1000` (120K: `…_120k_s1000`) | #77 + **상대 EEF 행동(state 유지, 원핫·ENV 없음)**, sandia(J73) — 변형 | 〃 | 10/20 §1 조건부: #63 첫 실기 정상일 때만 × 3. 명령은 #77 에서 repo 만. 로그 `[rel_eef] 상대 EEF 체크포인트 -- … (state_free=False …)` · `zero-pad only` | 1 조건부 (10/20 §1) | 2호기 투입 10/19 23:15 KST(역검증 OK) · **60K public 10/20 09:21 KST**(허브 068842c7, multi_manifest 있음) · 120K ≈20:30 KST [추정] | J73 |
| 79 | `NAJY_act_task03_c100_gist30_soloall_16D_60k_s1000` (120K: `…_120k_s1000`) | **t03 단계 전문가** — #51(J47) 레시피(c100 · GIST 0.43 · 원핫 없음 16D · base zero · lr 3e-5 일정) · **우리 데이터 전 ep(홀드아웃 없음) + GIST 학습 ∪ GIST 홀드아웃(no_motion 제외)** · **120K**(60K 를 먼저 public — 일정 lr 이라 60K 단독 학습과 갱신 경로가 같다). 우리 t03 117 ep + GIST task3 2,086 ep, DGX1(J74). state 유지 · GIST task3 유지 | 〃 | 10/20 §2 튜브 쥔 출발: (#79 → #2 → 팀 전문가) × 3. `LEROBOT_CHUNK_SEAM_BLEND_TICKS=10 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh kiroaiseoul/NAJY_act_task03_c100_gist30_soloall_16D_60k_s1000 3 60 1` · 로그 `zero-pad only` | **1 (10/20 §2)** | 학습 중 1호기(투입 23:08 KST, 역검증 OK) · 60K ≈10/20 10:45 [추정] | J74 |

**순서 제안**(옛 요약 — 위 「로봇 PC 전달」 S0~S11 이 정본): 우선 1 → #1 exec A/B(1 h) → #2·#12 선정 맞대기(≈3 h, #12 public 뒤) → #3 R1·R2(1 h) → 우선 2(#4·#5·#6·#9·#13·#14·#15) → 우선 3 → 선정 뒤 #17·#18. 무접촉 ① 은 모델마다 10분이라 **먼저 전부 돌려 두면** 이후 순서를 바꿀 근거가 생긴다.

## 결과 기록 (여기에 한 줄씩 — 상세는 결과 문서)
| 날짜 | # | 회차 | 성공/시도 | 실패 분류 | 결과 문서 |
|---|---|---|---|---|---|
| 10/11 | #1 J10 | 1011_2141 t01 · 2155 t02 | 0/2 | t01 목표 왼쪽(늦은 회전·기어감) · t02 `그리퍼`(왼쪽 우연히, 오른쪽 못 듦, 떨림) | [1011](eval_najy_results_1011.md) |
| 10/11 | #2 J14 | 1011_2143 t01 · 2157 t02 | 1/2 | t01 **성공(사람: 넷 중 최고)** · t02 `기타`(위치 못 감, 시늉·떨림) | [1011](eval_najy_results_1011.md) |
| 10/11 | #12 SmolVLA | 1011_214753 t01 · 220044 t02 | 0/2 | t01 도달 뒤 계속 감 · t02 덜 다가감·떨림 | [1011](eval_najy_results_1011.md) |
| 10/11 | #3 B | 1011_2149 t01 · 2203 t02 (러너 단일 단계) | 0/2 | t01 러너 timeout 으로 미도달 · t02 **거의 성공(사람: 넷 중 최고)**, 놓쳤는데 자동 완료(거짓 완료) | [1011](eval_najy_results_1011.md) |
| 10/13 | #2 J14 | 1013_1049 러너 · 1052 끔 · 1053 켬 · 1100 · 1105 · 1114 e100 (전부 t02) | 0/6 | `그리퍼` 빈손(왼 j2 시연보다 짧음) · 러너 그리퍼 가드 REFUSED 정상 · 블렌딩 켬 = 경계 튐 18×→0.8× | [1013](eval_najy_results_1013.md) |
| 10/13 | #32 gist30mv_sw | 1013_1057 · 1103 · 1112 e100 (t02, 블렌딩 켬) | 0/3 | `그리퍼` 빈손, 왼손 계속 재시도 · 사람: J14 보다 가능성 | [1013](eval_najy_results_1013.md) |
| 10/13 | #3 B | 1013_1050 (t02, eval_najy.sh) | 무효 | action 17D 로드 실패 — 러너로만 | [1013](eval_najy_results_1013.md) |
| 10/13 | #12·#35·#36 SmolVLA | t01 101354·101659·101830 · t02 111855·112129·112346 | 0/6 | t01 셋 다 82° 지나 계속 돎(203~339°) · t02 청크마다 뻗었다 당김, #35 최악·#36 가장 차분 | [1013](eval_najy_results_1013.md) |
| 10/13 | t03 — #2 J14 · #32 · #3 B 러너 · #36 SmolVLA | 1013_1139 · 1141 · 1143 · 114526 (빈손 출발) | **J14 1/1** · 나머지 0/3 | 셋 다 덜 돎(−44~−67°, 시연 −81°) — 빈손이라 팔 먼저 움직인 뒤 회전 + 베이스 저속 명령 손실 | [1013](eval_najy_results_1013.md) |
| 10/14 | t02 순서 1 — #2 · #44 · #43 · #44 (+재실행 #2·#44) | 1014_1606 · 1608 · 1611 · 1618 · 1624 · 1627 | 0/6 | 모델은 튜브를 보지 않음: #2 는 시연 닫힘 자세 그대로, #44 는 매번 같은 쪽으로 치우침. 사람: 튜브 구멍은 시연과 같다 | [1014](eval_najy_results_1014.md) |
| 10/14 | t02 시연 재생 ep102 (자리 맞추기) | 1014_1638 · 1644 · **1647** | 1647 양손 쥠 | 첫 자리에선 재생도 빈손 → 실패의 큰 몫은 로봇–랙 배치. 1647 자리 = 시연 자리 A | [1014](eval_najy_results_1014.md) |
| 10/14 | t02 자리 A — #2 · #44 · #43 | 1014_1657 · 1711 · 1714 | 0/3 (**#2 왼손 집어 올림**) | #2 오른손 안 뻗고 그리퍼 안 엶 · #44 치우친 자세로 못 잡음 · #43 살짝 쥐고 task03 동작으로 넘어감 | [1014](eval_najy_results_1014.md) |
| 10/15 | t01 — #2 · #40 · #39 · #29 · #27 · #25×2 | 1015_0054 · 005604_sv · 0100 · 0101 · 0104 · 0107 · 0110 | **#27 1/1** · 나머지 0 | #27 시연과 같은 도착·정지(이어서 task02·03 동작) · #39 도착 뒤 +7° · #2 +54° · #29 +30° · #25 회전하며 전진→충돌(원핫은 따름) · #40 SmolVLA 정지 실패 | [1015](eval_najy_results_1015.md) |
| 10/15 | t02 시연 튜브 A✓ — #2×2 · #39 · #29 · #27 · #25 · 전문가 14D · #40 | 1015_0128 · 0129 · 0130 · 0134 · 0135 · 0136 · 013856_sv · 0141 | **#40 1/1 (양손+정지)** · 전문가 왼손만 · 나머지 0 | 11단계 ACT 다섯은 같은 짧은 왼손 닫힘·오른손 거의 안 함. #40 SmolVLA+쿼리 양손 쥠·정지(떨림). 전문가 왼손 완벽·오른손 마지막 몇 cm | [1015](eval_najy_results_1015.md) |
| 10/15 | t01 확정 비교(재생 ep30 자리) — #27 · #2 · #49 · #1 · #39 · #27 (6/15) | 1015_1045 · 1047 · 1048 · 1050 · 1051 · 1053 | **#27 2/2** · 나머지 0/1 | #27 만 제자리 회전 먼저 → 도착·정지. #2·#49·#39 처음부터 전진해 충돌, #1 출발 못함. 자리 어긋난 6 회는 무효 | [1015](eval_najy_results_1015.md) |
| 10/15 | t06 — #40 SmolVLA+쿼리 | 1015_110222_sv (재생 ep93 자리) | 0/1 | 실패, 후진 0.40 m·−51°. 재생 ep93 도 −50° — 데이터 베이스 칸 확인 필요 | [1015](eval_najy_results_1015.md) |
| 10/19 | t01 1순위 11/16 — (전문가 → #27 e60 → #27 bb105 → #27 e30) × 3, e30 셋째 생략 | 1019_1406 · 1407 · 1410 · 1412 · 1414 · 1416 · 1420 · 1421 · 1427 · 1428 · 1429 · 1431 | 전문가 0/3 · #27 e60 1/3(+미기록 1) · bb105 0/3 · e30 0/2 | 전 설정 시연보다 7~36° 과회전. 전문가 출발 빠르나 +96~+102°·2 m. bb105 실측 +95~+97°(명령 기준 105°). e30 출발 막고 +109~+118°. 오늘 #27 출발 느림(출발 방향 몇 도 차 [추정]) | [1019](eval_najy_results_1019.md) |
| 10/15 | ↳ 정정(03:10 UTC, 1호기(DGX1) 재예측) | 1015_1059 | — | **1059 는 재생이 아니었다**(`LEROBOT_REPLAY_DATASET` 빠짐 → HOT=6/11 ACT 가 돎, 팔 대조로 #2 추정). 데이터 베이스 칸 정상(두 판 모두 −30° 아래 ep 0). −50° 는 정책 행동이며, 팔 시작 자세가 분포 밖(ep93 과 32°)이었다. t01: 「처음부터 전진」 은 모델 성질(#2 3 s 0.05~0.25 m vs #27 ≤ 0.04, 같은 장면) | [1015 DGX 분석](eval_najy_results_1015.md) |
