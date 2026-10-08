# 실기 Eval 큐 — 10/08 (DGX → Trossen PC1) · 안 해 본 것까지 한 장에

> 목적: **태스크별 70% 를 흔들린 시작에서** ([`najy_plan_1008.md`](najy_plan_1008.md) G0→G1). 체인(⑤)은 **아직 금지**(아래 R1·R2 가 와야 해제).
> 이 문서가 회차 순서의 정본이다. 끝낸 줄은 상태 칸을 `✅ <회차명>` 으로 바꾸고, 결과는 지금처럼 `eval_najy_results_1008.md` 에 적는다.
> 모든 실기는 **1 에피소드씩 · `DRY_RUN=1` 먼저 · 시작 장면을 시연 첫 프레임과 대조**(`scene_rows.py`) · e-stop 돌려 빼기 · 비상정지 손 닿는 곳.
> 기록(공통): 출발 시각(s) · 첫 2 s 팔 이동 · 성공/실패와 지점(사람 한 줄) · clamped · `phase=policy` 루프 Hz · 시작 자세 오차(pose_guide) · 이동 단계는 ∫θ(°)·∫x(m) 명령/실측.

## 0. 로봇 무접촉 — 먼저, GPU 만 (30분)
실기 프레임(1123·1128·1135·1040·1110·1101·1102)은 로봇 PC 에만 있다. 체크포인트는 전부 public.
| # | 무엇 | 체크포인트 | 볼 것 | 상태 |
|---|---|---|---|---|
| O1 | **레버 사다리**(§1 요청, 10/08 02:48) | `NAJY_act_all11_tph_27D_120k_s1000`(1라운드) · `NAJY_act_all11_r3_27D_60k_s1000` | 1128·1135 프레임에서 p 와 첫 청크 팔 이동 — 어느 레버가 p≈1 머물기를 만드나 | 대기 |
| O2 | **5라운드 4변형 + 같은 스텝 대조** | `r5_{bh_nopad,nopad,bh_aff}_27D_15k_s1000` · `r5_bh_nopad_27D_15k_s2000` · 대조 `r4_27D_15k_s1000` (전부 `kiroaiseoul/NAJY_act_all11_…`) | 같은 프레임에서 p·첫 청크·머물기. DGX 홀드아웃(sim §94.30.2): 시작을 끝으로 읽는 건 bh, t04 출발 붕괴는 pad+bh | 대기 |
| O3 | **청크 100** — c1 `kiroaiseoul/NAJY_act_all11_c100_m1_27D_15k_s1000` (**08:25 UTC public**, M1 레시피+청크 100, sha256 앞 `a9e6126272209c76`); c2~c4(`…c100_{drop,nopad}_27D_15k_s1000`·`…c100_m1_27D_15k_s2000`)는 1호기 ≈10:30 UTC | 위 이름 | **100스텝 청크**의 50~99 구간 팔 이동이 시연 크기인가(앞 30 은 작아도 된다, sim §94.31). 2호기 홀드아웃: 11단계 중 9단계가 100스텝 안에 시연만큼 간다 — **실기 프레임 1128·1135·1123 에서도 그런가** 가 O3 의 질문. **실기 회차는 exec 60**(기다림 40~56틱을 지나고 2.9 s 마다 재추론) 으로: `scripts/eval_najy.sh kiroaiseoul/NAJY_act_all11_c100_m1_27D_15k_s1000 5 60 1` | **c1 가능** |
| O4 | **SmolVLA 30K** 같은 프레임 | `kiroaiseoul/NAJY_smolvla_all11_30000` (지시문 = 매니페스트 문장, state 16D 베이스 0) | 첫 청크 50 의 팔 이동·방향. 흐름 모델이라 노이즈 4개 평균 | 대기 |
| O5 | **t03 로봇 PC 정의 재계산** (§2) | TPH 120K · R3 60K · R4 60K | 「첫 1 s 누적 경로」 로 1110 프레임 | 대기 |

## 1. 실행 층 — 「exec 이 작으면 시작 자세를 못 벗어난다」 (§6) · 장면 t05 → t04 순 (약 1.5 h)
| # | 명령 | 회수 | 볼 것 | 상태 |
|---|---|---|---|---|
| A1 | `scripts/eval_najy.sh kiroaiseoul/act_task05_tube_disposal_60000 5 30 1` | 2 | 같은 전문가 exec 30: 시작 2~3 s 뒤에도 머무나 (예측: 머문다) | 대기 |
| A2 | `scripts/eval_najy.sh kiroaiseoul/act_task05_tube_disposal_60000 5 100 1` | 2 | exec 100: 약 2 s 기다렸다 출발하나 · 성공 · 「너무 빠르다」 가 (a) 시작 자세 따라잡기 (b) 시연 자체 (c) 잔떨림 중 무엇인지 — `CHUNK_LOG=1`, 첫 2 s 와 그 뒤 `\|action−state\|` 따로 | 대기 |
| A3 | `scripts/eval_najy.sh kiroaiseoul/act_task05_tube_disposal_60000 5 60 1` | 1 | exec 60: 기다림(40~56틱)은 지나고 열린 루프는 2.9 s — 「빠르다」 가 줄어드나 | 선택 |
| A4 | `scripts/eval_najy.sh kiroaiseoul/act_task04_pour_liquid_from_tubes_to_beaker_60000 4 100 1` | 2 | 둘째 태스크에서도 기다렸다 출발하나 · 붓기 성공 | 대기 |
| R1 | `FROM_STAGE=4 TO_STAGE=4 scripts/eval_chain.sh kiroaiseoul/NAJY_act_all11_r4_27D_60k_s1000 r4_t04_travel` (1128 조건 재현) | 1 | 러너 가드용 `arm_travel_rad` 실측 — 거짓 완료를 0.4 가 거부하나 | 대기 |
| R2 | 위 A4 중 **사람이 성공으로 판정한 1건**의 `arm_travel_rad ÷ arm_travel_required_rad` | — | 가드 위 경계(롤아웃 경로 ÷ 시연 중앙). R1·R2 가 오면 체인 금지 해제 | 대기 |
조건: A1·A2 는 **같은 장면·지정 시작 자세**(1002_1422 는 자세가 어긋나 0.56 rad 튀며 정지점을 깼을 수 있다). 1135 재현도 t05 장면에서 R1 과 같은 식으로 1회 하면 좋다.

## 2. SmolVLA 첫 실기 (§7) · 같은 장면에서 1 과 번갈아 (약 1 h) — **새 `scripts/eval_smolvla.sh` push 뒤** (codex 검토 중; 옛 스크립트는 단계 인자를 거부한다)
| # | 명령 | 회수 | 볼 것 | 상태 |
|---|---|---|---|---|
| S1 | `scripts/eval_smolvla.sh 5 rec 1` | 2 | 출발 · 성공 · 떨림(방향 바뀜/s, 10/02 전문가 1.46 대비) · 추론 ms·지연 스텝 | 스크립트 대기 |
| S2 | `scripts/eval_smolvla.sh 5 long 1` | 1 | commit 45 가 출발·떨림·점프를 바꾸나 | 스크립트 대기 |
| S3 | `scripts/eval_smolvla.sh 4 rec 1` | 2 | t04 — A4 와 나란히 | 스크립트 대기 |

## 3. 이동 단계 — 한 번도 안 한 것 (§8) · 장면 t06 → t10 → t01 (약 1.5 h)
| # | 명령 | 회수 | 볼 것 | 상태 |
|---|---|---|---|---|
| V1 | `scripts/eval_najy.sh M3 6 30 1` | 2 | **t06 첫 실기**. 잡기 → 냉장고 앞: 출발(시연 베이스 출발 9 s 뒤) · ∫θ·∫x vs 21 Hz 환산 · 멈추는 자리 | 대기 |
| V3 | `scripts/eval_najy.sh kiroaiseoul/act_task06_pickup_beaker_and_move_to_refrigerator_14D_all_260923 6 100 1` | 1 | t06 전문가 e100, 같은 장면 | 대기 |
| V2 | `scripts/eval_najy.sh M3 10 30 1` | 2 | **t10 첫 실기**. 선반으로: ∫θ·∫x · 멈추는 자리 | 대기 |
| V3' | `scripts/eval_najy.sh kiroaiseoul/act_task10_move_to_beaker_shelf_14D_100k_fp32 10 100 1` | 1 | t10 전문가 e100 — DGX 오프라인(sim §94.33)에선 3.9 s 기다렸다 시연 크기로 돈다 | 대기 |
| V4 | `scripts/eval_najy.sh M3 1 30 1` — **비커 선반 앞**(t11 끝난 자리) | 1 | t01 을 맞는 장면에서 M3 로 (1156 의 빈 칸) | 선택 |
DGX 기대치(sim §94.33): M3 는 첫 프레임 출발에서 11단계와 같은 급. 멈추는 자리·과회전은 폐루프에서만 보인다. M3 는 `stage i/4 active` 로그 확인.

## 4. 한 번도 실기 안 한 조작 단계 — 실패 모양을 보려고 1회씩 (냉장고 장면 묶어서, 약 1 h)
| # | 명령 | 회수 | 볼 것 | 상태 |
|---|---|---|---|---|
| N1 | `scripts/eval_najy.sh kiroaiseoul/act_task07_open_refrigerator_14D_260916 7 100 1` | 1 | **t07 첫 실기** — 문 열기. 비커 든 채 | 대기 |
| N2 | `scripts/eval_najy.sh kiroaiseoul/act_task08_takeout_and_put_beaker_100k_251data 8 100 1` | 1 | **t08 첫 실기** — 그리퍼 복사 문제가 실기에서 어떻게 보이나 | 대기 |
| N3 | `scripts/eval_najy.sh kiroaiseoul/act_task09_close_refrigerator_100k_250data 9 100 1` | 1 | **t09 첫 실기** | 대기 |
| N4 | `scripts/eval_najy.sh M1 11 30 1` | 1 | **t11 첫 실기** — 전문가가 없어 M1(11/11). 선반에 비커 놓기 | 대기 |
| N5 | (보류) t02 | — | M1·R4 가 그리퍼를 못 열었다(1101·1102). 그리퍼 레버(2호기) 전까지 보류 | 보류 |
전문가 14D 는 `include_base_in_state=false` 로 돈다(스크립트가 폭으로 정한다). 성공 기준을 **한 줄로 적어 달라**(t07 「문이 열려 비커가 들어갈 틈」, t08 「든 비커가 냉장고 안, 꺼낸 비커가 손에」, t09 「문이 닫힘」, t11 「비커가 선반 위에 섰다」 — 틀리면 고쳐 달라).

## 5. 천장과 반경 (G0 끝 · G1) — 1~4 의 결과로 후보 하나를 고른 뒤
| # | 무엇 | 회수 | 산출 |
|---|---|---|---|
| C1 | 가장 나은 후보로 **시연 자리** t04·t05 | 각 5 | **천장** — 지금 데이터로 분포 안에서 몇 % |
| C2 | 같은 후보로 시작을 **10 cm 뒤 · 10° 틀기 · 20 cm 뒤** | 각 3 (t04 또는 t05) | **능력 반경** — 어디서 무너지나. 재수집 폭(±15 cm·±15° 임시)을 여기서 확정 |
| C3 | ACT 전문가 vs SmolVLA 를 같은 어긋남으로 | 각 1 | 흔들린 시작에서 누가 버티나 (§7 S4) |

## 6. 사람이 할 전송
- **1006·1008 eval 데이터셋**(`~/.cache/huggingface/lerobot/kiroaiseoul/eval_najy_*`)과 `~/eval_logs/1008_*` 를 DGX `/raid/kiro-ai/eval/real/datasets/` 로 — 위 0 절을 DGX 가 직접 돌리고, 실기 프레임으로 채점 정의를 검증하려면 필요하다. 아직 0개.
- 팀 묶음 전문가 `act_task06_task07_261006`·`…_task08_261006` 을 **누가 왜** 만들었는지.

## 7. 순서 제안 (오늘)
0 절(GPU) → 1 절 t05 장면(A1·A2·A3·1135 재현) → t04 장면(A4·R1·R2) → 2 절(스크립트 오면 t05·t04 장면 그대로) → 3 절 → 4 절 → 5 절.
장면 바꾸는 횟수를 줄이려고 같은 장면끼리 묶었다. 하루에 다 못 하면 **1 절·0 절이 먼저**다 — 다음 레시피(청크 100 exec 60·머물기 감독 제거)가 거기서 갈린다.
