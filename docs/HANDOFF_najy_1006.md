# 인수인계 — trossen-ai-simulation 세션 → lerobot_trossen (2026-10-06, DGX_1)

`trossen-ai-simulation`(DGX_1)에서 09-26~10-06 에 한 주행·11단계 정책 조사와 실기 준비를 **이 레포에서 이어 가기 위한** 문서다.
여기서 시작하는 세션은 이 문서 → [`eval_najy_session_guide.md`](eval_najy_session_guide.md) → [`eval_najy.md`](eval_najy.md) 순서로 읽는다.
학습·오프라인 판정의 정본은 private 레포 `Najongs/trossen-ai-simulation` 의 `docs/mobile_base_investigation.md`(§1~87, 2호기 S2-1~27)다.

## 1. 결론 — 지금 무엇을 쓰고 무엇을 안 쓰나

| 항목 | 결정 | 근거 (trossen-ai-simulation docs) |
|---|---|---|
| 정책 | **ACT** (SmolVLA 아님) — 같은 조건에서 3/4 태스크 우세, 추론 18배 빠름 | §64·§74·§77, S2-22·24 |
| 모델 구성 | **11단계를 한 모델로** (단계 원핫 입력). 이동·조작 모두 단계별 모델보다 같거나 낫다(오프라인) | §82·§83·§85, S2-26 |
| state 베이스 칸 | **0 (14D 학습)** — 속도를 넣으면 실연자 속도를 베낀다 | §58~61, S2-15·16 |
| 실행 주기 exec | **30** — 실기 task04 에서 exec 5 는 팔이 멈췄다. 시간 앙상블 금지 | §84·§87 |
| 학습 길이 | 11단계 120K (60K 에선 task01·06 이 밀림) | §83 |
| 오프라인 지표 | 녹화 재생(teacher-forced) 팔/베이스 MAE 는 **모델 간 순위 참고용**. 실행 설정·실기 성공 예측에 쓰지 않는다 (실기에서 두 번 틀림) | §84·§87 |
| 데이터 | 최대 레버는 그 단계 자신의 에피소드. 다른 현장(GIST)·서드파티 섞기는 해롭다 | §72·§80, §65 |

## 2. 만들어 둔 것 — 위치

| 무엇 | 어디 |
|---|---|
| 배포 모델 3개 (퍼블릭 HF, 원본과 sha256 일치) | `kiroaiseoul/NAJY_act_all11_hot_27D_120k_s1000` (M1) · `…_s2000` (M2) · `kiroaiseoul/NAJY_act_move4_hot_20D_60k_s1000` (M3) |
| 같은 모델 DGX 사본 + 체크섬 | `/raid/kiro-ai/deploy/<이름>/` |
| 원본 체크포인트 | `/raid/kiro-ai/outputs/act/exp_all11_hot_s{1000,2000}/checkpoints/120000` · `exp_move4_hot60k_s1000/checkpoints/060000` |
| 실기 원핫 패치 `LEROBOT_TASK_ONEHOT` | 이 레포 `packages/lerobot_robot_trossen/src/lerobot_robot_trossen/task_onehot_patch.py` (README 「Stage One-Hot」) — 학습측 변환과 비트 동일(오프라인), codex 교차 검토 반영 |
| 실기 회차 스크립트 · 분석 스크립트 | `scripts/eval_najy.sh` · `scripts/eval_motion_stats.py`(로봇 쪽 작성) · `scripts/eval_latency_stats.py` |
| 실기 순서표 · 세션 안내 · 결과 | `docs/eval_najy.md` · `docs/eval_najy_session_guide.md` · `docs/eval_najy_results_1002.md` |
| 오프라인 채점 도구 (trossen-ai-simulation) | `scripts/eval_rollout.py`(녹화 재생) · `scripts/closedloop_act.py`(재생 폐루프 근사, `--lag` `--start-offset` `--perfect`) · `scripts/eval_baseline.py` |
| 오프라인 결과 JSON | `/raid/kiro-ai/eval/` — `A11_s*_*_{60k,120k}.json`(녹화 재생) · `CL2_*`(폐루프 근사) · `CLso_*`(시작 자세 민감도) |
| 보고서 | trossen-ai-simulation `docs/report.html` = 웹 「주행 ACT 실험 현황판」 (claude.ai artifact, 비공개) |
| 볼트 | research_vault `research/ACT_Trossen.md` 갱신 로그 · CLM-20261002-trossen-exec-decide-on-robot-pcn8 · CLM-20261002-trossen-onehot-act-real-robot-rule-5m8p 등 (PR #103·#105) |

## 3. 실기 현황 (10/02, Trossen PC1)

- **0단계 기준선은 10/02 에 이미 쟀다** (`~/eval_logs/1002_step0_baseline.txt`, 로봇 PC): 텔레옵 녹화 20.9 Hz · 베이스 I/O 40.5 ms/프레임(상한 약 25 Hz) ·
  베이스 t63 회전 325 ms(t90 529) / 전진 163 ms(t90 285) · 정상상태 비 회전 1.064, 전진 0.954. 팔의 실기 지연은 아직 없다
  (학습 데이터에서는 state 가 action 을 2틱 늦게 따라간다).
- M1 task04 exec 30 **1/1 성공**, exec 5 0/2(정지) → exec 30 확정. task05 는 M1·M2 정지, 기존 전문가 1/3.
- DGX 근사: 팔 state 의 시작 자세를 0.35~0.7 rad 옮겨도 출력이 거의 안 변한다 → 시작 자세가 원인이면 **영상 경로**(손목 카메라에 보이는 것).
  그래서 순서표에 단계별 학습 시작 자세 표와 「허용」 거리를 넣었다.

## 4. 해야 할 일

### 사람 — 로봇에서 (명령은 Claude 가 만들어 넘긴다)
`docs/eval_najy.md` 순서대로: **A**(10/02 기록으로 지연 계산 — 로봇 무접촉) → **C**(task05 원인 가르기: 자세 맞춰 B* exec 100 · M1 · M2 각 3ep)
→ **D**(M1 exec 30, 11단계 각 3ep, 조작→이동) → **E**(실패 단계만 M2·M3·B* 비교) → **F**(단계별 승자). B 는 10/02 에 끝났다.

### Claude — DGX_1 에서 (데이터가 오면)
1. **실기 원자료 분석** — 사람이 `~/eval_logs` 와 `eval_najy_*` 데이터셋을 `/raid/kiro-ai/eval/real/` 로 옮기면:
   - 실패 회차의 실제 프레임(영상+state)을 같은 체크포인트에 다시 넣어 예측 청크를 본다 — 예측 자체가 「정지」 인가(장면 차이),
     예측은 움직이는데 실행이 멈추나(실행 쪽). 도구는 아직 없다 — `trossen-ai-simulation/scripts/eval_rollout.py` 의 `rollout()` 과
     `_eo.load_policy`·`tm.apply_task_onehot` 를 재사용해 「eval 데이터셋 한 에피소드를 teacher-forced 로 재생」 하면 된다
     (eval 데이터셋 state 는 14D 라 원핫 붙이기 전에 베이스 0 2칸을 끼워 16D 로 만든다).
   - 장면 차이 정도: 실기 첫 프레임과 학습 첫 프레임들의 영상 거리(예: ACT 백본 특징의 최근접 거리)를 단계별로.
2. 결과를 trossen-ai-simulation `docs/mobile_base_investigation.md` 새 절과 이 레포 `docs/eval_najy_results_<MMDD>.md` 양쪽에 남긴다.

### 제안만 해 둔 것 (사용자 승인 전 — 하지 마라)
- 영상 증강을 켠 11단계 ACT 재학습(1시드·120K) — 실기 장면 차이에 대한 보험. 오프라인에선 증강 효과가 판정 불가였다(§61).
- 학습 시작 자세로 팔을 보내는 리셋 스크립트 — 로봇을 움직이는 코드라 작성 후 codex 교차 검토, 실행은 사람.
- 원핫 패치를 조직 레포(kiro-ai-division)에 PR — 사용자 지시는 「Najongs fork main 에 직접」.
- 11단계 연속 실행(`stage_runner`, PR #48 미머지) — 리베이스·fps 21 타이머·원핫 전달 필드가 필요.
- 데이터 추가 수집 우선순위 task01·10·03(이동), 조작은 task02(에피소드가 가장 적다) — 사람 결정.

### 사람 — 행정
- 조직 Google 시트 「모델 체크포인트」 에 NAJY_ 3개 등록 (Claude 접근 불가).

## 5. DGX_1 에서 이 레포를 열 때 주의

- 원격이 둘이다: `origin` = kiro-ai-division(조직, **push 금지**), `najongs` = Najongs fork(**여기로 push**, main 직접). 로봇 PC 체크아웃은
  `origin` 이 fork 하나뿐이다 — 문서·스크립트에 remote 이름을 박지 마라 (메모리 push-to-najongs).
- DGX 의 `.venv` 는 lerobot 0.4.4. **`import lerobot_robot_trossen` 은 로봇 SDK(trossen_slate)까지 불러온다** — DGX 에서 테스트할 때는
  `task_onehot_patch.py` 처럼 파일 경로로 모듈만 로드한다. `eval_najy.sh` 는 `DRY_RUN=1` 로만.
- 학습·오프라인 채점은 이 레포가 아니라 `/home/kiro-ai/NAJY/trossen-ai-simulation` 에서 `uv run` 한다 (lerobot 0.4.1, `HF_HUB_OFFLINE=1`,
  데이터 `/raid/kiro-ai/lerobot/kiroaiseoul/…`). GPU·디스크 규칙은 전역 host.md(DGX_1).
- 로봇 PC 가 push 하는 결과 문서를 받으려면 `git fetch najongs && git merge --ff-only najongs/main`.
- 2호기(sandia)는 실험을 마무리했다(S2-27). 체크포인트는 sandia 로컬에만 있고 옮기지 않는다.
