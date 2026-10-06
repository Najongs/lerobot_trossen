# 인수인계 — DGX_1 에서 이 레포를 열 때 (2026-10-06)

`trossen-ai-simulation`(DGX_1)에서 09-26~10-06 에 한 주행·11단계 정책 조사·실기 준비를 **DGX_1 의 이 체크아웃**
(`/home/kiro-ai/NAJY/lerobot_trossen`)에서 이어 가기 위한 문서다. 로봇 PC(Trossen PC1)에서 연 세션에는 해당이 적다.

**현황·결론은 [`najy_overview.md`](najy_overview.md) 가 정본이다** (학습 결론 9개, 10/02·10/06 실기 결과, 다음 순서, 위치). 여기서 반복하지 않는다.
이 문서는 overview 에 없는 **DGX 쪽 사정**만 적는다: DGX 의 산출물 경로, 오프라인 도구, DGX 가 맡을 일, 주의사항.
읽는 순서: `CLAUDE.md` → [`najy_overview.md`](najy_overview.md) → 이 문서 → 가장 최근 `eval_najy_results_<MMDD>.md` 의 「다음」.

## 1. DGX_1 에 있는 것

| 무엇 | 경로 |
|---|---|
| 배포 모델 원본 체크포인트 | `/raid/kiro-ai/outputs/act/exp_all11_hot_s{1000,2000}/checkpoints/120000` (M1·M2) · `exp_move4_hot60k_s1000/checkpoints/060000` (M3) |
| 허브 업로드본과 바이트 동일한 사본 + `SHA256SUMS` + 학습 매니페스트 | `/raid/kiro-ai/deploy/act_all11_hot_27D_120k_s{1000,2000}/` · `act_move4_hot_20D_60k_s1000/` |
| 학습 데이터 (lerobot 0.4.1 v2 형식) | `/raid/kiro-ai/lerobot/kiroaiseoul/task0N_*` (11단계 원본) |
| 학습·평가 코드 | `/home/kiro-ai/NAJY/trossen-ai-simulation` — `scripts/train_multi.py`(원핫·`base_state`·`balance`), 매니페스트 `configs/datasets/all11_*.json` |
| 녹화 재생 채점 | `trossen-ai-simulation/scripts/eval_rollout.py` (`--self-base`, `--exec`, PROBE 교란) |
| 재생 폐루프 근사 | `trossen-ai-simulation/scripts/closedloop_act.py` (`--lag 2` = 실측 팔 지연, `--start-offset`, `--perfect` 검증) |
| 오프라인 결과 JSON | `/raid/kiro-ai/eval/` — `A11_s*_*_{60k,120k}.json`(녹화 재생) · `CL2_*`(폐루프, 조작 7단계 exec 30/10/5/1) · `CLso_*`(시작 자세 민감도) |
| 원핫 패치 동치 검사 (로봇 SDK 없이) | 패치 파일만 경로로 로드해 학습측 변환과 action chunk 를 비교 — 방식은 `docs/run_logs/2026-10-06_eval_najy/onehot_sens.py` 와 같다 |

## 2. DGX 가 맡을 일 (overview §4 의 2번)

1. **원핫이 실제로 단계를 고르게 만들기** — 10/06 실기·오프라인 교란에서 M1 은 원핫보다 카메라·팔 시작 자세로 단계를 정했다(task05 자리에서 task04 를 함).
   사람 결정: 시작 자세로 단계를 맞추는 우회는 채택하지 않는다. 학습 쪽 대책은 아직 설계 전이다. 후보와 먼저 잴 것:
   - ~~진단~~ **완료(10/06, sim 레포 §89, `scripts/stage_cond_diag.py`)**: 인접 경계 10곳 중 9곳에서 원핫은 단계를 고르지 않는다(M1·M2 공통). 유일한 예외는 t01→t02(이동→조작).
     대책은 task04/05 전용이 아니라 일반적이어야 한다. 2호기에 후속 셋(T8 증강 대조·fps·장면 유사도 행렬)을 넘겼다 — sim 레포 `docs/multi_server_setup.md` 「작업 지시 (10-06)」.
   - 학습 대책 후보 (설계 후 사용자 승인): ① 경계 구간 증강 — 앞 단계 마지막 N프레임을 다음 단계 원핫으로 「아무것도 안 함/다음 동작 시작」 라벨과 섞는다,
     ② 원핫을 state 끝 1칸 대신 더 강한 조건으로(FiLM 등 — ACT 구조 변경이라 비용 큼), ③ 원핫 드롭아웃의 반대 — 영상 일부를 가리고 원핫으로 맞히게.
     어느 것도 아직 재지 않았다. 1·2호기 기록에 같은 시도가 없다(원핫은 「성능 레버 아님」 으로만 판정됐다, §74·S2-18).
2. **fps 단서 확인** — 학습 데이터는 fps 30 으로 표기돼 있지만 실제 녹화 루프는 약 21 Hz 였을 수 있다(타임스탬프가 합성값).
   그러면 명령 적분 회전량이 −57°(30 기준) 가 아니라 −81°(21 기준)다. 실기 이동 단계(D) 해석에 필요하다. 녹화 런 로그(로봇 PC)나
   영상 프레임 간 시각 변화로 가를 수 있는지 본다.
3. **실기 원자료 분석** — 사람이 로봇 PC 의 `~/eval_logs`·`eval_najy_*` 데이터셋을 `/raid/kiro-ai/eval/real/` 로 옮기면, 실패 회차의 실제 프레임을
   같은 체크포인트에 넣어 「예측 자체가 정지/다른 단계인가」 를 본다(`eval_rollout.py` 의 `rollout()` 재사용, eval state 14D 에 베이스 0 2칸을 끼워 16D 로).
4. 결과는 trossen-ai-simulation `docs/mobile_base_investigation.md` 새 절(§88~)과 이 레포 `eval_najy_results_<MMDD>.md` 또는 overview 에 남긴다.

### 제안만 해 둔 것 (사용자 승인 전 — 하지 마라)
- 영상 증강을 켠 11단계 ACT 재학습 — 실기 장면 차이 보험(오프라인에선 판정 불가였다, §61).
- 원핫 패치를 조직 레포(kiro-ai-division)에 PR — 사용자 지시는 「Najongs fork main 에 직접」.
- 데이터 추가 수집 — 최대 레버, 사람 결정 대기 (overview §4.3).

## 3. DGX 에서 이 레포를 열 때 주의

- 원격이 둘이다: `origin` = kiro-ai-division(조직, **push 금지**), `najongs` = Najongs fork(**여기로**, main 직접). 로봇 PC 체크아웃은
  `origin` 이 fork 하나뿐이다 — 문서·스크립트에 remote 이름을 박지 마라. 받기: `git fetch najongs && git merge --ff-only najongs/main`.
- 로봇 PC 세션이 같은 문서(`eval_najy*.md`, overview, CLAUDE.md)를 수시로 고친다 — **고치기 전에 받고**, 충돌 나면 더 최근 실측을 남긴다.
- DGX 의 `.venv` 는 lerobot 0.4.4. **`import lerobot_robot_trossen` 은 로봇 SDK(trossen_slate)까지 불러온다** — 모듈 하나를 파일 경로로만 로드한다.
  `scripts/eval_najy.sh` 는 `DRY_RUN=1` 로만, `measure_base.py` 는 실행하지 않는다.
- 학습·오프라인 채점은 이 레포가 아니라 `trossen-ai-simulation` 에서 `uv run` 한다 (lerobot 0.4.1, `HF_HUB_OFFLINE=1`). GPU·디스크 규칙은 전역 host.md(DGX_1).
- 이 레포는 public — `/raid` 경로까지는 괜찮지만 서버 주소·토큰·IP 는 적지 않는다.
- 2호기(sandia)는 실험을 마무리했다(S2-27). 체크포인트는 sandia 로컬에만 있다.
