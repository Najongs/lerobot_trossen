# NAJY 다단계 ACT — 인계 (2026-10-09 저녁, DGX_1 세션 → 다음 세션)

> **10/09 저녁 추가 — 정책 선정(ACT vs SmolVLA) 진행 중**: 설계·굽는 순서·실기 회차·다음 할 일은 [`model_selection_1009.md`](model_selection_1009.md)(§10 이 다음 세션 순서). 학습 큐 J18(SmolVLA gist30 60K)이 J14 뒤 1호기 8장에 대기, J15 는 그 뒤로 밀렸다.
>
> **새 세션이 처음 읽는 문서.** 서버 이름: **1호기 = DGX1**(`kiro-dgx1`, `/home/kiro-ai/NAJY`, 이 문서를 쓴 쪽) · **2호기 = sandia**(`/home/sandia/NAJY`, RTX 3090 ×4) · 로봇 PC = Trossen PC1. 10/08~09 이틀의 결론·현재 돌아가는 것·다음 할 일을 한 장에 모았다. 세부는 각 정본으로 간다.
> 읽는 순서: 이 문서 → [`najy_plan_1008.md`](najy_plan_1008.md)(목표·관문·결정) → [`eval_queue_1008.md`](eval_queue_1008.md)(로봇 PC 가 돌릴 회차 전부) → sim 레포 `docs/mobile_base_investigation.md` §94.27~§94.38(측정 원문) → `docs/multi_server_setup.md`(2호기(sandia) 루프 큐).
> 머신 사정은 `claude-dotfiles/hosts/DGX_1/host.md` 가 정본이다. 10/06 이전 인계는 [`HANDOFF_najy_1006.md`](HANDOFF_najy_1006.md).

## 1. 목표 (사용자 결정, 10/08)
- **태스크별 성공률 ≥ 70% 를 흔들린 시작(±15 cm·±15°)에서.** 체인 전체 성공률은 지금 걱정하지 않는다(단계 재시도를 전제로 한 숫자다).
- 파일럿 태스크 **t04 붓기**. 모델은 **ACT 먼저**, SmolVLA 는 실기 결과를 본 뒤. **재수집은 보류** — 지금 데이터로 실행 층을 닫고(G0) 능력 반경(G1)을 잰 뒤 연다.
- 관문: G0 실행 층 걷어내기·천장 측정 → G1 반경 → G2 재수집(또는 G2' GIST 혼합) → G3 확장·체인.
- **10/09 저녁 추가(병렬 DGX1 세션, 사용자 합의)**: 그 전에 **ACT 냐 VLA 기반(SmolVLA)이냐를 먼저 선정** — 같은 데이터(gist30)·같은 유효 배치·60K·fp32 한 쌍 **J14(ACT, s1000) vs J18(SmolVLA, s1000)** 을 실기 t05·t04·t03 으로 맞댄다. 설계·회차·판정 규칙은 [`model_selection_1009.md`](model_selection_1009.md). **J15(GIST 50%)는 선정 뒤.** ACT 팔은 s1000 으로 미리 고정(사후 선택 편향 방지).

## 2. 이틀 동안 확정된 것 (근거는 sim §, 볼트 교훈 6개는 `session/DGX_1-20261008-1654-mcp` 브랜치 — **push 는 사람 몫**)
| # | 교훈 | 근거 |
|---|---|---|
| 1 | **출발 실패의 원인은 청크 길이다.** 시연이 시작 뒤 2~2.7 s(39~56틱) 기다리므로 청크 30 모델은 정지 시작 프레임에서 청크 전체가 「기다림」 이 되어 정지점에 갇힌다. 청크 100 은 한 청크 안에서 기다렸다 출발한다(전문가·11단계 두 시드·레시피 셋 재현). **학습은 청크 100, 실기는 exec 60.** t03 도 같은 문제(시연 회전은 40틱 뒤), t01 은 기다림 108틱이라 100 으로도 못 담는다 → **청크 150 이면 t01 첫 프레임 h150 비 1.02 로 시연만큼 간다**(sandia J16 15K). 다만 같은 15K·시드에서 t02·t03·t08 의 h100 이 c100 보다 낮다(8/10 vs 9/10) — 청크를 늘린 비용인지 시드 폭인지는 60K(J19)가 답한다 | §94.31·§94.33 정정·§94.34·§94.37 |
| 2 | **머물기 감독(pad_hold·boundary_hold)은 둘 다 뺀다.** bh 는 공유 장면 경계에서 시작 장면을 끝(p≈1)으로 읽게 하고 학습할수록 커진다, pad+bh 는 t04 출발을 깎는다. 끝 검출은 러너 가드(정지 + 누적 이동량) | §94.30.1~2 |
| 3 | **실기 후보는 홀드아웃이 아니라 실기 프레임 시험으로 고른다.** 시연 프레임에선 정상인 모델이 실기 장면을 끝으로 읽었다. 옛 출발 지표 idle% 는 과대 보고 → 비 중앙 ≥0.5 ∧ <0.3 비율 ≤20% | §94.27~29 |
| 4 | **청크 100 예측 청크는 시연의 1.6~5배 떨린다(학습량 따라 감소).** 팔 12관절 3틱 중심 이동 평균·양 끝 고정으로 시연 수준·L1 불변 → 러너 `LEROBOT_CHUNK_SMOOTH_TICKS=3`. 청크 30 의 경계 점프는 시연 1틱의 10~60배 | §94.32·§94.37, 2호기(sandia) chunk_smooth_check |
| 5 | **trim_idle(nopad) 레시피는 기다림을 안 배우고 t06·t11 첫 프레임이 붕괴**(두 시드). 체인용 17D 는 M1+드롭아웃+진행도(J7/B)로 — 진행도 칸은 출발을 깨지 않는다 | §94.34.1 |
| 6 | **GIST 혼합(가중치 고정 30%)은 우리 홀드아웃을 안 떨어뜨리고 다른 현장 출발을 올린다**(15K 만으로 7~8/10, 우리 데이터만 본 A 60K 는 3/10). 「섞으면 나빠진다」(§65·66)는 희석의 결과였다 | §94.38·§94.38.1 |
| 7 | 60K 에서 **레시피 차이보다 시드 폭이 크다**(M1 두 시드 10/10 vs 8/10). 증강(J4/J9)은 60K 에서 이득 없음. t11 약함은 첫 프레임 지표의 성질(출발 52틱·끝 자세 분산), 레시피 레버 아님 | §94.37, 2호기(sandia) J13 |
| 8 | [함정] 16D state 체크포인트를 `include_base_in_state=false` 로 돌리면 로봇 관측이 14D 라 정규화 첫 프레임에서 죽는다 → 원핫 패치 `LEROBOT_TASK_ONEHOT=0/0`(0 채움 전용). `train_multi` 매니페스트 사본은 `steps == save_freq` 런에서 경쟁으로 안 남던 것을 고쳤다(414c371) | README §Stage One-Hot |

## 3. 지금 돌아가는 것 (10/09 16:40 UTC — **최신은 sim `docs/loop_board.md` 종합 표**: 결과·큐·실기 이관 한 장, 알림마다 거기를 갱신한다)
| 머신 | 런 | 무엇 | 끝 | 그 뒤 자동 |
|---|---|---|---|---|
| 1호기(DGX1) GPU 0~7 | `exp_all11_c100_gist30_long_s{1000,2000}` (J14) | GIST 30% + M1 + 청크 100, 60K, 두 시드 — **Claude 가 직접 투입**(허용 규칙). s1000 이 선정 맞대기의 ACT 팔 | ≈10/10 02:20 UTC (0.64 s/step 실측) | public `NAJY_act_all11_c100_gist30_27D_60k_s*` + 우리/GIST 홀드아웃 출발 표 → `/raid/kiro-ai/eval/c100_gist_long_status.txt` |
| 2호기(sandia) GPU 0·1 | `t24_all11_c100_m1_60k_s3000` (J17) | 배포 후보 셋째 시드 60K | ≈22:50 UTC | 2호기(sandia)가 push |
| 2호기(sandia) GPU 2·3 | ~~J16 청크 150 15K~~ **끝(17:50 UTC)** → public `…c150_m1_27D_15k_s1000`, t01 h150 1.02 · 나머지 h100 8/10. 다음 **J19 = 청크 150 M1 60K s1000**(큐에 적음, sandia 가 잡는다) | — | sandia 가 push |
| 1호기(DGX1) GPU 0~7 | **J18** `sv_all11_gist30_s1000` — SmolVLA 팔(`smolvla_base` → `all11_tr_nohot_gist30.json`, 8장×4·60K, expert 는 bf16(lerobot 기본 — 정밀도 (a)/(b) 사용자 결정 대기)). **10/10 02:20 UTC 투입**(스모크·codex 뒤), 0.544 s/step | ≈10/10 11:30 UTC | waiter `sim scripts/dgx/sv_gist30_after.sh` → public `NAJY_smolvla_all11_gist30_60k_s1000` → `/raid/kiro-ai/eval/sv_gist30_status.txt` |
감시: `scripts/dgx/fork_watch.sh`(sim 레포, 3분 간격, 2 h 상한 → 다시 건다). 새 세션은 먼저 `/raid/kiro-ai/eval/*_status.txt` 와 두 fork 의 `git log najongs/main` 을 본다.

## 4. 후보 체크포인트 (전부 public `kiroaiseoul/…`)
| 용도 | 이름 | 홀드아웃 출발(10단계) | 비고 |
|---|---|---|---|
| **실기 1순위** | `NAJY_act_all11_c100_m1_27D_60k_s2000` (J10) | 10 | M1 레시피 + 청크 100. 실기는 `HOT=5/11 LEROBOT_CHUNK_SMOOTH_TICKS=3 scripts/eval_najy.sh <id> 5 60 1` (repo id 로 부르면 `HOT=<단계>/11` 필수) |
| 같은 레시피 둘째 | `…c100_m1_27D_60k_s1000` (A) | 8(+경계 2) | GIST 홀드아웃 3/10 — 시드 복권 |
| **체인용 17D** | `…c100_drop_prog_27D_60k_s1000` (B) · `…_s2000` (J12) | 9 · 8 | 러너 `EXEC=60`, 진행도 재현율 80%·오검출 0% |
| 장면 강건성 후보 | `…c100_gist30_27D_15k_s{1000,2000}` → 60K 진행 중 | 9·8 / GIST 7·8 | 실기 프레임(큐 O6)이 판정 |
| 증강 | `…c100_m1_aff_27D_60k_s1000` (J9) | 8 | 60K 에서 이득 없음 — 실기 프레임에서만 |
| 보류 | nopad 계열, 4·5라운드(청크 30), 4라운드 120K | — | 청크 30 은 출발 문제로 실기 보류 |
| SmolVLA | `NAJY_smolvla_all11_30000` | 출발 M1 보다 좋음(§94.11) | `scripts/eval_smolvla.sh <단계> rec` (0 채움 자동) |

## 5. 로봇 PC (Trossen PC1) — 받은 회신 없음 (10/08 11:43 KST 이후)
- 돌릴 것은 전부 [`eval_queue_1008.md`](eval_queue_1008.md): 무접촉 O1~O6 → 실행 층 A1~A6(전문가 exec A/B, **c100 60K exec 30/60/100**, 다듬기 on/off) → 러너 가드 R1·R2(**체인 금지 해제 조건**) → SmolVLA S1~S3 → 이동 단계 첫 실기 V1~V4 → 미실기 조작 N1~N4 → 천장·반경 C1~C3 → 백로그 §10.
- 누적 성공률은 [`eval_scoreboard.md`](eval_scoreboard.md) 한 장(사람 판정만). 상시 세트는 큐 §9.
- 러너·스크립트 변경(전부 codex 검토·push 됨): 조작 단계 거짓 완료 가드(`configs/chain/README.md`), `LEROBOT_CHUNK_SMOOTH_TICKS`, 원핫 패치 `0/0`, `scripts/eval_smolvla.sh` 일반화, `eval_chain.sh` EXEC 상한 100.
- 사람 몫 전송: eval 데이터셋(1006·1007·1008)·`~/eval_logs` → DGX `/raid/kiro-ai/eval/real/` — 아직 0개.

## 6. 두 서버 루프 (어떻게 도나)
- 큐는 sim `docs/multi_server_setup.md` 「GPU 송수신 루프」 표. **2호기(sandia)는 GPU 가 비면 「대기」 첫 줄을 잡아 상태를 먼저 push 하고 띄운다**, 1호기(DGX1)는 fork 감시로 받아 결과를 적고 큐를 채운다. 지금 큐: J14(1호기(DGX1))·J16·J17(2호기(sandia)) 진행, J15(GIST 50%)는 **1호기(DGX1) 몫**(GIST 데이터가 1호기(DGX1) `/raid` 에만 있다).
- 1호기(DGX1) 투입 규칙(host.md, 10/09): `scripts/launch_*.sh` 는 Claude 가 직접 띄운다 — 큐에 먼저 적고, 런처 자체 검사(GPU 여유·학습 프로세스·포트·중복)는 그대로, 띄운 뒤 로그 역검증(파라미터·프레임·플래그·시드)을 같은 세션에 남긴다.
- 역검증 기준값: M1 청크 100 = 51,628,944 파라미터·511,777 프레임 · +진행도 = 51,629,969·511,447 · +ENV·trim = 51,636,625·506,039 · 청크 150 = 51,654,544 · GIST30 = 6,461,644 프레임 + `Weighted sampling active`.

## 7. 열린 결정 (사용자)
1. **GIST 데이터(17 GB, 16셋)를 2호기(sandia)로 보낼지** — 허브 private 데이터셋 업로드 또는 직접 복사. GIST 소유 데이터라 사용자 판단. 그 전까지 GIST 런은 1호기(DGX1)만.
2. 재수집 시점(G2) — G0 천장·G1 반경 숫자 뒤. GIST 혼합이 실기 프레임에서 먹으면 폭을 줄인다.
3. 이동 단계 스크립트화(G3) · 팀 묶음 전문가(`act_task06_task07_261006` 등) 출처.
4. 볼트 브랜치 `session/DGX_1-20261008-1654-mcp`(교훈 6, 커밋 2) push.

## 8. 다음 세션이 할 일 (순서) — 정책 선정이 먼저. 진행 상태는 sim `docs/loop_board.md` 「할 일 S1~S11」 표가 정본
1. **J14 끝 처리(S1·S2)**: `/raid/kiro-ai/eval/c100_gist_long_status.txt` → GIST 60K 둘의 우리/GIST 홀드아웃 표를 sim §94.38.1 아래에 적고 종합 표 J14 행·실기 큐 O6 갱신.
2. **J18 SmolVLA 팔 투입(S3~S6)**: `/preflight` + codex 검토(새 레시피·60K, host.md ④) → `SMOKE=1 MIN_FREE_MIB=15000 scripts/launch_sv_gist30.sh`(20스텝, 로그 4줄 확인, `_smoke` 삭제) → 본런 → 역검증(smolvla·chunk 50·steps 60000·decay 60000·`num_frames=6,461,644`·`Weighted sampling active`·`multi_manifest.json` 에 `task_onehot` 0건·state 16) → Monitor. 명령·확인 항목 원문은 [`model_selection_1009.md`](model_selection_1009.md) §2.
3. **J18 60K(S7)**: 자동 public 아님 — `hub_upload_ckpt.py … --name NAJY_smolvla_all11_gist30_60k_s1000 --public` 수동, 스냅샷 `multi_manifest.json` 확인 → `model_selection_1009.md` §3·`eval_queue_1008.md` §8 을 「가능」 → 로봇 PC 에 알림.
4. **실기 결과(S8~S10)**: 로봇 PC 가 M1~M5(같은 시작 짝, ACT `HOT=<단계>/11` 필수)를 돌리면 모델별 A·B 성공·짝·실패 분류·출발 s·Hz 표를 `eval_najy_results_<MMDD>.md` 에, scoreboard 에 열 둘. **판정은 사용자**(문턱 없음·모델 혼합 없음).
5. 그 밖의 로봇 PC 회신(O3·O6·A5·R1·R2)은 그대로 처리 — 실기 프레임 표로 레시피를 가르고 scoreboard 갱신. 체인 금지는 R1·R2 뒤.
6. sandia 결과(J20 c150 s2000 ≈04:10 UTC · J21 c150 drop_prog ≈06:00 UTC)는 종합 표 행 갱신 + 빈 GPU 에 표의 「다음 빈 GPU 후보」 순서로 큐. J15(GIST 50%)는 선정 뒤.
7. 볼트: 새 사실이 생기면 `create_node`(DGX_1 은 커밋까지, push 는 사람). MCP 가 파일만 쓰고 멈추는 일이 있었다 — `git status graph/` 로 미추적 파일을 확인하고 `project: "[[ACT_Trossen]]"` 이중 괄호·`machine: [DGX_1]`·`brief:` 를 손봐 커밋.

## 9. 어디에 무엇이
| 무엇 | 위치 |
|---|---|
| 학습 산출물·로그 | 1호기(DGX1) `/raid/kiro-ai/outputs/act/<run>` · `/raid/kiro-ai/logs/act/<run>.log` (launch 줄에 git SHA·매니페스트 sha256) |
| 채점 원자료 | `/raid/kiro-ai/eval/` — `STARTCHUNK_*`(출발 청크) · `START_check_*` · `PROG_*` · `HOT_diag_*` · `score*_*.log` · `move_start_m3*.json` · `chunk_profile.json` · `gist_ho_baseline.log` |
| GIST 데이터·홀드아웃 | sim `data/gist/` (17 GB, 1호기(DGX1)만) · `configs/datasets/gist_ho/` · 혼합 매니페스트 `all11_tr_hot_gist30.json` |
| 런처 | sim `scripts/launch_c100_{short,extra,long,gist,gist_long}.sh` · `launch_v5_short.sh` |
| 자동화 | sim `scripts/dgx/`(감시·채점·후처리 템플릿·GIST 기준선, README) |
| 전문가 체크포인트(0.4.1 용 config 사본) | `/raid/kiro-ai/hf_experts_041/<name>` (`use_peft` 한 칸만 뺌) · 허브 사본 `/raid/kiro-ai/hub/` |
| 측정 도구 | sim `scripts/start_chunk_check.py`(first·trim, h30/100/150) · `stage_start_check.py` · `progress_check.py` · `end_stationary_check.py` · `stage_cond_diag.py` · 2호기(sandia) `chunk_smooth_check.py --fix-ends` · `t03_turn_check.py` |
| 메모리 | `~/.claude/projects/-home-kiro-ai-NAJY-lerobot-trossen/memory/` (`najy-act-state-1006.md`, `robot-observation-first.md`) — dotfiles 로 공유, push 됨 |
