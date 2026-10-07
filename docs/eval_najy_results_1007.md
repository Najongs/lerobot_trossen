# 다단계 ACT 실기 Eval 결과 — 2026-10-07 (Trossen PC1)

순서표 단계: **G. 체인 러너 bring-up ①·②** (DGX 10/07 아침 지시 A 의 첫 둘). 새 모델 없음(M1 설정, ② 는 가중치를 로드하지 않음).
원자료: `~/eval_logs/1007_1045_chain_resets_e30.*`, `outputs/stage_runner/1007_1045_chain_resets_e30/`(events·config), eval 데이터셋
`~/.cache/huggingface/lerobot/kiroaiseoul/eval_1007_1045_chain_resets_e30`(로컬만). 사본: [`run_logs/2026-10-07_eval_najy/`](run_logs/2026-10-07_eval_najy/).
수치 출처: 러너 events.jsonl·chain.md·로그 실측. 사람 관찰: 사용자 보고는 「돌려봤다」 — 충돌·이상 보고 없음(상세 관찰은 미기록).

## 받은 것 (10/07 아침, 1호기 26커밋)

체인 러너(`packages/stage_runner`, `configs/chain/`, `scripts/eval_chain*.{sh,py}`), `eval_najy.sh` 의 TPH/ENV 분기, 원핫 패치의 env 토큰·`set_stage`,
pose_guide 의 지정 자세. 지시는 `eval_najy_results_1006.md` 「DGX_1 → Trossen PC1 (10/07 아침)」: **A(bring-up ①~④, M1) → B(짧은 실험 3) → C(전송)**.

## 환경 (①)

- `uv lock && uv sync`: lock 에 `stage-runner` 하나만 추가(커밋 `0118a61`). lerobot 0.4.4 · torch 2.10.0+cu128 그대로. `import stage_runner` OK.
- 러너 테스트(unittest, mock) **221건 OK** — 이 환경에서.
- `DRY_RUN=1 scripts/eval_chain.sh M1` **통과**: `state 27D · action 16D (진행도 없음) · chunk 30 · exec 30 · fps 21 · env 토큰 없음 (1라운드)`,
  단계 파라미터 `11단계 · fps 21 · sim a07c939`, 원핫 패치·`set_stage` 확인, 조립 명령에 `--chain.*` 5개.
- 세션은 x11(ESC/→ 가능). **tmux 는 없다** — 로컬 터미널에서 창을 닫지 않고 돌렸다(ESC 로 끝냄).

## ② 리셋만 · 빈손 · 10경계 — `1007_1045_chain_resets_e30` ([chain.md](run_logs/2026-10-07_eval_najy/1007_1045_chain_resets_e30.chain.md))

`RESET_ONLY=1 scripts/eval_chain.sh M1 resets`. 시작은 대기 자세(task01 지정 자세에서 0.045 rad). 종료 코드 0, completed, 에피소드 저장(591프레임·59.2 s), 팔·카메라 disconnect 정상.

| 통과 기준 (G절) | 결과 | 판정 |
|---|---|---|
| 모든 경계 `reached`, 사유 「도달」 | 11/11 (최초 1 + 경계 10) | ✅ |
| 도달 오차 < tol 0.05 | 최대 **0.0019 rad** (0.0004~0.0019) | ✅ |
| `clamped` 0 | 전 리셋 0 | ✅ |
| Δmax < 상한 | 최대 **1.048** (t10→11), 0.981 (t08→09), 0.636 (t02→03), 0.556 (t07→08), 0.505 (t04→05); 최초 0.045 (상한 0.6) | ✅ |
| `POSE` 「목표까지」 단조 감소 → 0 | 모든 리셋에서 1초마다 줄어 0.00~0.03 → 0.00 | ✅ |
| 베이스 ∫x·∫θ ≈ 0 | 명령 적분 0 / 실측 적분 +0.0017 m · +0.27° (49 s, 잡음 수준; \|meas\| 최대 0.007·0.009) | ✅ |
| `trial_start.policies` 빈 리스트 | 체크포인트 **없음**(리셋 전용) | ✅ |
| 베이스 정지 `stop_base_path` | 11곳 전부 `primary`(확인용 `set_cmd_vel(0,0)` 수락) | ✅ |
| 루프 Hz (display_data false) | **phase=reset mean 11.9~13.1 Hz, min 7.5~11.4** — 아래 | ⚠️ 기록 |

- 리셋 T 는 1.5 s(틱 기준 32틱)이고 큰 두 곳은 1.87·2.00 s(40·43틱). 루프가 12.5 Hz 라 실제 벽시계는 2.6~3.4 s(설계대로 「느린 쪽이 안전」, reset_policy.py:49).
- 리셋 한 번의 경과는 4.0~5.0 s(램프 + 정착 21틱), 10경계 합 약 49 s.

### 루프 주기 — 리셋 구간이 12.5 Hz (목표 21)

| 조건 | mean Hz | per-frame (ms) |
|---|---|---|
| **이 회차, phase=reset** (11창) | 11.9~13.1 (min 7.5~11.4) | arms 1+3 · base 10+10(rearm active) · cam 0 · **other 51~59** |
| 10/06 1350 `eval_najy.sh` phase=policy (ACT 추론 포함) | 21.0 | arms 1+2 · base 10+10 · cam 0 · other 24 |
| 10/06 1350 phase=teleop (리셋 구간, lerobot-record) | 20.8 | arms 1+2 · base 20+20 · cam 0 · other 4 |
| **mock 로봇·mock 정책으로 같은 리셋 전용 체인** ([`mock_reset_hz.py`](run_logs/2026-10-07_eval_najy/mock_reset_hz.py) → [출력](run_logs/2026-10-07_eval_najy/mock_reset_hz.txt)) | **20.9** (11 리셋 모두) | — |

- I/O 는 정상이다(팔 4 ms·베이스 20 ms, rearm 켜짐). 남는 시간은 계측 밖 `other` **55 ms** 로, ACT 추론을 돌리는 policy 구간(24 ms)의 두 배다.
  mock 로봇에선 20.9 Hz 가 나오므로 **러너의 리셋 코드 자체(보간·판정)가 느린 것은 아니고, 실로봇 경로에서 리셋 구간에만 붙는 무언가**다 [원인 미확인 — 로봇 PC 에서 프로파일 안 함].
  창마다 min 이 7.5~7.9 Hz(≈130 ms 틱) 라 주기적으로 느린 틱이 하나씩 있다.
- 영향: 리셋이 느려지는 쪽이라 안전엔 문제 없었다. 다만 ④ 의 「eval_najy 회차와 같은 수치」 비교는 **policy 구간끼리** 해야 하고, reset 구간 Hz 는 이 값이 첫 기준선이다.
- 다른 보고 줄: rearm 은 리셋 구간에서 「active … 21 fps」 로 켜지고 경계마다 「off again (no record_loop phase tag)」 뒤 다시 켜진다 — 리셋당 재등록 104회(=52틱×2 트랜잭션), 정상.

## ②' 리셋만 재실행 — Hz 수정 역검증 · `1007_1532_chain_resets2_e30` ([chain.md](run_logs/2026-10-07_eval_najy/1007_1532_chain_resets2_e30.chain.md) · [events](run_logs/2026-10-07_eval_najy/1007_1532_chain_resets2_e30.events.jsonl))

`a22b560` 을 받은 뒤(`uv sync` 없이 editable, 테스트 232건 OK) 같은 조건으로: `RESET_ONLY=1 scripts/eval_chain.sh M1 resets2`, 빈손, 대기 자세 시작. 종료 코드 0 · completed · 저장 591프레임 · **40.7 s**(② 는 59.2 s).

| DGX 가 요구한 셋 | ② (수정 전) | ②' (수정 후) | 판정 |
|---|---|---|---|
| 보고서 「경계 리셋」 `Hz` ≈ 21, `⚠` 없음 | 11.9~13.1 | **20.9~21.0**, ⚠ 없음 (루프 줄 mean 20.9~21.2, min 20.4~20.7) | ✅ |
| `reason_detail.predict_path == "state_only"` | (없음) | 11/11 `state_only`, `predict_calls_upstream` 0 | ✅ |
| `predict_calls ≈ frames` | — | 52/52 ×9 · 60/60 · 63/63 (= 프레임) | ✅ |

- per-frame: arms 1+3 · base 10+10 · cam 0 · **other 22~24 ms**(② 55 ms) — 리셋 구간 `other` 가 policy 구간(24 ms)과 같아졌다. 벽시계 리셋 경과 2.5 s(② 4.2 s).
- ② 의 다른 관문은 그대로 통과: 11/11 reached · 도달 오차 ≤0.0015 · clamped 0 · `stop_base_path` 11곳 `primary` · Δmax 동일(최대 1.048) · 베이스 명령 최대 0.0000 / 실측 적분 +0.0002 m · -0.08° (30 s).
- 경고·에러 줄 없음, 팔·카메라 disconnect 정상. **수정이 실기 경로에서 먹었다.** 리셋 구간 Hz 기준선은 이 값(21.0)으로 바꾼다.

## ④' 시그널 실증 — `1007_1551_chain_hup2_e30` ([chain.md](run_logs/2026-10-07_eval_najy/1007_1551_chain_hup2_e30.chain.md) · [events](run_logs/2026-10-07_eval_najy/1007_1551_chain_hup2_e30.events.jsonl) · [로그 발췌](run_logs/2026-10-07_eval_najy/1007_1551_chain_hup2_e30.signal_path.txt))

`RESET_ONLY=1 scripts/eval_chain.sh M1 hup2`(빈손, 10경계)를 띄우고 두 번째 터미널에서 `kill -HUP $(pgrep -f '[.]venv/bin/python3 -m stage_runner')`.
HUP 은 `reset_to_02` 램프 도중(14프레임, 0.9 s)에 들어갔다 — 베이스는 리셋 전용이라 정지 상태.

| 기대 (G절·codex 치명 지적) | 로그·events | 판정 |
|---|---|---|
| 러너가 HUP 을 받아 teardown 으로 간다 | `ERROR … SIGHUP received: stopping the trial. The base will be zeroed and the robot disconnected on the way out.` (cli.py:200) | ✅ |
| stop_base → 확인 | `stop_base commanded` → `stop_base confirmed: base.set_cmd_vel(0.0, 0.0) was accepted`, events `stop_base_path: primary`, `direct_ok: true` | ✅ |
| disconnect 까지 | 팔 2대 `disconnected` → 카메라 3대 `disconnected` (4~9 s 뒤) | ✅ |
| 베이스 제자리 | basevel 68행·3.4 s: 명령 최대 0.0000, 실측 적분 +0.0001 m · +0.06° | ✅ |
| 팔 제자리(토크 유지) | 사람 확인 요청 — 로그상 마지막 stop_base 의 팔 action 은 그 순간의 자세(hold) | (사람) |

- events: `reset_to_02` terminator **`error` — `SystemExit(129)`**, trial_end reason **`exception`**, completed False, 에피소드 저장 안 됨. 보고서 집계는 `{'error': 1} (중단·오류)`.
  → **시그널 종료가 events 에서 예외·크래시와 구분되지 않는다** — NaN 게이트와 ESC 가 같은 `stop_recording` 인 것과 같은 꼴의 집계 구멍. 로그의 `SIGHUP received` 줄만 둘을 가른다. DGX 에 알린다(결함이라기보다 표기 요청).
- 앞선 시도 3회(`1007_1539`·`1540`·`1542` `chain_hup`)는 **무효** — 세션이 준 pgrep 패턴이 `.venv/bin/python` 이었는데 `uv run` 의 자식은 `.venv/bin/python3` 이라 PID 를 못 찾았다(`kill: usage`). 러너엔 신호가 안 갔고 셋 다 리셋 2개를 정상 완주했다(② 조건의 정상 회차로 센다).
  G절·eval_chain.sh 의 안내에는 **`pgrep -f '[.]venv/bin/python3 -m stage_runner'`** 로 적어야 한다.

## 읽은 것

- bring-up ①·② **통과**. 리셋 궤적(관절공간 직선, 시연에 없는 경로)은 빈손에서 전 경계 도달·상한 안·클램프 0 이었다. 큰 전이 다섯 곳의 Δmax 는 사전 계산(stage_params)과 일치한다.
- ~~열린 것 하나: 리셋 구간 루프 12.5 Hz 의 원인.~~ → DGX 가 원인(리셋 번들의 CPU 영상 변환)을 고쳤고 ②' 에서 21.0 Hz 로 확인됐다(위).
- 사람 관찰이 비어 있다 — `~/eval_logs/eval_chain_results.csv` 는 정책 단계 라벨용(`run_id,stage_id,outcome,failure_phase`)이라 리셋 전용 회차엔 쓰지 않았다.

## 1호기(DGX_1)로 넘기는 것 — 10/07

1. **②의 세 관문**: `stop_base_direct_ok` 11/11 `primary` · `clamped` 0 · 도달 오차 ≤0.0019. ③(그리퍼)·④'(시그널)는 아직.
2. ~~**리셋 구간 루프 12.5 Hz**~~ → **닫힘(②', 21.0 Hz · state_only · calls=frames)**. 원래 요청: (`other` 55 ms, mock 20.9 Hz) — 러너의 실로봇 리셋 경로에서 틱당 ~30 ms 가 어디서 드는지. 후보(미확인):
   리셋 정책 `select_action`/후처리의 실관측 경로, pose_guide `set_stage` 뒤 틱당 계산, 리셋 구간의 데이터셋 프레임 추가·finite 게이트. 로봇 PC 에서 `cProfile` 을 걸어 달라면 건다.
2b. **④' 결과**: 시그널 래치는 실기에서 동작(위 절). 요청 하나 — events/보고서에서 시그널 종료를 `error/exception` 이 아니라 따로 표기해 주면 집계에서 크래시와 갈린다.
3. 사람 몫 그대로: **C. 1006 회차 원자료 전송**(데이터셋 6개 + `~/eval_logs/1006_*`) 아직 안 됨.

## DGX_1 → Trossen PC1 (10/07 저녁) — 지금 돌릴 Eval (우선순위 순)

2라운드(env 단계 토큰) 는 60K 채점 중(≈06:30 UTC 결과) · 120K ≈12:00 UTC 종료 → **허브 업로드는 그 뒤 결정**(사용자). 그때까지 **새 모델 없이 M1·M2 로** 아래를 돈다.
오프라인에서 새로 확정된 것(sim §94.9~94.12): 1라운드 TPH 는 출발만 해결 · 증강(T8)·SmolVLA 모두 「원핫/지시문이 단계를 고른다」 에 못 미침 · cam_high 가 t03→04 단계 선택을 지배하고 손목 카메라도 30~100% 기여.
→ 실기에서 지금 필요한 것은 **러너 관문 닫기 + 오프라인 채점의 실기 라벨**이다. 전부 tmux(없으면 `nohup`) 안에서, **처음은 1ep, ESC 와 베이스 e-stop 을 함께**.

**DGX 가 답한 것 (위 「1호기로 넘기는 것」 2번)** — 리셋 구간 12.5 Hz 의 원인은 리셋 번들(`device=cpu`)이 매 틱 카메라 3장을 CPU 에서 변환한 것(DGX 실측 틱당 324 ms vs 0.55 ms). 리셋 단계 동안만 `predict_action` 을 state 전용으로 바꾸고 끝나면 복원(main `14e07c9`→`a22b560`, 테스트 232건).
**받기**: `git fetch najongs && git merge --ff-only najongs/main` — 이번엔 패키지 코드만 바뀌어 **`uv lock`/`uv sync` 불필요**(workspace 멤버는 editable). `git log --oneline -3 -- packages/stage_runner` 에 `a22b560` 이 보이면 됐다.

| # | 무엇 (명령) | 기록할 것 | 닫히는 결정 | 시간 |
|---|---|---|---|---|
| **1** | **②' 리셋만 재실행** — `RESET_ONLY=1 scripts/eval_chain.sh M1 resets2` (빈손, ② 와 같은 조건·같은 시작 자세) | 보고서 「경계 리셋」 표의 **`Hz ≈ 21`** · `⚠` 없음 · `events.jsonl` 의 `reason_detail.predict_path == "state_only"` · `predict_calls ≈ frames` | Hz 수정이 **실기 경로에서 먹었나** — 셋이 같이 와야 한다. 하나라도 아니면(`bypassed` 포함) 멈추고 그 값 그대로 보고 | 10분 |
| **2** | **④' 시그널** — `RESET_ONLY=1 FROM_STAGE=1 TO_STAGE=2 scripts/eval_chain.sh M1 hup`, 베이스가 **정지한 틈**에 두 번째 터미널에서 `kill -HUP $(pgrep -f '[s]tage_runner')` | 로그에 stop_base → disconnect · 팔·베이스 제자리 · 종료 코드 | 시그널 래치 실증(codex 치명 지적) | 5분 |
| **3** | **③ 물체 든 채 리셋** — 가벼운 튜브 먼저 `RESET_ONLY=1 scripts/eval_chain.sh M1 resets_tube`, 통과하면 비커 `… resets_beaker` | 경계별 **그리퍼가 놓치나 / 더 쥐나** · reach_err · clamped | 「그리퍼 관측값 재명령」 이 파지력을 유지하나 — 유일하게 설계 변경이 필요할 수 있는 관문. 놓치면 거기서 멈추고 보고 | 20분 |
| **4** | **④ M1 + 러너 단계 1개** — `FROM_STAGE=4 TO_STAGE=4 scripts/eval_chain.sh M1 one4` (task04 장면 세팅, 1ep → 괜찮으면 2ep) | policy 구간 Hz · rearm `active` · clamped · FIRED **vs 10/06 1350** · **종료 사유**(`complete` / `manual` / `timeout` / `never_departed`) · 출발 s · 정지 감지 s | 러너가 실행 계층을 안 바꿨나 + **완료 감시의 첫 실기 데이터** — 오프라인 예측은 「M1 은 끝 장면에서 안 멈춘다」(§93.2) → `complete` 가 안 나고 `→` 로 넘기게 되는지 | 15분 |
| **5** | **⑤-M1 체인 1→3** — `TO_STAGE=3 scripts/eval_chain.sh M1 s1_3` (`→` 로 넘긴다, 자동 완료를 기대하지 않음) | 단계별 종료(자동/수동/타임아웃) · 리셋 T·Δmax·도달 · **단계별 ∫θ·∫x**(베이스 드리프트) · **리셋 뒤 다음 단계가 출발하나** | 리셋+정책+전환이 한 프로세스에서 끝까지 도나 · 베이스 드리프트가 리셋 범위 밖에 쌓이는지(계획 §7 위험) · 지정 자세 리셋 뒤 M1 출발률(B1 을 체인 안에서 한 번 더) | 30분 |
| 6 | **B1 지정 자세 출발** — `configs/chain/stage_params.json` `stages["5"]`·`["2"]` 의 `start_pose_rad_arm12` 에 팔을 맞추고(`POSE` 줄로 거리 확인) `scripts/eval_najy.sh M1 5 30 3` · `scripts/eval_najy.sh M1 2 30 3` | 출발 ep / 정지 ep · 시작 자세 거리 | 오프라인 출발 지도(M1 t05 20%·t02 26% 정지 예측, §94.9)의 **실기 라벨** — 채점을 믿어도 되는지 | 20분 |
| 7 | **B2 M2 끝 정지** — `scripts/eval_najy.sh M2 4 30 2`, 부은 뒤 손대지 말고 30초 관찰 | 끝에서 머무나 / 움직이나(어느 관절·방향) | 끝 정지 지표(M2 120K t07 96%·t11 95%, §94.9)의 실기 검증 — 맞으면 M2 가 16D 폴백 참조군 | 10분 |
| 8 | B3 task03·task01 재시험 — `eval_najy_results_1006.md` 「다음」 1·2 그대로 | 회전·전진량 vs 기준(task03 −81°, task01 +82°·1.64 m, 21 Hz 환산) | M1 이동 단계 실제 성공률 | 30분 |
| 9 | **B4 (선택) 장면 변화 민감도** — task04 를 ④ 와 같은 세팅으로 `scripts/eval_najy.sh M1 4 30 1`, 그다음 **cam_high 시야 안에 작업과 무관한 물체 하나를 추가**(또는 조명만 바꿈)하고 같은 명령 1ep | 성공/실패 · 동작이 달라진 시점·관절 · 무엇을 바꿨나(사진) | 오프라인: t03→04 단계 선택을 cam_high 가 지배(M1 교환 0.175 vs 손목 0.07), 증강 모델은 그 의존을 손목으로 옮길 뿐(§94.12) — **환경 변화 취약성의 첫 실기 데이터**. 탐색용이지 판정 아님 | 10분 |
| C | **전송(사람)** — 1006 데이터셋 6개 + `~/eval_logs/1006_*` + 위 1~9 의 `eval_chain_*`·`eval_najy_1007_*` → DGX `/raid/kiro-ai/eval/real/` (명령 `eval_najy.md` 「결과 넘기기」) | — | 1350 ep0/ep1/ep2 · C-2 프레임으로 출발 지도·진행도·원핫 진단을 **실기 라벨로 검증** → 2라운드 후보를 믿고 고른다. **여전히 가장 큰 도움** | — |

- 순서의 이유: 1 은 DGX 수정의 역검증이라 가장 먼저(10분) · 2·3 은 ⑤ 전에 닫혀야 하는 안전 관문 · 4·5 는 러너 전체를 M1 로 끝까지 한 번 · 6·7 은 오프라인 채점의 라벨 · 8·9 는 시간이 남을 때.
- **하지 말 것**: `scripts/eval_chain.sh TPH|ENV` — 허브에 없어 다운로드에서 멈춘다(정상, 올라가면 이 문서에 적는다) · M1 로 자동 완료를 기대하고 ⑤ 전체(1→11)를 돌리기 · 추가 수집.
- **기록**: 회차마다 `eval_chain_report.py`/`eval_najy_post.sh` 산출을 `docs/run_logs/2026-10-07_eval_najy/` 에, 이 문서 「다음」 에 한 줄씩. **1(Hz)·3(그리퍼)·2(시그널)** 은 결과가 어느 쪽이든 바로 commit·push — DGX 가 보고 러너를 고친다.

## 다음

0. **(10/07 저녁) 위 「DGX_1 → Trossen PC1 (10/07 저녁)」 표 1→9 순.** 1(②' Hz 역검증)이 먼저다 — 아래 1~3 은 그 표의 2·3·4 와 같다.
0. ~~②' Hz 역검증~~ ✅ 통과(10/07 15:32, 위 절). 1호기 목록의 1번 닫힘.
1. ~~④' 시그널 실증~~ ✅ 통과(10/07 15:51, 위 절). HUP → `SIGHUP received` → stop_base primary → disconnect. 자식 프로세스 이름은 `.venv/bin/python3` — pgrep 패턴 주의.
2. **③ 물체 든 채 리셋** — 가벼운 플라스틱 튜브 먼저(`RESET_ONLY=1 scripts/eval_chain.sh M1 resets_tube`), 그다음 비커. 그리퍼가 놓치거나 더 쥐면 멈추고 보고.
3. **④ M1 + 러너로 단계 1개** — `FROM_STAGE=4 TO_STAGE=4 scripts/eval_chain.sh M1 one4` (task04 장면 세팅). policy 구간 Hz·rearm·clamped·FIRED 를 10/06 1350 과 비교.
4. B1~B3(지정 자세 출발 task05/02 · M2 task04 끝 정지 · task03/01 재시험) → `eval_najy_results_1006.md` 「다음」.
5. C. 전송(사람).
