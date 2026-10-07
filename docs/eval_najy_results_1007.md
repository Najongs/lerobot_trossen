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

## 로봇 PC 가 러너를 바꿨다 (10/07 오후) — 리더암 텔레옵 구간

**왜**: bring-up ③·④·⑤ 는 전부 「물체를 쥔 채」 시작해야 하는데, 러너는 연결 즉시 자기 리셋을 시작해 사람이 리더암으로 튜브를 쥐게 할 틈이 없었다
(lerobot-record 회차에서는 `←` 로 리셋 구간에 들어가 리더암으로 잡았다). 사용자 지시: 「이전 Eval 실행 코드처럼 리셋·저장·텔레옵이 되게」.

**무엇** (`packages/stage_runner/`, `scripts/eval_chain.sh`, 테스트 239건 통과 — 절차는 `eval_najy.md` G절 「텔레옵 구간」):
- `StageRunnerConfig.teleop`(lerobot `TeleoperatorConfig`)·`teleop_time_s`(300)·`teleop_base_from_leader`(false). `eval_chain.sh` 가 `TELEOP=1` 기본으로 리더(.3/.2)를 넘긴다.
- `runner._run_teleop_phase`: trial_start 직후·첫 리셋 전에 `record_loop(policy=None, dataset=None, teleop=리더)` 한 번. `→` 끝 · `←` 다시 · ESC 중단 · **타임아웃도 중단**.
  연결 중 눌린 ESC/`←` 는 구간을 돌지 않는다. 끝마다 `transitions.stop_base`(실패면 exit 3). events `teleop_start`/`teleop_end`(`ended_by`, `stop_base_path`, `t_mono`), 보고서 한 줄.
- **베이스는 구간 동안 0 으로 덮는다**(`TeleopBaseZeroStep`, teleop_action 파이프라인에 구간 동안만 설치) — 리더 action 의 x.vel/theta.vel 은 `get_latest_base_velocity()`, 즉 같은 틱의
  베이스 실측이라 토크 켜진 베이스에 되먹임되는 구조다(basevel.csv teleop 행 cmd==meas). eval_najy 리셋 구간도 같은 구조였다 — **DGX 확인 요청**: 의도된 설계인가.
- `cli`: 리더 connect 는 `robot.connect()` 뒤, disconnect 는 `robot.disconnect()` 의 finally 에서. `mock_teleop.py`(테스트용).
- **첫 실기 `1007_1626_chain_resets_tube_e30`**: 텔레옵 구간 32.4 s · 리더로 튜브 집음 · `→` · stop_base primary — 구간 자체는 동작. 그 뒤 최초 리셋이
  `largest joint gap 0.874 rad > initial_max_jump_rad 0.6` 으로 **거부(아무것도 안 움직임, exit 4)** — `FROM_STAGE=1` 의 목표가 빈손 대기 자세라 튜브 든 팔로는 못 들어간다.
  사용자: 「로그 보면서 맞추는 게 쉽지 않다」 → 두 가지를 더했다: ① **`→` 는 팔이 최초 리셋 상한 안일 때만 받는다**(`arrow_not_ready` 로 기록하고 구간 계속, 로그 `RIGHT ARROW IGNORED`)
  ② `POSE` 줄 앞에 `[→ 가능 ✔]` / `[아직 ✘ R j3 50° > 34°]` 표시 + 가능해지는 순간 터미널 벨(`pose_guide.set_stage(gate_deg=…)`). 물체를 든 시작은 그 물체를 드는 단계부터(`FROM_STAGE=3`/`6`).
- 단계 **사이**의 텔레옵(정책이 못 집은 물체를 사람이 쥐여 주고 이어 가기)은 넣지 않았다 — `←` 의 의미(체인 중단)를 바꾸는 일이라 DGX 와 설계할 것.

**읽기 전용 리뷰(Claude, 10/07) 반영**: 수용 ① 구간 전 ESC/`←` 가 켜져 있으면 루프에 안 들어감(치명 — 리더가 300 s 팔을 끌 뻔) ② 정상 경로 `stop_base` + `base_is_stopped` 합산 ③ 베이스 0 덮기(설정으로 끔)
④ ESC 테스트에 경과 단언·「구간 전 ESC」 테스트 ⑤ 타임아웃 = 중단 ⑥ 테스트 클래스를 `__main__` 앞으로·SDK 누출 tearDown ⑦ `teleop.disconnect` 를 finally 로 ⑧ 구간 중 예외도 `teleop_end` 로 기록, `t_mono`.
기각 없음. 보류: 단계 사이 텔레옵(위). codex 교차 검토는 이 머신에서 안 한다(host 규칙) — **DGX 에서 한 번 돌려 달라**(로봇을 움직이는 코드).

## ③ 물체 든 채 리셋 — 진행 (10/07 저녁)

| 회차 | 러너 | 텔레옵 구간 | 그 뒤 | 판정 |
|---|---|---|---|---|
| `1007_1626_chain_resets_tube_e30` (FROM 1) | `d4bc8f8` | 32.4 s · 튜브 집음 · `→` · stop_base primary | 최초 리셋 거부 — 관절 최대 간격 **0.874 rad > 0.6** (task01 지정 자세 = 빈손 대기 자세). 아무것도 안 움직임, exit 4 | 무효 (시작 단계 잘못) |
| `1007_1630_chain_resets_tube_e30` (FROM 3 TO 5) | `d4bc8f8` | 36.2 s · 튜브 집음 · `→` · primary | 최초 리셋 거부 — **0.807 rad > 0.6** (task03 지정 자세까지 손목·팔꿈치가 멂). 아무것도 안 움직임 | 무효 (자세 미달 — 사람이 로그 숫자로 맞추기 어려움) |
| `1007_1723_chain_resets_tube_e30` (FROM 3 TO 5) | `f6083df` | — | 왼팔 192.168.1.5 `No route to host` 로 connect 실패(21 s), 프로세스 종료. 로봇 미구동 | 무효 (네트워크 일시 장애, 재시도로 해결) |
| **`1007_1731_chain_resets_beaker_e30`** (FROM 6 TO 11) | `f6083df` | **23.2 s** · `[아직 ✘]` 8줄 뒤 `[→ 가능 ✔]` · `→` 1회 수락 · stop_base primary | **리셋 6개(최초 Δ0.37 · t06→07 · t07→08 · t08→09 Δ0.98 · t09→10 · t10→11 Δ1.05) 전부 reached**, 오차 ≤0.0015, clamped 0, stop_base 6곳 primary, 21 Hz, 베이스 ∫ ≈0, 저장 331프레임 | **✅ ③ 비커 통과** — 사람: 「비커 안 떨어진다」 |
| **`1007_1725_chain_resets_tube_e30`** (FROM 3 TO 5) | `f6083df`(→ 준비 표시) | **76.6 s** · `[아직 ✘ R j0 36°…]` → 리더로 맞춤 → `→` 1회에 수락(`arrow_not_ready` 0회) · stop_base primary | **최초 리셋(Δmax 0.542 < 0.6) · t03→04 · t04→05 전부 reached**(오차 ≤0.0011, clamped 0, stop_base 3곳 primary), 21 Hz, 에피소드 저장 156프레임 | **✅ ③ 튜브 통과** — 사람: 「튜브 든 상태로 성공적으로 동작」 |

원출력: [`1626.chain.md`](run_logs/2026-10-07_eval_najy/1007_1626_chain_resets_tube_e30.chain.md) · [`1630.chain.md`](run_logs/2026-10-07_eval_najy/1007_1630_chain_resets_tube_e30.chain.md) (+ events).

- 텔레옵 구간은 두 회차 모두 설계대로 돌았다(리더가 팔로워를 끌고, `→` 로 끝, 베이스 정지 확인, 녹화 없음). 거부는 **러너의 최초 리셋 상한(0.6 rad, 관절별)** 이 그대로 작동한 것이다 — 올리지 않았다.
- 사람 피드백: 「로그 보면서 맞추는 게 쉽지 않다」 → `f6083df`: `→` 는 상한 안일 때만 받고(멀면 `arrow_not_ready` 로 적고 창 계속), `POSE` 줄 앞에 `[→ 가능 ✔]`/`[아직 ✘ R j3 50° > 34°]` + 가능해지는 순간 터미널 벨. 1723 회차가 이 판의 첫 실기.
- DGX `9393512` 받음(시그널 분리·`nan_gate`·`emergency_stop_base_*`·`check_teleop_window`·mock 리더 베이스 노브). 로봇 PC 테스트는 1723 회차가 끝난 뒤 돌린다(제어 루프와 CPU 경합 회피). 리뷰 결론 ①~⑦ 성립 확인, 감사.
- DGX 의 [추정] 「텔레옵으로 쥔 조임이 창이 끝나는 순간 실측 위치 재명령으로 풀릴 수 있다」 — **1725 에서 안 풀렸다** [1회, 플라스틱 튜브]: 에피소드 156프레임 내내 그리퍼 명령 = 실측(L 0.0066 / R 0.0064, 차 0.0000), 리셋 3개 뒤에도 튜브를 들고 있었다(사람 관찰).
  튜브는 플라스틱이라 조임 손실이 있어도 안 보일 수 있다 — 비커(무거움·유리)에서 다시 본다. 풀리면 후보: 리셋·hold 의 그리퍼 재명령을 실측이 아니라 창의 마지막 명령값으로.
- 준비 표시(`[아직 ✘ R j0 36° > 34°]` → 리더로 맞춤 → `→`)로 사람이 로그 숫자를 읽지 않고 들어갔다. `→` 무시(`arrow_not_ready`)는 이번엔 0회.
- 1723 은 왼팔 컨트롤러 TCP `No route to host`(21 s 재시도 뒤 CRITICAL) — 재시도(1725)에서 정상. 러너는 connect 전이라 아무것도 안 움직였고 events 에 trial_end 가 없어 보고서가 「프로세스가 먼저 죽었다」 로 표시(정상).
- **비커(1731)도 통과** — 가장 큰 전이 둘(t08→09 0.98 rad, t10→11 1.05 rad)을 비커를 든 채 지났다. 그리퍼 명령 = 실측(331프레임 내내, 차 0). → **③ 관문 닫힘**: 「매 틱 관측값 재명령」 으로 파지가 유지된다 [튜브 1회·비커 1회].
- 다음: **④ M1 + 러너 단계 1개(task04)** `FROM_STAGE=4 TO_STAGE=4 scripts/eval_chain.sh M1 one4`.

## ④ M1 + 러너로 단계 1개 (task04 붓기) — `1007_1735_chain_one4_e30` ([chain.md](run_logs/2026-10-07_eval_najy/1007_1735_chain_one4_e30.chain.md) · [events](run_logs/2026-10-07_eval_najy/1007_1735_chain_one4_e30.events.jsonl) · [로그 발췌](run_logs/2026-10-07_eval_najy/1007_1735_chain_one4_e30.policy_stage.txt))

`FROM_STAGE=4 TO_STAGE=4 scripts/eval_chain.sh M1 one4`. 텔레옵 창 84.8 s(튜브 집고 비커 앞) → `→` → 최초 리셋 reached(Δ0.43) → **M1 정책 단계 t04**.
**사람 판정: 붓기 동작함(양팔), 물을 바닥에 조금 흘림, 오른팔은 한 번이 아니라 여러 번 반복한 느낌** — 10/02 1150(eval_najy, 1/1 성공, 왼팔 3~4회 반복)과 같은 양상.

| 볼 것 (DGX 표 4번) | 이번 (러너) | 10/06 1350 (`eval_najy.sh`, policy 구간) | 판정 |
|---|---|---|---|
| 원핫 | `stage 4/11 active -- policy state 14 -> 27` (set_stage) | `installed` + `stage 3/11 active` | ✅ 같은 패치 경로 |
| policy 구간 Hz | mean 20.9~21.1 (min 16.3~18.4), per-frame arms 1+2 · base 10+10 · cam 0 | 21.0 (min 15.3~17.0), 같은 per-frame | ✅ 동일 |
| rearm | `active for this policy loop at 21 fps` (reset 구간 포함) | active | ✅ |
| clamped (0.1 rad 상한) | 10 arm-tick / 476 프레임 | 1350: 0 · 1347(task01): 83 | ✅ 범위 안 |
| FIRED (페이싱) | 0 | 0 | ✅ |
| 종료 사유 | **`timeout`** 22.7 s (= p90 17.5 × 1.3) — 사람이 `→` 를 안 눌렀고 완료 감시자도 안 울림 | — | 예상대로(§93.2 「M1 은 끝에서 안 멈춤」) |
| 출발 | `departed` **1.26 s** (팔 0.15 rad 이탈, 3틱 연속) | — | ✅ 출발 래치 첫 실기 |
| 베이스 | ∫θ 명령 +1.4° / 실측 0.0, ∫x 0.001 m, stop_base primary | — | ✅ |

- 결과는 `chain_failed`(exit 4, 에피소드 528프레임 저장 = 리셋 52 + 정책 476). **러너 검증으로는 통과** — 실행 계층 수치가 eval_najy 회차와 같고, 원핫·출발·정지·저장이 전부 찍혔다. 모델 평가가 아니다.
- 팔 움직임(`eval_motion_stats`, 에피소드 전체 25.1 s — 앞 2.5 s 는 리셋): 프레임당 0.042 rad(학습 중앙 0.060), 방향 바뀜 0.40/s(학습 0.29). 「청크 경계 점프 0.8×」 는 **쓰지 않는다** — 에피소드 앞의 리셋 52프레임 때문에 30프레임 경계 위상이 어긋나 지표가 경계를 못 짚는다(체인 데이터셋에 `eval_motion_stats` 를 쓰려면 정책 구간만 잘라야 한다 — 보류).
- 완료 감시자가 22.7 s 안에 안 울린 이유는 로그상 「정지 3 s」 조건이 한 번도 안 맞은 것 [추정 — M1 이 반복 붓기로 계속 움직임]. ⑤ 에서는 **사람이 `→` 로 넘긴다**(DGX 지시).

## 1호기(DGX_1)로 넘기는 것 — 10/07 저녁 요약 (오늘 Eval 종료)

**오늘 한 것** (전부 M1 = 1라운드 체크포인트, 새 모델 없음; 회차별 원자료는 `run_logs/2026-10-07_eval_najy/`)

| 관문 | 결과 | 회차 |
|---|---|---|
| ① DRY_RUN·환경 | ✅ (uv sync, 테스트 242) | — |
| ② / ②' 빈손 리셋 10경계 | ✅ 11/11 도달, clamped 0, stop_base primary · ②' 에서 Hz 수정 역검증(21.0, state_only, calls=frames) | 1045 · 1532 |
| ④' 시그널 | ✅ HUP → stop_base primary → disconnect (pgrep 패턴은 `python3`) | 1551 |
| ③ 물체 든 채 리셋 | ✅ 튜브(3→5)·비커(6→11) 전 경계 도달, 그리퍼 명령=실측 유지 | 1725 · 1731 |
| ④ M1 + 러너 단계 1개(task04) | ✅ 러너 검증 — 원핫·21 Hz·rearm·clamped 10·FIRED 0 이 eval_najy 와 동일, 출발 1.26 s, timeout(사람 → 없음). 붓기 동작(흘림·오른팔 반복) | 1735 |
| ⑤ M1 체인 | 전환 기계는 확인(정책→stop_base→리셋→원핫 재설정→정책 출발), **그러나 M1 이 1·2·3 단계를 못 해 체인은 못 감** | 1743·1746·1751·1756 |

**러너 쪽 발견 (DGX 가 고칠 것)**
1. **[중요] 이동 단계 거짓 완료** (1756 t03): 베이스가 안 돌았는데 팔 0.10 rad 출발 + 3 s 정지 + 팔 끝 자세 NN 으로 `complete` → 붓기로 넘어감. 이동 단계(1·3·6·10)는 출발·완료를 **베이스 누적**으로.
2. 텔레옵 구간(로봇 PC 추가, DGX 리뷰 반영됨) — 실기 6회 사용, 문제 없음. `→` 는 상한 안일 때만 받고 `[→ 가능 ✔]` 표시·벨. 물체 든 시작은 그 물체를 드는 단계부터(`FROM_STAGE`).
3. `TIMEOUT_FACTOR` 환경변수(러너 검증용 상한 배수) — 본선 판정은 1.3.
4. 리셋 구간 12.5 Hz → DGX 수정으로 21.0 Hz(②').

**모델 쪽 발견 (M1 실기, 에피소드 1~2개 — 증상)**
- task01: 선반 앞에서 시작해도 12.8 s 에 6 cm (1743). task02: 다가갔다 물러났다 반복, 그리퍼 안 염(0.011~0.019 vs 시연 0.041), 2/2 실패 (1746·1751). task03: 회전 0 (1756; 어제 1/3). task04: 붓기 됨 (1735; 어제 1/1).
- 공통 양상: **이동·집기 단계에서 「출발」 이 안 된다** — 어제 오프라인 교체(cam_high 가 출발을 정함)와 같은 방향. 사용자 가설: 원핫이 단계를 못 고르는 문제라 3라운드(env 단계 토큰)에서 개선될 것. 세션 판단: 출발 문제는 1라운드 TPH 가 오프라인에서 해결했다고 보고된 것(§94.9, 시작 정지 예측 ≤4%)이고 토큰 강화는 그 위에 얹는 것이라 **방향은 맞지만 실기 확인 전까지 가설** [오프라인 지표가 실기를 두 번 틀린 규칙].

**다음 (로봇 PC)**: 새 체크포인트(TPH/ENV)가 허브에 올라오면 ④(단계 1개: task03·task01·task02) → ⑤ 체인. 그 전엔 B1·B2(M2 끝 정지)·B3·B4 가 남아 있으나 오늘은 종료.
**사람 몫**: C 전송 — 이제 `eval_najy_1006_*` 6개 + `eval_1007_*` 체인 데이터셋(1725·1731·1735·1743·1746·1751·1756) + `~/eval_logs/1006_*, 1007_*`.

## 읽은 것

- bring-up ①·② **통과**. 리셋 궤적(관절공간 직선, 시연에 없는 경로)은 빈손에서 전 경계 도달·상한 안·클램프 0 이었다. 큰 전이 다섯 곳의 Δmax 는 사전 계산(stage_params)과 일치한다.
- ~~열린 것 하나: 리셋 구간 루프 12.5 Hz 의 원인.~~ → DGX 가 원인(리셋 번들의 CPU 영상 변환)을 고쳤고 ②' 에서 21.0 Hz 로 확인됐다(위).
- 사람 관찰이 비어 있다 — `~/eval_logs/eval_chain_results.csv` 는 정책 단계 라벨용(`run_id,stage_id,outcome,failure_phase`)이라 리셋 전용 회차엔 쓰지 않았다.

## 1호기(DGX_1)로 넘기는 것 — 10/07

1. **②의 세 관문**: `stop_base_direct_ok` 11/11 `primary` · `clamped` 0 · 도달 오차 ≤0.0019. ③(그리퍼)·④'(시그널)는 아직.
2. ~~**리셋 구간 루프 12.5 Hz**~~ → **닫힘(②', 21.0 Hz · state_only · calls=frames)**. 원래 요청: (`other` 55 ms, mock 20.9 Hz) — 러너의 실로봇 리셋 경로에서 틱당 ~30 ms 가 어디서 드는지. 후보(미확인):
   리셋 정책 `select_action`/후처리의 실관측 경로, pose_guide `set_stage` 뒤 틱당 계산, 리셋 구간의 데이터셋 프레임 추가·finite 게이트. 로봇 PC 에서 `cProfile` 을 걸어 달라면 건다.
2a. **러너 변경(텔레옵 구간)을 받아 달라** — 위 「로봇 PC 가 러너를 바꿨다」. 베이스 되먹임 구조(리더 x.vel/θ.vel = 실측)가 의도인지 확인 + codex 교차 검토.
2d. **[중요] 이동 단계 거짓 완료** — 1756 t03: 베이스가 안 돌았는데 팔 0.10 rad 출발 + 3 s 정지 + NN 0.120 으로 `complete`. 이동 단계는 베이스 누적으로 출발·완료를 판정해 달라(위 ⑤ 네 번째).
   → **DGX 반영(10/07 저녁, main)**: `stage_params.json` 에 `kind`(move = 1·3·6·10)와 시연 베이스 이동량 `base_fwd_total_m`·`base_rot_total_rad`(중앙/p10/p90, 21 Hz)를 실었고(sim `export_chain_params.py`), 러너는 move 단계에서 **팔로 출발을 래치하지 않으며**(베이스 0.17 rad / 0.10 m 만) 완료에 「시연 중앙 이동량의 50%」 를 축마다 요구한다(16D 는 NN 대신 `stall_base`, 17D 는 진행도∧정지∧이동량). task03 이면 \|∫θ\| ≥ 0.71 rad(≈41°) 가 있어야 완료. 기준: `configs/chain/README.md` 「단계 kind 와 베이스 이동량」. **받은 뒤 `DRY_RUN=1` 로 파일이 읽히는지(kind 4개) 확인**. 1756 재현 조건(회전 0)에서는 이제 `never_departed` 로 timeout 이 난다.
2b. **④' 결과**: 시그널 래치는 실기에서 동작(위 절). 요청 하나 — events/보고서에서 시그널 종료를 `error/exception` 이 아니라 따로 표기해 주면 집계에서 크래시와 갈린다.
   → **DGX 반영(10/07 저녁, main)**: `terminator: signal`(reason `SIGHUP`/`SIGTERM`) · `trial_end.reason: signal` · 텔레옵 창 `ended_by: signal` — 종료 코드 128+signum 은 그대로. 보고서 「그 밖」 은 `(중단·시그널·오류)`.
2c. **텔레옵 구간(`d4bc8f8`) DGX 읽기 전용 리뷰 결과** — ①~⑦ 안전 보장은 전부 성립(코드 판독; ①의 NaN 게이트는 체인 런 한정). 고친 것(main): [중요] 텔레옵 창 안에서 예외·시그널이 나면 비상 stop_base 결과가 events 에 안 남던 것 → `trial_end` 에 `emergency_stop_base_{path,is_stopped,error}` 추가 · [중요] 베이스 0 덮기가 e2e 로 검증되지 않던 것(mock 리더가 늘 0 을 보냄) → mock 리더에 `x_vel/theta_vel` 노브, 전체 cli 경로에서 기본 0 / `teleop_base_from_leader=true` 면 통과를 검증하는 테스트 · `teleop_time_s<=0` 을 preflight 가 거부(exit 2) · 창 안 NaN 트립이 `esc` 로 찍히던 것 → `ended_by: nan_gate` · connect 중 누른 `→` 폐기를 경고로 · 주석·docstring 정정(해제 순서, 한 틱 상한의 실제 자리).
   **남긴 것(보류)**: `←` 재시작 횟수 상한 없음(ESC 로 탈출 가능) · 상한 직전 0.25 s 안의 `→` 는 timeout 으로 분류(안전한 방향) · `EVENT_SCHEMA_VERSION` 유지.
   **[추정] ③ 에서 꼭 볼 것**: 텔레옵으로 쥔 「조임」(명령 위치가 접촉점보다 더 닫힘)은 창이 끝나는 순간 `build_hold_action`·리셋 램프가 그리퍼를 **실측 위치로 재명령**하면서 사라질 수 있다 — 물체를 쥔 채 첫 리셋에서 미끄러지는지가 이 구간의 존재 이유를 가른다. 튜브부터, 놓치면 멈추고 보고.
   참고: `scripts/eval_chain.sh` 의 `TELEOP` 기본값이 1 이라 **RESET_ONLY 포함 모든 회차의 기본 동작이 바뀌었다** — 옛 경로는 `TELEOP=0`.
3. 사람 몫 그대로: **C. 1006 회차 원자료 전송**(데이터셋 6개 + `~/eval_logs/1006_*`) 아직 안 됨.

## DGX_1 → Trossen PC1 (10/07 저녁) — 지금 돌릴 Eval (우선순위 순)

2라운드(env 단계 토큰) 는 60K 채점에서 **단계 토큰이 전혀 읽히지 않아**(ENV 토큰 교환 효과 두 시드 0.000~0.003, 15K 와 동일 — sim §94.13) 60K 에서 끊었고, **3라운드(영상 모달리티 드롭아웃 `dropall30`, §94.14)** 를 07:26 UTC 에 띄웠다(15K 조기 신호 ≈09:05 UTC, 120K ≈20:00 UTC). **허브 업로드는 3라운드 결과를 보고 결정**(사용자). 그때까지 **새 모델 없이 M1·M2 로** 아래를 돈다.
오프라인에서 새로 확정된 것(sim §94.9~94.12): 1라운드 TPH 는 출발만 해결 · 증강(T8)·SmolVLA 모두 「원핫/지시문이 단계를 고른다」 에 못 미침 · cam_high 가 t03→04 단계 선택을 지배하고 손목 카메라도 30~100% 기여.
→ 실기에서 지금 필요한 것은 **러너 관문 닫기 + 오프라인 채점의 실기 라벨**이다. 전부 tmux(없으면 `nohup`) 안에서, **처음은 1ep, ESC 와 베이스 e-stop 을 함께**.

**DGX 가 답한 것 (위 「1호기로 넘기는 것」 2번)** — 리셋 구간 12.5 Hz 의 원인은 리셋 번들(`device=cpu`)이 매 틱 카메라 3장을 CPU 에서 변환한 것(DGX 실측 틱당 324 ms vs 0.55 ms). 리셋 단계 동안만 `predict_action` 을 state 전용으로 바꾸고 끝나면 복원(main `14e07c9`→`a22b560`, 테스트 232건).
**받기**: `git fetch najongs && git merge --ff-only najongs/main` — 이번엔 패키지 코드만 바뀌어 **`uv lock`/`uv sync` 불필요**(workspace 멤버는 editable). `git log --oneline -3 -- packages/stage_runner` 에 `a22b560` 이 보이면 됐다.

| # | 무엇 (명령) | 기록할 것 | 닫히는 결정 | 시간 |
|---|---|---|---|---|
| **1** | **②' 리셋만 재실행** — `RESET_ONLY=1 scripts/eval_chain.sh M1 resets2` (빈손, ② 와 같은 조건·같은 시작 자세) | 보고서 「경계 리셋」 표의 **`Hz ≈ 21`** · `⚠` 없음 · `events.jsonl` 의 `reason_detail.predict_path == "state_only"` · `predict_calls ≈ frames` | Hz 수정이 **실기 경로에서 먹었나** — 셋이 같이 와야 한다. 하나라도 아니면(`bypassed` 포함) 멈추고 그 값 그대로 보고 | 10분 |
| **2** | **④' 시그널** — `RESET_ONLY=1 FROM_STAGE=1 TO_STAGE=2 scripts/eval_chain.sh M1 hup`, 베이스가 **정지한 틈**에 두 번째 터미널에서 `kill -HUP $(pgrep -f '[.]venv/bin/python3 -m stage_runner')` (`[s]tage_runner` 패턴은 `uv run` 자식 이름과 안 맞아 PID 를 못 찾는다 — 로봇 PC 10/07 실측) | 로그에 stop_base → disconnect · 팔·베이스 제자리 · 종료 코드 | 시그널 래치 실증(codex 치명 지적) — ✅ 통과(위 ④' 절). **DGX 반영**: 시그널 종료는 이제 events 에 `terminator: signal`(reason `SIGHUP`/`SIGTERM`) · `trial_end.reason: signal` 로 찍히고 보고서 「그 밖」 이 `(중단·시그널·오류)` 로 갈린다 | 5분 |
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
2. **③ 물체 든 채 리셋** — 튜브는 `RESET_ONLY=1 FROM_STAGE=3 TO_STAGE=5 scripts/eval_chain.sh M1 resets_tube`, 비커는 `FROM_STAGE=6 TO_STAGE=11 … resets_beaker` (물체를 드는 단계부터). **튜브 ✅(1725) · 비커 ✅(1731) — ③ 닫힘**(위 ③ 절).
3. ~~④ M1 + 러너로 단계 1개~~ ✅ 러너 검증 통과(1735, 위 ④ 절) — 붓기 동작, timeout 종료(사람 `→` 없이), 실행 계층 수치 동일.
3b. **⑤-M1 체인** — 첫 시도 `1007_1743_chain_s1_3_e30`(1→3): 텔레옵 14.6 s → 최초 리셋 reached → **t01 에서 끊김**: M1 이 12.8 s 상한(p90 9.86×1.3) 안에 전진 0.06 m·회전 +2.4° 만 하고 `timeout`(출발 7.4 s). exec 30·21 Hz·원핫 정상 — 모델이 느린 것(10/06 task01 과 같은 꼴), 러너 문제 아님. 텔레옵 창 중 베이스 e-stop 이 잠깐 걸렸다 풀림(경고 2줄, 정책 전).
    두 번째 `1007_1746_chain_s2_4_e30`(2→4, 랙 앞 빈손): 텔레옵 3.1 s → 최초 리셋 reached(Δ0.16) → **t02(튜브 집기) 37.2 s timeout** — 체인 끊김. 사람: 「초기 자세로 넘어가는 게 부드럽지 않았다」 · 「타겟으로 다가가는 게 잘 안 된다」.
    - 리셋(데이터): 1.5 s·32틱 최소저크, 틱당 실측 이동 0.002→0.017→0.001 rad 의 종 모양, 추종 오차 최대 0.019 rad, 첫 틱 명령 단차 0.078 rad(텔레옵 마지막 명령 → 램프 첫 명령) — 수치상 매끄럽다. 「안 부드럽다」 가 램프인지 그 뒤 정책 첫 청크인지 사람 확인 필요 [추정: 정책 시작].
    - t02(M1): 출발 3.1 s, 10→15 s 사이에 팔이 시작 자세에서 **2.1 rad** 까지 크게 휘둘렀고 시연 최근접 거리가 0.3~0.4 rad 로 분포 밖에 머묾. **그리퍼를 열지 않았다** — 명령 최대 0.011(시연은 0.041 까지 열어 집음). 청크 경계 점프 1.2×(작음), clamped 0, 베이스 ≈0. → M1 task02 실기 첫 데이터: 실패(모델). 러너 문제 아님.
    세 번째 `1007_1751_chain_s2_4_e30`(2→4 재시도, `TIMEOUT_FACTOR=2`): 텔레옵 40 s → 최초 리셋 reached(Δ0.59) → **t02 57.2 s timeout**. 사람: 「꿈틀꿈틀하며 움직인다」.
    - 실행 계층: 20.9 Hz 일정, clamped 4틱, FIRED 0, 청크 경계 점프 1.3×(작음), 명령-실측 추종 오차 중앙 0.007 — **떨림의 원인이 아니다.**
    - 정책: 첫 3 s 에 0.8~1.1 rad/s 로 크게 간 뒤 50 s 동안 0.1~0.3 rad/s 의 작은 왕복(관절당 방향 반전 1.6/s, 특히 L3·R3 3.8/s ≈ 2 Hz) — **다가갔다 물러났다를 반복하며 집기로 가지 않는 「망설임」**. 그리퍼는 L 최대 0.019 / R 0.0 (시연은 0.041 까지 염). 틱당 이동 중앙 0.0062(시연 0.0077)로 느리지도 않다 — 방향을 못 정하는 것.
    - → M1 task02 실기 2/2 실패(모델). 러너 문제 아님. DGX 에: task02 시작 장면(랙 앞 빈손)에서 M1 이 집기 동작으로 수렴하지 않는 이유 — t01→t02 는 §89 에서 원핫이 단계를 고르는 유일한 경계였다 [연결 추정].
    네 번째 **`1007_1756_chain_s3_4_e30`**(3→4, `TIMEOUT_FACTOR=2`, 튜브 든 채 랙 앞) — [chain.md](run_logs/2026-10-07_eval_najy/1007_1756_chain_s3_4_e30.chain.md) · [events](run_logs/2026-10-07_eval_najy/1007_1756_chain_s3_4_e30.events.jsonl) · [발췌](run_logs/2026-10-07_eval_najy/1007_1756_chain_s3_4_e30.stages.txt):
    텔레옵 27.6 s → `reset_pre_03` reached → **t03 `complete`(stall_nn, 8.05 s)** → `reset_to_04` reached(Δ0.41) → **t04 departed 0.92 s → 35 s timeout**(사람 `→` 없음). 사람: 「회전을 안 한다, 팔만 움직거린다」.
    - **M1 task03: 베이스 회전 0** — 8 s 동안 θ 명령 최대 0.014 rad/s, 적분 −1.0°(10/06 ep1·ep2 와 같은 「출발 안 함」; ep0 은 −85° 성공). 오늘 0/1. 실행 계층 정상(20.9 Hz, clamped 0).
    - **[중요 — 러너] 이동 단계의 거짓 완료**: 팔이 0.10 rad 꿈틀한 것을 「출발」(6.0 s)로 받고, 그 뒤 3 s 정지 + 끝 자세 NN 0.120 으로 `complete` 를 냈다 — **베이스가 한 번도 안 돈 task03 을 완료로 판정**하고 체인이 붓기로 넘어갔다.
      원인은 출발 래치가 「팔 0.10 rad **또는** 베이스 0.17 rad/0.10 m」 인 것 — 이동 단계(1·3·6·10)는 시연에서도 팔이 움직이므로(task03 0.021/프레임) 팔 조건이 늘 먼저 걸린다. **제안(DGX)**: 이동 단계는 출발·완료 둘 다 **베이스 누적(회전 또는 전진)** 을 요구하고, 완료의 끝 자세 NN 에 베이스 ∫θ·∫x(stage_params 의 시연 중앙값, 21 Hz 환산)를 포함.
    - 전환 자체는 돌았다: 정책 → stop_base primary → 리셋 reached → 원핫 4/11 재설정 → 정책 출발 0.92 s. **⑤ 의 「리셋+정책+전환이 한 프로세스에서」 는 확인** — 다만 그 전환이 거짓 완료로 열린 것이라 모델 평가로는 무효.
    ⑤ 는 여기까지. M1 은 이동 단계(1·3)를 못 하고 task02 도 못 하므로 체인 완주는 새 모델(3라운드) 뒤. ← **지금 여기**
4. B1~B3(지정 자세 출발 task05/02 · M2 task04 끝 정지 · task03/01 재시험) → `eval_najy_results_1006.md` 「다음」.
5. C. 전송(사람).
