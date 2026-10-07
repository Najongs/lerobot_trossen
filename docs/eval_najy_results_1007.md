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

## 읽은 것

- bring-up ①·② **통과**. 리셋 궤적(관절공간 직선, 시연에 없는 경로)은 빈손에서 전 경계 도달·상한 안·클램프 0 이었다. 큰 전이 다섯 곳의 Δmax 는 사전 계산(stage_params)과 일치한다.
- 열린 것 하나: 리셋 구간 루프 12.5 Hz 의 원인. DGX 가 러너를 소유하므로 넘긴다(아래).
- 사람 관찰이 비어 있다 — `~/eval_logs/eval_chain_results.csv` 는 정책 단계 라벨용(`run_id,stage_id,outcome,failure_phase`)이라 리셋 전용 회차엔 쓰지 않았다.

## 1호기(DGX_1)로 넘기는 것 — 10/07

1. **②의 세 관문**: `stop_base_direct_ok` 11/11 `primary` · `clamped` 0 · 도달 오차 ≤0.0019. ③(그리퍼)·④'(시그널)는 아직.
2. **리셋 구간 루프 12.5 Hz** (`other` 55 ms, mock 20.9 Hz) — 러너의 실로봇 리셋 경로에서 틱당 ~30 ms 가 어디서 드는지. 후보(미확인):
   리셋 정책 `select_action`/후처리의 실관측 경로, pose_guide `set_stage` 뒤 틱당 계산, 리셋 구간의 데이터셋 프레임 추가·finite 게이트. 로봇 PC 에서 `cProfile` 을 걸어 달라면 건다.
3. 사람 몫 그대로: **C. 1006 회차 원자료 전송**(데이터셋 6개 + `~/eval_logs/1006_*`) 아직 안 됨.

## 다음

1. **④' 시그널 실증** — 짧은 리셋 회차 `RESET_ONLY=1 FROM_STAGE=1 TO_STAGE=2 scripts/eval_chain.sh M1 hup` 을 띄우고 두 번째 터미널에서
   `kill -HUP $(pgrep -f '[s]tage_runner')`. 로그에 stop_base → disconnect, 팔·베이스 제자리.
2. **③ 물체 든 채 리셋** — 가벼운 플라스틱 튜브 먼저(`RESET_ONLY=1 scripts/eval_chain.sh M1 resets_tube`), 그다음 비커. 그리퍼가 놓치거나 더 쥐면 멈추고 보고.
3. **④ M1 + 러너로 단계 1개** — `FROM_STAGE=4 TO_STAGE=4 scripts/eval_chain.sh M1 one4` (task04 장면 세팅). policy 구간 Hz·rearm·clamped·FIRED 를 10/06 1350 과 비교.
4. B1~B3(지정 자세 출발 task05/02 · M2 task04 끝 정지 · task03/01 재시험) → `eval_najy_results_1006.md` 「다음」.
5. C. 전송(사람).
