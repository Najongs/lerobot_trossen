# 체인 회차 1007_1626_chain_resets_tube_e30

- 체크포인트: **없음** (리셋 전용 — 가중치를 로드하지 않았다. `chain.reset.only` / `RESET_ONLY=1`)
- 데이터셋: `kiroaiseoul/eval_1007_1626_chain_resets_tube_e30` @ 21 fps · 로봇 `mobileai_robot` (state 14, action 16)
- lerobot 0.4.4 · stage_runner 0.1.0 · config v2
- 결과: **chain_failed** · completed=False · 저장=False · 1/11 단계 · 0 프레임 · 32.7 s
- 텔레옵(리더암) 구간: 32.4 s → arrow (stop_base primary) -- 녹화 안 함
- **체인이 끊긴 단계: `reset_pre_01`** (재시도 없음 — 이후 단계는 돌지 않았다)

## 정책 단계

없음 — 리셋 전용 회차다 (bring-up ②·③).

| 단계 | 종료 | 출발 s | 경과 s (p10/p90) | Hz | p 끝값·유지 s | 정지 s | NN rad | ∫θ 명령/실측 ° | ∫x 명령/실측 m | clamped |
|---|---|---|---|---|---|---|---|---|---|---|

## 경계 리셋

| 리셋 | 종료 | 사유 | T s | Δmax rad (상한) | 보폭 상한 rad | 도달 오차 rad (tol) | 경과 s (상한) | Hz | 프레임 | clamped |
|---|---|---|---|---|---|---|---|---|---|---|
| `reset_pre_01` (최초) | **not_reached** | - | - | - (-) | - | - (-) | 0.0 (-) | 0.0 | 0 | - |

## 집계

- 정책 단계 0개: 없음
- 리셋 1개 (경계 0 + 최초 1): not_reached 1
- **자동 완료 0 / 수동 완료 0**: 수동이 있으면 그만큼은 모델이 끝을 스스로 알리지 못한 것이다 (사용자 결정 10/06 — 두 모델 모두 `→` 를 켜 두고 따로 센다)
- 경계 1곳의 베이스 정지: 전부 primary (확인용 set_cmd_vel 까지 수락됨)
- max_relative_target 클램프: 미측정 (로봇 밖)
- basevel.csv 681 행

> 성공·실패 **판정은 여기 없다.** 영상과 사람이 적은 표에서 오고 `~/eval_logs/eval_chain_results.csv` 로 들어간다. 이 보고서는 러너가 측정한 것만 담는다.

