# 데이터 레이아웃 함정 — 차원은 맞는데 조용히 틀리는 것 (2026-10-02)

녹화·학습·eval 사이에서 텐서 모양은 맞지만 의미가 어긋나 **에러 없이** 틀린 결과를 내는 경우들.
출처: `Najongs/trossen-ai-simulation` 의 `docs/foundation_dataset_manifest.md`, `docs/action_observation_dim.md`,
`docs/project_status.md` §9, `docs/mobile_base_investigation.md`(§번호), `docs/corpus_distribution_audit.md`.

## 1. 16D 배치 — 실데이터와 sim 이 다르다

| 출처 | 배치 | 그리퍼 |
|---|---|---|
| **실데이터 (이 레포로 녹화, v3.0)** | `[left 6관절+그리퍼, right 6관절+그리퍼, x.vel, theta.vel]` — 베이스가 **뒤** | 캐리지 이동 미터 (0~0.04) |
| sim 텔레옵 (`trossen_ai_sim`) | `[v, w, left7, right7]` — 베이스가 **앞** | [0, 1] 정규화 |
| sim gym env | 베이스가 뒤 | — |
| Stationary (팔 2개) | 14D, 카메라 4개 | — |

- 카메라: `cam_high`, `cam_left_wrist`, `cam_right_wrist` · 480×640 · 30 fps 메타데이터
- sim 문서들의 차원 표를 실기 16D 의 근거로 인용하지 마라. sim 데이터를 실데이터와 섞으면 차원만 맞고 베이스·팔 칸이 뒤바뀐다

## 2. 베이스 action 은 명령이 아니라 실측의 복사본이다

- 녹화는 passive 다: 리더암이 팔을 움직이고, 베이스는 사람이 조이스틱으로 몰며, 기록되는 베이스 action 은
  `base.get_vel()` 실측값이다(§1)
- 그래서 텔레옵 데이터에서 `action[14:16] == state[14:16]` 이 **100%** 다. state 에 베이스 속도를 넣으면 정책이
  「지금 속도를 그대로 내보내기」 라는 지름길을 배운다(§41, §76.1)
- ACT 는 학습한 에피소드를 암기하고, 못 본 에피소드에서는 state 의 베이스 칸을 베낀다(§42). 실기는 전부 못 본 에피소드다
- 베이스 state 표현 비교: vel 이 최하위, odom 은 zero 와 구분되지 않음 → **베이스 칸 0(14D 기반)** 이 기본(§61.2, S2-27 #1).
  이 레포의 `include_base_in_state=false` 기본값(a015c68)이 이것이다

## 3. timestamp 는 합성값이다

- parquet 의 timestamp 는 30 fps 로 찍힌 합성값이다. **실제 녹화 루프는 약 21 Hz** 이고, 진짜 주기는 녹화 로그의
  `Control loop rate` 줄에만 남는다(§8)
- 베이스 속도 명령은 다음 명령까지 유지되므로 **과회전 배수 ≈ 녹화 루프 Hz ÷ eval 루프 Hz**. eval 루프가 느리면 베이스가 더 돈다
- 그래서 eval 전에 기준을 잡는다: 녹화 로그의 rate, `scripts/measure_base.py rate`·`latency`.
  10-02 기준선: 텔레옵 녹화 20.9 Hz 평균, 베이스 I/O 40.5 ms/프레임(→ 약 25 Hz 상한)
- 비상정지가 걸려 있으면 `set_cmd_vel` 은 성공을 반환하고 `get_vel` 은 0 이다 — 「명령은 갔는데 안 움직임」 의 1순위 원인

## 4. 데이터셋 포맷 세대

| 포맷 | state | 비고 |
|---|---|---|
| v2.1 (`trossen_ai_mobile`) | 19D = `[팔 14, odom 3, vel 2]` | `[0:16]` 으로 자르면 **odom 이 vel 자리에** 들어간다 |
| v3.0 | 16D = `[팔 14, vel 2]` | 현재 녹화 |
| `*_14D` repo | state 14 / action 16 | 베이스 state 를 뺀 슬라이스 (`scripts/slice_feature_dims.py`) |

허브 데이터셋 메타데이터 자체에도 결함이 있는 것이 있다 — 학습 전 `meta/info.json` 의 feature 이름·shape 를 실제 parquet 과 대조한다.

## 5. 서드파티 16D 데이터 (파운데이션 사전학습용 참고)

ACT 배포 모델(M1·M2·M3)은 `kiroaiseoul/task01~11` 만 쓰므로 무관하다. SmolVLA 류 사전학습에 외부 데이터를 섞을 때만 해당.

- 좌측 그리퍼 단위가 라디안인 셋이 있다(미터와 ×0.017453 차이) — `corpus_distribution_audit.md` §1
- `mrrl-emcnei` 의 열 회전: 매니페스트는 2칸, 실측 오프셋은 3칸 — **미해결 충돌**
- 목록: sim 레포 `docs/trossen_16d_public_datasets.md`

## 6. 베이스가 기록되지 않은 에피소드 — 에피소드 통째로 0 (2026-10-14 DGX1 실측)
녹화 세션 단위로 베이스 속도가 안 잡혀 **에피소드 내내 x.vel·theta.vel 이 정확히 0** 인 구간이 있다. action 과 state 베이스 칸이 둘 다 0 이다(§2 — state 는 action 의 복사). 팔·영상은 정상이라 에러 없이 학습에 들어간다.
- **GIST**: 라벨 `*_labels.csv` 의 `data_issue = mobile_base_not_captured(frozen_at=[0.0, 0.0])` 가 이것이다. `make_manifest.py:39` 가 `success==1 & data_issue 없음` 만 남겨 **학습 매니페스트에는 0 ep** 이다. task9 원본(`kiroaiseoul/task09_close_refrigerator_gist`, 1,897 ep)은 1,895 ep 가 플래그라 사실상 빠지고 `_0804`(정상 998 ep)만 쓴다.
- **우리 데이터에는 라벨이 없어 걸러지지 않았다** — `all11_tr_hot_gist30.json`(#2 J14) 학습분 기준:

| 데이터셋 | 내내 0 인 ep (학습분) | 번호 | 비고 |
|---|---|---|---|
| `task05_tube_disposal` | 5 / 59 | 0, 53~56 | 베이스 거의 정지 단계 |
| `task06_pickup_beaker_and_move_to_refrigerator` | 5 / 160 | 48~52 | **이동 단계 — ep 50 영상은 냉장고 앞까지 이동, 명령은 0 → 라벨이 틀림** |
| `task07_open_refrigerator` | 67 / 223 | 43~52, 125~181 | 움직임 p99 0.017 |
| `task08_takeout_and_put_beaker` | 103 / 201 | 0~100, 200~204 | 움직임 p99 0.004 |
| 그 밖의 우리 7셋 | 0 | — | — |

- **14D 판은 같은 에피소드의 사본**이라 똑같이 물려받는다: `task05_tube_disposal_14D`(5)·`task06_…_14D`·`_14D_all`(48~52)·`task07_open_refrigerator_14D`(68)·`task08_…_14D`(109)·`_14D_333data`(106). 깨끗한 새 녹화는 `task08_takeout_and_put_beaker_0930`(90 ep, 0 없음 — `_14D_333data` 의 새 90 ep 와 같다)과 task06 `_14D_new_3` 계열(문구가 「shake」 로 다름).
- 에피소드 **중간** 미기록은 찾지 못했다: 이동 단계에서 2 s 이상 내부 0 구간은 그동안 팔 변위도 0.02~0.13 rad 라 실제 정지로 판단 [중간 끊김을 영상으로 확인하지는 않음].
- 홀드아웃 셋에도 이런 ep 가 섞여 있을 수 있다 — 오프라인 베이스 지표가 조금 흐려진다.
- 찾는 법: 에피소드별 `max |action[14:16]| == 0`. 처치는 학습 매니페스트 `episodes` 에서 빼기 — sim `configs/datasets/all11_tr_hot_gist30_basefix.json`(J49).
