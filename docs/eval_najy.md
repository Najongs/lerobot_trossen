# 다단계 ACT 실기 Eval — 순서표 (2026-10-02)

DGX_1 에서 학습한 다단계 ACT(11단계 단일 모델 등)를 로봇에서 하나씩 평가하는 순서다.
명령은 `scripts/eval_najy.sh` 가 조립한다 — **모델·단계·exec 만 바꾼다.**
배경·오프라인 근거: `trossen-ai-simulation` 레포 `docs/real_robot_eval.md`, `docs/mobile_base_investigation.md` §82~86.

## 시작 전 (한 번)

```shell
cd ~/NAJY/lerobot_trossen          # Trossen PC1 의 실기 체크아웃 (홈의 ~/lerobot_trossen 아님 — 원핫 패치가 없다)
git pull --ff-only && uv sync
grep -n "^apply_task_onehot_patch()" packages/lerobot_robot_trossen/src/lerobot_robot_trossen/__init__.py   # 한 줄 나와야 한다
```

- 팔 4개 ping · RealSense 시리얼 3개 · **베이스 비상정지 버튼 해제** (README §3-1).
- 모델은 스크립트가 허브에서 받는다 (퍼블릭):

| 기호 | 허브 | state | 단계 | 오프라인에서 |
|---|---|---|---|---|
| M1 | `kiroaiseoul/NAJY_act_all11_hot_27D_120k_s1000` | 27 | 1~11 | 이동 단계 우세 |
| M2 | `kiroaiseoul/NAJY_act_all11_hot_27D_120k_s2000` | 27 | 1~11 | 조작 단계 우세 |
| M3 | `kiroaiseoul/NAJY_act_move4_hot_20D_60k_s1000` | 20 | 1·3·6·10 | 이동 전용, M1 과 동률 |
| B* | 기존 `kiroaiseoul/act_task0N_*` | 14 또는 16 | 해당 단계 | 기준선 (원핫 없이, chunk 100). 16D 면 스크립트가 `include_base_in_state=true` 로 돌린다 |

- 아무것도 움직이기 전에 명령만 보려면: `DRY_RUN=1 scripts/eval_najy.sh M1 4 30 3`

## 매 회차 공통 규칙

- 실행 로그 첫머리에 **`LEROBOT_TASK_ONEHOT=<i>/<K> installed`**, 첫 에피소드에 **`stage i/K active`** 가 보여야 한다.
  **없으면 ESC 로 즉시 중단** — 패치 없이 돈 런은 결과로 쓰지 않는다. (M1·M2·M3 만 해당. B* 는 원핫 없음)
- 리셋 구간은 리더암으로 시작 자세를 만들고 `→`. 에피소드 구간이 정책 구동이다.
- 에피소드가 끝날 때마다 `~/eval_logs/eval_najy_results.csv` 에 한 줄: `run,episode,success(0/1),비고(어디서 어떻게 실패)`.
  `run` 은 스크립트가 출력하는 회차 이름(`1002_1430_m1_t04_e30` 꼴).
- 위험하면 비상정지 → 그 회차는 중단 기록. 스크립트 종료 코드가 0 이 아니면 요약 파일과 함께 남긴다.
- 회차가 끝나면 `~/eval_logs/<회차>.summary.txt` 에 루프 주기·페이싱 발동 수가 정리된다.

## 순서

각 단계는 앞 단계가 통과해야 넘어간다. 에피소드 수는 첫 판단용 최소치다.

### 0. 정책 없이 재기 — 해석의 기준선

| 할 일 | 명령 | 남길 것 |
|---|---|---|
| 0-1 녹화 주기 | 각 단계 **녹화 런**의 로그에서 `grep "Control loop rate" <녹화로그> \| grep phase=teleop \| tail -1` | `mean` Hz (단계별). 로그가 없으면 같은 조건으로 텔레옵 녹화를 30초 해서 잰다 |
| 0-2 베이스 왕복 | `uv run --extra pi0 python scripts/measure_base.py rate` | 출력 전체 |
| 0-3 베이스 응답 지연 | `uv run --extra pi0 python scripts/measure_base.py latency` ⚠️ **베이스가 움직인다** — 주변 1 m 비우기 | t63·t90 중앙값 |

> 베이스 속도 명령은 다음 명령까지 유지되므로 **(정책 런 루프 주기) ÷ (녹화 주기) 의 역수만큼 과회전**한다.
> 0-1 이 없으면 이동 단계 결과를 해석할 수 없다.

### 1. 연결 점검 — M1 · task04 · exec 30 · 3ep

```shell
scripts/eval_najy.sh M1 4 30 3
```
task04(붓기)는 베이스가 거의 정지하고, 오프라인 폐루프 근사에서 유일하게 진행도가 높았다(exec 30 에서 0.97, `mobile_base_investigation.md` §84 — 실기 성공률의 예측은 아니다).
**통과**: `installed`·`stage 4/11 active` 확인 · 팔이 시연과 같은 방향으로 움직임 · 요약의 `mean` Hz 가 0-1 녹화 주기의 ±10% 안.
**멈춤**: 팔이 엉뚱한 관절로 가면(정규화·원핫 오류 신호) 즉시 비상정지하고 로그를 보낸다.

### 2. exec 결정 — M1 · task04·05 · exec 30 vs 5 · 각 5ep

```shell
scripts/eval_najy.sh M1 4 30
scripts/eval_najy.sh M1 4 5
scripts/eval_najy.sh M1 5 30
scripts/eval_najy.sh M1 5 5
```
오프라인 두 평가가 반대 결론이라(녹화 재생은 5 우세, 폐루프 근사는 30 우세) 여기서 정한다.
**비교**: 성공 수 · 요약의 `min` Hz(exec 5 는 5프레임마다 추론이 몰려 `min` 만 떨어진다) · 페이싱 `FIRED` 횟수(청크 이음매가 팔 속도 한계를 때리는 신호).
**판정**: 성공 수가 같으면 exec 30 유지(덜 바꾼 쪽). 이후 단계는 이긴 exec 로 — 아래 `<E>`.

### 3. 첫 이동 단계 — M1 · task03 · 5ep

```shell
scripts/eval_najy.sh M1 3 <E>
```
제자리 회전(−57°)만 있어 가장 단순하다. **볼 것**: 회전량(정답 대비), 멈춘 뒤 제자리 유지(떨림/한 방향 표류).
베이스 CSV(`<회차>.basevel.csv`)에 명령·실측 속도가 남는다.

### 4. 이동 단계 — M1 vs M3 · task01·06·10 · 각 5ep

```shell
for s in 1 6 10; do scripts/eval_najy.sh M1 $s <E>; scripts/eval_najy.sh M3 $s <E>; done
```
11단계 모델(M1)과 이동 전용(M3) — 오프라인 동률. 기존 전문가가 있으면 같은 날 B* 도:
`scripts/eval_najy.sh kiroaiseoul/act_task06_pickup_beaker_and_move_to_refrigerator_14D_all_260923 6 <E>`
(B* 는 chunk 100 이지만 같은 `<E>` 로 돌려 조건을 맞춘다. ⚠️ `act_task10_..._14D_baseprogress_60k_fp32` 는 이름과 달리
state 가 16D 이고 베이스 칸의 의미(진행도?)를 확인하지 못했다 — 기준선에서 뺀다).

### 5. 조작 단계 — M1 vs M2 · task02·07·08·09·11 · 각 5ep

```shell
for s in 2 7 8 9 11; do scripts/eval_najy.sh M1 $s <E>; scripts/eval_najy.sh M2 $s <E>; done
```
오프라인에서는 M2 가 조작 3단계 모두 나았고(task08 0.0186 vs 0.0349 rad), 11단계 모델이 단계별 조작 ACT 보다 팔 MAE 가 33~63% 낮았다(task02 0.0309 vs 0.0839, §85 — 「3.0~10.2×」 는 개선폭을 시드폭으로 나눈 값이지 오차 비가 아니다).
기존 전문가가 있는 단계는 B* 도 같은 회차에 — `kiroaiseoul/act_task04_pour_liquid_from_tubes_to_beaker_60000`,
`kiroaiseoul/act_task05_tube_disposal_60000` (둘 다 state 16D·chunk 100, 베이스 실측 속도를 state 에 넣고 학습된 구세대).

### 6. 정리

- 단계마다 승자(M1/M2/M3/B*)를 정한다. 이동은 M1, 조작은 M2 처럼 갈리면 **단계마다 체크포인트를 나눠 쓴다** (둘 다 `/11`).
- 11단계 연속 실행은 `stage_runner`(PR #48, 미머지)가 리베이스·fps 21 타이머 재산정·원핫 전달을 갖춘 뒤.

## 결과 넘기기

`~/eval_logs/` 의 `*.summary.txt` · `*.log` · `*.basevel.csv` · `eval_najy_results.csv` 를 DGX 로 옮기면
(`scp ~/eval_logs/* kiro-ai@<DGX>:/raid/kiro-ai/eval/real/`) 오프라인 지표와 대조해 판정한다.
eval 데이터셋(`kiroaiseoul/eval_najy_*`)은 lerobot-record 가 로컬에 남긴다 — 허브 업로드는 필요할 때만.

## 문제가 생기면

| 증상 | 원인 후보 | 할 일 |
|---|---|---|
| `installed` 줄이 없음 | 패치 미반영 (`uv sync` 안 함, 다른 체크아웃) | 중단 → 「시작 전」 다시 |
| 첫 프레임에 `normalizer`·shape 에러 | 원핫 미적용 / 잘못된 모델 | 로그 첫 50줄 보내기 |
| `robot state is N wide` | `include_velocity` 등 관측 옵션이 켜짐 | 기본 설정으로 |
| 베이스가 과하게 돈다 | 루프 주기 < 녹화 주기 | 요약의 `mean` Hz 와 0-1 비교 |
| 팔이 청크 경계마다 튄다 | 이음매 | `FIRED` 수 기록, exec 비교에 반영 |
| `FileExistsError` | 같은 repo_id 재사용 | 스크립트는 분 단위 이름 — 1분 기다리고 재실행 |
