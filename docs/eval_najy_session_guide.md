# 다단계 ACT 실기 Eval — Claude 세션 진행 안내

다음 Claude 세션이 이 문서 하나로 「지금 어디까지 왔고, 다음에 무엇을, 누가, 어떻게」 를 알 수 있게 쓴 것이다.
**순서·명령의 정본은 [`eval_najy.md`](eval_najy.md)** 이고, 이 문서는 그것을 세션에서 진행하는 방법이다.
안전 규칙은 `CLAUDE.md` 「실기 안전」 — 여기서 반복하지 않는다.

## 0. 세션을 열면 (5분)

```shell
cd ~/NAJY/lerobot_trossen && git pull --ff-only          # 다른 서버(DGX_1)가 문서를 고쳤을 수 있다
echo $CLAUDE_CONFIG_DIR                                    # /home/trossen-ai/NAJY/.claude-najy 여야 한다 (CLAUDE.md 머리말)
grep -n "^apply_task_onehot_patch()" packages/lerobot_robot_trossen/src/lerobot_robot_trossen/__init__.py
ls -t ~/eval_logs/ | head -20                              # 마지막으로 돈 회차
ls docs/eval_najy_results_*.md                             # 결과 기록 — 날짜가 가장 늦은 것이 현재 위치
```

DGX_1 에서 열었으면 먼저 [`HANDOFF_najy_1006.md`](HANDOFF_najy_1006.md) (로봇 PC 와 다른 점·DGX 에서 할 분석).

읽는 순서: `CLAUDE.md` → 이 문서 → `eval_najy.md` 「지금까지」 표 → 가장 최근 `eval_najy_results_<MMDD>.md` 의 「다음」 절.

**현재 위치 정하기** — 가장 최근 결과 문서의 「다음」 절이 가리키는 단계부터 한다. 결과 문서에 없는 회차가 `~/eval_logs` 에
있으면(사람이 혼자 돌린 것) 먼저 그것부터 2절 B 로 정리한다.

## 1. 누가 무엇을 하나

| 하는 일 | Claude 가 직접 | 사람에게 명령을 넘김 |
|---|---|---|
| 기록 분석 — `scripts/eval_najy_post.sh` (motion·latency·base·report) | ✅ (파일만 읽는다) | |
| 명령 확인 — `DRY_RUN=1 scripts/eval_najy.sh …` | ✅ (로봇 무접촉, `~/eval_logs`·HF 다운로드는 한다) | |
| 결과 문서 작성·커밋·push | ✅ | |
| 실기 회차 — `scripts/eval_najy.sh …` (DRY_RUN 없이) | ❌ | ✅ |
| `scripts/measure_base.py rate|latency|odom` | ❌ | ✅ (`latency`·`odom` 은 베이스가 움직인다) |
| 시작 자세 맞추기·장면 세팅·비상정지·성공 판정 | ❌ | ✅ |

넘길 때는 **복사해서 바로 칠 수 있는 한 줄**과, 그 회차에서 사람이 볼 것(통과/멈춤 기준)을 같이 준다. 예:

> `scripts/eval_najy.sh M1 5 30 3`
> 리셋 구간에 시작 자세를 task05 표(왼팔 j0 −14° j1 +90° …)에 맞춰 주세요. 로그 첫머리에 `LEROBOT_TASK_ONEHOT=5/11 installed` 가
> 없으면 ESC. 에피소드마다 성공/실패와 어디서 멈췄는지 알려 주세요.

## 2. 한 회차를 진행하는 법

**A. 돌리기 전 (Claude)**
1. `DRY_RUN=1 scripts/eval_najy.sh <모델> <단계> <exec> <ep>` — 체크포인트 폭·원핫·exec≤chunk 확인이 통과하는지 본다.
2. 그 단계의 학습 시작 자세(`eval_najy.md` 표)와 「허용」 값을 사람에게 준다. task08·task10 은 중앙값이 아니라 시연 하나의 첫 프레임.
3. 명령을 넘긴다.

**B. 돈 뒤 (Claude)** — 사람이 「끝났다」 고 하면:
```shell
cat ~/eval_logs/<회차>.report.md          # eval_najy.sh 가 끝에 eval_najy_post.sh 로 만든 한 장 보고서
scripts/eval_najy_post.sh <회차>           # 중단 등으로 안 만들어졌을 때만 (로봇 무접촉)
```
- `<회차>` = `MMDD_HHMM_<m1|m2|m3|base>_t<NN>_e<exec>` (스크립트가 실행 시작에 찍는다). 원자료는 `<회차>.{summary,motion,latency,base}.txt`.
- 학습 데이터셋은 post 스크립트가 단계 번호로 고른다(아래 표). 로컬 캐시(`~/.cache/huggingface/lerobot/kiroaiseoul/`)에 없으면 학습 대조만 빠진다 —
  report 에 그렇게 적힌다(내려받지 말 것: 수 GB).
- report 의 「사람 판정」 열은 `results.csv` 에서 온다 — 먼저 사람 관찰을 CSV 에 적고 post 를 다시 돌리면 채워진다.
- 확인할 것과 판정:

| 볼 것 | 어디서 | 기준 |
|---|---|---|
| 원핫 적용 | `<회차>.log` 의 `installed`·`stage i/K active` | 없으면 그 회차 **무효** |
| 루프 주기 | summary 의 policy `mean` Hz | 녹화 주기(현재 20.9 Hz)의 ±10% 밖이면 이동 단계 결과는 해석 보류 |
| 시작 자세 | motion 의 `시작점 최근접` | 「허용」 이하인 에피소드만 모델 판정에 쓴다. 넘으면 「자세 불일치」 |
| 움직임 | motion 의 `프레임당 이동` | 학습 중앙값의 1/5 미만이면 「정지」 (10/02 task05: 0.005 vs 0.053) |
| 청크 경계 튐 | motion 의 `청크 경계 점프` 비 · summary 의 `FIRED` | 기록만 한다 |
| 지연 | latency 출력 | 베이스 유효 지연·t63/t90, 팔 지연(학습 데이터는 2틱) |

- 에피소드별 결과를 `~/eval_logs/eval_najy_results.csv` 에 한 줄씩(사람이 안 적었으면 Claude 가 대화 내용으로): `run,episode,success(0/1),시작점최근접,비고`.

**C. 기록 (Claude)** — 그날 회차를 `docs/eval_najy_results_<MMDD>.md` 에 정리한다(형식은 4절). 커밋·push 후,
`eval_najy.md` 의 「지금까지」 표에서 바뀐 줄만 고친다.

## 3. 판단 규칙 (eval_najy.md 와 같은 것을 한곳에)

- **exec 는 30.** 10/02 task04 에서 exec 5 는 팔이 멈췄다. exec 5·시간 앙상블을 다시 제안하지 마라 (B* 의 chunk 100 확인용 exec 100 만 예외).
- **에피소드 1~3개는 성공률이 아니다.** 「된다 / 정지 / 엉뚱함」 분류와 증상만 적는다. 두 모델을 성공 수 1개 차이로 가르지 않는다.
- **오프라인 지표(팔 MAE·base_step_mae)로 실기 결과를 예측하거나 반박하지 마라.** 녹화 재생 지표는 실기에서 두 번 틀렸다(exec 5, s2000 우세).
- **시작 자세가 「허용」 밖인 에피소드로 모델을 판정하지 않는다.** 10/02 task05 정지 회차는 전부 허용 밖이었다.
- 순서는 `eval_najy.md` 의 A → B → C → D → E → F. C 의 결과표가 D 로 갈지, DGX 분석으로 넘길지를 정한다.
- 단계 하나에서 3ep 모두 정지·엉뚱함이면 그 단계는 E(비교)로 미루고 다음 단계로 간다 — 한 단계에 회차를 쏟지 않는다.

## 4. 결과 문서 형식 — `docs/eval_najy_results_<MMDD>.md`

[`eval_najy_results_1002.md`](eval_najy_results_1002.md) 를 따른다. 최소한:

```markdown
# 다단계 ACT 실기 Eval 결과 — 2026-MM-DD (Trossen PC1)
순서표 단계: <예: C·D 일부>. 원자료: ~/eval_logs/<회차>.*, eval 데이터셋 ~/.cache/huggingface/lerobot/kiroaiseoul/eval_najy_<회차> (로컬만)
수치 출처: 성공/실패·증상은 사람 관찰, 움직임·지연은 스크립트 실측.

## 회차
| 회차 | 모델 | 단계 | 결과 | 시작점 최근접(허용) | 프레임당 이동 | 비고 |

## 지연 (eval_latency_stats.py)
| 회차 | 베이스 유효 지연 | t63/t90 | 잔여 회전 | 팔 지연(학습 대비) | 추론 스파이크 |

## 읽은 것
(판단 규칙에 비춰 무엇이 확인됐고 무엇이 아직 아닌가. 추정은 [추정])

## 다음
1. (eval_najy.md 의 어느 단계, 어떤 명령)
```

커밋: `git add docs/eval_najy_results_<MMDD>.md docs/eval_najy.md` (파일 명시) → `docs(eval): MM/DD 실기 Eval 결과(<단계>)` → `git push`.

## 5. DGX_1 로 넘기기

DGX 에서는 실패 회차의 실제 카메라 프레임을 모델에 다시 넣어 원인을 가른다(「모델 예측 자체가 정지인가」 vs 「실행이 멈추나」).
C 에서 「B* 만 됨」·「셋 다 정지」 가 나오거나, D 에서 정지 단계가 둘 이상이면 사람에게 이 전송을 요청한다:

```shell
scp -r ~/eval_logs kiro-ai@<DGX>:/raid/kiro-ai/eval/real/
scp -r ~/.cache/huggingface/lerobot/kiroaiseoul/eval_najy_* kiro-ai@<DGX>:/raid/kiro-ai/eval/real/datasets/
```
(`<DGX>` 주소는 사람이 안다 — 레포는 public 이라 적지 않는다.) 결과 문서 「다음」 절에 「DGX 분석 요청: <회차들>」 을 남긴다.

## 6. 학습 단계 · 데이터셋 대응표

| 단계 | 원핫 (M1·M2) | M3 | 학습 데이터셋 (`kiroaiseoul/…`) | 기존 전문가 B* |
|---|---|---|---|---|
| 1 | `1/11` | `1/4` | task01_move_to_tube_rack | — |
| 2 | `2/11` | — | task02_pickup_tubes | — |
| 3 | `3/11` | `2/4` | task03_turn_to_face_beaker | — |
| 4 | `4/11` | — | task04_pour_liquid_from_tubes_to_beaker | act_task04_pour_liquid_from_tubes_to_beaker_60000 (16D) |
| 5 | `5/11` | — | task05_tube_disposal | act_task05_tube_disposal_60000 (16D) |
| 6 | `6/11` | `3/4` | task06_pickup_beaker_and_move_to_refrigerator | act_task06_pickup_beaker_and_move_to_refrigerator_14D_all_260923 (14D) |
| 7 | `7/11` | — | task07_open_refrigerator | — |
| 8 | `8/11` | — | task08_takeout_and_put_beaker | — |
| 9 | `9/11` | — | task09_close_refrigerator | — |
| 10 | `10/11` | `4/4` | task10_move_to_beaker_shelf | (`act_task10_..._baseprogress` 는 쓰지 않는다) |
| 11 | `11/11` | — | task11_place_beaker_on_the_shelf | — |

`eval_najy.sh` 는 원핫을 단계 번호로 알아서 붙인다 — 위 원핫 열은 로그 대조용이다.
