# 다단계 ACT 체크포인트 — 무엇을 받아서 돌리는가 (2026-10-02)

로봇 PC 에서 `scripts/eval_najy.sh` 로 돌리는 체크포인트(M1·M2·M3·B*)가 어떻게 학습됐고 무엇을 입력으로 기대하는지.
실기 순서는 [`eval_najy.md`](eval_najy.md), 오프라인 판정은 [`offline_eval_findings_act.md`](offline_eval_findings_act.md),
차원 함정은 [`data_layout_traps.md`](data_layout_traps.md).
학습 정본: `Najongs/trossen-ai-simulation` 레포 `docs/mobile_base_investigation.md` §74~86, `docs/real_robot_eval.md`
(이하 §번호는 mobile_base_investigation 기준).

## 1. 체크포인트 한눈에

| 기호 | 허브 (public) | state | action | 단계 | 시드 | 오프라인에서 |
|---|---|---|---|---|---|---|
| M1 | `kiroaiseoul/NAJY_act_all11_hot_27D_120k_s1000` | 27 | 16 | 1~11 | 1000 | 이동 단계 4/4 우세 (§83) |
| M2 | `kiroaiseoul/NAJY_act_all11_hot_27D_120k_s2000` | 27 | 16 | 1~11 | 2000 | 조작 단계 3/3 우세 (§85) |
| M3 | `kiroaiseoul/NAJY_act_move4_hot_20D_60k_s1000` | 20 | 16 | 1·3·6·10 | 1000 | 이동 전용, M1 과 동률 |
| B* | `kiroaiseoul/act_task0N_*` (구세대 단계별 전문가) | 14 또는 16 | 16 | 해당 단계 | — | 기준선. chunk 100, 원핫 없음 |

M1·M2 는 같은 설정에 시드만 다르다. 시드 차이가 단계 종류에 따라 반대로 나서(이동은 s1000, 조작은 s2000) 둘 다 실기에 올린다.

## 2. 학습 설정 (M1, 체크포인트의 `config.json`·`train_config.json` 에서 확인)

- ACT · resnet18 백본 · VAE(kl_weight 10) · dim_model 512
- `chunk_size=30`, `n_action_steps=30`, `temporal_ensemble_coeff=null` — 앙상블 없이 학습·배포
- AdamW lr 3e-5 · weight decay 1e-4 · grad clip 10 · 스케줄러 없음(상수 LR)
- 배치 8/GPU × 4 GPU = 유효 32 · 120K step · 15K 마다 저장 · 영상 증강 off · `video_backend=pyav`
- 입력: `observation.state[27]` + 카메라 3개(`cam_high`, `cam_left_wrist`, `cam_right_wrist`, 3×480×640)
- 출력: `action[16]`

lr 3e-5 는 「최적」이라는 근거가 없다. 그 판정(§21~36)은 전처리기 버그로 통째 철회됐고 재측정에서는 1e-5 가 같거나 낫다(§37).
M1 이 3e-5 를 쓰는 것은 기존 설정을 유지한 것뿐이다.

## 3. 학습 데이터

- 매니페스트 `configs/datasets/all11_tr_hot.json` (sim 레포): `kiroaiseoul/task01` ~ `task11` 11개, 학습 에피소드 합 1,392
- 분할: 이동 4단계(01·03·06·10)는 move4 와 같은 분할, 조작 7단계는 시간순 80/20 (§81.2)
- knob: `base_state: zero` · `balance: true` · `task_onehot: [i, 11]` (**0-based** — task01 = `[0, 11]`)
- **체크포인트의 `train_config.json` 의 `repo_id` 에는 task01 하나만 보인다.** 실제로 쓴 데이터 목록은 동봉된 `multi_manifest.json` 이다

## 4. 입력·출력 차원

```
state 27 = [ left 6관절+그리퍼 (7) | right 6관절+그리퍼 (7) | x.vel=0 | theta.vel=0 | 원핫 11 ]
state 20 = [ 팔 14 | 0 | 0 | 원핫 4 ]          (M3)
action 16 = [ left 7 | right 7 | x.vel | theta.vel ]
```

- **베이스 state 칸은 항상 0이다**(이름의 「14D 기반」). action 은 16D 그대로라 정책은 베이스를 계속 조종한다(§76.1).
  0으로 두는 이유는 [`data_layout_traps.md`](data_layout_traps.md) §2.
- 원핫은 로봇 PC 에서 `LEROBOT_TASK_ONEHOT=<i>/<K>` 로 붙인다 — **env 는 1-based**(`4/11` = task04), 매니페스트는 0-based.
  패치가 변환하며, 오프라인에서 학습 입력과 비트 동일함을 검증했다(real_robot_eval §2).
  패치 없이 27D/20D 체크포인트를 돌리면 첫 프레임 정규화에서 죽는다.
- 원핫은 **성능 때문에 넣은 게 아니다.** 세 번 비교에서 이득이 없었다(S2-18·S2-24, §74). 실기가 시간 기반으로
  단계를 넘기므로 단계 번호를 이미 안다는 배포 요건 때문이다(§74.3, §76.3).
- B*: 구세대 16D(9/16 이전) 체크포인트는 베이스 실측 속도를 state 에 넣고 학습됐다 → `include_base_in_state=true` 필요.
  `eval_najy.sh` 가 16D 를 감지해 켠다. `act_task10_…_14D_baseprogress_60k_fp32` 는 이름과 달리 16D 이고 베이스 칸의
  의미를 알 수 없어 기준선에서 뺐다(eval_najy.md).

## 5. 버전

- 학습: lerobot **0.4.1** (sim 레포). 로봇 PC: lerobot **0.4.4** (`uv.lock`, `.venv` 실측)
- 전처리기 json 의 0.4.1 → 0.4.4 호환은 문서상 확인 기록이 없다. 원핫 패치의 「비트 동일」 검증이 어느 버전에서 됐는지도 미기재.
  첫 실기에서 팔이 엉뚱한 관절로 가면 이 차이를 먼저 의심한다
- 추론 지연(V100 측정): ACT 28.7 ms, SmolVLA 530.7 ms (§74.2). **RTX 5090 Laptop 에서의 ACT forward 는 미측정**

## 6. 정책 선택 배경

- 배포 주력은 ACT. 같은 서버 비교에서 SmolVLA 와 같거나 낫고(S2-22·24, 잠정), SmolVLA 는 exec=5 를 실기 예산(167 ms)
  안에 못 돈다(§78.1)
- SmolVLA 를 다시 쓴다면 출발점은 우리 데이터만 쓴 `foundation_smolvla_kiro` 가 최선이고, 외부 42셋을 섞은 v3 가
  넷 중 최악이었다(§65). 「사전학습 데이터는 많을수록 좋다」 는 반증됐다(§65.2)
- pi0: 이 레포에는 README 부록의 명령뿐이고 학습·평가 결과는 없다
- 구세대 단계별 전문가(B*): chunk 100. task01·03 만 theta.vel 자기상관을 근거로 chunk 30. hue 증강은 weight 0
  (색이 물체 식별 단서라서)

## 7. 학습을 다시 돌릴 때의 함정

- lerobot 0.4.1 DDP 로그의 `smpl`·`ep`·`epch` 는 `num_processes` 배 부풀려 찍힌다 — step 수로 판단
- torchcodec 은 `num_workers>0` 에서 워커가 죽는다 → `video_backend=pyav` (§40)
- DDP 랭크 하나가 영상 디코딩 오류로 죽으면 나머지 랭크가 무한 대기한다 — GPU util 이 0 인데 프로세스가 살아 있으면 의심
- 직접 추론 코드를 쓸 때는 `predict_action_chunk(pre(batch))` → `post(...)`. 전처리기를 빠뜨려도 **에러 없이** 정규화 안 된
  입력으로 돈다 — §21~36 철회의 원인(§37)
- 큰 레버는 「평가하는 단계 자신의 에피소드 수」다(§61.3, §76.2, §80.1). 합본 이득은 자기 데이터가 적을 때만.
  추가 수집 우선순위: 이동 task01·10·03, 조작 task02 (S2-27.4)
