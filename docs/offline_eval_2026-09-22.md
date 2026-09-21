# SmolVLA 오프라인 평가: task02·task06 (2026-09-22, ibomcom)

로봇 없이 데이터셋만으로 "지금 명령이 시연과 얼마나 다르게 움직이는가, 무엇을 바꾸면 나아지는가"를 잰 기록이다.
대상 체크포인트는 `kiroaiseoul/smolvla_190k_30k_aug_80k_kirogist`(로봇 PC의 `~/daehee/models/…_80k_kirogist`와 같은 것),
데이터는 `kiroaiseoul/task02_pickup_tubes`와 `kiroaiseoul/task06_pickup_beaker_and_move_to_refrigerator`다.
이전 기록([smolvla_inference_experiments.md](smolvla_inference_experiments.md), E1~E22)의 후속이다.

모든 수치는 **오프라인 실측**(ibomcom, RTX 3090·A6000, 2026-09-22)이다. 로봇 로그에서 온 값은 "로봇 로그"라고 따로 적었다.
원본 산출물(예측 청크, 시뮬레이션 결과, 실행 명령이 담긴 json)은 `docs/run_logs/2026-09-22_offline_eval/`에 있다.

## 요약

- **추론 설정으로 얻을 수 있는 개선은 작다.** 이 체크포인트는 task02·task06 시연을 "가만히 있기"보다 35~45% 나은 정도로만
  재현한다. 같은 task06 에피소드에서 이 데이터로 학습한 ACT 는 팔 오차가 1/3 이다. 성공률을 크게 올리려면 **과제 전용 파인튜닝**이
  필요하다(학습 쪽 제안 절).
- 지금 명령(`commit=50`, K=7, noise 0.9)은 큰 틀이 맞다. 폐루프 근사에서 청크를 길게 실행하는 스케줄(commit 30~50)이 연속 TA
  (coeff 0.01)보다 확실히, 짧은 commit 보다 대체로 낫고, 30~50 사이에는 차이가 없다.
- 다만 commit 50 은 청크가 45스텝에서 바닥나 **크로스페이드 없이 갈아타고, 교체 순간 관절이 한 스텝에 최대 0.64 rad 튄다.**
  새 옵션 `--ensemble.fade=5` 로 없앤다. 권장은 **`commit=30, fade=5`**. commit 40 은 교체 시점에 남은 겹침이 4~5스텝뿐이라
  fade 를 줘도 점프가 0.15~0.25 rad 남는다. commit 30 은 0.10~0.13 rad 다.
- **노이즈를 0.9 밑으로 줄이지 않는다.** 결정적 샘플링은 움직임을 1/2~1/3 로 줄이고, 폐루프 근사 완주를 5/12 에서 3/12 로 떨어뜨린다.
  K=4·noise 1.0 은 K=7·noise 0.9 와 정확도가 같고 루프가 더 빠르다(로봇 로그 19.6 vs 18.5 Hz). **K=4 를 권한다.**
- **프롬프트는 데이터셋 문장 그대로.** 다른 과제 문장을 넣으면 정책이 가만히 있기 수준으로 무너진다. GIST 판 문장도 더 나쁘다.
- 정책은 현재 state 에 강하게 붙어 있다. 첫 액션이 state 를 거의 복사하고, 어깨(joint_1)를 일관되게 0.01 rad 낮게 명령한다.
  베이스도 현재 속도를 복사해서 **정지에서 출발하는 순간을 자주 놓친다**(시연이 출발할 때 같이 출발 37~60%, ACT 78~88%).
- task06 은 베이스 속도가 루프 주기에 따라 과회전한다(K=7 에서 1.16배 추정). K=4 로 줄이고, 필요하면 새 옵션
  `--ensemble.base_rate_hz=21.5` 로 보정한다(로봇 미검증).

## 권장 명령

실행 스크립트로 정리했다. 레포 루트에서:

```bash
bash scripts/eval_smolvla.sh t02 rec        # task02 권장 설정
bash scripts/eval_smolvla.sh t02 cur        # task02 지금까지 설정 (비교용)
bash scripts/eval_smolvla.sh t06 rec        # task06 권장 설정
bash scripts/eval_smolvla.sh t06 rec_base   # task06 권장 + 베이스 과회전 보정
```

로그는 `docs/run_logs/<날짜>_smolvla/<task>_<variant>_<시각>.log` 에 남는다. 아래는 스크립트가 만드는 명령의 전문이다.

공통 인자(로봇·카메라·텔레옵·데이터셋)는 지금 명령과 같다. 달라진 줄에 표시를 달 수 없어서(bash 는 `\` 줄에 주석 불가) 아래에 적는다.

- `--ensemble.commit=30` (50 → 30), `--ensemble.fade=5` (새 옵션), `--ensemble.samples=4` (7 → 4), `--ensemble.noise_scale=1.0` (0.9 → 1.0)
- 로그를 `/tmp` 대신 레포 안 `docs/run_logs/<날짜>_smolvla/` 에 받는다(git 에 남는다)

task02:

```bash
rm -rf ~/.cache/huggingface/lerobot/kiroaiseoul/eval_foundation_smolvla_v1
mkdir -p docs/run_logs/$(date +%F)_smolvla

uv run python scripts/record_ensemble.py \
  --ensemble.commit=30 \
  --ensemble.fade=5 \
  --ensemble.coeff=-1.5 \
  --ensemble.amp=bf16 \
  --ensemble.noise_scale=1.0 \
  --ensemble.samples=4 \
  --policy.num_steps=10 \
  --robot.type=mobileai_robot \
  --robot.left_arm_ip_address=192.168.1.5 \
  --robot.right_arm_ip_address=192.168.1.4 \
  --robot.id=follower \
  --robot.cameras="{
    cam_high: {type: intelrealsense, serial_number_or_name: \"230422273501\", width: 640, height: 480, fps: 30},
    cam_left_wrist: {type: intelrealsense, serial_number_or_name: \"230422271234\", width: 640, height: 480, fps: 30},
    cam_right_wrist: {type: intelrealsense, serial_number_or_name: \"230322274369\", width: 640, height: 480, fps: 30}
  }" \
  --robot.enable_base_motor_torque=true \
  --robot.left_arm_max_relative_target=0.1 \
  --robot.right_arm_max_relative_target=0.1 \
  --robot.include_base_in_state=true \
  --teleop.type=mobileai_leader_teleop \
  --teleop.left_arm_ip_address=192.168.1.3 \
  --teleop.right_arm_ip_address=192.168.1.2 \
  --teleop.id=leader \
  --dataset.repo_id=kiroaiseoul/eval_foundation_smolvla_v1 \
  --dataset.num_episodes=1 \
  --dataset.reset_time_s=90 \
  --dataset.single_task="Pick up the two tubes from the rack with both hands" \
  --dataset.push_to_hub=false \
  --policy.device=cuda \
  --policy.path=/home/trossen-ai/daehee/models/smolvla_190k_30k_aug_80k_kirogist/pretrained_model \
  2>&1 | tee -i docs/run_logs/$(date +%F)_smolvla/t02_c30f5_k4_$(date +%H%M%S).log
```

task06: 위와 같고 두 줄만 다르다. 베이스 보정은 넣은 런과 뺀 런을 비교한다.

```bash
  --dataset.single_task="Grab and shake the filled beaker, then move to the front of the fridge" \
  --ensemble.base_rate_hz=21.5 \
```

로그 파일명도 `t06_…` 로 바꾼다. task06 문장은 데이터셋 200개 에피소드 중 177개가 쓰는 문장이다. 나머지 23개의 "Pick up the beaker and
move to the refrigerator" 로 통일하면 오프라인 오차가 커진다(2절).

## 무엇을 쟀나

1. **teacher-forced 청크 예측** (`scripts/offline_eval/predict_chunks.py`). 시연 에피소드 12개의 모든 프레임에서
   카메라 3장·state·task 문장을 넣고, 예측 청크(50×16, 실제 단위)를 시연의 다음 50 액션과 비교한다. 추론 경로는
   로봇과 같다: 체크포인트 preprocessor → `_get_action_chunk`(bf16) → postprocessor. 샘플링(K, noise)은
   `record_ensemble.py`의 함수를 그대로 가져다 쓴다. 에피소드는 전체에서 고르게 뽑았다(task02 0,12,…,132 / task06 0,18,…,199).
2. **실행 스케줄 시뮬레이션** (`simulate.py`). 저장한 청크로 `record_ensemble.py`의 비동기 경로(commit, merge, coeff,
   fade, 지연 L)를 스텝 단위로 재현한다. 병합·크로스페이드 함수는 `record_ensemble.py`에서 import 한다.
   관측이 시연에서 오므로 "같은 예측을 실행 방식이 얼마나 늦추고 뭉개는가"만 잰다.
3. **재생 폐루프 근사** (`closedloop_eval.py`). 로봇 state 가 실행한 액션을 따라가게 하고, 이미지는 시연에서
   그 state 와 가장 가까운 프레임을 가져다 넣는다. 매칭된 프레임 번호가 진행도다. 완벽한 정책(시연 청크)을 넣으면
   모든 방식이 100% 완주하도록 검증했다. **로봇이 시연 경로를 크게 벗어나면 이미지와 state 가 어긋나므로** 절대값보다
   방식 간 비교로 읽는다. 베이스 위치를 이미지에 반영할 수 없어 task02 에만 썼다.

지표:

| 이름 | 뜻 |
|---|---|
| 팔 RMSE | 팔 12관절, 시연 액션 대비 (rad) |
| 이동 비율 (gain) | 예측 변위를 시연 변위 방향에 사영한 길이 / 시연 변위. 1 이면 시연만큼, 작으면 덜 간다 |
| 방향 코사인 | 예측 변위와 시연 변위의 방향 일치 |
| 최대 점프, jerk | 실행 궤적의 스텝 간 최대 변화(rad), 2차 차분 RMS. 시연 자체의 jerk 는 0.0038 |
| 나이 | 지금 실행하는 액션을 만든 관측이 몇 스텝 전 것인가 |

한계:

- **성공률이 아니다.** 시연과 다른 궤적으로도 성공할 수 있고 그 반대도 있다.
- task02·task06 이 이 체크포인트의 학습 데이터에 들어갔는지 기록으로 확인하지 못했다(`train_config.json`에는
  `gist/task1-move-to-tube-rack` 하나만 적혀 있다). 볼트의 파운데이션 기록상 kiroaiseoul 자체 수집 10개가 코퍼스의 20% 라
  들어갔을 가능성이 높다. 그렇다면 여기 수치는 **학습 데이터 안의 평가**다.
- 3090 의 추론 시간은 로봇 PC(RTX 5090 Laptop)와 다르다. 지연(L)은 로봇 로그 값을 반올림해 썼다(10 디노이징 스텝에서
  K≤2 가 3.0, K=4 가 3.6, K=7 이 4.9~5.0 스텝 → K=4 는 4, K=7 은 5).
- 폐루프 근사의 36개 에피소드는 task02 전체 133개에서 고르게 뽑았고, 12개 실행과 일부 겹친다.

## 결과

### 1. 모델이 시연을 얼마나 재현하나

"가만히 있기"는 청크 50칸을 모두 현재 state 로 채운 기준선이다. 이동 비율과 방향은 1.7초(24스텝) 뒤 기준이다.

| 과제 | 모델 | 팔 RMSE | 베이스 RMSE | 이동 비율 | 방향 코사인 |
|---|---|---|---|---|---|
| task02 | 가만히 있기 | 0.197 | 0.007 | | |
| task02 | SmolVLA 80k, K=7 noise 0.9 | 0.112 | 0.005 | 0.58 | 0.89 |
| task06 | 가만히 있기 | 0.100 | 0.163 | | |
| task06 | SmolVLA 80k, K=7 noise 0.9 | 0.065 | 0.076 | 0.35 | 0.71 |
| task06 | ACT `task06_…_260907` (이 데이터로 학습) | 0.021 | 0.029 | 0.98 | 0.98 |

- SmolVLA 는 "가만히 있기"보다 35~45% 나을 뿐이다. 같은 에피소드에서 ACT 는 팔 오차가 1/3 이다. ACT 는 이 데이터셋으로
  학습된 학습 데이터 안의 평가라 공정한 비교는 아니다. 그래도 볼트 기록상 단일 과제 SmolVLA(`tube_disposal_smolvla_kiro2`)도
  자기 학습 데이터에서 채널 std 대비 2~5% 오차를 냈다. **같은 구조가 과제 전용으로 학습되면 이 수준까지 맞출 수 있다는 뜻이고,
  지금 체크포인트의 격차는 추론 설정으로 메울 수 있는 크기가 아니다.**
- 계보 비교(파운데이션 190000 → 30k_aug → kirogist 10k → 80k, 모두 noise 1.0·K=1)에서 task02 팔 오차는 0.124~0.126 으로 같다.
  task06 베이스만 kirogist 단계에서 나아졌다(0.116 → 0.094 → 0.086). 추가 학습이 이 두 과제의 팔 동작을 거의 바꾸지 못했다.
- GIST 판 시연(`*_gist`, 같은 과제를 다른 곳에서 16배 더 모은 것)에서도 "가만히 있기" 대비 개선은 26% 로 비슷하다.
  GIST 장면에 끌려갔다는 가설은 기각이다.
- 정체(시연은 움직이는데 정책이 25% 미만만 움직이는 프레임)는 task02 에서 **에피소드 시작(38%)과 첫 파지 직후(40% 지점, 34%)**에
  몰린다. 노이즈를 0 으로 하면 정체가 19% 에서 35% 로 늘어난다.

### 2. 프롬프트

| 과제 | 넣은 문장 | 팔 RMSE | 방향 코사인 |
|---|---|---|---|
| task02 | 데이터셋 문장 "Pick up the two tubes from the rack with both hands" | 0.123 | 0.83 |
| task02 | GIST 판 문장 "Pick two tubes out of the rack with both hands" | 0.144 | 0.73 |
| task02 | task06 문장 | 0.192 | 0.31 |
| task06 | 데이터셋 문장 (대부분 "Grab and shake the filled beaker, then move to the front of the fridge") | 0.070 | 0.64 |
| task06 | 전부 "Pick up the beaker and move to the refrigerator" | 0.087 | 0.49 |
| task06 | task02 문장 | 0.122 | 0.16 |

정책은 문장에 강하게 반응한다. 다른 과제 문장을 넣으면 "가만히 있기" 수준으로 무너진다. **데이터셋 문장을 한 글자도 바꾸지 말고
쓴다.** 로봇에서 쓰던 두 문장(T1, T4)이 맞다. 09-21 R10~R14 의 깨진 문장("…t fridge")은 이 영향을 받았다.

### 3. 샘플링 (noise_scale, K)

| 설정 | task02 팔 RMSE | task02 이동 비율 | task02 방향 | task06 팔 RMSE | task06 이동 비율 | task06 방향 |
|---|---|---|---|---|---|---|
| noise 1.0, K=1 (기본) | 0.124 | 0.60 | 0.84 | 0.071 | 0.38 | 0.64 |
| noise 0 (결정적) | 0.126 | 0.40 | 0.83 | 0.071 | 0.17 | 0.60 |
| noise 0.5, K=4 | 0.120 | 0.46 | 0.86 | 0.069 | 0.22 | 0.64 |
| **noise 0.9, K=7 (지금)** | 0.112 | 0.58 | 0.89 | 0.065 | 0.35 | 0.71 |
| **noise 1.0, K=4** | 0.113 | 0.61 | 0.88 | 0.065 | 0.39 | 0.71 |
| noise 1.0, K=4, medoid | 0.121 | 0.59 | 0.83 | 0.069 | 0.37 | 0.64 |
| noise 1.2, K=1 | 0.131 | 0.66 | 0.80 | 0.077 | 0.46 | 0.62 |

- **노이즈를 0.9 밑으로 줄이면 움직임이 줄어든다.** 결정적 샘플링은 이동 비율을 1/2~1/3 로 깎는다. 이전 기록 E22 가 RMSE 로
  설정을 골랐는데, RMSE 는 평균 쪽으로 움츠러든 예측을 벌하지 않는다.
- noise 0.9~1.0 에서 K 샘플 평균은 움직임을 거의 깎지 않으면서 방향을 맞춘다. **K=4·noise 1.0 은 K=7·noise 0.9 와 정확도가 같다.**
  로봇 로그(10 디노이징 스텝)에서 제어 루프는 K=7 이 18.5~18.6 Hz(R37, R40), K=4 가 19.6 Hz(R16), K≤2 가 20.5~20.7 Hz(R31~R33)였다.
  같은 정확도라면 루프가 빠른 K=4 가 낫다.
- K 개 중 하나를 고르는 medoid(`--ensemble.reduce=medoid`)는 움직임을 보존하려고 넣어 봤지만 평균보다 나을 게 없었다(방향·jerk 악화).
  기각. 옵션은 남겨 두었다.

### 4. 실행 스케줄 (teacher-forced)

task02, K=7 noise 0.9, 지연 5스텝. 시연 자체의 jerk 는 0.0038, 스텝당 변화 RMS 는 0.011 이다.

| 스케줄 | 팔 RMSE | 최대 점프 (rad/스텝) | jerk | 나이 |
|---|---|---|---|---|
| **지금: commit 50, coeff −1.5** | 0.115 | **0.64** | 0.028 | 26 |
| 연속 TA, coeff 0.01 | 0.096 | 0.09 | 0.010 | 7 |
| 연속 TA, coeff −1.0 | 0.064 | 0.26 | 0.019 | 7 |
| commit 5 + fade 5 | 0.066 | 0.13 | 0.007 | 7 |
| commit 30 + fade 5 | 0.114 | 0.11 | 0.008 | 23 |
| commit 30 (fade 없음) | 0.120 | 0.07 | 0.006 | 23 |

- **commit 50 은 청크가 45스텝(50 − 지연 5)에서 바닥나 크로스페이드 없이 갈아탄다.** 교체 순간 관절 하나가 한 스텝에 최대 0.64 rad
  (≈12 rad/s) 튄다. 로봇에서는 `max_relative_target=0.1` 이 이것을 잘라 줬을 뿐이다(R40 클램프 6회).
- 원래 commit 모드의 크로스페이드는 남은 청크 전체(commit 5 면 약 40스텝)에 걸쳐 새 청크가 거의 반영되지 않았다. 페이드 길이를
  commit 과 분리하는 `--ensemble.fade=F` 를 넣었다.
- teacher-forced 에서는 짧은 commit 이 정확도에서 이긴다. **하지만 이 재생은 state 를 시연에서 가져오므로 폐루프에서 로봇이
  느려지거나 흘러가는 효과를 못 본다.** 다음 절이 그것을 본다.

### 5. 재생 폐루프 근사 (task02)

검증: 완벽한 정책(매칭 프레임의 시연 청크)을 넣으면 모든 스케줄이 100% 완주한다. 시연의 멈춤 구간을 건너뛰도록 프레임을 매칭하므로
완주 시간은 시연보다 짧게 나온다(0.5~0.7배). 시간은 같은 스케줄의 완벽 정책 대비로만 읽는다.

**36개 에피소드, K=4 noise 1.0, 지연 4스텝, `max_relative_target=0.1` 클램프 포함.** 진행도 차이는 같은 에피소드끼리 비교한 값이다.

| 스케줄 | 완주 | 평균 진행도 | 지금 명령 대비 진행도 차이 (±표준오차) | jerk | 최대 점프 |
|---|---|---|---|---|---|
| 지금: commit 50 | 16/36 | 0.67 | | 0.010 | 0.099 (클램프에 걸림) |
| commit 40 + fade 5 | 15/36 | 0.68 | +0.015 ± 0.031 | 0.009 | 0.090 |
| commit 30 + fade 5 | 15/36 | 0.65 | −0.021 ± 0.028 | 0.008 | 0.085 |
| commit 30 + fade 10 | 14/36 | 0.63 | −0.041 ± 0.031 | 0.007 | 0.080 |
| commit 5 + fade 5 | 13/36 | 0.61 | −0.061 ± 0.036 | 0.008 | 0.089 |
| 연속 TA, coeff 0.01 | 13/36 | 0.56 | **−0.109 ± 0.029** | 0.005 | 0.060 |

12개 에피소드, K=7 noise 0.9, 지연 5스텝에서는 commit 30 + fade 5 가 7/12, commit 50 이 6/12, 짧은 commit·연속 TA(coeff −1.0)가 5/12,
**noise 0 이 3/12** 였다. medoid 는 같은 스케줄의 평균과 완주가 같고 jerk 만 컸다.

- **긴 commit(30~50) 끼리는 완주율에 차이가 없다.** 연속 TA(coeff 0.01)는 확실히, 짧은 commit 은 대체로 나쁘다. 6절의 "따라 하기" 때문이다. 자주
  재계획하면 가까운 계획(현재 자세를 이어 가는 부분)만 실행해서 로봇이 느려지고 편향이 쌓인다. teacher-forced 결과(4절)와 반대다.
- 그러니 **지금 명령의 큰 틀(청크를 길게 실행)은 맞다.** 바꿀 것은 교체 순간의 점프다. teacher-forced 에서 교체 점프(최대, rad/스텝)는
  다음과 같다. 로봇에서는 `max_relative_target=0.1` 클램프가 이를 잘라 왔다.

  | 조건 | commit 50 | commit 40 + fade 5 | commit 35 + fade 5 | commit 30 + fade 5 |
  |---|---|---|---|---|
  | task02, K=7 noise 0.9, 지연 5 | 0.64 | 0.25 | 0.11 | 0.11 |
  | task02, K=4 noise 1.0, 지연 4 | 0.75 | 0.16 | 0.13 | 0.13 |
  | task06, K=4 noise 1.0, 지연 4 | 0.54 | 0.15 | 0.12 | 0.10 |
- 모든 스케줄에서 실패하는 에피소드가 있다(12개 중 5개: 시작 직후 출발하지 않거나 첫 파지 뒤 멈춤). 스케줄로는 못 고친다.
- 한계: 로봇이 시연 경로에서 평균 0.6~1.0 rad 벗어나면 이미지와 state 가 어긋난다. 절대 완주율은 믿지 말고 스케줄 간 비교로만 쓴다.

### 6. 정책이 state 에 붙어 있다 (따라 하기)

- state 를 15프레임 과거 것으로 바꿔 넣으면 첫 액션이 그 차이만큼 그대로 따라 움직인다(비율 0.98). 계획 전체의 평행이동 비율은
  9스텝 0.9, 24스텝 0.6~0.75, 49스텝 0.4~0.5 다. **가까운 계획은 현재 자세에서 이어 가고, 먼 계획일수록 시각이 정한 목표로
  수렴한다.** 그래서 너무 자주 재계획하면 로봇이 자기 자세를 따라가기만 해서 느려지고, 아래 편향이 쌓인다.
- 첫 액션에 관절별 편향이 있다. 어깨(joint_1)를 시연보다 낮게 명령한다: task02 왼쪽 −0.014 rad(12개 에피소드 모두 같은 방향),
  오른쪽 −0.009, GIST 판 −0.020 / −0.011, task06 −0.004. ACT 는 +0.001~0.002 로 편향이 없다. 폐루프 근사에서 commit 5 는
  어깨가 이 방향으로 −0.10~−0.13 rad 흘러갔다. 관절별 상수 보정(왼 +0.014, 오른 +0.009, 절반 에피소드로 추정해 나머지에서 확인)을
  넣으면 흘러감은 사라지지만(−0.10 → +0.06, −0.13 → 0.00) **완주율은 그대로였다.** 기제는 맞지만 실패의 주원인은 아니다.
  그래서 로봇 코드에는 넣지 않았다.
- 베이스도 같은 성질이다. 첫 베이스 명령이 현재 베이스 속도를 복사한다(부호 100%, 크기 0.97배). 시연이 정지에서 출발하는 순간
  SmolVLA 는 전진 37%, 회전 60% 만 같이 출발한다(ACT 88%, 78%). state 의 베이스 속도를 0 으로 가려 넣어 봤지만 출발률은 그대로이고
  주행 중 베이스 오차만 3배가 됐다(0.078 → 0.263). 기각.

### 7. 베이스 과회전 (task06)

베이스 속도 명령은 다음 `send_action` 까지 유지되므로, 제어 루프가 녹화(21.5 Hz, 로봇 로그)보다 느리면 스텝마다 더 멀리 돈다.
K=7 의 18.5 Hz 에서 1.16배, K=4 의 19.6 Hz 에서 1.10배, K≤2 의 20.6 Hz 에서 1.04배다(로봇 로그의 루프 주파수로 계산한 추정).
두 가지로 대응한다.

- K 를 7 에서 4 로 줄인다.
- `--ensemble.base_rate_hz=21.5` (새 옵션, 기본 꺼짐): 측정한 루프 주기로 베이스 속도를 `실제 Hz / 21.5` 배로 줄인다. 늘리지는 않는다.
  로봇에서는 아직 안 돌려 봤다.

### 8. RTC (real-time chunking) — 오프라인으로 판정 불가

lerobot 0.4.4 에 SmolVLA 용 RTC 가 들어 있다(`policies/rtc`). 새 청크를 남은 이전 계획에 맞춰 디노이징한다. teacher-forced 재생에서는
가이던스가 약해도 오차가 누적됐다(task02 에피소드 2개, 가중치 1 에서 팔 RMSE 0.083, 10 에서 0.62, 교체만 하면 0.066). 새 청크가 prefix 에서 이어지려 해서
관측보다 자기 계획을 따라가기 때문이다. 이것은 실행 이력을 가진 방식을 teacher-forced 로 재는 한계와 겹쳐서, 좋고 나쁨을 판정하지
못했다. `record_ensemble.py` 에는 연결하지 않았다.

## 코드 변경 (`scripts/record_ensemble.py`)

세 옵션 모두 기본값에서는 기존 동작과 같다.

| 옵션 | 기본값 | 동작 | 상태 |
|---|---|---|---|
| `--ensemble.fade=F` | `0` | commit 모드에서 새 청크로 갈아탈 때 최대 F 스텝에 걸쳐 크로스페이드한다. 0 이면 예전처럼 남은 청크 전체(average) 또는 즉시 교체(latest) | **권장 명령에 쓴다** |
| `--ensemble.reduce=mean\|medoid` | `mean` | K 샘플을 평균하거나, 다른 샘플과 거리 합이 가장 작은 샘플 하나를 고른다 | medoid 는 기각, 옵션만 남김 |
| `--ensemble.base_rate_hz=H` | `0`(끔) | 루프가 녹화 주파수 H 보다 느리면 베이스 속도 명령을 `실제 Hz / H` 배로 줄인다(0.7~1.0). 1초 넘는 공백 뒤에는 측정을 다시 시작한다 | 로봇 미검증, task06 A/B 용 |

- 크로스페이드는 모듈 함수 `_crossfade` 로 빼서 오프라인 시뮬레이터가 같은 코드를 쓴다. `fade=0` 의 결과가 리팩터링 전과 같음을 확인했다.
- 로봇 없이 실제 경로(`install` → 프로세스 워커 → `select_action`)를 task02 에피소드 0 의 150 프레임으로 돌렸다. 권장 설정
  (commit 30, fade 5, K=4)에서 예외·NaN 없이 최대 점프 0.083 rad, 지금 설정(commit 50)은 0.331 rad 였다.
- codex(gpt-5.6-sol) 교차 검토를 두 번 받았고 지적 3건을 모두 수용해 고쳤다.
  1. 베이스 보정의 주기 추정이 에피소드 사이 공백을 넘어 이어져, 리셋 구간의 배율이 다음 에피소드 초반에 쓰였다.
  2. 시뮬레이터가 3프레임 간격 파일을 받으면 시간축이 조용히 틀어졌다. 이제 거부한다.
  3. 1의 수정 뒤, 공백 직후 호출이 마침 300번째면 로그 계산에서 예외가 나 로봇 루프가 멈출 수 있었다. 재현 후 고쳤다.

오프라인 도구는 `scripts/offline_eval/` 에 있다.

| 스크립트 | 하는 일 |
|---|---|
| `predict_chunks.py` | 프레임마다 예측 청크를 저장 (teacher-forced). 프롬프트 덮어쓰기, state 지연, state 차원 가리기 옵션 |
| `simulate.py` | 저장된 청크로 비동기 실행 스케줄 재현 |
| `closedloop_eval.py` | 재생 폐루프 근사. `--perfect` 로 검증, `--offset` 으로 관절 보정 실험 |
| `analyze_openloop.py`, `analyze_motion.py`, `optimal_gain.py`, `summarize.py` | 지표 계산·표 |
| `act_chunks.py` | ACT 체크포인트를 같은 형식으로 (기준선) |
| `rtc_eval.py` | RTC 재생 (판정 불가, 기록용) |
| `fetch_episodes.py` | 버전 태그가 없는 허브 데이터셋에서 필요한 에피소드 파일만 받기 |
| `run_*.sh` | 이번에 돌린 실행 전부 (GPU UUID 지정) |

## 로봇에서 확인할 순서

오프라인 지표는 성공률이 아니다. 아래 순서로 로봇에서 확인한다. 각 런은 **오른쪽 화살표로 저장**하고, 로그는 `tee -i` 로 레포 안
`docs/run_logs/<날짜>_smolvla/` 에 받는다. 끝나면 관찰 한 줄을 로그 끝에 붙인다(기존 절차).

1. task02 에서 지금 명령(commit 50)과 권장 명령(commit 30 + fade 5, K=4)을 같은 시작 자세로 번갈아 5회씩. 완주 여부, 멈춘 지점을 적는다.
2. 권장 명령에서 K=4 와 K=7(noise 0.9)을 비교한다. 로그의 `Control loop rate` 로 루프 주파수도 확인한다.
3. task06 에서 권장 명령에 `--ensemble.base_rate_hz=21.5` 를 넣고 뺀다. 냉장고 앞 도착 위치·방향을 비교한다.
4. 볼 것: 에피소드 시작에서 출발하는가(오프라인에서 가장 흔한 정체 지점), 첫 파지 뒤 들어 올리는가, 어깨가 서서히 내려가는가.

## 학습 쪽 제안 (실행하지 않음)

추론 설정으로 줄일 수 있는 부분은 작다. 가장 큰 격차는 모델이 이 두 과제를 충분히 맞추지 못하는 데서 온다(1절).

- **80k 체크포인트에서 kiro 판 task02·task06 으로 과제 전용 파인튜닝**을 권한다. 단일 과제 SmolVLA 가 자기 데이터에서 잘 맞았다는
  볼트 기록이 근거다. 에피소드 10% 를 held-out 으로 떼어 두고, 이 레포의 `predict_chunks.py`·`closedloop_eval.py` 로 먼저 확인한다.
  목표선은 ACT 수준(task06 팔 RMSE 0.02~0.03, 이동 비율 ~1)이다.
- 정책이 state 에 강하게 붙는 문제(6절)는 학습 때 state 에 잡음을 넣는 증강으로 줄일 수 있다는 가설이 있다. 이번에는 확인하지 않았다.
- 프롬프트는 데이터셋 문장을 그대로 쓴다. GIST 판을 섞으면 task02 문장이 두 가지가 된다.
- SmolVLA 학습 코드는 DGX_1 에만 있고 git 미추적이다(볼트 CLM-20260914-canon-dgx-training-code-1tfo). 학습을 띄우기 전에 `/preflight` 를 거친다.

## 기각·미결

| 항목 | 판정 | 근거 |
|---|---|---|
| GIST 장면에 끌려가서 kiro 에서 못 한다 | 기각 | GIST 판에서도 개선폭이 같다 (1절) |
| K 샘플 평균이 움직임을 줄인다 | 기각 (noise 0.9~1.0 에서) | 이동 비율이 기본과 같다. 줄이는 것은 noise 축소다 (3절) |
| medoid 로 움직임 보존 | 기각 | 평균보다 방향·jerk 가 나쁘다 (3절) |
| 베이스 속도 state 가리기 | 기각 | 출발률 그대로, 주행 오차 3배 (6절) |
| 어깨 편향 보정 | 기제 확인, 효과 없음 | 흘러감은 사라지지만 완주율 그대로 (6절) |
| RTC | 판정 불가 | teacher-forced 로는 잴 수 없다 (8절) |
| task06 폐루프 | 미결 | 베이스 위치를 이미지에 반영할 수 없어 근사를 못 만들었다 |
| task02·task06 이 학습에 들어갔나 | 미결 | 학습 매니페스트가 DGX_1 에만 있다 |
