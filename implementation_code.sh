#!/usr/bin/env bash
# SmolVLA eval — temporal ensembling (비동기)
#
# ⚠️ 2026-09-22 이후 권장 실행은 scripts/eval_smolvla.sh 다 (task02/task06, 설정 A/B).
#    아래 연속 TA(coeff 0.01) 설정은 오프라인 폐루프 근사에서 청크를 길게 실행하는 설정보다 못 끝냈다.
#    근거: docs/offline_eval_2026-09-22.md
#
# 매번 바꾸는 3줄: --policy.path (모델) / --dataset.repo_id / --ensemble.coeff
# (bash 는 \ 로 이어진 줄에 주석을 못 달아서 표시를 못 박았다)
#
# 원래 쓰시던 명령에서 달라진 것 여섯:
#   + uv run --extra pi0                  transformers 가 pi0 extra 에만 있다. 빼면 ImportError
#   - --robot.*_max_relative_target=0.1   삭제. 이 fork 의 속도 페이싱이 대체하고
#                                         README §3-3 정식 eval 명령에도 없다
#   ~ --ensemble.every 를 주지 않는다      비동기에서는 1 이 유일하게 맞는 값이다.
#                                         올려도 이미 천장인 제어율을 못 사면서 평활만 버린다
#   ! --robot.include_base_in_state=true  16-dim 체크포인트일 때만. 14-dim 이면 그 줄을 지운다
#                                         (확인법은 맨 아래)
#   + --policy.num_steps=5                 10 에서 내려도 정확도 손실이 없고 추론이 1.57배
#                                         빨라진다 (오프라인 실측, 근거는 「추론 시간 줄이기」
#                                         절). 비행 중 지연 k 도 같은 배수로 준다
#   ~ --policy.path                        파운데이션 lab 런 체크포인트로 교체. 기존 190k+30k
#                                         대비 팔 MAE -18.4%, 청크 끝 오차 -21.5% (실측)
#
# 🚨 베이스 진행 방향을 비우고, 비상정지에 손이 닿는 위치에서.
#    베이스 비상정지 버튼은 돌려 빼 둘 것 — 걸린 채면 connect() 에서 RuntimeError 로 즉사한다.

cd ~/lerobot_trossen

uv run --extra pi0 python scripts/record_ensemble.py \
  --ensemble.coeff=0.01 \
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
  --robot.include_base_in_state=true \
  --teleop.type=mobileai_leader_teleop \
  --teleop.left_arm_ip_address=192.168.1.3 \
  --teleop.right_arm_ip_address=192.168.1.2 \
  --teleop.id=leader \
  --dataset.repo_id=kiroaiseoul/eval_foundation_smolvla_v1 \
  --dataset.num_episodes=2 \
  --dataset.reset_time_s=90 \
  --dataset.single_task="Move to the beaker storage shelf" \
  --dataset.push_to_hub=false \
  --policy.device=cuda \
  --policy.num_steps=5 \
  --policy.path=/home/trossen-ai/daehee/models/foundation_smolvla_lab_35k/pretrained_model \
  2>&1 | tee ~/eval_$(date +%y%m%d_%H%M).log
#  ↑ tee 를 붙여도 → ← ESC 는 살아 있다 (pynput 이 stdin 이 아니라 X 를 훅한다).
#    로그가 필요한 이유는 맨 아래 「끝나고 볼 것」.


# ═══════════════════════════════════════════════════════════════════════════
# 돌리기 전에 (한 번씩)
# ═══════════════════════════════════════════════════════════════════════════
#
# 팔 4개:
#   for ip in 192.168.1.5 192.168.1.4 192.168.1.3 192.168.1.2; do
#     ping -c1 -W1 $ip >/dev/null 2>&1 && echo "OK $ip" || echo "NG $ip"; done
#
# 체크포인트 가져오기 — 학습 머신(kiro-dgx1)에서 최신 lab 체크포인트를 받는다.
# 학습이 5000 스텝마다 저장하므로 회차를 올려 가며 받는다. checkpoints/last 는 최신 회차를
# 가리키는 심볼릭 링크라, 회차를 모르면 먼저 확인한다:
#   ssh kiro-ai@172.16.201.166 readlink /raid/kiro-ai/outputs/foundation_smolvla_lab/checkpoints/last
#   DST=/home/trossen-ai/daehee/models/foundation_smolvla_lab_35k
#   mkdir -p $DST      # 없으면 scp 가 실패한다
#   scp -r kiro-ai@172.16.201.166:/raid/kiro-ai/outputs/foundation_smolvla_lab/checkpoints/035000/pretrained_model \
#          $DST/
#   ⚠️ 회차마다 DST 를 새로 만들 것. 이미 pretrained_model 이 있는 곳에 다시 scp -r 하면
#      덮어쓰지 않고 pretrained_model/pretrained_model 로 한 단계 더 들어간다.
#
# 체크포인트 state 차원 — 16 이면 --robot.include_base_in_state=true 를 두고, 14 면 지운다:
#   POLICY=/home/trossen-ai/daehee/models/foundation_smolvla_lab_35k/pretrained_model
#   python3 -c "import json;print(json.load(open('$POLICY/config.json'))['input_features']['observation.state']['shape'])"
#
# eval repo 이름이 이미 있으면 FileExistsError 로 즉사한다 (덮어쓰지 않는다):
#   ls ~/.cache/huggingface/lerobot/kiroaiseoul/ | grep eval_foundation_smolvla_v1
#   → 있으면 --dataset.repo_id 에 회차를 붙인다 (..._v1_0918_2)
#
# ── --dataset.single_task 는 아래 11개 중 하나를 그대로 써야 한다 ──────────────
# 학습이 28개 데이터셋 전부를 이 문자열에 고정했다. 한 글자라도 다르게 쓰면 언어
# 임베딩이 학습 분포 밖으로 나간다. 새 문장을 지어내지 말 것.
#
#   Move to the tube rack
#   Pick up the two tubes from the rack with both hands
#   Turn in place to face the beaker
#   Pour the liquid from the tubes into the leftmost beaker
#   Drop the two empty tubes into the tube disposal tray
#   Grab and shake the filled beaker, then move to the front of the fridge
#   Open the fridge door while holding the beaker
#   Take out the stored beaker and put the held beaker into the fridge
#   Close the fridge door while holding the beaker
#   Move to the beaker storage shelf
#   Place the beaker on the shelf


# ═══════════════════════════════════════════════════════════════════════════
# 끝나고 볼 것 — base 과회전 배수
# ═══════════════════════════════════════════════════════════════════════════
#
# 이 로봇은 send_action 사이 간격이 곧 base 속도 명령의 적분 구간이라,
# 루프가 느려진 만큼 그대로 과회전한다. 취득 때 로그와 나눠 본다:
#
#   grep "Control loop rate" <취득로그>   | grep phase=policy | tail -1   # 예: mean=21.5 Hz
#   grep "Control loop rate" ~/eval_*.log | grep phase=policy | tail -1   # 예: mean=20.6 Hz
#                                                                        # → 1.04x 면 정상
#
# 그리고 종료 줄의 「비행 중 지연」:
#   추론 39회 · 평균 170.2 ms (최악 ...) · 비행 중 지연 평균 4.1 스텝 (최악 5) · ...
#   → 4 근처면 정상. 훨씬 크거나 「너무 늦어 버린 청크」가 쌓이면 추론 시간(Δ)을 깎는다.
#     Δ 를 줄이면 지연·평균 창·관측 노후화가 한꺼번에 줄어드는 유일한 레버다:
#
#       --policy.num_steps=5     플로우매칭 디노이징 10→5. 가장 확실하고 코드 변경 불필요.
#                                (체크포인트 config 위에 덮어써진다 — 확인함)
#                                정확도 대가가 없다. lab 035000 을 kiro 11개 태스크 620 프레임
#                                에서 시작 노이즈를 고정해 재 본 팔 12축 MAE (2026-09-19):
#                                  ns=1  0.04785  (+8.3%)  ← 너무 거칠다. 쓰지 말 것
#                                  ns=2  0.04265  (-3.5%)
#                                  ns=3  0.04262  (-3.6%)  ← 최소
#                                  ns=5  0.04322  (-2.2%)  ← 여유를 둔 값
#                                  ns=10 0.04420  (기준)
#                                  ns=20 0.04493  (+1.7%)
#                                ⚠️ 낮을수록 좋아 보이는 건 지표 탓도 있다. 적분 단계가 적으면
#                                   예측이 조건부 평균 쪽으로 눌리고 MAE 는 평균을 편든다.
#                                   읽을 수 있는 건 「10→5 가 손해가 아니다」까지고, 「5 가 더
#                                   나은 정책이다」가 아니다. 2~3 까지 내리려면 실기 성공률로
#                                   확인할 것.
#                                속도 이득 (같은 체크포인트/GPU 교차 측정 5라운드 중앙값):
#                                  ns=10  1.00배     ns=3  1.90배
#                                  ns=5   1.57배     ns=2  2.26배
#                                선형 적합 T(ns) = 고정비 + ns x (단계당). 고정비가 ns=10 에서
#                                31% 를 차지한다 -- num_steps 는 액션 전문가 디노이징 루프만
#                                줄이고, SigLIP 3카메라 prefix 인코딩은 한 번 하고 캐시하므로
#                                (use_cache=true) 절대 절반이 되지 않는다. 절대 ms 는 이 GPU
#                                기준이라 로봇 PC 로 안 옮겨가고, 배수만 근사적으로 옮겨간다.
#                                → 종료 줄이 170 ms / 지연 4.1 스텝이었다면 ns=5 에서
#                                  약 108 ms / 2.6 스텝을 기대한다. 실제 값은 종료 줄로 확인할 것.
#       --ensemble.amp=bf16      청크 forward 를 autocast 로. 효과는 재봐야 안다 (아래).
#
#     ⚠️ --policy.use_amp=true 는 여기서 아무 일도 안 한다. lerobot 은 autocast 를 제어
#        스레드의 predict_action 안에서 켜는데 autocast 상태는 thread-local 이라 워커의
#        forward 에 안 닿는다. 그래서 --ensemble.amp 이 따로 있다.
#     ⚠️ --ensemble.amp 은 bf16 을 쓴다. lerobot 의 autocast 는 dtype 을 안 줘서 fp16 이
#        되는데, fp16 오버플로는 「Joint 0 position input contains NaN」으로 나타나고
#        README 는 그걸 normalizer stats 손상으로 설명해 둬서 오진하기 딱 좋다.
#     ⚠️ SmolVLA 는 레이어마다 활성값을 가중치 dtype 으로 되돌리고 attention 을 fp32 로
#        강제한다(smolvlm_with_expert.py:222,307,528). GEMM 은 빨라져도 캐스팅이 늘어
#        순이득이 불확실하다 — 종료 줄의 「평균 ms」를 켜고/끄고 비교할 것.


# ═══════════════════════════════════════════════════════════════════════════
# ACT 태스크 전문가 평가 — temporal ensembling 필수  (2026-09-25)
# ═══════════════════════════════════════════════════════════════════════════
#
# SmolVLA 와 별개로 task01/02/03 전담 ACT 를 구웠다. ACT 는 52M 라 SmolVLA(450M)의 1/9,
# 추론도 훨씬 빨라 매 스텝 재질의해도 30Hz 를 지킨다.
#
# 🔑 **--policy.temporal_ensemble_coeff 를 반드시 켤 것.** 오프라인 롤아웃 실측에서
#    회전 오차가 27% -> 3~5% 로 줄었다 (task01 은 2%). 학습을 건드리지 않는 추론 설정이다.
#
#   cd ~/lerobot_trossen
#   uv run --extra pi0 lerobot-record \
#     --policy.path=/home/trossen-ai/daehee/models/act_task03/pretrained_model \
#     --policy.temporal_ensemble_coeff=0.01 \
#     --policy.n_action_steps=1 \
#     --robot.type=mobileai_robot \
#     ... (로봇/카메라/teleop 인자는 SmolVLA 명령과 동일) \
#     --dataset.single_task="Turn in place to face the beaker"
#
# ⚠️ ACT 는 temporal_ensemble_coeff 를 쓰면 n_action_steps==1 을 강제한다
#    (configuration_act.py:148). 둘을 같이 줘야 한다.
# ⚠️ record_ensemble.py 는 **필요 없다.** 그건 SmolVLA 가 앙상블을 지원하지 않아 만든
#    패치다. ACT 는 lerobot 이 네이티브로 갖고 있으므로 stock lerobot-record 를 쓴다.
#
# ── 왜 앙상블이 필요한가 (실측 근거) ──────────────────────────────────────────
#
# task03 60k 체크포인트, 기록된 에피소드 10개를 끝까지 재생한 결과:
#
#   실행 방식                        팔 MAE    회전 절대오차   총회전 비율
#   청크 30 열고 진행 (기본)          0.198     15.3° (27%)    0.88  (덜 돔)
#   매 스텝 재질의, 앙상블 없음        0.206     24.6° (44%)    1.29  (과회전)
#   매 스텝 재질의 + 앙상블           0.0073     2.7° (5%)     1.01
#
# **재질의만 자주 하면 오히려 나빠진다.** 매 프레임 재질의하면 청크 앞부분만 반복
# 실행되는데 거기가 속도 절정이라 과회전한다. 앙상블이 겹친 청크를 평균해 상쇄한다.
# coeff 는 +0.01 / 0.0 / -0.01 이 소수점까지 같았다 -- 가중 방식이 아니라 **평균한다는
# 사실 자체**가 핵심이므로 논문 기본값 0.01 을 쓰면 된다.
#
# ⚠️ 위 수치는 **기록된 관측**을 재생한 것이다. 실기는 정책 출력이 다음 관측을 바꾸는
#    진짜 폐루프라 오차가 더 클 수 있다. 세 설정을 같은 조건에서 비교했으므로
#    상대 순위는 유효하지만, 절대값을 기대치로 삼지 말 것.
#
# 체크포인트 가져오기:
#   DST=/home/trossen-ai/daehee/models/act_task03
#   mkdir -p $DST
#   scp -r kiro-ai@172.16.201.166:/raid/kiro-ai/outputs/act/task03/checkpoints/060000/pretrained_model $DST/
#
# 배경: trossen-ai-simulation/docs/mobile_base_investigation.md Part 2


# ═══════════════════════════════════════════════════════════════════════════
# 베이스 계측 — scripts/measure_base.py  (2026-09-24)
# ═══════════════════════════════════════════════════════════════════════════
#
# 베이스가 덜 가거나 과회전하는 원인 중, 오프라인 데이터로 답이 안 나온 부분을 잰다.
# 기록된 parquet 의 timestamp 는 lerobot 이 명목 fps 로 만든 합성값이라(dt std=0.00ms)
# 취득 시 실제 제어율을 알 수 없고, 명령 지연도 데이터에 안 남는다.
#
# 팔은 절대 관절각이라 늦어도 목표에 도달할 뿐이지만, 베이스는 속도라
# 「위치 = 속도 x Δt」 다. 제어 주기가 곧 적분 구간이라 느려진 만큼 그대로 과회전한다.
#
#   uv run --extra pi0 python scripts/measure_base.py rate       # 안전 (안 움직임)
#   uv run --extra pi0 python scripts/measure_base.py latency    # 🚨 제자리 회전/전진
#   uv run --extra pi0 python scripts/measure_base.py odom --turns 6   # 🚨 90도 왕복
#   uv run --extra pi0 python scripts/measure_base.py all --out ~/base_meas.json
#
# 베이스만 쓰므로 팔 IP 가 필요 없다. --dry-run 으로 흐름만 먼저 볼 수 있다.
# 🚨 표시된 두 모드는 실제로 움직인다. 진행 방향을 비우고 비상정지에 손이 닿는 곳에서.
#    비상정지가 걸려 있으면 set_cmd_vel 은 성공을 반환하면서 get_vel 이 계속 0 이라
#    「지연 무한대」 라는 무의미한 값이 나온다 -- 스크립트가 먼저 감지해 중단한다.
#
# ── 읽는 법 ─────────────────────────────────────────────────────────────────
#
# latency: t63 = 계단 입력 후 목표 속도의 63% 에 닿는 시간.
#   Mobile ALOHA 의 BASE_DELAY=13 (260ms @50Hz) 에 해당하는 우리 값이다. 그쪽은
#   추론 시 베이스 채널만 13스텝 앞당겨 쏜다(학습 데이터는 안 건드린다):
#     all_actions = torch.cat([all_actions[:, :-13, :-2], all_actions[:, 13:, -2:]], dim=2)
#   우리 값이 나오면 30Hz 기준 몇 스텝인지 스크립트가 같이 찍어 준다.
#   steady_ratio 가 1 에서 멀면 명령 대비 실제 속도에 스케일 오차가 있다는 뜻
#   (Mobile ALOHA 의 angular_vel *= 0.9 가 그 보정이다).
#
# odom: get_pose() 가 보고하는 각도의 일관성. 왕복이 짝수면 누적이 0 에 가까워야 한다.
#   ⚠️ 이건 pose 가 스스로 보고한 숫자지 실제 각도가 아니다. 바닥에 표시하고 눈으로
#      대조해야 오도메트리 자체의 정확도를 안다. 드리프트가 크면 pose 를 관측에 넣어도
#      소용없다 -- 넣을지 말지의 전제 조건이다.
#
# rate: set_cmd_vel/get_vel 왕복 시간. 제어 루프에서 베이스가 차지하는 몫의 하한선.
#
# ── 과회전 배수 계산 ────────────────────────────────────────────────────────
#
#   grep "Control loop rate" <취득로그>   | grep phase=policy | tail -1   # 예: 21.5 Hz
#   grep "Control loop rate" ~/eval_*.log | grep phase=policy | tail -1   # 예: 20.6 Hz
#
# 취득율 / 평가율 = 과회전 배수. 속도 명령이 다음 호출까지 유지되므로 루프가 느린
# 만큼 정비례로 더 돈다. SmolVLA 170ms(5.9Hz)면 21.5/5.9 = 3.6배다.
# 이 배수가 1 에 가까운데도 회전이 틀리면 제어율이 원인이 아니라는 뜻이고,
# 그때는 데이터 쪽(왕복 정렬 태스크, task01 의 회전량 상수성)을 봐야 한다.
# 배경은 trossen-ai-simulation/docs/mobile_base_investigation.md 에 정리해 뒀다.


# ═══════════════════════════════════════════════════════════════════════════
# A/B — 위 명령에서 --ensemble.* 줄만 갈아끼운다 (repo_id 도 겹치지 않게 바꿀 것)
# ═══════════════════════════════════════════════════════════════════════════
#
#   앙상블 없이 (기준선)   : record_ensemble.py 대신 `lerobot-record`, --ensemble.* 전부 삭제
#   동기 앙상블           : --ensemble.async=false --ensemble.every=20
#   정렬만 끄기           : --ensemble.align=false
#   추론 시간 줄이기       : --policy.num_steps=5  /  --ensemble.amp=bf16
#   구형 체크포인트        : --policy.path=.../foundation_smolvla_v1_190k_30k_aug/pretrained_model
#
# 오프라인 체크포인트 비교 (kiro 11개 태스크 620 프레임, 시드 고정, 2026-09-19):
#   190k+30k (구)     팔 12축 MAE 0.05418   오른팔 0.06530   청크 끝 0.0639
#   lab 035000 (신)              0.04420           0.04736           0.0502
#                                 -18.4%            -27.5%            -21.5%
#   → 청크 끝 오차가 줄었다는 건 이음매 떨림 자체가 작아졌다는 뜻이다. 앙상블과 별개로 더해진다.
#   → 단 train set 기준 fitting 오차다. 가중치가 나아졌다는 뜻이지 성공률 보장이 아니다.
#
# 스텁 측정(실기 아님) — 이음매 최대 점프 / 제어율:
#   앙상블 없이   0.328 / 20.1 Hz
#   동기 every=1  0.149 /  5.7 Hz   ← 원래 명령이 이 자리였다
#   비동기 정렬X  0.199 / 20.8 Hz   ← 앙상블 안 한 것보다 나쁘다
#   비동기 정렬O  0.143 / 20.9 Hz   ← 위 명령
