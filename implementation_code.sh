#!/usr/bin/env bash
# SmolVLA eval — temporal ensembling (비동기)
#
# 매번 바꾸는 3줄: --policy.path (모델) / --dataset.repo_id / --ensemble.coeff
# (bash 는 \ 로 이어진 줄에 주석을 못 달아서 표시를 못 박았다)
#
# 원래 쓰시던 명령에서 달라진 것 넷:
#   + uv run --extra pi0                  transformers 가 pi0 extra 에만 있다. 빼면 ImportError
#   - --robot.*_max_relative_target=0.1   삭제. 이 fork 의 속도 페이싱이 대체하고
#                                         README §3-3 정식 eval 명령에도 없다
#   ~ --ensemble.every 를 주지 않는다      비동기에서는 1 이 유일하게 맞는 값이다.
#                                         올려도 이미 천장인 제어율을 못 사면서 평활만 버린다
#   ! --robot.include_base_in_state=true  16-dim 체크포인트일 때만. 14-dim 이면 그 줄을 지운다
#                                         (확인법은 맨 아래)
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
  --policy.path=/home/trossen-ai/daehee/models/foundation_smolvla_v1_190k_30k_aug/pretrained_model \
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
# 체크포인트 state 차원 — 16 이면 --robot.include_base_in_state=true 를 두고, 14 면 지운다:
#   POLICY=/home/trossen-ai/daehee/models/foundation_smolvla_v1_190k_30k_aug/pretrained_model
#   python3 -c "import json;print(json.load(open('$POLICY/config.json'))['input_features']['observation.state']['shape'])"
#
# eval repo 이름이 이미 있으면 FileExistsError 로 즉사한다 (덮어쓰지 않는다):
#   ls ~/.cache/huggingface/lerobot/kiroaiseoul/ | grep eval_foundation_smolvla_v1
#   → 있으면 --dataset.repo_id 에 회차를 붙인다 (..._v1_0918_2)


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
# A/B — 위 명령에서 --ensemble.* 줄만 갈아끼운다 (repo_id 도 겹치지 않게 바꿀 것)
# ═══════════════════════════════════════════════════════════════════════════
#
#   앙상블 없이 (기준선)   : record_ensemble.py 대신 `lerobot-record`, --ensemble.* 전부 삭제
#   동기 앙상블           : --ensemble.async=false --ensemble.every=20
#   정렬만 끄기           : --ensemble.align=false
#   추론 시간 줄이기       : --policy.num_steps=5  /  --ensemble.amp=bf16
#
# 스텁 측정(실기 아님) — 이음매 최대 점프 / 제어율:
#   앙상블 없이   0.328 / 20.1 Hz
#   동기 every=1  0.149 /  5.7 Hz   ← 원래 명령이 이 자리였다
#   비동기 정렬X  0.199 / 20.8 Hz   ← 앙상블 안 한 것보다 나쁘다
#   비동기 정렬O  0.143 / 20.9 Hz   ← 위 명령
