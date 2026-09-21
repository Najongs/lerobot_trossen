#!/usr/bin/env bash
# SmolVLA 실로봇 eval — task02 / task06, 설정 A/B 용
#
# 사용법 (레포 루트에서):
#   bash scripts/eval_smolvla.sh <task> <variant>
#     task    : t02 | t06
#     variant : rec      권장 (commit 30, fade 5, K=4, noise 1.0)
#               cur      09-22 까지 쓰던 설정 (commit 50, K=7, noise 0.9) — 비교용
#               rec_base 권장 + 베이스 과회전 보정 (--ensemble.base_rate_hz=21.5) — task06 A/B 용
#   예: bash scripts/eval_smolvla.sh t02 rec
#
# 근거: docs/offline_eval_2026-09-22.md
# 끝나면: 저장할 에피소드는 오른쪽 화살표(왼쪽=버리고 재녹화, Ctrl+C=저장 안 함).
#        관찰 한 줄을 로그 끝에 붙인다:  echo "관찰: ..." >> <로그 경로>
#
# 🚨 베이스 진행 방향을 비우고, 비상정지에 손이 닿는 위치에서.
#    베이스 비상정지 버튼은 돌려 빼 둘 것 — 걸린 채면 connect() 에서 RuntimeError 로 즉사한다.

set -euo pipefail

TASK="${1:-}"
VARIANT="${2:-rec}"

POLICY=/home/trossen-ai/daehee/models/smolvla_190k_30k_aug_80k_kirogist/pretrained_model
DATASET=kiroaiseoul/eval_foundation_smolvla_v1

case "$TASK" in
  # 문장은 데이터셋 문장 그대로. 한 글자라도 바꾸면 정책이 무너진다 (오프라인 평가 2절)
  t02) PROMPT="Pick up the two tubes from the rack with both hands" ;;
  t06) PROMPT="Grab and shake the filled beaker, then move to the front of the fridge" ;;
  *) echo "task 는 t02 | t06" >&2; exit 2 ;;
esac

case "$VARIANT" in
  rec)      ENS=(--ensemble.commit=30 --ensemble.fade=5 --ensemble.samples=4 --ensemble.noise_scale=1.0) ;;
  cur)      ENS=(--ensemble.commit=50 --ensemble.samples=7 --ensemble.noise_scale=0.9) ;;
  rec_base) ENS=(--ensemble.commit=30 --ensemble.fade=5 --ensemble.samples=4 --ensemble.noise_scale=1.0
                 --ensemble.base_rate_hz=21.5) ;;
  *) echo "variant 는 rec | cur | rec_base" >&2; exit 2 ;;
esac

cd "$(dirname "$0")/.."
LOGDIR="docs/run_logs/$(date +%F)_smolvla"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/${TASK}_${VARIANT}_$(date +%H%M%S).log"

# eval 데이터셋은 로컬에만 두는 일회용이다 (push_to_hub=false). 이전 런 것을 지우고 시작한다.
rm -rf ~/.cache/huggingface/lerobot/"$DATASET"

echo "로그: $LOG"
echo "설정: task=$TASK variant=$VARIANT ${ENS[*]}" | tee "$LOG"

uv run python scripts/record_ensemble.py \
  "${ENS[@]}" \
  --ensemble.coeff=-1.5 \
  --ensemble.amp=bf16 \
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
  --dataset.repo_id="$DATASET" \
  --dataset.num_episodes=1 \
  --dataset.reset_time_s=90 \
  --dataset.single_task="$PROMPT" \
  --dataset.push_to_hub=false \
  --policy.device=cuda \
  --policy.path="$POLICY" \
  2>&1 | tee -ia "$LOG"
#  ↑ tee -i: Ctrl+C 에도 마지막 "추론 N회 …" 요약 줄이 잘리지 않는다.

# 로그에서 볼 줄:
#   grep "추론" "$LOG"                                       # 추론 시간, 비행 중 지연
#   grep "Control loop rate" "$LOG" | grep phase=policy      # 제어 루프 주파수 (K=4 면 ~19.6 Hz 기대)
#   grep "base rate compensation" "$LOG"                     # rec_base 일 때 베이스 배율
#   grep -c "had to be clamped" "$LOG"                       # 클램프 (리셋 구간 것은 정책과 무관)
