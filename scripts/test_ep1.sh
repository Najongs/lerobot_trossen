#!/usr/bin/env bash
# 새 로봇 점검 — 리더암으로 한 에피소드를 찍고(record), 그 에피소드를 팔로워로 다시 돌린다(replay).
#
#   scripts/test_ep1.sh record [길이 s=30]    # kiroaiseoul/TEST_EP1 에 1 에피소드 (허브 업로드 안 함)
#   scripts/test_ep1.sh replay [에피소드=0]   # 🚨 기록 궤적대로 팔·베이스 실제 구동
#   scripts/test_ep1.sh compare               # 가장 최근 record·replay 의 베이스 이동·루프 속도 비교 (로봇 무접촉)
#
#   예)  TROSSEN_PC=pc2 DRY_RUN=1 scripts/test_ep1.sh record   # 명령만 출력
#        TROSSEN_PC=pc2 scripts/test_ep1.sh record 20
#        TROSSEN_PC=pc2 scripts/test_ep1.sh replay
#        REPO=kiroaiseoul/TEST_EP2 scripts/test_ep1.sh record    # 다른 이름으로
#        TIMING=fixed scripts/test_ep1.sh replay                 # 예전처럼 표기 fps(21)로 고정 재생
#        HEADING_KP=0 scripts/test_ep1.sh replay                 # 방향 보정 끔 (open-loop)
#        HEADING_MIN_RATE=0 scripts/test_ep1.sh replay           # 정지 중 최소 회전 속도 끔
#
# 녹화는 베이스 토크 끔(손으로 밀고 실측 속도가 action 으로 남는다), 재생은 토크 켬(그 속도를 명령으로 보낸다).
# 재생은 명령만 따라간다(open-loop) — 녹화를 시작한 자리·자세에 맞춰 놓고 돌려야 같은 궤적이 나온다.
# 재생은 녹화 때의 실제 틱 시각을 따른다(test_ep1_replay.py) — 리더암 녹화는 표기 21 fps 보다 느린
# ≈19.5 Hz 로 돌아서, 21 fps 로 고정 재생하면 7~8 % 빨라진다(10/26 PC2). 시각은 가장 최근 record 로그에서 읽는다.
# 같은 로그의 실측 방향을 목표로 회전 명령을 보정한다(HEADING_KP, 기본 1.5 /s — test_ep1_replay.py 방향 보정).
# 베이스는 정지에서 작은 회전 명령을 무시하므로 정지 중 보정은 최소 회전 속도로 올린다
# (HEADING_MIN_RATE °/s, 기본 4 — 10/26 PC2 measure_base.py deadband: ≤2°/s 0 · 3°/s 문턱 · 4~5°/s ≈2°/s 양방향 일정).
# 같은 이름이 이미 있으면 lerobot-record 가 거부한다 — 지우거나 REPO 를 바꾼다.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/robot_hw.sh

MODE=${1:?record | replay}
REPO=${REPO:-kiroaiseoul/TEST_EP1}
FPS=21
DRY_RUN=${DRY_RUN:-0}
DATA_DIR="${HF_LEROBOT_HOME:-$HOME/.cache/huggingface/lerobot}/$REPO"
# 베이스 명령·실측 속도 CSV (base_vel_log.py — 메모리에 모았다가 끝날 때 한 번 쓴다, 동작 무영향)
LOGDIR=~/eval_logs/test_ep1; mkdir -p "$LOGDIR"
LOG_PREFIX="$LOGDIR/${REPO//\//_}"

case "$MODE" in
  record)
    LEN=${2:-30}
    [[ "$LEN" =~ ^[0-9]+$ && "$LEN" -gt 0 ]] || { echo "길이는 양의 정수(s)" >&2; exit 2; }
    if [[ -e "$DATA_DIR" && "$DRY_RUN" != 1 ]]; then
      echo "!! $DATA_DIR 가 이미 있다 — 지우고 다시 찍으려면: rm -rf '$DATA_DIR'  (또는 REPO=… 로 다른 이름)" >&2
      exit 3
    fi
    CMD=(uv run lerobot-record
      --robot.type=mobileai_robot
      "--robot.left_arm_ip_address=$ROBOT_LEFT_IP"
      "--robot.right_arm_ip_address=$ROBOT_RIGHT_IP"
      --robot.id=follower
      "--robot.cameras=$ROBOT_CAMERAS"
      --robot.enable_base_motor_torque=false   # 취득은 베이스를 손으로 민다(README §1-2) — 켜면 바퀴가 버텨 브레이크처럼 된다
      --teleop.type=mobileai_leader_teleop
      "--teleop.left_arm_ip_address=$LEADER_LEFT_IP"
      "--teleop.right_arm_ip_address=$LEADER_RIGHT_IP"
      --teleop.id=leader
      --display_data=true
      "--dataset.repo_id=$REPO"
      "--dataset.single_task=new robot test"
      --dataset.num_episodes=1
      "--dataset.episode_time_s=$LEN"
      --dataset.reset_time_s=10
      "--dataset.fps=$FPS"
      --dataset.push_to_hub=false)
    BASE_LOG="${LOG_PREFIX}_record_$(date +%m%d_%H%M%S).csv"
    echo "== $TROSSEN_PC · 녹화 $REPO · ${LEN}s · ${FPS} fps → $DATA_DIR"
    ;;
  replay)
    EP=${2:-0}
    [[ "$EP" =~ ^[0-9]+$ ]] || { echo "에피소드는 0 이상 정수" >&2; exit 2; }
    [[ -d "$DATA_DIR" || "$DRY_RUN" == 1 ]] || { echo "!! $DATA_DIR 없음 — 먼저 record" >&2; exit 3; }
    TIMING=${TIMING:-recorded}
    TIMING_ARG=()
    case "$TIMING" in
      recorded)
        TIMING_CSV=$(ls -1 "${LOG_PREFIX}"_record_*.csv 2>/dev/null | sort | tail -n 1 || true)
        [[ -n "$TIMING_CSV" ]] || { echo "!! 녹화 로그(${LOG_PREFIX}_record_*.csv) 없음 — 다시 record 하거나 TIMING=fixed" >&2; exit 3; }
        HEADING_KP=${HEADING_KP:-1.5}
        HEADING_MIN_RATE=${HEADING_MIN_RATE:-4}
        TIMING_ARG=("--timing_csv=$TIMING_CSV" "--heading_kp=$HEADING_KP" "--heading_min_rate_deg_s=$HEADING_MIN_RATE") ;;
      fixed)
        [[ "${HEADING_KP:-0}" == 0 ]] || { echo "방향 보정은 녹화 로그가 필요하다 — TIMING=fixed 와 같이 못 쓴다" >&2; exit 2; }
        HEADING_KP=0 ;;
      *) echo "TIMING 은 recorded 또는 fixed" >&2; exit 2 ;;
    esac
    CMD=(uv run python scripts/test_ep1_replay.py
      --robot.type=mobileai_robot
      "--robot.left_arm_ip_address=$ROBOT_LEFT_IP"
      "--robot.right_arm_ip_address=$ROBOT_RIGHT_IP"
      --robot.id=follower
      --robot.enable_base_motor_torque=true
      "--dataset.repo_id=$REPO"
      "--dataset.episode=$EP"
      "${TIMING_ARG[@]}")
    BASE_LOG="${LOG_PREFIX}_replay_$(date +%m%d_%H%M%S).csv"
    echo "== $TROSSEN_PC · 재생 $REPO 에피소드 $EP · 시각 ${TIMING_CSV:-고정 ${FPS} fps} · 방향 보정 Kp $HEADING_KP · 정지 중 최소 ${HEADING_MIN_RATE:-0}°/s  🚨 팔·베이스가 움직인다 — 녹화 시작 자리에 두었나?"
    ;;
  compare)
    exec uv run python scripts/test_ep1_compare.py "$LOG_PREFIX" ;;
  *) echo "record · replay · compare" >&2; exit 2 ;;
esac

if [[ "$DRY_RUN" == 1 ]]; then
  echo "-- DRY_RUN: 실행하지 않는다"; printf '%q ' env LEROBOT_BASE_VEL_LOG="$BASE_LOG" "${CMD[@]}"; echo
  exit 0
fi
check_arms
echo "   베이스 로그 $BASE_LOG"
exec env LEROBOT_BASE_VEL_LOG="$BASE_LOG" "${CMD[@]}"
