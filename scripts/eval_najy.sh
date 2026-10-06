#!/usr/bin/env bash
# 다단계 ACT 실기 Eval 한 회차 — 모델·단계·exec 만 주면 README §3-3 명령을 조립한다.
#
#   scripts/eval_najy.sh <모델> <단계 1-11> <exec 30|5> [에피소드 수=5]
#
#   모델  M1 = kiroaiseoul/NAJY_act_all11_hot_27D_120k_s1000  (11단계, 이동 우세)
#         M2 = kiroaiseoul/NAJY_act_all11_hot_27D_120k_s2000  (11단계, 조작 우세)
#         M3 = kiroaiseoul/NAJY_act_move4_hot_20D_60k_s1000   (이동 4단계 전용: 단계 1·3·6·10 만)
#         그 밖의 문자열은 허브 repo id 로 보고 원핫 없이 돌린다 (기존 단계별 전문가 = 기준선)
#
#   예)  DRY_RUN=1 scripts/eval_najy.sh M1 4 30 3     # 명령만 출력, 로봇 안 건드림
#        scripts/eval_najy.sh M1 4 30 3               # 실행
#        MAX_REL=none scripts/eval_najy.sh M1 4 30 3  # 팔 한 틱 이동 제한을 끈다 (10/06 이전 회차와 같은 조건)
#        POSE_GUIDE=0 scripts/eval_najy.sh M1 4 30 3  # 리셋 중 시작 자세 안내 로그(POSE …)를 끈다
#
#   MAX_REL  팔 관절 한 틱 이동 상한(rad, 기본 0.1). 정책 action 이 현재 자세에서 이만큼 넘게 떨어지면
#            그 틱엔 0.1 만 간다 -- 청크 경계·시작 순간의 큰 점프로 팔이 다치는 것을 막는다(10/06 부터).
#            21 Hz 에서 관절당 최대 약 2.1 rad/s. 그리퍼(m)에도 같은 값이 걸리지만 행정이 작아 영향 없다.
#
# 순서와 판정 기준: docs/eval_najy.md. 로그·CSV 는 ~/eval_logs/<회차>.* 에 남는다.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${1:?모델 (M1|M2|M3|<허브 repo id>)}
STAGE=${2:?단계 1-11}
EXEC=${3:?exec (30 또는 5)}
EPISODES=${4:-5}
DRY_RUN=${DRY_RUN:-0}
MAX_REL=${MAX_REL:-0.1}
[[ "$MAX_REL" == none ]] || awk -v v="$MAX_REL" 'BEGIN{exit !(v ~ /^[0-9]*\.?[0-9]+$/ && v+0 > 0)}' \
  || { echo "MAX_REL 은 양수(rad) 또는 none" >&2; exit 2; }

TASKS=(
  ""
  "Move to the tube rack"
  "Pick up the two tubes from the rack with both hands"
  "Turn in place to face the beaker"
  "Pour the liquid from the tubes into the leftmost beaker"
  "Drop the two empty tubes into the tube disposal tray"
  "Pick up the beaker and move to the refrigerator"
  "Open the fridge door while holding the beaker"
  "Take out the stored beaker and put the held beaker into the fridge"
  "Close the fridge door while holding the beaker"
  "Move to the beaker storage shelf"
  "Place the beaker on the shelf"
)

[[ "$STAGE" =~ ^([1-9]|1[01])$ ]] || { echo "단계는 1~11" >&2; exit 2; }
[[ "$EXEC" =~ ^[0-9]+$ ]] && (( EXEC >= 1 )) || { echo "exec 는 1 이상 (체크포인트 chunk 이하 -- 아래에서 확인)" >&2; exit 2; }

ONEHOT=""
case "$MODEL" in
  M1) REPO=kiroaiseoul/NAJY_act_all11_hot_27D_120k_s1000; ONEHOT="$STAGE/11"; TAG=m1 ;;
  M2) REPO=kiroaiseoul/NAJY_act_all11_hot_27D_120k_s2000; ONEHOT="$STAGE/11"; TAG=m2 ;;
  M3)
    REPO=kiroaiseoul/NAJY_act_move4_hot_20D_60k_s1000; TAG=m3
    case "$STAGE" in 1) ONEHOT=1/4 ;; 3) ONEHOT=2/4 ;; 6) ONEHOT=3/4 ;; 10) ONEHOT=4/4 ;;
      *) echo "M3 는 이동 단계(1·3·6·10)만 학습했다" >&2; exit 2 ;; esac ;;
  */*) REPO=$MODEL; TAG=base ;;
  *) echo "모델은 M1|M2|M3|<허브 repo id>" >&2; exit 2 ;;
esac

NN=$(printf "%02d" "$STAGE")
RUN="$(date +%m%d_%H%M)_${TAG}_t${NN}_e${EXEC}"
LOGDIR=~/eval_logs; mkdir -p "$LOGDIR"

echo "== 모델 $REPO · 단계 task$NN · exec $EXEC · ${EPISODES}ep · 원핫 ${ONEHOT:-없음} · 팔 한 틱 상한 $MAX_REL"

# 1) 체크포인트 받기 (루트형/pretrained_model 중첩형 모두)
POLICY=$(uv run python -c "
from huggingface_hub import snapshot_download
from pathlib import Path
d = Path(snapshot_download('$REPO'))
print(d if (d/'config.json').exists() else d/'pretrained_model')
" | tail -1)
echo "   POLICY=$POLICY"

# 2) 체크포인트 폭 ↔ 원핫 대조, exec ≤ chunk (로봇에 연결하기 전에 막는다)
#    원핫 없는 16D 체크포인트(기존 전문가 일부)는 로봇이 베이스 실측 속도를 state 에 넣어야 한다.
STATE_DIM=$(uv run python - "$POLICY" "$ONEHOT" "$EXEC" <<'PY'
import json, sys
cfg = json.load(open(f"{sys.argv[1]}/config.json"))
dim = cfg["input_features"]["observation.state"]["shape"][0]
onehot, ex, chunk = sys.argv[2], int(sys.argv[3]), cfg["chunk_size"]
if cfg.get("temporal_ensemble_coeff") is not None:
    sys.exit("체크포인트에 temporal_ensemble_coeff 가 박혀 있다 -- n_action_steps>1 과 같이 못 쓴다")
if ex > chunk:
    sys.exit(f"exec {ex} > chunk {chunk}")
if onehot:
    k = int(onehot.split("/")[1])
    if dim != 16 + k:
        sys.exit(f"체크포인트 state={dim}D 인데 원핫 {onehot} 은 16+{k}D 를 요구한다")
elif dim not in (14, 16):
    sys.exit(f"체크포인트 state={dim}D -- 원핫 모델이면 M1/M2/M3 로 불러라")
print(f"   체크포인트 state {dim}D · chunk {chunk} 확인", file=sys.stderr)
print(dim)
PY
)
INCLUDE_BASE=false
if [[ -z "$ONEHOT" && "$STATE_DIM" == 16 ]]; then
  INCLUDE_BASE=true
  echo "   16D 기준선: --robot.include_base_in_state=true (state 베이스 칸 = 실측 속도, 학습 때와 같은 조건)"
fi

# 3) 원핫 패치가 이 체크아웃에 있는지 (로봇 SDK 를 import 하지 않는다 -- 패치 파일만 읽는다)
if [[ -n "$ONEHOT" ]]; then
  P=packages/lerobot_robot_trossen/src/lerobot_robot_trossen
  if ! grep -q "^apply_task_onehot_patch()" "$P/__init__.py" 2>/dev/null; then
    echo "!! 이 체크아웃에 원핫 패치가 없다 -- git pull --ff-only && uv sync  (Najongs/lerobot_trossen main)" >&2; exit 3
  fi
  uv run python - "$P/task_onehot_patch.py" "$ONEHOT" <<'PY'
import importlib.util, sys
spec = importlib.util.spec_from_file_location("tp", sys.argv[1]); tp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tp); tp._parse(sys.argv[2])
PY
  echo "   원핫 패치 확인 (실행 로그의 'installed' 줄로 다시 확인할 것)"
fi

CMD=(uv run lerobot-record
  --robot.type=mobileai_robot
  --robot.left_arm_ip_address=192.168.1.5
  --robot.right_arm_ip_address=192.168.1.4
  --robot.id=follower
  "--robot.cameras={
    cam_high: {type: intelrealsense, serial_number_or_name: \"230422273501\", width: 640, height: 480, fps: 30},
    cam_left_wrist: {type: intelrealsense, serial_number_or_name: \"230422271234\", width: 640, height: 480, fps: 30},
    cam_right_wrist: {type: intelrealsense, serial_number_or_name: \"230322274369\", width: 640, height: 480, fps: 30}
}"
  --robot.enable_base_motor_torque=true
  "--robot.include_base_in_state=$INCLUDE_BASE"
  --teleop.type=mobileai_leader_teleop
  --teleop.left_arm_ip_address=192.168.1.3
  --teleop.right_arm_ip_address=192.168.1.2
  --teleop.id=leader
  --display_data=true
  "--dataset.repo_id=kiroaiseoul/eval_najy_${RUN}"
  "--dataset.single_task=${TASKS[$STAGE]}"
  "--policy.path=$POLICY"
  "--policy.n_action_steps=$EXEC"
  --dataset.episode_time_s=120
  --dataset.reset_time_s=90
  "--dataset.num_episodes=$EPISODES"
  --dataset.fps=21
  --dataset.push_to_hub=false)
[[ "$MAX_REL" != none ]] && CMD+=("--robot.left_arm_max_relative_target=$MAX_REL" "--robot.right_arm_max_relative_target=$MAX_REL")

# 베이스 명령·실측 CSV 는 순수 기록이라 항상 켠다. 청크 실행 로그는 청크 실행기 자체를 설치해
# 실행 경로가 바뀌므로 기본은 끈다 -- 필요하면 CHUNK_LOG=1.
ENVS=(LEROBOT_BASE_VEL_LOG="$LOGDIR/$RUN.basevel.csv")
[[ "${CHUNK_LOG:-0}" == 1 ]] && ENVS+=(LEROBOT_CHUNK_EXECUTION_LOG="$LOGDIR/$RUN.chunks.csv")
[[ -n "$ONEHOT" ]] && ENVS+=(LEROBOT_TASK_ONEHOT="$ONEHOT")
# 리셋 구간에 1초마다 「현재 → 학습 시작 자세」 를 로그로 찍는다 (pose_guide.py, 로봇 동작은 안 바뀜). 끄려면 POSE_GUIDE=0
[[ "${POSE_GUIDE:-1}" != 0 ]] && ENVS+=(LEROBOT_POSE_GUIDE="$STAGE")

if [[ "$DRY_RUN" == 1 ]]; then
  echo "-- DRY_RUN: 실행하지 않는다"; printf '%q ' env "${ENVS[@]}" "${CMD[@]}"; echo
  exit 0
fi

echo "== 로그 $LOGDIR/$RUN.log  (리셋 구간은 리더암, 끝나면 →)"
set +e
env "${ENVS[@]}" "${CMD[@]}" 2>&1 | tee "$LOGDIR/$RUN.log"
rc=${PIPESTATUS[0]}
set -e

# 4) 회차 요약 -- 판정에 쓰는 줄만
{
  echo "== $RUN 종료 코드 $rc · 팔 한 틱 상한 $MAX_REL"
  grep -h "LEROBOT_TASK_ONEHOT" "$LOGDIR/$RUN.log" | head -3 || true
  echo "-- 루프 주기 (phase=policy 마지막 3줄 -- mean·min 을 녹화 주기와 비교)"
  grep -h "Control loop rate" "$LOGDIR/$RUN.log" | grep "phase=policy" | tail -3 || true
  echo "-- 팔 페이싱 발동 횟수: $(grep -c "FIRED" "$LOGDIR/$RUN.log" || true)"
  echo "-- 팔 한 틱 상한에 잘린 틱: $(grep -c "had to be clamped" "$LOGDIR/$RUN.log" || true)"
  echo "-- 베이스 시리얼 재등록: $(grep -h "LEROBOT_BASE_SERIAL_REARM" "$LOGDIR/$RUN.log" | tail -1)"
} | tee "$LOGDIR/$RUN.summary.txt"
echo "에피소드별 성공/실패는 $LOGDIR/eval_najy_results.csv 에 손으로 한 줄씩: run,episode,success(0/1),비고"
exit "$rc"
