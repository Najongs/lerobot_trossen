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
#        LOOP_HZ_WINDOW=105 scripts/eval_najy.sh …     # 루프 주기 요약을 105프레임(≈5초)마다. 기본 30 — 바꾸면 eval_latency_stats 의
#                                                       # 추론 스파이크(창마다 최장 틱)가 10/02 기준선과 비교 불가가 된다
#
# 순서와 판정 기준: docs/eval_najy.md. 로그·CSV 는 ~/eval_logs/<회차>.* 에 남는다.
# 끝나면 scripts/eval_najy_post.sh <회차> 가 요약·움직임·지연·베이스 측정과 한 장 보고서(.report.md)를 만든다 —
# 그 단계가 안 돌았으면(중단 등) 손으로 다시 부른다.
# 로그 소음 억제: 상한 경고는 loop_rate_log 가 걸러 clamped=N 으로 센다, 인코더 배너는 SVT_LOG=2.
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
# 2라운드 체크포인트는 같은 원핫을 ACT 인코더의 별도 토큰으로도 받는다:
# observation.environment_state (type ENV, shape [K]) 가 추가로 선언되고 state 는 그대로 16+K 다.
# 거부하지 않는다 -- 폭만 본다. 패치가 그 키를 전처리 단계에서 채운다.
env_ft = (cfg.get("input_features") or {}).get("observation.environment_state")
env_k = None
if env_ft is not None:
    if env_ft.get("type") != "ENV":
        sys.exit(f"observation.environment_state 가 type={env_ft.get('type')!r} -- ENV 여야 한다")
    env_k = int(env_ft["shape"][0])
if onehot:
    k = int(onehot.split("/")[1])
    if dim != 16 + k:
        sys.exit(f"체크포인트 state={dim}D 인데 원핫 {onehot} 은 16+{k}D 를 요구한다")
    if env_k is not None and env_k != k:
        sys.exit(f"env 토큰 {env_k}D 인데 원핫 {onehot} 은 K={k} 다 -- env 토큰은 state 꼬리와 같은 원핫이다")
elif env_k is not None:
    # state=16 이어도 여기서 걸린다: 원핫이 없으면 그 키를 채우는 것이 없고
    # ACT 는 첫 프레임에 읽는다 -- robot.connect() 뒤다.
    sys.exit(
        f"체크포인트가 env 토큰({env_k}D, 2라운드)을 선언했는데 원핫이 없다. "
        "이 스크립트는 M1/M2/M3 분기에서만 ONEHOT 을 세우고 그 셋은 1라운드 repo 에 "
        "고정돼 있다 -- 이 모델용 case 를 추가하라 (REPO=…; ONEHOT=\"$STAGE/11\"). "
        "체인은 scripts/eval_chain.sh <repo id> 로 바로 된다"
    )
elif dim not in (14, 16):
    sys.exit(f"체크포인트 state={dim}D -- 원핫 모델이면 M1/M2/M3 로 불러라")
# 위 K 검사와 별개로, 체크포인트가 스스로 선언한 두 값끼리도 대조한다.
if env_k is not None and env_k != dim - 16:
    sys.exit(f"env 토큰 {env_k}D 인데 state={dim}D 의 원핫 꼬리는 {dim - 16}D 다 -- 같은 벡터여야 한다")
print(
    f"   체크포인트 state {dim}D · chunk {chunk}"
    + (f" · env 토큰 {env_k}D (2라운드)" if env_k is not None else "")
    + " 확인",
    file=sys.stderr,
)
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
ENVS=(LEROBOT_BASE_VEL_LOG="$LOGDIR/$RUN.basevel.csv"
  PYTHONUNBUFFERED=1                          # 로그를 파일로 보내므로 (아래) 버퍼링 없이
  SVT_LOG=2                                   # libsvtav1 인코더의 info 배너(에피소드 저장마다 ~20줄) 끄기 — warning 이상만
  LEROBOT_LOOP_HZ_WINDOW="${LOOP_HZ_WINDOW:-30}")    # 루프 주기 요약 간격(프레임). 30 이 10/02 기준선과 같은 조건 (스파이크 지표)
[[ "${CHUNK_LOG:-0}" == 1 ]] && ENVS+=(LEROBOT_CHUNK_EXECUTION_LOG="$LOGDIR/$RUN.chunks.csv")
[[ -n "$ONEHOT" ]] && ENVS+=(LEROBOT_TASK_ONEHOT="$ONEHOT")
# 리셋 구간에 1초마다 「현재 → 학습 시작 자세」 를 로그로 찍는다 (pose_guide.py, 로봇 동작은 안 바뀜). 끄려면 POSE_GUIDE=0
[[ "${POSE_GUIDE:-1}" != 0 ]] && ENVS+=(LEROBOT_POSE_GUIDE="$STAGE")

if [[ "$DRY_RUN" == 1 ]]; then
  echo "-- DRY_RUN: 실행하지 않는다"; printf '%q ' env "${ENVS[@]}" "${CMD[@]}"; echo
  exit 0
fi

echo "== 로그 $LOGDIR/$RUN.log  (리셋 구간은 리더암, 끝나면 →)"
# 로그는 파일로 직접 쓰고 tail 로 보여 준다. `| tee` 를 쓰면 --display_data 가 띄운 rerun 뷰어가 파이프를
# 물려받아, 창을 닫기 전엔 tee 가 끝나지 않아 아래 정리 단계가 안 돈다 (10/06 C-1). lerobot 은 전경에 둔다 —
# Ctrl-C·ESC 가 그대로 간다. 백그라운드 tail 은 끝난 뒤 죽인다.
: > "$LOGDIR/$RUN.log"
tail -n +1 -f "$LOGDIR/$RUN.log" & TAIL_PID=$!
trap 'kill "$TAIL_PID" 2>/dev/null || true' EXIT     # Ctrl-C 두 번 등으로 아래를 못 지나도 고아 tail 을 남기지 않는다
set +e
env "${ENVS[@]}" "${CMD[@]}" > "$LOGDIR/$RUN.log" 2>&1
rc=$?
set -e
sleep 1; kill "$TAIL_PID" 2>/dev/null || true; wait "$TAIL_PID" 2>/dev/null || true   # tail 이 먼저 죽어 있어도 set -e 에 안 걸리게

# 4) 회차 뒤 정리 — 요약·움직임·지연·베이스 측정·한 장 보고서 (scripts/eval_najy_post.sh, 로봇 무접촉)
RC=$rc MAX_REL=$MAX_REL scripts/eval_najy_post.sh "$RUN" || echo "!! 정리 단계 실패 — 손으로: scripts/eval_najy_post.sh $RUN" >&2
exit "$rc"
