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
#        BASE_BUDGET_DEG=105 scripts/eval_najy.sh <repo> 1 30 1   # task01 베이스 회전 예산(base_budget.py): 에피소드마다 순회전이
#                                                       # 105° 에 닿으면 같은 방향 theta.vel 만 0 (되돌리기·x.vel·팔 그대로). task01 전용.
#                                                       # 회차 이름에 _bb105 가 붙는다. 비우면(기본) 끔. LEROBOT_BASE_THETA_BUDGET_DEG 가 셸에 남아 있어도 덮어쓴다 —
#                                                       # BASE_BUDGET_DEG 자체는 export 하지 말고 명령 앞에만 붙여라
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
BASE_BUDGET_DEG=${BASE_BUDGET_DEG:-}
[[ -z "$BASE_BUDGET_DEG" ]] || awk -v v="$BASE_BUDGET_DEG" 'BEGIN{exit !(v ~ /^[0-9]*\.?[0-9]+$/ && v+0 > 0)}' \
  || { echo "BASE_BUDGET_DEG 는 양수(도) 또는 빈 값" >&2; exit 2; }
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
# 예산 값(105°)은 task01 시연(순회전 76~89°)에서 나온 숫자다 — 다른 단계에 걸면 정당한 회전을 막는다
[[ -z "$BASE_BUDGET_DEG" || "$STAGE" == 1 ]] || { echo "BASE_BUDGET_DEG 는 task01 전용 (단계 $STAGE)" >&2; exit 2; }
# 재생(시연 명령이 실기에서 얼마나 도나)을 예산이 자르면 재생의 목적이 가려진다
[[ -z "$BASE_BUDGET_DEG" || -z "${LEROBOT_REPLAY_DATASET:-}" ]] || { echo "BASE_BUDGET_DEG 는 재생 회차(LEROBOT_REPLAY_DATASET)와 같이 못 쓴다" >&2; exit 2; }
[[ "$EXEC" =~ ^[0-9]+$ ]] && (( EXEC >= 1 )) || { echo "exec 는 1 이상 (체크포인트 chunk 이하 -- 아래에서 확인)" >&2; exit 2; }

ONEHOT=""
case "$MODEL" in
  M1) REPO=kiroaiseoul/NAJY_act_all11_hot_27D_120k_s1000; ONEHOT="$STAGE/11"; TAG=m1 ;;
  M2) REPO=kiroaiseoul/NAJY_act_all11_hot_27D_120k_s2000; ONEHOT="$STAGE/11"; TAG=m2 ;;
  M3)
    REPO=kiroaiseoul/NAJY_act_move4_hot_20D_60k_s1000; TAG=m3
    case "$STAGE" in 1) ONEHOT=1/4 ;; 3) ONEHOT=2/4 ;; 6) ONEHOT=3/4 ;; 10) ONEHOT=4/4 ;;
      *) echo "M3 는 이동 단계(1·3·6·10)만 학습했다" >&2; exit 2 ;; esac ;;
  TPH) REPO=kiroaiseoul/NAJY_act_all11_tph_27D_120k_s1000; ONEHOT="$STAGE/11"; TAG=tph ;;   # 1라운드 재학습(17D 진행도) — 허브 업로드 뒤 유효
  ENV) REPO=kiroaiseoul/NAJY_act_all11_tph_env_27D_120k_s1000; ONEHOT="$STAGE/11"; TAG=env ;; # 2라운드(env 단계 토큰) — 허브 업로드 뒤 유효
  */*) REPO=$MODEL; TAG=base; [ -n "${HOT:-}" ] && { ONEHOT="$HOT"; TAG=hot; } ;;   # 맨 repo id: HOT=<i>/<K> 를 주면 원핫 패치 켬(ENV 모델도 같은 패치). 태그 base=16D 전문가 · hot=원핫 다단계(10/11 로봇 PC 오표기 지적)
  *) echo "모델은 M1|M2|M3|TPH|ENV|<허브 repo id>" >&2; exit 2 ;;
esac

NN=$(printf "%02d" "$STAGE")
RUN="$(date +%m%d_%H%M)_${TAG}_t${NN}_e${EXEC}"
# RUN_SUFFIX: 회차 이름 끝에 붙인다 (영숫자·_ 만). 시연 재생(LEROBOT_REPLAY_DATASET)은 repo 이름에 'replay' 가
# 있어야 켜지므로 RUN_SUFFIX=replay_ep102 처럼 **사람이 명시**한다 — 자동으로 붙이면 남은 export 가드가 무의미해진다.
if [[ -n "${RUN_SUFFIX:-}" ]]; then
  [[ "$RUN_SUFFIX" =~ ^[A-Za-z0-9_]+$ ]] || { echo "RUN_SUFFIX 는 영숫자·_ 만" >&2; exit 2; }
  RUN="${RUN}_${RUN_SUFFIX}"
fi
[[ -n "$BASE_BUDGET_DEG" ]] && RUN="${RUN}_bb${BASE_BUDGET_DEG//./p}"
LOGDIR=~/eval_logs; mkdir -p "$LOGDIR"

echo "== 모델 $REPO · 단계 task$NN · exec $EXEC · ${EPISODES}ep · 원핫 ${ONEHOT:-없음} · 팔 한 틱 상한 $MAX_REL · 베이스 회전 예산 ${BASE_BUDGET_DEG:-끔}"

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
# state-free 체크포인트(J57, train_multi state_free)는 observation.state 를 선언하지 않는다 → dim 0.
st_ft = (cfg.get("input_features") or {}).get("observation.state")
dim = int(st_ft["shape"][0]) if st_ft else 0
onehot, ex, chunk = sys.argv[2], int(sys.argv[3]), cfg["chunk_size"]
if cfg.get("temporal_ensemble_coeff") is not None:
    sys.exit("체크포인트에 temporal_ensemble_coeff 가 박혀 있다 -- n_action_steps>1 과 같이 못 쓴다")
if ex > chunk:
    sys.exit(f"exec {ex} > chunk {chunk}")
# action 폭: lerobot-record 는 녹화 데이터셋의 action(16D)으로 정책 출력 차원을 덮어쓴다 --
# 17D(진행도 칸) 체크포인트는 가중치 로드에서 size mismatch 로 죽는다(10/13 1050_hot, 로봇 미구동).
# 진행도 모델은 러너(scripts/eval_chain.sh)만 싣는다.
act_dim = cfg["output_features"]["action"]["shape"][0]
if act_dim != 16:
    sys.exit(f"체크포인트 action={act_dim}D -- 이 경로는 16D 만 싣는다. 17D(진행도) 모델은 scripts/eval_chain.sh <repo id> 로")
# 2라운드 체크포인트는 같은 원핫을 ACT 인코더의 별도 토큰으로도 받는다:
# observation.environment_state (type ENV, shape [K]) 가 추가로 선언되고 state 는 그대로 16+K 다.
# 거부하지 않는다 -- 폭만 본다. 패치가 그 키를 전처리 단계에서 채운다.
env_ft = (cfg.get("input_features") or {}).get("observation.environment_state")
env_k = None
if env_ft is not None:
    if env_ft.get("type") != "ENV":
        sys.exit(f"observation.environment_state 가 type={env_ft.get('type')!r} -- ENV 여야 한다")
    env_k = int(env_ft["shape"][0])
if dim == 0:
    # state 없이 단계는 env 토큰으로만 받는다 — 원핫이 꼭 있어야 하고 K 가 맞아야 한다.
    if env_k is None or not onehot or int(onehot.split("/")[1]) != env_k:
        sys.exit(f"state-free 체크포인트(env 토큰 {env_k}D)는 HOT=<단계>/{env_k} 가 필요하다 (지금 원핫 {onehot or '없음'})")
elif onehot:
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
if dim and env_k is not None and env_k != dim - 16:
    sys.exit(f"env 토큰 {env_k}D 인데 state={dim}D 의 원핫 꼬리는 {dim - 16}D 다 -- 같은 벡터여야 한다")
# 학습 때 state 베이스 칸을 무엇으로 채웠나 — 체크포인트의 multi_manifest.json(우리 sim 학습본)에서 읽는다.
# 팀 허브 전문가처럼 매니페스트가 없으면 "" (예전 규칙 그대로). 항목마다 다르면 거부.
from pathlib import Path
ck = Path(sys.argv[1]); base = ""
man = next((d / "multi_manifest.json" for d in (ck, ck.parent) if (d / "multi_manifest.json").exists()), None)
if man is not None:
    entries = json.loads(man.read_text()); entries = entries if isinstance(entries, list) else entries.get("entries", entries)
    modes = {e.get("base_state") for e in entries}
    if modes == {"zero"} or modes == {"vel"}:
        base = modes.pop()
    elif modes - {None}:
        sys.exit(f"multi_manifest.json 의 base_state 가 항목마다 다르다 {sorted(map(str, modes))} -- 로봇에서 한 가지로 만들 수 없다")
# 상대 EEF 체크포인트(J57/J58, action_space rel_eef)는 이 체크아웃의 rel_eef_patch 가 청크를 관절로 바꿔야 한다.
space = "joint"
if man is not None:
    spaces = {e.get("action_space", "joint") for e in entries}
    if len(spaces) != 1:
        sys.exit(f"multi_manifest.json 의 action_space 가 항목마다 다르다 {sorted(spaces)}")
    space = spaces.pop()
if space == "rel_eef":
    init = Path("packages/lerobot_robot_trossen/src/lerobot_robot_trossen/__init__.py")
    if "_install_rel_eef_hook()" not in init.read_text():
        sys.exit("상대 EEF 체크포인트인데 이 체크아웃에 rel_eef_patch 가 없다 -- git pull 하라 (관절로 안 바꾸면 변위가 관절 목표로 나간다)")
    print("   상대 EEF 체크포인트 — 실행 로그에 '[rel_eef] 상대 EEF 체크포인트' 줄이 없으면 중단", file=sys.stderr)
elif space != "joint":
    sys.exit(f"모르는 action_space {space!r}")
print(
    f"   체크포인트 state {dim or '없음(state-free)'}D · chunk {chunk} · 행동 {space}"
    + (f" · env 토큰 {env_k}D (2라운드)" if env_k is not None else "")
    + (f" · 학습 base_state={base}" if base else " · 학습 base_state 기록 없음")
    + " 확인",
    file=sys.stderr,
)
print(f"{dim}\t{base}")
PY
)
BASE_STATE=$(cut -f2 <<<"$STATE_DIM"); STATE_DIM=$(cut -f1 <<<"$STATE_DIM")
# 베이스 칸 규약 (10/14 — J45 vel·J46~48 16D zero 실기 짝):
#  - 원핫 모델: 기본은 베이스 칸 0. 학습이 vel 이면 include_base_in_state=true + 패치가 실측 속도를 칸에 넣는다(LEROBOT_TASK_ONEHOT_BASE=vel)
#  - 원핫 없는 16D: 학습이 zero 면 14D + 0 채움(LEROBOT_TASK_ONEHOT=0/0). vel 이거나 기록이 없으면(팀 16D 전문가) 실측 속도(true) — 예전 규칙
INCLUDE_BASE=false; ONEHOT_BASE=""
if [[ -n "$ONEHOT" && "$BASE_STATE" == vel ]]; then
  INCLUDE_BASE=true; ONEHOT_BASE=vel
  echo "   원핫 + 학습 base_state=vel: --robot.include_base_in_state=true, 패치가 state 14-15 에 실측 속도를 넣는다 (로그 'base slots = measured velocity' 확인)"
elif [[ -z "$ONEHOT" && "$STATE_DIM" == 16 && "$BASE_STATE" == zero ]]; then
  ONEHOT="0/0"
  echo "   16D + 학습 base_state=zero: 로봇 state 14D + 0 채움(LEROBOT_TASK_ONEHOT=0/0) — 실측 속도를 넣지 않는다 (로그 'zero-pad only' 확인)"
elif [[ -z "$ONEHOT" && "$STATE_DIM" == 16 ]]; then
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
[[ -n "$ONEHOT" ]] && ENVS+=(LEROBOT_TASK_ONEHOT_BASE="${ONEHOT_BASE:-zero}")   # 셸에 남은 값이 섞이지 않게 항상 명시
# 리셋 구간에 1초마다 「현재 → 학습 시작 자세」 를 로그로 찍는다 (pose_guide.py, 로봇 동작은 안 바뀜). 끄려면 POSE_GUIDE=0
[[ "${POSE_GUIDE:-1}" != 0 ]] && ENVS+=(LEROBOT_POSE_GUIDE="$STAGE")
# 베이스 회전 예산 — 끔일 때도 빈 값으로 명시해 셸에 남은 export 가 「끔」 회차를 켜지 못하게 한다
ENVS+=(LEROBOT_BASE_THETA_BUDGET_DEG="$BASE_BUDGET_DEG")
# 예산은 record_loop phase 태그(LEROBOT_LOOP_HZ_LOG)로 policy 구간을 안다 — 셸에 남은 0 이 예산을 조용히 끄지 않게
[[ -n "$BASE_BUDGET_DEG" ]] && ENVS+=(LEROBOT_LOOP_HZ_LOG=1)

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
