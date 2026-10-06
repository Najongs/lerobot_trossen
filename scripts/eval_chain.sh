#!/usr/bin/env bash
# 11단계 자율 체인 한 회차 — 모델만 주면 stage_runner 명령을 조립한다.
#
#   scripts/eval_chain.sh <M1|TPH|<허브 repo id>> [회차태그]
#
#   M1   = kiroaiseoul/NAJY_act_all11_hot_27D_120k_s1000  (16D, 진행도 없음 — 참조군)
#   TPH  = kiroaiseoul/NAJY_act_all11_tph_27D_120k_s1000  (17D, 진행도 칸 — 본선)
#   그 밖의 문자열은 허브 repo id 로 보고 TPH 쪽 YAML 로 돌린다 (폭은 체크포인트에서 읽는다)
#
#   예)  DRY_RUN=1 scripts/eval_chain.sh M1            # 명령만 출력, 로봇 안 건드림
#        scripts/eval_chain.sh M1 bringup4            # 실행
#        TO_STAGE=3 scripts/eval_chain.sh TPH s1_3     # 1~3 단계만 (bring-up ⑤ 의 첫 회차)
#        RESET_ONLY=1 scripts/eval_chain.sh M1 resets  # 리셋만: 정책을 한 단계도 돌리지 않는다 (bring-up ②·③)
#
# 환경변수
#   DRY_RUN=1     조립한 명령만 찍고 끝낸다. 체크포인트 다운로드·폭 검사는 **한다**
#                 (eval_najy.sh DRY_RUN 과 같은 범위: 로봇 무접촉, HF 다운로드는 함)
#   FROM_STAGE/TO_STAGE   기본 1 / 11
#   RESET_ONLY=1  정책 단계를 전부 빼고 리셋만 연달아 돌린다. 내부적으로
#                 `--chain.completion.timeout_factor` 를 건드리지 않고 러너의
#                 단계 전개에서 정책을 뺄 방법이 없으므로, 경계마다 사람이
#                 `→` 로 넘기는 대신 **단계 범위를 하나씩** 돌리는 것을 쓴다 —
#                 아래에서 안내만 하고 자동으로 돌리지는 않는다
#   MANUAL=0      `→` 수동 완료를 끈다 (기본 켬: 사용자 결정 10/06, 두 모델 공통)
#   LOOP_HZ_WINDOW  루프 주기 요약 간격(프레임). 기본 30 — 바꾸면 10/02 기준선과 비교 불가
#
# ⚠️ 이 스크립트는 **로봇을 움직인다.** DGX_1 에서 실행하지 마라 (로봇 패키지를
#    import 하고 카메라·팔 포트를 연다). bring-up 순서는 docs/eval_najy.md 「단계 체이닝」.
#
# 로그·CSV 는 ~/eval_logs/<회차>.* 에, 이벤트·설정 스냅샷은
# outputs/stage_runner/<run_id>/ 에 남는다. 끝나면 scripts/eval_chain_report.py 가
# 단계별 표를 만든다.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${1:?모델 (M1|TPH|<허브 repo id>)}
TAG=${2:-}
DRY_RUN=${DRY_RUN:-0}
FROM_STAGE=${FROM_STAGE:-1}
TO_STAGE=${TO_STAGE:-11}
MANUAL=${MANUAL:-1}
RESET_ONLY=${RESET_ONLY:-0}

[[ "$FROM_STAGE" =~ ^([1-9]|1[01])$ ]] || { echo "FROM_STAGE 는 1~11" >&2; exit 2; }
[[ "$TO_STAGE" =~ ^([1-9]|1[01])$ ]] || { echo "TO_STAGE 는 1~11" >&2; exit 2; }
(( FROM_STAGE <= TO_STAGE )) || { echo "FROM_STAGE <= TO_STAGE" >&2; exit 2; }

case "$MODEL" in
  M1)  REPO=kiroaiseoul/NAJY_act_all11_hot_27D_120k_s1000; YAML=configs/chain/chain_m1_all11.yaml;  SHORT=m1 ;;
  TPH) REPO=kiroaiseoul/NAJY_act_all11_tph_27D_120k_s1000; YAML=configs/chain/chain_tph_all11.yaml; SHORT=tph ;;
  */*) REPO=$MODEL; YAML=configs/chain/chain_tph_all11.yaml; SHORT=base ;;
  *)   echo "모델은 M1|TPH|<허브 repo id>" >&2; exit 2 ;;
esac
[[ -f "$YAML" ]] || { echo "!! $YAML 이 없다" >&2; exit 2; }

PARAMS=configs/chain/stage_params.json
if [[ ! -f "$PARAMS" ]]; then
  echo "!! $PARAMS 이 없다 — sim 레포 scripts/export_chain_params.py 가 만들어 여기로 복사한다." >&2
  echo "   스키마·거부 조건은 configs/chain/README.md." >&2
  exit 3
fi

RUN="$(date +%m%d_%H%M)_chain_${TAG:-$SHORT}_e30"
LOGDIR=~/eval_logs; mkdir -p "$LOGDIR"

echo "== 체인 $REPO · 단계 ${FROM_STAGE}~${TO_STAGE} · 회차 $RUN"
echo "   설정 $YAML · 단계 파라미터 $PARAMS"

# 1) 체크포인트 받기 (루트형/pretrained_model 중첩형 모두) — eval_najy.sh:76-82 와 같은 방식
POLICY=$(uv run python -c "
from huggingface_hub import snapshot_download
from pathlib import Path
d = Path(snapshot_download('$REPO'))
print(d if (d/'config.json').exists() else d/'pretrained_model')
" | tail -1)
echo "   POLICY=$POLICY"

# 2) 체크포인트 폭 ↔ 원핫·exec·앙상블 검사 (로봇에 연결하기 전에 막는다).
#    eval_najy.sh:86-104 의 블록을 체인용으로: action 폭 16(진행도 없음)과
#    17(progress)을 모두 허용하고, 어느 쪽인지 찍는다.
uv run python - "$POLICY" "$YAML" <<'PY'
import json, sys, re
cfg = json.load(open(f"{sys.argv[1]}/config.json"))
state = cfg["input_features"]["observation.state"]["shape"][0]
action = cfg["output_features"]["action"]["shape"][0]
chunk = cfg["chunk_size"]
yaml_text = open(sys.argv[2], encoding="utf-8").read()

def key(name, default=None):
    m = re.search(rf"^\s*{name}:\s*(\S+)\s*$", yaml_text, re.M)
    return m.group(1) if m else default

if cfg.get("temporal_ensemble_coeff") is not None:
    sys.exit("체크포인트에 temporal_ensemble_coeff 가 박혀 있다 -- n_action_steps>1 과 같이 못 쓴다")
exec_steps = int(key("n_action_steps", "30"))
if exec_steps > chunk:
    sys.exit(f"n_action_steps {exec_steps} > chunk {chunk}")
k = key("onehot_k", "null")
if k != "null":
    if state != 16 + int(k):
        sys.exit(f"체크포인트 state={state}D 인데 원핫 K={k} 는 16+{k}D 를 요구한다")
elif state not in (14, 16):
    sys.exit(f"체크포인트 state={state}D -- 원핫 모델이면 YAML 의 chain.model.onehot_k 를 채워라")
has_progress = key("has_progress", "null")
expected = {16: "false", 17: "true"}.get(action)
if expected is None:
    sys.exit(f"체크포인트 action={action}D -- 체인은 16(진행도 없음)과 17(progress)만 안다")
if has_progress not in ("null", expected):
    sys.exit(f"YAML 의 has_progress={has_progress} 인데 체크포인트 action={action}D 는 {expected} 다")
print(f"   체크포인트 state {state}D · action {action}D "
      f"({'progress 있음' if action == 17 else '진행도 없음'}) · chunk {chunk} · exec {exec_steps} 확인",
      file=sys.stderr)
PY

# 3) 원핫 패치가 이 체크아웃에 있는지 (로봇 SDK 를 import 하지 않는다 -- 패치 파일만 읽는다).
#    체인은 한 프로세스에서 단계를 바꾸므로 `set_stage` 가 **있어야** 한다.
P=packages/lerobot_robot_trossen/src/lerobot_robot_trossen
if grep -q '^\s*onehot_k: [0-9]' "$YAML"; then
  grep -q "def set_stage" "$P/task_onehot_patch.py" || {
    echo "!! 이 체크아웃의 원핫 패치에 set_stage 가 없다 -- 체인은 한 프로세스에서 단계를 바꾼다." >&2
    echo "   git pull --ff-only && uv sync  (Najongs/lerobot_trossen main)" >&2; exit 3; }
  echo "   원핫 패치·set_stage 확인 (실행 로그의 'stage i/K active' 가 단계마다 다시 찍히는지 볼 것)"
fi
grep -q "def set_stage" "$P/pose_guide.py" || echo "!! pose_guide 에 set_stage 가 없다 — 리셋 중 POSE 줄이 옛 중앙값 표를 쓴다" >&2

# 4) 단계 파라미터 검증 (로봇 SDK 를 import 하지 않는다 -- chain_params 는 stdlib only)
FPS=$(grep -E '^\s*fps:' "$YAML" | head -1 | awk '{print $2}')
uv run python -c "
import sys
sys.path.insert(0, 'packages/stage_runner/src')
from stage_runner.chain_params import load_chain_params
p = load_chain_params('$PARAMS', expected_fps=$FPS)
print(f'   단계 파라미터 OK: {len(p.stages)}단계 · fps {p.fps} · sim {p.source.get(\"sim_commit\", \"?\")[:12]}')
"

ARGS=(uv run python -m stage_runner
  --config_path "$YAML"
  "--chain.model.policy_path=$POLICY"
  "--chain.params_path=$PARAMS"
  "--chain.from_stage=$FROM_STAGE"
  "--chain.to_stage=$TO_STAGE"
  "--dataset.repo_id=kiroaiseoul/eval_${RUN}"
  "--output.run_id=$RUN")
[[ "$MANUAL" == 1 ]] || ARGS+=("--chain.completion.allow_manual_complete=false")

# env: eval_najy.sh:155-162 와 같은 묶음. 베이스 명령·실측 CSV 는 순수 기록이라
# 항상 켠다 -- 체인 보고서가 단계별 ∫θ·∫x 를 이 CSV 에서 뽑는다(t_mono 로 범위를
# 자른다). 청크 실행 로그는 실행 경로를 바꾸므로 기본 off.
ENVS=(LEROBOT_BASE_VEL_LOG="$LOGDIR/$RUN.basevel.csv"
  PYTHONUNBUFFERED=1
  SVT_LOG=2
  LEROBOT_LOOP_HZ_WINDOW="${LOOP_HZ_WINDOW:-30}")
[[ "${CHUNK_LOG:-0}" == 1 ]] && ENVS+=(LEROBOT_CHUNK_EXECUTION_LOG="$LOGDIR/$RUN.chunks.csv")
# LEROBOT_TASK_ONEHOT 은 **주지 않는다.** 그 변수는 `lerobot-record` 진입점을
# 패치해 한 프로세스 = 한 단계를 가정한다. 체인은 stage_runner 가 번들을 한 번
# 만들고 `TaskOneHotStep.set_stage` 로 단계를 바꾼다 -- 둘을 같이 쓰면 패치가
# 두 번 끼워진다.
# LEROBOT_POSE_GUIDE 도 주지 않는다: 러너가 단계마다 `pose_guide.set_stage` 로
# 지정 리셋 자세를 직접 넘긴다(중앙값 표가 아니라).

if [[ "$RESET_ONLY" == 1 ]]; then
  cat >&2 <<'MSG'
-- RESET_ONLY: 러너에는 「리셋만」 모드가 없다. 정책 단계를 빼면 체인이 아니라
   자세 이동 시험이고, 그건 단계 범위를 하나씩 돌려서 한다:

     for k in 2 3 4 5 6 7 8 9 10 11; do
       FROM_STAGE=$k TO_STAGE=$k scripts/eval_chain.sh <모델> reset_$k
     done

   각 회차는 「k 의 시작 자세로 리셋 → k 정책 1단계」 다. 리셋만 보려면 리셋이
   `reached` 로 끝난 직후 ESC 를 눌러라 — 정책 단계는 0 프레임으로 끝나고
   체인은 실패로 기록된다(의도된 결과다: 사람이 중단했다).
   큰 전이 5곳을 먼저: t02→03 · t03→04 · t04→05 · t07→08 · t10→11.
MSG
  exit 0
fi

if [[ "$DRY_RUN" == 1 ]]; then
  echo "-- DRY_RUN: 실행하지 않는다"; printf '%q ' env "${ENVS[@]}" "${ARGS[@]}"; echo
  exit 0
fi

echo "== 로그 $LOGDIR/$RUN.log"
echo "   사람이 할 것: ESC = 중단(e-stop 과 **함께**) · → = 이 단계 완료, 다음으로"
echo "   ⚠️ 베이스 e-stop 은 팔(별도 이더넷)을 멈추지 않는다 [추정] — ESC 와 e-stop 을 함께."
# `| tee` 금지: --display_data 가 띄운 rerun 뷰어가 파이프를 물려받아 창을 닫기
# 전엔 tee 가 끝나지 않는다 (10/06 C-1). 파일로 쓰고 tail 로 보여 준다.
: > "$LOGDIR/$RUN.log"
tail -n +1 -f "$LOGDIR/$RUN.log" & TAIL_PID=$!
trap 'kill "$TAIL_PID" 2>/dev/null || true' EXIT
set +e
env "${ENVS[@]}" "${ARGS[@]}" > "$LOGDIR/$RUN.log" 2>&1
rc=$?
set -e
sleep 1; kill "$TAIL_PID" 2>/dev/null || true; wait "$TAIL_PID" 2>/dev/null || true

# exit code: 0 정상 · 1 사람 중단 · 2 preflight · 3 베이스 정지 실패 · 4 체인 실패
case "$rc" in
  0) echo "== 체인 완주 (모든 단계가 요구 종료값으로 끝났다)" ;;
  1) echo "== 사람이 중단했다 (ESC 또는 ←)" ;;
  2) echo "== preflight 가 거부했다 — 위 ERROR 한 줄이 고칠 키를 말한다" >&2 ;;
  3) echo "!! 베이스를 멈추지 못한 경계가 있다 — 로봇을 확인하라" >&2 ;;
  4) echo "== 체인이 끊겼다 (타임아웃 또는 리셋 미도달). 에피소드는 저장됐다." ;;
  *) echo "== 종료 코드 $rc" >&2 ;;
esac

# 회차 뒤 정리 — 단계별 표 (로봇 무접촉)
RUN_DIR="outputs/stage_runner/$RUN"
if [[ -f "$RUN_DIR/events.jsonl" ]]; then
  uv run python scripts/eval_chain_report.py "$RUN_DIR" \
    --basevel "$LOGDIR/$RUN.basevel.csv" \
    --out "$LOGDIR/$RUN.chain.md" \
    || echo "!! 보고서 생성 실패 — 손으로: uv run python scripts/eval_chain_report.py $RUN_DIR" >&2
else
  echo "!! $RUN_DIR/events.jsonl 이 없다 — 러너가 run 디렉터리를 만들기 전에 끝났다" >&2
fi
exit "$rc"
