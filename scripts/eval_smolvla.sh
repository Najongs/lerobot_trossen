#!/usr/bin/env bash
# SmolVLA 실로봇 eval — 단계 1~11, 체크포인트·실행 설정을 인자로 (record_ensemble.py 경유).
#
# 사용법 (레포 루트에서):
#   scripts/eval_smolvla.sh <단계 1-11> <variant> [에피소드 수=1]
#     variant : rec   권장 (commit 30 + fade 5, K=4, noise 1.0 — docs/offline_eval_2026-09-22.md)
#               long  commit 40 + fade 5, K=4, noise 1.0 — 청크 50 을 최대한 열린 루프로. 비동기 도착 지연(4~5스텝)을 빼면
#                     commit 40 이 fade 5 를 보장하는 상한이다(codex 10/08: commit + fade + 지연 5 ≤ chunk). 시연의 시작
#                     2 s 기다림(40~56틱)을 한 청크 안에서 지나는지 보는 용도 (sim §94.31)
#               cur   09-22 이전 설정에 가장 가까운 것 (commit 45, fade 0, K=7, noise 0.9) — 비교용. 옛 commit 50 은 도착 지연 때문에
#                     이 스크립트의 불변식(commit + fade + 지연 5 ≤ chunk)을 못 넘어 45 로 낮췄다. 크로스페이드 없이 갈아탄다(점프 큼)
#   환경변수:
#     POLICY        체크포인트 (허브 id 또는 로컬 경로). 기본 kiroaiseoul/NAJY_smolvla_all11_30000
#                   (2호기 11단계 언어 조건, chunk 50, state 16D, base_state zero)
#     INCLUDE_BASE  --robot.include_base_in_state (true|false). 기본은 체크포인트 multi_manifest.json 의 base_state 로 정한다:
#                   zero → false + **state 를 14D→16D 로 0 채움**(LEROBOT_TASK_ONEHOT=0/0, 원핫 없이 베이스 칸만 0) ·
#                   vel → true(실측 속도) · 그 밖의 표현(odom 등)은 거부. 매니페스트가 없으면 직접 준다 —
#                   옛 80k kirogist 는 true 로 돌렸다. false 를 주면 항상 0 채움이 붙는다(16D 모델에 14D 가 들어가면 죽는다)
#     PROMPT        지시문. 기본은 체크포인트 multi_manifest.json 의 그 단계 task 문장 **그대로**. 매니페스트가 있는데
#                   다른 문장을 주면 거부한다(한 글자 차이로 정책이 무너진다, offline_eval 2절). 매니페스트가 없을 때만 자유 입력
#     MAX_REL       팔 한 틱 상한 (기본 0.1, none 이면 해제)   DRY_RUN=1  명령만 출력하고 로봇은 안 건드린다 (0|1 만 받는다)
#   예: DRY_RUN=1 scripts/eval_smolvla.sh 5 rec        # 명령 확인
#       scripts/eval_smolvla.sh 5 rec 1                # task05, 1 에피소드
#       POLICY=/home/trossen-ai/daehee/models/smolvla_190k_30k_aug_80k_kirogist/pretrained_model INCLUDE_BASE=true \
#         PROMPT="Pick up the two tubes from the rack with both hands" scripts/eval_smolvla.sh 2 rec
#
# 끝나면: 저장할 에피소드는 오른쪽 화살표(왼쪽=버리고 재녹화, Ctrl+C=저장 안 함).
#        관찰 한 줄을 로그 끝에 붙인다:  echo "관찰: ..." >> <로그 경로>
# 로그에서 볼 줄:  grep "추론" <로그>  ·  grep "Control loop rate" <로그> | grep phase=policy  ·  grep -c "had to be clamped" <로그>
#                 0 채움이면 "LEROBOT_TASK_ONEHOT=0/0 installed" 와 "zero-pad only" 줄이 있어야 한다
#
# 🚨 베이스 진행 방향을 비우고, 비상정지에 손이 닿는 위치에서. 베이스 비상정지 버튼은 돌려 빼 둘 것 —
#    걸린 채면 connect() 에서 RuntimeError 로 즉사한다. 청크 하나를 commit 스텝만큼 열린 루프로 실행한다(long 은 약 1.9 s).

set -euo pipefail

(( $# <= 3 )) || { echo "인자는 <단계> <variant> [에피소드 수] 셋까지" >&2; exit 2; }
STAGE_RAW="${1:?단계 1-11}"
VARIANT="${2:-rec}"
EPISODES="${3:-1}"
POLICY="${POLICY:-kiroaiseoul/NAJY_smolvla_all11_30000}"
MAX_REL="${MAX_REL:-0.1}"
DRY_RUN="${DRY_RUN:-0}"

[[ "$STAGE_RAW" =~ ^[0-9]+$ ]] || { echo "단계는 1~11 의 정수" >&2; exit 2; }
STAGE=$((10#$STAGE_RAW)); (( STAGE >= 1 && STAGE <= 11 )) || { echo "단계는 1~11" >&2; exit 2; }
NN=$(printf "%02d" "$STAGE")
[[ "$EPISODES" =~ ^[1-9][0-9]*$ ]] || { echo "에피소드 수는 양의 정수" >&2; exit 2; }
[[ "$MAX_REL" == none || ( "$MAX_REL" =~ ^[0-9]*\.?[0-9]+$ && "$MAX_REL" =~ [1-9] ) ]] || { echo "MAX_REL 은 none 또는 0 보다 큰 수" >&2; exit 2; }
[[ "$DRY_RUN" == 0 || "$DRY_RUN" == 1 ]] || { echo "DRY_RUN 은 0 또는 1 (받은 값: $DRY_RUN)" >&2; exit 2; }
[[ -z "${INCLUDE_BASE:-}" || "${INCLUDE_BASE}" == true || "${INCLUDE_BASE}" == false ]] || { echo "INCLUDE_BASE 는 true|false" >&2; exit 2; }

LAG_BUDGET=5   # 비동기 워커의 도착 지연 상한(스텝). record_ensemble 은 도착 시 lag 만큼 청크를 잘라 fade 가 줄어든다
case "$VARIANT" in
  rec)  COMMIT=30; FADE=5; ENS=(--ensemble.samples=4 --ensemble.noise_scale=1.0) ;;
  long) COMMIT=40; FADE=5; ENS=(--ensemble.samples=4 --ensemble.noise_scale=1.0) ;;
  cur)  COMMIT=45; FADE=0; ENS=(--ensemble.samples=7 --ensemble.noise_scale=0.9) ;;
  *) echo "variant 는 rec | long | cur" >&2; exit 2 ;;
esac

cd "$(dirname "$0")/.."

# 체크포인트에서 읽는다: 종류·chunk·폭(state 16·action 16)·그 단계의 지시문·base_state. 지시문은 base64 (개행·탭 보존).
# 학습에 쓴 정본은 체크포인트 옆(또는 pretrained_model 의 상위) multi_manifest.json 이다. 없으면 PROMPT·INCLUDE_BASE 를 요구한다.
META=$(uv run python - "$POLICY" "$STAGE" <<'PY'
import base64, json, re, sys
from pathlib import Path
policy, stage = sys.argv[1], int(sys.argv[2])
p = Path(policy)
root = p if p.exists() else Path(__import__("huggingface_hub").snapshot_download(policy))
cands = [c for c in (root, root / "pretrained_model") if (c / "config.json").exists()]
if not cands:
    sys.exit(f"config.json 이 없다: {root}")
ck = cands[0]
cfg = json.loads((ck / "config.json").read_text())
if cfg.get("type") != "smolvla":
    sys.exit(f"type={cfg.get('type')!r} — 이 스크립트는 SmolVLA 전용 (ACT 는 eval_najy.sh)")
sd = cfg["input_features"]["observation.state"]["shape"][0]
ad = cfg["output_features"]["action"]["shape"][0]
if sd != 16 or ad != 16:
    sys.exit(f"state {sd}D / action {ad}D — 16D/16D 만 받는다")
chunk = cfg.get("chunk_size")
if not isinstance(chunk, int) or chunk <= 0:
    sys.exit(f"chunk_size 가 이상하다: {chunk!r}")
if p.exists() and ck != p:
    sys.exit(f"POLICY 는 config.json 이 있는 폴더를 가리켜야 한다: {ck}")
man = next((d / "multi_manifest.json" for d in (ck, ck.parent) if (d / "multi_manifest.json").exists()), None)
prompt = base = ""
if man is not None:
    entries = json.loads(man.read_text()); entries = entries if isinstance(entries, list) else entries.get("entries", entries)
    hit = [e for e in entries if re.match(rf"^task{stage:02d}(_|$)", e["repo_id"].split("/")[-1])]
    if len(hit) != 1:
        sys.exit(f"매니페스트에서 task{stage:02d} 항목이 {len(hit)}개 — 단계를 특정할 수 없다")
    prompt = hit[0].get("task")
    if not isinstance(prompt, str) or not prompt.strip():
        sys.exit(f"매니페스트 task{stage:02d} 의 task 문장이 없거나 비어 있다 — 학습 지시문을 알 수 없다")
    b = hit[0].get("base_state"); base = "" if b is None else str(b)
    if "\n" in prompt or "\r" in prompt:
        sys.exit("매니페스트 지시문에 개행이 있다 — 데이터셋 문장이 아니다")
print(f"{chunk}\t{base}\t{base64.b64encode(prompt.encode()).decode()}")
PY
)
CHUNK=$(cut -f1 <<<"$META"); BASE_STATE=$(cut -f2 <<<"$META"); MAN_PROMPT=$(cut -f3 <<<"$META" | base64 -d)
[[ "$CHUNK" =~ ^[1-9][0-9]*$ ]] || { echo "chunk 를 못 읽었다: $CHUNK" >&2; exit 2; }

# 지시문: 매니페스트가 있으면 그 문장만 허용한다
if [[ -n "$MAN_PROMPT" ]]; then
  if [[ -n "${PROMPT:-}" && "$PROMPT" != "$MAN_PROMPT" ]]; then
    echo "PROMPT 가 매니페스트 문장과 다르다 — 거부한다 (한 글자 차이로 정책이 무너진다)" >&2
    echo "  준 것:      \"$PROMPT\"" >&2; echo "  매니페스트: \"$MAN_PROMPT\"" >&2; exit 2
  fi
  PROMPT="$MAN_PROMPT"
else
  [[ -n "${PROMPT:-}" ]] || { echo "체크포인트에 multi_manifest.json 이 없다 — PROMPT=\"...\" 를 직접 줘라 (데이터셋 문장 그대로)" >&2; exit 2; }
fi

# 베이스 칸 규약: 학습이 zero 면 로봇은 베이스를 빼고(14D) 0 채움으로 16D 를 만든다. vel 이면 실측 속도(true). 그 밖은 거부.
case "$BASE_STATE" in
  zero) WANT=false ;;
  vel)  WANT=true ;;
  "")   WANT="" ;;
  *) echo "학습 base_state=$BASE_STATE 는 이 스크립트가 로봇에서 만들 수 없다 (zero|vel 만)" >&2; exit 2 ;;
esac
if [[ -z "${INCLUDE_BASE:-}" ]]; then
  [[ -n "$WANT" ]] || { echo "체크포인트에 base_state 정보가 없다 — INCLUDE_BASE=true|false 를 직접 줘라 (false 면 0 채움이 붙는다)" >&2; exit 2; }
  INCLUDE_BASE="$WANT"
elif [[ -n "$WANT" && "$INCLUDE_BASE" != "$WANT" ]]; then
  echo "학습 base_state=$BASE_STATE 인데 INCLUDE_BASE=$INCLUDE_BASE — state 베이스 칸이 학습과 다르게 들어간다. 거부한다" >&2; exit 2
fi
ZERO_PAD=""
[[ "$INCLUDE_BASE" == false ]] && ZERO_PAD="LEROBOT_TASK_ONEHOT=0/0"   # 14D → [arms, 0, 0] (원핫 없음), 정규화 전

(( COMMIT + FADE + LAG_BUDGET <= CHUNK )) || { echo "commit $COMMIT + fade $FADE + 지연 $LAG_BUDGET > chunk $CHUNK — fade 가 보장되지 않는다" >&2; exit 2; }
ENS+=(--ensemble.commit=$COMMIT); (( FADE > 0 )) && ENS+=(--ensemble.fade=$FADE)

RUN="$(date +%m%d_%H%M%S)_sv_t${NN}_${VARIANT}_$$"
LOGDIR="$HOME/eval_logs"; mkdir -p "$LOGDIR"
LOG="$LOGDIR/$RUN.log"; DSET="kiroaiseoul/eval_smolvla_${RUN}"
for f in "$LOG" "$LOGDIR/$RUN.basevel.csv" "$HOME/.cache/huggingface/lerobot/$DSET"; do [[ -e "$f" ]] && { echo "이미 있다: $f" >&2; exit 2; }; done

CMD=(uv run python scripts/record_ensemble.py
  "${ENS[@]}"
  --ensemble.coeff=-1.5
  --ensemble.amp=bf16
  --policy.num_steps=10
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
  "--dataset.repo_id=$DSET"
  "--dataset.single_task=$PROMPT"
  "--dataset.num_episodes=$EPISODES"
  --dataset.episode_time_s=120
  --dataset.reset_time_s=90
  --dataset.fps=21
  --dataset.push_to_hub=false
  --policy.device=cuda
  "--policy.path=$POLICY")
[[ "$MAX_REL" != none ]] && CMD+=("--robot.left_arm_max_relative_target=$MAX_REL" "--robot.right_arm_max_relative_target=$MAX_REL")

ENVS=(LEROBOT_BASE_VEL_LOG="$LOGDIR/$RUN.basevel.csv" PYTHONUNBUFFERED=1 SVT_LOG=2
  LEROBOT_LOOP_HZ_WINDOW="${LOOP_HZ_WINDOW:-30}" LEROBOT_POSE_GUIDE="$STAGE")
[[ -n "$ZERO_PAD" ]] && ENVS+=("$ZERO_PAD")   # 빈 배열 확장을 피한다 (bash 4.3 이하 + set -u, codex 10/08)

echo "== SmolVLA $POLICY · 단계 task$NN · $VARIANT (${ENS[*]}) · ${EPISODES}ep · include_base=$INCLUDE_BASE${ZERO_PAD:+ + 0 채움(14D→16D)} · 팔 한 틱 상한 $MAX_REL · chunk $CHUNK"
echo "   지시문: \"$PROMPT\"   (매니페스트: \"${MAN_PROMPT:-없음}\" · base_state ${BASE_STATE:-없음})"
echo "   로그: $LOG"
if [[ "$DRY_RUN" == 1 ]]; then
  echo "-- DRY_RUN: 실행하지 않는다"; printf '%q ' env "${ENVS[@]}" "${CMD[@]}"; echo; exit 0
fi
{ echo "설정: policy=$POLICY stage=$STAGE variant=$VARIANT include_base=$INCLUDE_BASE zero_pad=${ZERO_PAD:-no} prompt=\"$PROMPT\" ${ENS[*]}"; } > "$LOG"
env "${ENVS[@]}" "${CMD[@]}" 2>&1 | tee -ia "$LOG"
#  ↑ tee -i: Ctrl+C 에도 마지막 "추론 N회 …" 요약 줄이 잘리지 않는다.
