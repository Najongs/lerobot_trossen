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
#   RESET_ONLY=1  정책 단계를 **전개에서 빼고** 리셋만 연달아 돌린다
#                 (`--chain.reset.only=true`). bring-up ②·③ 용. 「리셋이
#                 reached 로 끝난 직후 ESC」 로 대신하지 않는 이유: 사람이
#                 반응하기 전에 정책이 이미 몇 틱을 보낸다
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

# 2) 체크포인트 폭 ↔ YAML 검사 + 단계 파라미터 검증 (로봇에 연결하기 전에 막는다).
#    eval_najy.sh:86-104 의 블록을 체인용으로: action 폭 16(진행도 없음)과
#    17(progress)을 모두 허용하고, 어느 쪽인지 찍는다.
#
#    YAML 은 **파싱**한다 -- grep 하지 않는다. `grep -E '^\s*fps:' | head -1` 은
#    카메라 블록의 `fps: 30` 을 먼저 집어 `dataset.fps: 21` 에 닿지 못한다
#    (이 스크립트의 첫 판이 실제로 그랬고, 그러면 단계 파라미터 로더가 fps
#    불일치로 런을 거부한다 -- `bash -n` 으로는 안 잡힌다). 들여쓰기로 중첩을
#    표현하는 파일에서 키 이름만 보는 것은 같은 이름이 두 블록에 있는 순간 틀린다.
uv run python scripts/_chain_preflight.py "$POLICY" "$YAML" "$PARAMS" > "$LOGDIR/.$RUN.preflight"
cat "$LOGDIR/.$RUN.preflight"

# 3) 원핫 패치가 이 체크아웃에 있는지 (로봇 SDK 를 import 하지 않는다 -- 패치 파일만 읽는다).
#    체인은 한 프로세스에서 단계를 바꾸므로 `set_stage` 가 **있어야** 한다.
#    `_chain_preflight.py` 가 YAML 의 onehot_k 를 파싱해 마지막 줄에 찍어 준다.
P=packages/lerobot_robot_trossen/src/lerobot_robot_trossen
if grep -q "ONEHOT_K=" "$LOGDIR/.$RUN.preflight"; then
  grep -q "def set_stage" "$P/task_onehot_patch.py" || {
    echo "!! 이 체크아웃의 원핫 패치에 set_stage 가 없다 -- 체인은 한 프로세스에서 단계를 바꾼다." >&2
    echo "   git pull --ff-only && uv sync  (Najongs/lerobot_trossen main)" >&2; exit 3; }
  echo "   원핫 패치·set_stage 확인 (실행 로그의 'stage i/K active' 가 단계마다 다시 찍히는지 볼 것)"
fi
grep -q "def set_stage" "$P/pose_guide.py" || echo "!! pose_guide 에 set_stage 가 없다 -- 리셋 중 POSE 줄이 옛 중앙값 표를 쓴다" >&2

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
# 만들고 `TaskOneHotStep.set_stage` 로 단계를 바꾼다.
# 주면 **아무 일도 일어나지 않는다**(패치가 두 번 끼워지는 것이 아니다):
# `apply_task_onehot_patch` 는 `lerobot_record.make_pre_post_processors` 를
# 리바인드하는데(task_onehot_patch.py:240,267), `stage_runner.policies` 는 그
# 이름을 `lerobot.policies.factory` 에서 **직접** import 한다(policies.py:18) --
# 리바인드가 닿지 않는다. 그래서 「원핫이 설치됐다」 는 로그만 남고 체인의
# 전처리기에는 안 들어간다. 결론은 같다: 주지 마라.
# LEROBOT_POSE_GUIDE 도 주지 않는다: 러너가 단계마다 `pose_guide.set_stage` 로
# 지정 리셋 자세를 직접 넘긴다(중앙값 표가 아니라).

if [[ "$RESET_ONLY" == 1 ]]; then
  # 정책 단계를 **전개에서 뺀다**. 「리셋이 reached 로 끝난 직후 ESC」 로 대신하지
  # 않는 이유: 사람이 반응하기 전에 정책이 이미 몇 틱을 보낸다. 러너를 처음 돌리는
  # 회차(bring-up ②)에서 그건 「리셋만」이 아니다.
  ARGS+=("--chain.reset.only=true")
  echo "   RESET_ONLY: 정책 단계 없음 -- 리셋만 ${FROM_STAGE}~${TO_STAGE} 연달아"
  echo "   큰 전이 5곳을 먼저: t02→03 · t03→04 · t04→05 · t07→08 · t10→11"
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
