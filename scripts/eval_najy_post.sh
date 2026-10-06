#!/usr/bin/env bash
# 회차 뒤 정리 — eval_najy.sh 가 끝에 부르고, 손으로도 부른다 (요약이 안 남았을 때, 지난 회차를 다시 볼 때).
#
#   scripts/eval_najy_post.sh <회차>            # 예: 1006_1126_m1_t01_e30
#
# 로봇 무접촉: ~/eval_logs/<회차>.{log,basevel.csv} 와 로컬 HF 캐시의 데이터셋만 읽는다 (내려받지 않음).
# 산출 (~/eval_logs/<회차>.*):
#   .summary.txt  로그에서 뽑은 판정 줄 (원핫·루프 주기·잘린 틱·페이싱·시리얼 재등록)
#   .motion.txt   eval_motion_stats  — 시작 자세 일치·팔 움직임·청크 경계 점프 (학습 데이터셋이 캐시에 있을 때)
#   .latency.txt  eval_latency_stats — 베이스·팔 지연, 추론 스파이크
#   .base.txt     eval_base_stats    — 에피소드별 베이스 회전·전진 (명령/실측) vs 학습 적분
#   .report.md    위 넷 + 로그 + results.csv 를 한 장으로 (eval_najy_report.py)
# 같은 다섯 파일을 docs/run_logs/<YYYY-MM-DD>_eval_najy/ 에도 복사한다 (다른 서버가 git 으로 본다). RUN_LOGS=0 이면 안 한다.
# 성공/실패는 사람이 ~/eval_logs/eval_najy_results.csv 에 적는다 — 이 스크립트는 판정하지 않는다.
set -uo pipefail
cd "$(dirname "$0")/.."

RUN=${1:?회차 이름 (MMDD_HHMM_<m1|m2|m3|base>_tNN_eE)}
LOGDIR=~/eval_logs
LOG=$LOGDIR/$RUN.log
[[ -f "$LOG" ]] || { echo "로그 없음: $LOG" >&2; exit 2; }
[[ "$RUN" =~ ^([0-9]{2})([0-9]{2})_[0-9]{4}_[a-z0-9]+_t([0-9]{2})_e([0-9]+)$ ]] \
  || { echo "회차 이름에서 날짜·단계·exec 를 못 읽음: $RUN" >&2; exit 2; }
MM=${BASH_REMATCH[1]}; DD=${BASH_REMATCH[2]}; STAGE=$((10#${BASH_REMATCH[3]})); EXEC=${BASH_REMATCH[4]}
# 회차 이름엔 연도가 없다 — 오늘보다 뒤의 MMDD 면 작년 회차로 본다 (12/31 회차를 1/1 에 정리하는 경우)
YEAR=$(date +%Y); [[ "$MM$DD" > "$(date +%m%d)" ]] && YEAR=$((YEAR - 1))

TRAIN_DS=( ""
  task01_move_to_tube_rack task02_pickup_tubes task03_turn_to_face_beaker
  task04_pour_liquid_from_tubes_to_beaker task05_tube_disposal task06_pickup_beaker_and_move_to_refrigerator
  task07_open_refrigerator task08_takeout_and_put_beaker task09_close_refrigerator
  task10_move_to_beaker_shelf task11_place_beaker_on_the_shelf )
TRAIN=kiroaiseoul/${TRAIN_DS[$STAGE]}
EVAL=kiroaiseoul/eval_najy_$RUN
CACHE=~/.cache/huggingface/lerobot

# 1) 요약 — 판정에 쓰는 줄만. clamped= 는 loop_rate_log 가 창마다 찍는 arm-tick 수의 합.
{
  echo "== $RUN 종료 코드 ${RC:-?} · 팔 한 틱 상한 ${MAX_REL:-?}"
  grep -h "LEROBOT_TASK_ONEHOT" "$LOG" | head -3
  echo "-- 루프 주기 (phase=policy 마지막 3줄 — mean 을 녹화 주기 20.9 Hz 와 비교)"
  grep -h "Control loop rate" "$LOG" | grep "phase=policy" | tail -3
  echo "-- 팔 페이싱 발동 횟수: $(grep -c "FIRED" "$LOG")"
  # policy 구간만 센다 (리셋 구간의 리더암→정책 전환 점프도 잘리지만 그건 판정 대상이 아니다).
  # phase 끝 줄(Arm relative-target clamps this policy phase: N arm-ticks)이 완전한 값; 그 줄이 없는 로그(중단·옛 포맷)는 창 합으로.
  CLAMP_PHASE=$(awk '/Arm relative-target clamps this policy phase/ {for(i=1;i<=NF;i++) if($i=="phase:") s+=$(i+1)} END{print s+0}' "$LOG")
  CLAMP_WIN=$(awk -F'clamped=' '/Control loop rate/ && /phase=policy/ && NF>1 {split($2,a," "); s+=a[1]} END{print s+0}' "$LOG")
  echo "-- 팔 한 틱 상한에 잘린 arm-tick (policy): $CLAMP_PHASE (phase 끝 줄 합; 창 합 $CLAMP_WIN; 필터 전 경고줄 $(grep -c "had to be clamped" "$LOG"))"
  echo "-- 베이스 시리얼 재등록: $(grep -h "LEROBOT_BASE_SERIAL_REARM" "$LOG" | tail -1)"
  echo "-- 에피소드 시작 자세 (POSE-START = 첫 정책 틱, POSE = 리셋 마지막 줄):"
  awk '/ POSE task/ {pose=$0} /Recording episode/ {print "   " $0; if (pose) print "   " pose; pose=""} / POSE-START / {print "   " $0}' "$LOG"
} > "$LOGDIR/$RUN.summary.txt"
cat "$LOGDIR/$RUN.summary.txt"

# 2) 데이터셋 기반 측정 — 학습 데이터셋이 없으면 학습 대조만 빠진다 (내려받지 않는다: 수 GB)
PY=(uv run --no-sync python)
HAVE_TRAIN=0; [[ -d "$CACHE/$TRAIN" ]] && HAVE_TRAIN=1
if [[ -d "$CACHE/$EVAL" ]]; then
  if (( HAVE_TRAIN )); then
    "${PY[@]}" scripts/eval_motion_stats.py "$TRAIN" "$EVAL" --exec "$EXEC" > "$LOGDIR/$RUN.motion.txt" 2>&1 \
      || echo "!! eval_motion_stats 실패 — $LOGDIR/$RUN.motion.txt 확인" >&2
  else
    echo "학습 데이터셋 $TRAIN 이 로컬 캐시에 없다 — 시작 자세·움직임의 학습 대조 생략 (내려받지 않음)" > "$LOGDIR/$RUN.motion.txt"
  fi
else
  echo "eval 데이터셋 $EVAL 이 없다 (저장 전 중단?) — motion 생략" > "$LOGDIR/$RUN.motion.txt"
fi
TRAIN_OPT=(); (( HAVE_TRAIN )) && TRAIN_OPT=(--train "$TRAIN")
# ${arr[@]+"${arr[@]}"}: bash 4.4 미만에서 set -u 아래 빈 배열 확장이 셸을 죽이는 것을 피한다 (이 머신은 5.2)
"${PY[@]}" scripts/eval_latency_stats.py "$RUN" ${TRAIN_OPT[@]+"${TRAIN_OPT[@]}"} > "$LOGDIR/$RUN.latency.txt" 2>&1 \
  || echo "!! eval_latency_stats 실패 — $LOGDIR/$RUN.latency.txt 확인" >&2
"${PY[@]}" scripts/eval_base_stats.py "$RUN" ${TRAIN_OPT[@]+"${TRAIN_OPT[@]}"} > "$LOGDIR/$RUN.base.txt" 2>&1 \
  || echo "!! eval_base_stats 실패 — $LOGDIR/$RUN.base.txt 확인" >&2

# 3) 한 장 보고서
"${PY[@]}" scripts/eval_najy_report.py "$RUN" > "$LOGDIR/$RUN.report.md" 2>"$LOGDIR/$RUN.report.err" \
  && rm -f "$LOGDIR/$RUN.report.err" || echo "!! eval_najy_report 실패 — $LOGDIR/$RUN.report.err 확인" >&2
echo; cat "$LOGDIR/$RUN.report.md" 2>/dev/null

# 4) 레포로 복사 (git 으로 다른 서버에 간다 — 커밋은 사람이/Claude 가)
if [[ "${RUN_LOGS:-1}" != 0 ]]; then
  DEST="docs/run_logs/${YEAR}-${MM}-${DD}_eval_najy"
  mkdir -p "$DEST"
  for f in summary.txt motion.txt latency.txt base.txt report.md; do
    [[ -f "$LOGDIR/$RUN.$f" ]] && cp "$LOGDIR/$RUN.$f" "$DEST/"
  done
  echo "-- 복사: $DEST/$RUN.{summary,motion,latency,base,report}"
fi
echo "에피소드별 성공/실패는 $LOGDIR/eval_najy_results.csv 에 한 줄씩: run,episode,success(0/1),비고"
