#!/usr/bin/env bash
# 2단계: GIST 판 시연 평가 + 프롬프트 민감도 + state 의존도 (80k 체크포인트, 3프레임 간격)
set -u
cd "$(dirname "$0")/../.."
PY=/home/najo/NAS/lerobot_trossen/.venv/bin/python
OUT=docs/run_logs/2026-09-22_offline_eval/chunks
LOG=docs/run_logs/2026-09-22_offline_eval/logs
GA=GPU-05b804ff-3b02-39f4-cf62-b848e189ebdd
M80=kiroaiseoul/smolvla_190k_30k_aug_80k_kirogist
T02=kiroaiseoul/task02_pickup_tubes
T06=kiroaiseoul/task06_pickup_beaker_and_move_to_refrigerator
L=/home/najo/.cache/huggingface/lerobot/kiroaiseoul
C2="stock:10:1.0:1,det:10:0.0:1"
run() { local gpu=$1 tag=$2; shift 2; CUDA_VISIBLE_DEVICES=$gpu $PY scripts/offline_eval/predict_chunks.py --out $OUT --tag $tag "$@" > $LOG/$tag.log 2>&1; echo "EXIT $tag $?" >> $LOG/_status.txt; }
run $GA m80_t02gist --policy $M80 --dataset kiroaiseoul/task02_pickup_tubes_gist --root $L/task02_pickup_tubes_gist --n-episodes 12 --stride 3 --configs $C2
run $GA m80_t06gist --policy $M80 --dataset kiroaiseoul/task06_pickup_beaker_and_move_to_refrigerator_gist --root $L/task06_pickup_beaker_and_move_to_refrigerator_gist --n-episodes 12 --stride 3 --configs $C2
run $GA m80_t02_pGIST --policy $M80 --dataset $T02 --n-episodes 12 --stride 3 --configs $C2 --task "Pick two tubes out of the rack with both hands"
run $GA m80_t02_pWRONG --policy $M80 --dataset $T02 --n-episodes 12 --stride 3 --configs $C2 --task "Grab and shake the filled beaker, then move to the front of the fridge"
run $GA m80_t06_pWRONG --policy $M80 --dataset $T06 --n-episodes 12 --stride 3 --configs $C2 --task "Pick up the two tubes from the rack with both hands"
run $GA m80_t06_pPICK --policy $M80 --dataset $T06 --n-episodes 12 --stride 3 --configs $C2 --task "Pick up the beaker and move to the refrigerator"
run $GA m80_t02_shift15 --policy $M80 --dataset $T02 --n-episodes 12 --stride 3 --configs $C2 --state-shift 15
run $GA m80_t06_shift15 --policy $M80 --dataset $T06 --n-episodes 12 --stride 3 --configs $C2 --state-shift 15
echo "PHASE2 ALLDONE" >> $LOG/_status.txt
