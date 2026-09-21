#!/usr/bin/env bash
# 4단계: 베이스 속도 state 를 가리면 출발을 더 잘 하는가 (task06, 80k, 3프레임 간격)
set -u
cd "$(dirname "$0")/../.."
PY=/home/najo/NAS/lerobot_trossen/.venv/bin/python
OUT=docs/run_logs/2026-09-22_offline_eval/chunks
LOG=docs/run_logs/2026-09-22_offline_eval/logs
GA=GPU-05b804ff-3b02-39f4-cf62-b848e189ebdd
until grep -q "PHASE2 ALLDONE" $LOG/_status.txt; do sleep 20; done
run() { local gpu=$1 tag=$2; shift 2; CUDA_VISIBLE_DEVICES=$gpu $PY scripts/offline_eval/predict_chunks.py --out $OUT --tag $tag "$@" > $LOG/$tag.log 2>&1; echo "EXIT $tag $?" >> $LOG/_status.txt; }
run $GA m80_t06_zbase --policy kiroaiseoul/smolvla_190k_30k_aug_80k_kirogist --dataset kiroaiseoul/task06_pickup_beaker_and_move_to_refrigerator --n-episodes 12 --stride 3 --configs stock:10:1.0:1,det:10:0.0:1 --zero-state-dims 14,15
run $GA m80_t06_s3ref --policy kiroaiseoul/smolvla_190k_30k_aug_80k_kirogist --dataset kiroaiseoul/task06_pickup_beaker_and_move_to_refrigerator --n-episodes 12 --stride 3 --configs stock:10:1.0:1,det:10:0.0:1
echo "PHASE4 ALLDONE" >> $LOG/_status.txt
