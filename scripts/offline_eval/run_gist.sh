#!/usr/bin/env bash
# 같은 체크포인트를 GIST 판 시연에 평가 -- kiro 판보다 오차가 훨씬 낮으면 GIST 장면에 끌려간 것
set -u
cd "$(dirname "$0")/../.."
PY=/home/najo/NAS/lerobot_trossen/.venv/bin/python
OUT=docs/run_logs/2026-09-22_offline_eval/chunks
LOG=docs/run_logs/2026-09-22_offline_eval/logs
GA=GPU-05b804ff-3b02-39f4-cf62-b848e189ebdd
M80=kiroaiseoul/smolvla_190k_30k_aug_80k_kirogist
run() { local gpu=$1 tag=$2; shift 2; CUDA_VISIBLE_DEVICES=$gpu $PY scripts/offline_eval/predict_chunks.py --out $OUT --tag $tag "$@" > $LOG/$tag.log 2>&1; echo "EXIT $tag $?" >> $LOG/_status.txt; }
run $GA m80_t02gist --policy $M80 --dataset kiroaiseoul/task02_pickup_tubes_gist --n-episodes 12 --stride 3 --configs stock:10:1.0:1,det:10:0.0:1
run $GA m80_t06gist --policy $M80 --dataset kiroaiseoul/task06_pickup_beaker_and_move_to_refrigerator_gist --n-episodes 12 --stride 3 --configs stock:10:1.0:1,det:10:0.0:1
