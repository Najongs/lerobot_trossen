#!/usr/bin/env bash
# 3단계: K 샘플을 평균 대신 medoid 로 줄이면 움직임이 보존되는가 (80k, 모든 프레임, 1단계와 같은 에피소드)
set -u
cd "$(dirname "$0")/../.."
PY=/home/najo/NAS/lerobot_trossen/.venv/bin/python
OUT=docs/run_logs/2026-09-22_offline_eval/chunks
LOG=docs/run_logs/2026-09-22_offline_eval/logs
G1=GPU-c3d180c2-f92c-fe13-b7b8-337247e36a33
G4=GPU-1cdd7bc8-7c5a-dc09-d4fc-5dcfe92e104d
M80=kiroaiseoul/smolvla_190k_30k_aug_80k_kirogist
CFG="k4med:10:1.0:4:medoid,k7med:10:0.9:7:medoid,k4mean1:10:1.0:4:mean,ns12:10:1.2:1"
run() { local gpu=$1 tag=$2; shift 2; CUDA_VISIBLE_DEVICES=$gpu $PY scripts/offline_eval/predict_chunks.py --out $OUT --tag $tag "$@" > $LOG/$tag.log 2>&1; echo "EXIT $tag $?" >> $LOG/_status.txt; }
( run $G1 m80_t02_p3 --policy $M80 --dataset kiroaiseoul/task02_pickup_tubes --n-episodes 12 --configs $CFG --seed 1 ) &
( run $G4 m80_t06_p3 --policy $M80 --dataset kiroaiseoul/task06_pickup_beaker_and_move_to_refrigerator --n-episodes 12 --configs $CFG --seed 1 --batch 24 ) &
wait
echo "PHASE3 ALLDONE" >> $LOG/_status.txt
