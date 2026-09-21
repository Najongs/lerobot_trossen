#!/usr/bin/env bash
# 1단계: teacher-forced 청크 예측. GPU 는 UUID 로 지정한다 (ibomcom: CUDA 순서와 nvidia-smi 순서가 다르다).
set -u
cd "$(dirname "$0")/../.."
PY=/home/najo/NAS/lerobot_trossen/.venv/bin/python
OUT=docs/run_logs/2026-09-22_offline_eval/chunks
LOG=docs/run_logs/2026-09-22_offline_eval/logs
G0=GPU-7ff6997b-14c1-9283-5119-251c9c899b8e   # nvidia-smi 0, 3090
GA=GPU-05b804ff-3b02-39f4-cf62-b848e189ebdd   # nvidia-smi 3, A6000
G1=GPU-c3d180c2-f92c-fe13-b7b8-337247e36a33   # nvidia-smi 1, 3090
G4=GPU-1cdd7bc8-7c5a-dc09-d4fc-5dcfe92e104d   # nvidia-smi 4, 3090
M80=kiroaiseoul/smolvla_190k_30k_aug_80k_kirogist
T02=kiroaiseoul/task02_pickup_tubes
T06=kiroaiseoul/task06_pickup_beaker_and_move_to_refrigerator
CFG="stock:10:1.0:1,det:10:0.0:1,k4h:10:0.5:4,k7:10:0.9:7"
run() { local gpu=$1 tag=$2; shift 2; CUDA_VISIBLE_DEVICES=$gpu $PY scripts/offline_eval/predict_chunks.py --out $OUT --tag $tag "$@" > $LOG/$tag.log 2>&1; echo "EXIT $tag $?" >> $LOG/_status.txt; }
# 80k 체크포인트, 모든 프레임 (실행 방식 시뮬레이션용)
( run $G0 m80_t02 --policy $M80 --dataset $T02 --n-episodes 12 --configs $CFG ) &
( run $GA m80_t06 --policy $M80 --dataset $T06 --n-episodes 12 --configs $CFG --batch 24 ) &
# 계보 비교 (3프레임 간격, 열린 루프 지표용)
( for m in smolvla_190k_30k_aug_10k_kirogist foundation_smolvla_v1_190k_30k_aug foundation_smolvla_v1_190000; do
    run $G1 ${m}_t02 --policy kiroaiseoul/$m --dataset $T02 --n-episodes 12 --stride 3 --configs stock:10:1.0:1,det:10:0.0:1
  done ) &
( for m in smolvla_190k_30k_aug_10k_kirogist foundation_smolvla_v1_190k_30k_aug foundation_smolvla_v1_190000; do
    run $G4 ${m}_t06 --policy kiroaiseoul/$m --dataset $T06 --n-episodes 12 --stride 3 --configs stock:10:1.0:1,det:10:0.0:1
  done ) &
wait
echo "PHASE1 ALLDONE" >> $LOG/_status.txt
