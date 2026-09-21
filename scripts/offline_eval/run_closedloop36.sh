#!/usr/bin/env bash
# 재생 폐루프, 에피소드 36개 (task02, 80k, K=4 noise 1.0). 상위 후보만. 같은 스케줄의 완벽 정책 기준선도 남긴다.
set -u
cd "$(dirname "$0")/../.."
PY=/home/najo/NAS/lerobot_trossen/.venv/bin/python
OUT=docs/run_logs/2026-09-22_offline_eval/closedloop
LOG=docs/run_logs/2026-09-22_offline_eval/logs
M=kiroaiseoul/smolvla_190k_30k_aug_80k_kirogist
T=kiroaiseoul/task02_pickup_tubes
G1=GPU-c3d180c2-f92c-fe13-b7b8-337247e36a33
GA=GPU-05b804ff-3b02-39f4-cf62-b848e189ebdd
G4=GPU-1cdd7bc8-7c5a-dc09-d4fc-5dcfe92e104d
V1="A_c50:4:50:average:-1.5:0:4:1.0:mean;F_c30f5:4:30:average:-1.5:5:4:1.0:mean"
V2="D_ta.01:4:0:average:0.01:0:4:1.0:mean;B_c5f5:4:5:average:-1.5:5:4:1.0:mean"
V3="K_c40f5:4:40:average:-1.5:5:4:1.0:mean;L_c30f10:4:30:average:-1.5:10:4:1.0:mean"
run() { local gpu=$1 tag=$2; shift 2; CUDA_VISIBLE_DEVICES=$gpu $PY scripts/offline_eval/closedloop_eval.py --policy $M --dataset $T --n-episodes 36 "$@" --out $OUT/$tag.jsonl > $LOG/$tag.log 2>&1; echo "EXIT $tag $?" >> $LOG/_status.txt; }
( run $G1 cl36_a --variants "$V1"; run $G1 cl36_perfect --perfect --variants "$V1;$V2;$V3" ) &
( run $GA cl36_b --variants "$V2" ) &
( run $G4 cl36_c --variants "$V3" ) &
wait
echo "CL36 ALLDONE" >> $LOG/_status.txt
