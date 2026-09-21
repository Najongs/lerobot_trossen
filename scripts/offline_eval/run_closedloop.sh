#!/usr/bin/env bash
# 재생 폐루프 (task02, 80k). 같은 스케줄의 완벽 정책(시연 청크) 기준선을 같이 남긴다.
set -u
cd "$(dirname "$0")/../.."
PY=/home/najo/NAS/lerobot_trossen/.venv/bin/python
OUT=docs/run_logs/2026-09-22_offline_eval/closedloop
LOG=docs/run_logs/2026-09-22_offline_eval/logs
G0=GPU-7ff6997b-14c1-9283-5119-251c9c899b8e
V="A_user_c50:5:50:average:-1.5:0:7:0.9:mean;B_c5f5:5:5:average:-1.5:5:7:0.9:mean;C_ta-1:5:0:average:-1.0:0:7:0.9:mean;D_ta.01:5:0:average:0.01:0:7:0.9:mean;E_c20f5:5:20:average:-1.5:5:7:0.9:mean;F_c30f5:5:30:average:-1.5:5:7:0.9:mean;G_k1_c5f5:5:5:average:-1.5:5:1:1.0:mean;H_med4_c5f5:5:5:average:-1.5:5:4:1.0:medoid;I_med4_c30f5:5:30:average:-1.5:5:4:1.0:medoid;J_det_c5f5:5:5:average:-1.5:5:1:0.0:mean"
CUDA_VISIBLE_DEVICES=$G0 $PY scripts/offline_eval/closedloop_eval.py --policy kiroaiseoul/smolvla_190k_30k_aug_80k_kirogist --dataset kiroaiseoul/task02_pickup_tubes --perfect --variants "$V" --out $OUT/t02_perfect.jsonl > $LOG/cl_t02_perfect.log 2>&1; echo "EXIT cl_t02_perfect $?" >> $LOG/_status.txt
CUDA_VISIBLE_DEVICES=$G0 $PY scripts/offline_eval/closedloop_eval.py --policy kiroaiseoul/smolvla_190k_30k_aug_80k_kirogist --dataset kiroaiseoul/task02_pickup_tubes --variants "$V" --out $OUT/t02_m80.jsonl > $LOG/cl_t02_m80.log 2>&1; echo "EXIT cl_t02_m80 $?" >> $LOG/_status.txt
