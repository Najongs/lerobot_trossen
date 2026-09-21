#!/usr/bin/env bash
# 재생 폐루프: 어깨(joint_1) 편향 보정을 넣으면 시연 경로 이탈(off)과 완주율이 나아지는가
set -u
cd "$(dirname "$0")/../.."
PY=/home/najo/NAS/lerobot_trossen/.venv/bin/python
OUT=docs/run_logs/2026-09-22_offline_eval/closedloop
LOG=docs/run_logs/2026-09-22_offline_eval/logs
GA=GPU-05b804ff-3b02-39f4-cf62-b848e189ebdd
V="A_user_c50:5:50:average:-1.5:0:7:0.9:mean;B_c5f5:5:5:average:-1.5:5:7:0.9:mean;F_c30f5:5:30:average:-1.5:5:7:0.9:mean"
CUDA_VISIBLE_DEVICES=$GA $PY scripts/offline_eval/closedloop_eval.py --policy kiroaiseoul/smolvla_190k_30k_aug_80k_kirogist --dataset kiroaiseoul/task02_pickup_tubes --offset 1:0.014,8:0.009 --variants "$V" --out $OUT/t02_m80_offset.jsonl > $LOG/cl_t02_m80_offset.log 2>&1; echo "EXIT cl_t02_m80_offset $?" >> $LOG/_status.txt
