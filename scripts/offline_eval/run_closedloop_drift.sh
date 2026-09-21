#!/usr/bin/env bash
# 재생 폐루프 드리프트 진단: 관절별 부호 있는 이탈을 기록한다 (보정 없음 / 어깨 보정)
set -u
cd "$(dirname "$0")/../.."
PY=/home/najo/NAS/lerobot_trossen/.venv/bin/python
OUT=docs/run_logs/2026-09-22_offline_eval/closedloop
LOG=docs/run_logs/2026-09-22_offline_eval/logs
G1=GPU-c3d180c2-f92c-fe13-b7b8-337247e36a33
V="A_user_c50:5:50:average:-1.5:0:4:1.0:mean;B_c5f5:5:5:average:-1.5:5:4:1.0:mean"
M=kiroaiseoul/smolvla_190k_30k_aug_80k_kirogist
CUDA_VISIBLE_DEVICES=$G1 $PY scripts/offline_eval/closedloop_eval.py --policy $M --dataset kiroaiseoul/task02_pickup_tubes --variants "$V" --out $OUT/t02_drift_nooff.jsonl > $LOG/cl_t02_drift_nooff.log 2>&1; echo "EXIT cl_t02_drift_nooff $?" >> $LOG/_status.txt
CUDA_VISIBLE_DEVICES=$G1 $PY scripts/offline_eval/closedloop_eval.py --policy $M --dataset kiroaiseoul/task02_pickup_tubes --offset 1:0.014,8:0.009 --variants "$V" --out $OUT/t02_drift_off.jsonl > $LOG/cl_t02_drift_off.log 2>&1; echo "EXIT cl_t02_drift_off $?" >> $LOG/_status.txt
