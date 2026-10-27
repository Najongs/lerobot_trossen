#!/bin/bash
# ACT training on KISTI Hangang (GH200 144G x4 per node, aarch64) -- README §2-6.
#
# Submit from /scratch or /appsdata (Slurm refuses jobs submitted elsewhere):
#   cd /scratch/$USER && sbatch --export=ALL,DATASET=kiroaiseoul/<repo>,RUN=<name> \
#       ~/lerobot_trossen/scripts/train_hangang.sbatch
# Without STEPS it is the 10-step smoke run of §2-1 (no Hub push). For the real run:
#   $ sbatch --time=12:00:00 --export=ALL,DATASET=...,RUN=...,STEPS=60000,SAVE_FREQ=20000,\
#       POLICY_REPO=kiroaiseoul/act_<stage>_<steps> ~/lerobot_trossen/scripts/train_hangang.sbatch
#
#SBATCH --job-name=act-train
#SBATCH --partition=normal
#SBATCH --comment="field=robotics;appl=pytorch"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

: "${DATASET:?set DATASET=<hub repo id>}"
: "${RUN:?set RUN=<run name>}"
REPO=${REPO:-$HOME/lerobot_trossen}
STEPS=${STEPS:-10}
SAVE_FREQ=${SAVE_FREQ:-$STEPS}
BATCH=${BATCH:-16}                 # per GPU; x4 GPUs = effective 64 (README "유효 배치")
NPROC=${NPROC:-4}
OUT_ROOT=${OUT_ROOT:-/appsdata/$USER/lerobot_trossen/outputs}
OUTPUT_DIR=$OUT_ROOT/$RUN

# uv venv and caches off the home quota (64G, 500k files). /scratch deletes files
# not accessed for 15 days, so only re-downloadable things go there.
export UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT:-/appsdata/$USER/venvs/lerobot_trossen}
export UV_CACHE_DIR=/scratch/$USER/.cache/uv
export UV_PYTHON_INSTALL_DIR=/appsdata/$USER/uv-python
export UV_PYTHON_PREFERENCE=only-managed
export HF_HOME=/scratch/$USER/.cache/huggingface
export TORCH_HOME=/scratch/$USER/.cache/torch   # ACT's resnet18 backbone weights
# The token stays where `hf auth login` on the login node wrote it.
export HF_TOKEN_PATH=${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}
# Single node: no OFI plugin (it fails NCCL init on one node).
unset NCCL_NET
export NCCL_NET_PLUGIN=none

if [[ "$STEPS" == 10 ]]; then
  PUSH_ARGS=(--policy.push_to_hub=false)
else
  : "${POLICY_REPO:?set POLICY_REPO=<hub repo id> for a real run (or STEPS=10 for the smoke run)}"
  PUSH_ARGS=(--policy.repo_id="$POLICY_REPO")
fi
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "refusing: $OUTPUT_DIR exists (lerobot_train would stop; pick a new RUN or resume per §2-3)" >&2
  exit 2
fi
mkdir -p "$OUT_ROOT" "$HF_HOME"

cd "$REPO"
echo "job=$SLURM_JOB_ID node=$SLURM_JOB_NODELIST start=$(date -Is)"
echo "repo=$REPO git=$(git rev-parse HEAD) dirty=$(git status --porcelain | wc -l)"
echo "venv=$UV_PROJECT_ENVIRONMENT dataset=$DATASET output=$OUTPUT_DIR steps=$STEPS batch=$BATCH x $NPROC"

# Warm start (README §2-4): POLICY_PATH=<checkpoint or hub id> replaces --policy.type.
if [[ -n "${POLICY_PATH:-}" ]]; then
  POLICY_ARGS=(--policy.path="$POLICY_PATH")
else
  POLICY_ARGS=(--policy.type=act)
fi
# Fetch the dataset in ONE process first. Inside accelerate the other ranks wait
# for rank 0 at NCCL init with a 10 min limit, and a slow Hub download (a 477 MB
# dataset stalled 9 min on one file, job 1154115) kills the run there.
uv run --frozen --no-sync python -c \
  "import sys; from lerobot.datasets.lerobot_dataset import LeRobotDataset as D; d = D(sys.argv[1]); print('dataset', sys.argv[1], d.num_episodes, 'episodes', d.num_frames, 'frames')" \
  "$DATASET"

MULTI=()
[[ "$NPROC" -gt 1 ]] && MULTI=(--multi_gpu)

# --no-sync: the lock's torchcodec / pyrealsense2 have no aarch64 wheels, so the
# venv is synced once without them (README §2-6) and must not be re-synced here.
# lerobot then decodes video with pyav.
srun --mpi=none --kill-on-bad-exit=1 \
  uv run --frozen --no-sync accelerate launch \
    "${MULTI[@]}" --num_processes="$NPROC" \
    -m lerobot.scripts.lerobot_train \
    "${POLICY_ARGS[@]}" \
    --policy.device=cuda \
    "${PUSH_ARGS[@]}" \
    --dataset.repo_id="$DATASET" \
    --output_dir="$OUTPUT_DIR" \
    --job_name="$RUN" \
    --batch_size="$BATCH" --steps="$STEPS" --save_freq="$SAVE_FREQ" \
    --wandb.enable=false \
    ${TRAIN_ARGS:-}

echo "end=$(date -Is)"
