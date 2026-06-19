#!/bin/bash
#SBATCH --job-name=abdiff_embed
#SBATCH --partition=savio3_gpu
#SBATCH --qos=savio_lowprio
#SBATCH --account=co_nilah
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/global/scratch/users/sergiomar10/TCera/abdiff/logs/%x_%j.out
#SBATCH --error=/global/scratch/users/sergiomar10/TCera/abdiff/logs/%x_%j.err
set -euo pipefail
cd /global/scratch/users/sergiomar10/TCera/abdiff
PY=/clusterfs/nilah/sergio/miniconda3/envs/tttppi/bin/python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

echo "[abdiff] $(date) host=$(hostname)"
DEVICE=cuda; "$PY" -c "import torch,sys;sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null || DEVICE=cpu
echo "[abdiff] device=$DEVICE"

# 1) build sequence corpus (Fab / scFv / VHH) — cheap, stdlib only
srun "$PY" -u scripts/build_corpus.py

# 2) per-chain ESM2-150M residue embeddings
srun "$PY" -u scripts/embed_plm.py --model 150M --device "$DEVICE"

echo "[abdiff] done $(date)"
