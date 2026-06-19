#!/bin/bash
#SBATCH --job-name=abdiff_train
#SBATCH --partition=savio3_gpu
#SBATCH --qos=savio_lowprio
#SBATCH --account=co_nilah
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/global/scratch/users/sergiomar10/TCera/abdiff/logs/%x_%j.out
#SBATCH --error=/global/scratch/users/sergiomar10/TCera/abdiff/logs/%x_%j.err
set -euo pipefail
cd /global/scratch/users/sergiomar10/TCera/abdiff
PY=/clusterfs/nilah/sergio/miniconda3/envs/tttppi/bin/python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

echo "[abdiff] $(date) host=$(hostname)"
DEVICE=cuda; "$PY" -c "import torch,sys;sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null || DEVICE=cpu
echo "[abdiff] device=$DEVICE"

# 1) build aligned (AntiBERTy-emb, backbone-coords) training tensors (CIFs prefetched locally)
N_TRAIN=$(find data/train -name '*.pt' 2>/dev/null | wc -l)
if [ "$N_TRAIN" -lt 150 ]; then
  srun "$PY" -u scripts/prep_structures.py --device "$DEVICE"
fi

# 2) train the AF3-style coordinate-diffusion model (AntiBERTy conditioning, dim=512)
srun "$PY" -u scripts/train.py --device "$DEVICE" --epochs 300 --bs 4 --n-block 8

echo "[abdiff] done $(date)"
