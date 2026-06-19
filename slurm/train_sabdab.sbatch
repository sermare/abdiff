#!/bin/bash
#SBATCH --job-name=abdiff_sabdab
#SBATCH --partition=savio3_gpu
#SBATCH --qos=savio_lowprio
#SBATCH --account=co_nilah
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=/global/scratch/users/sergiomar10/TCera/abdiff/logs/%x_%j.out
#SBATCH --error=/global/scratch/users/sergiomar10/TCera/abdiff/logs/%x_%j.err
set -euo pipefail
cd /global/scratch/users/sergiomar10/TCera/abdiff
ENVBIN=/clusterfs/nilah/sergio/miniconda3/envs/tttppi/bin
PY=$ENVBIN/python
export PATH=$ENVBIN:$PATH          # ANARCI needs hmmscan on PATH
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

echo "[sabdab] $(date) host=$(hostname)"
DEVICE=cuda; "$PY" -c "import torch,sys;sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null || DEVICE=cpu
echo "[sabdab] device=$DEVICE hmmscan=$(command -v hmmscan)"

# 1) ANARCI-trimmed Fv tensors for the full SAbDab corpus (AntiBERTy embeddings)
N=$(find data/train_sabdab -name '*.pt' 2>/dev/null | wc -l)
echo "[sabdab] existing tensors: $N"
if [ "$N" -lt 12000 ]; then
  srun "$PY" -u scripts/prep_structures_anarci.py --device "$DEVICE"
fi

# 2) train the AF3-style coordinate-diffusion model on the Fv set
srun "$PY" -u scripts/train.py --data data/train_sabdab --ckpt-dir checkpoints/sabdab \
     --device "$DEVICE" --epochs 120 --bs 8 --n-block 8 --max-tok 320

echo "[sabdab] done $(date)"
