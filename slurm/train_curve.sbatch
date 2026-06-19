#!/bin/bash
#SBATCH --job-name=abdiff_curve
#SBATCH --partition=savio3_gpu
#SBATCH --account=co_nilah
#SBATCH --qos=savio_lowprio
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/global/scratch/users/sergiomar10/TCera/abdiff/logs/%x_%j.out
#SBATCH --error=/global/scratch/users/sergiomar10/TCera/abdiff/logs/%x_%j.err
set -euo pipefail
cd /global/scratch/users/sergiomar10/TCera/abdiff
PY=/clusterfs/nilah/sergio/miniconda3/envs/tttppi/bin/python
export PATH=/clusterfs/nilah/sergio/miniconda3/envs/tttppi/bin:$PATH
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "[curve] $(date) host=$(hostname)"
srun "$PY" -u scripts/train.py --data data/train_sabdab --ckpt-dir checkpoints/sabdab_curve \
     --device cuda --epochs 140 --bs 8 --n-block 8 --max-tok 320 \
     --save-every 20 --eval-every 20 --gen-eval-n 10 --gen-eval-steps 100
echo "[curve] done $(date)"
