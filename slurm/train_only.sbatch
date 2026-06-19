#!/bin/bash
#SBATCH --job-name=abdiff_traind
#SBATCH --partition=savio3_gpu
#SBATCH --qos=savio_lowprio
#SBATCH --account=co_nilah
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/global/scratch/users/sergiomar10/TCera/abdiff/logs/%x_%j.out
#SBATCH --error=/global/scratch/users/sergiomar10/TCera/abdiff/logs/%x_%j.err
set -euo pipefail
cd /global/scratch/users/sergiomar10/TCera/abdiff
ENVBIN=/clusterfs/nilah/sergio/miniconda3/envs/tttppi/bin
PY=$ENVBIN/python
export PATH=$ENVBIN:$PATH
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "[train] $(date) host=$(hostname) part=$SLURM_JOB_PARTITION"
N=$(find data/train_sabdab -name '*.pt' 2>/dev/null | wc -l)
echo "[train] train_tensors=$N (cuda auto-detected inside srun)"
srun "$PY" -u scripts/train.py --data data/train_sabdab --ckpt-dir checkpoints/sabdab \
     --device cuda --epochs 150 --bs 8 --n-block 8 --max-tok 320
echo "[train] done $(date)"
