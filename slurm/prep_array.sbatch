#!/bin/bash
#SBATCH --job-name=abdiff_prep
#SBATCH --partition=savio3_gpu,savio2_1080ti
#SBATCH --qos=savio_lowprio
#SBATCH --account=co_nilah
#SBATCH --gres=gpu:1
#SBATCH --array=0-9
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=/global/scratch/users/sergiomar10/TCera/abdiff/logs/%x_%A_%a.out
#SBATCH --error=/global/scratch/users/sergiomar10/TCera/abdiff/logs/%x_%A_%a.err
set -euo pipefail
cd /global/scratch/users/sergiomar10/TCera/abdiff
ENVBIN=/clusterfs/nilah/sergio/miniconda3/envs/tttppi/bin
PY=$ENVBIN/python
export PATH=$ENVBIN:$PATH          # ANARCI needs hmmscan
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "[prep ${SLURM_ARRAY_TASK_ID}] $(date) host=$(hostname) part=$SLURM_JOB_PARTITION"
# NB: detect CUDA INSIDE srun (a pre-check outside srun has no GPU binding -> false cpu).
srun "$PY" -u scripts/prep_structures_anarci.py --device cuda \
     --out-dir data/train_sabdab_cdr \
     --shard "$SLURM_ARRAY_TASK_ID" --nshards 10
echo "[prep ${SLURM_ARRAY_TASK_ID}] done $(date)"
