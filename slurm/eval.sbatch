#!/bin/bash
#SBATCH --job-name=abdiff_eval
#SBATCH --partition=savio2_1080ti
#SBATCH --account=ac_nilah
#SBATCH --qos=savio_normal
#SBATCH --gres=gpu:GTX1080TI:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/global/scratch/users/sergiomar10/TCera/abdiff/logs/%x_%j.out
#SBATCH --error=/global/scratch/users/sergiomar10/TCera/abdiff/logs/%x_%j.err
set -euo pipefail
cd /global/scratch/users/sergiomar10/TCera/abdiff
PY=/clusterfs/nilah/sergio/miniconda3/envs/tttppi/bin/python
echo "[eval] $(date) host=$(hostname)"
srun "$PY" -u scripts/eval_sample.py --ckpt checkpoints/sabdab/abdiff_best.pt \
     --data data/train_sabdab --n-eval 40 --n-samples 4 --steps 200 \
     --device cuda --write-pdb results/eval_pdbs
echo "[eval] done $(date)"
