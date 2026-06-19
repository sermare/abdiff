#!/bin/bash
#SBATCH --job-name=abdiff_evalcdr
#SBATCH --partition=savio2_1080ti
#SBATCH --account=ac_nilah
#SBATCH --qos=savio_normal
#SBATCH --gres=gpu:GTX1080TI:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/global/scratch/users/sergiomar10/TCera/abdiff/logs/%x_%j.out
#SBATCH --error=/global/scratch/users/sergiomar10/TCera/abdiff/logs/%x_%j.err
set -euo pipefail
cd /global/scratch/users/sergiomar10/TCera/abdiff
PY=/clusterfs/nilah/sergio/miniconda3/envs/tttppi/bin/python
echo "[evalcdr] $(date) host=$(hostname)"
srun "$PY" -u scripts/eval_sample.py --ckpt checkpoints/sabdab/abdiff_final_v1.pt \
     --data data/train_sabdab_cdr --n-eval 60 --n-samples 4 --steps 200 \
     --device cuda --write-pdb results/eval_pdbs_cdr
echo "[evalcdr] done $(date)"
