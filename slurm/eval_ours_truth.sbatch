#!/bin/bash
#SBATCH --job-name=abdiff_ourstruth
#SBATCH --partition=savio2_1080ti
#SBATCH --account=ac_nilah
#SBATCH --qos=savio_normal
#SBATCH --gres=gpu:GTX1080TI:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:40:00
#SBATCH --output=/global/scratch/users/sergiomar10/TCera/abdiff/logs/%x_%j.out
#SBATCH --error=/global/scratch/users/sergiomar10/TCera/abdiff/logs/%x_%j.err
set -euo pipefail
cd /global/scratch/users/sergiomar10/TCera/abdiff
PY=/clusterfs/nilah/sergio/miniconda3/envs/tttppi/bin/python
srun "$PY" -u scripts/eval_ours_truth.py --truth data/bench_truth.pt --device cuda --steps 200 --n-samples 4
