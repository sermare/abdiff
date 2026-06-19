#!/bin/bash
#SBATCH --job-name=abdiff_esmfold
#SBATCH --partition=savio2_1080ti
#SBATCH --account=ac_nilah
#SBATCH --qos=savio_normal
#SBATCH --gres=gpu:GTX1080TI:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:50:00
#SBATCH --output=/global/scratch/users/sergiomar10/TCera/abdiff/logs/%x_%j.out
#SBATCH --error=/global/scratch/users/sergiomar10/TCera/abdiff/logs/%x_%j.err
set -euo pipefail
cd /global/scratch/users/sergiomar10/TCera/abdiff
# ESMFold lives in the struct-evo env (fair-esm with esmfold)
PY=/clusterfs/nilah/sergio/miniconda3/envs/struct-evo/bin/python
echo "[esmfold] $(date) host=$(hostname)"
srun "$PY" -u scripts/bench_esmfold.py --device cuda
echo "[esmfold] done $(date)"
