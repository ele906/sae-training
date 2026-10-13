#!/bin/bash
#SBATCH --job-name=flower_sae_auc
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/n/fs/goose/sae-training/logs/%x-%j.out

set -euo pipefail
cd /n/fs/goose
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
/n/fs/goose/el8403/conda-envs/reno/bin/python -u sae-training/compare_flower_auc.py
