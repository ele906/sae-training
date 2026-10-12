#!/bin/bash
#SBATCH --job-name=sae_box_frozen
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/n/fs/goose/sae-training/logs/%x-%j.out

set -euo pipefail
cd /n/fs/goose
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
/n/fs/goose/el8403/conda-envs/reno/bin/python -u sae-training/train_box_sae.py \
  --freeze-sae --output sae-training/runs/frozen_sae_box_baseline_002
