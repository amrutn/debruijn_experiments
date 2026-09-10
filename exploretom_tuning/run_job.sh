#!/bin/bash
#SBATCH --job-name=exploretom
#SBATCH --output=logs/%x-%j.out       # %x=job name, %j=job id (unique per run)
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=dgx-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1                       # <-- WAS MISSING. Request 1 B200. (try --gres=gpu:1 if this is rejected)
#SBATCH --cpus-per-task=28
#SBATCH --mem=224G
#SBATCH --time=20:00:00                # check the partition's max walltime; the job is resumable (see below)

set -euo pipefail
mkdir -p logs                          # --output dir must exist or the job dies silently

# --- environment: packages live in a conda env, not `module load <pkg>` ---
module load anaconda3                  # exact name via `module avail` on Betty (maybe miniconda/miniforge)
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ngram-experiments-cu130 # create once from environment2.yml (cu130 x86 -> B200 OK)

# --- GPQA is gated; supply a token WITHOUT hard-coding it in the script ---
export HF_TOKEN=$(cat ~/.hf_token)     # or run `huggingface-cli login` once on the login node

cd "$HOME/ngram_experiments/exploretom_tuning_4people"                  # directory containing entropy_exp.py


export HF_HOME=/ceph/projects/pratikac/cot-reasoning/hf
export HF_HUB_DISABLE_XET=1
export PYTHONUNBUFFERED=1
# SLURM already sets CUDA_VISIBLE_DEVICES to your one allocated GPU, so DON'T pass
# --devices (it would override SLURM's pin). The script uses the single visible GPU.
python run_experiments.py
