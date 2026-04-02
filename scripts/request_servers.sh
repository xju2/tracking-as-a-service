#!/bin/bash
#SBATCH -A mxxx
#SBATCH --job-name=tritonserver
#SBATCH -C "gpu&hbm80g"
#SBATCH -N 1
#SBATCH --time=13:00:00
#SBATCH --output=tritonserver-%j.out
#SBATCH --error=tritonserver-%j.err
#SBATCH --exclusive
#SBATCH -q premium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mail-type=ALL

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256
OUTPUT="tracking-as-a-service/jobs/tracking_ready.json" 
MODEL_NAME="DoubleMetricLearning"
#MODEL_NAME="ModuleMap"
echo "Loading model: $MODEL_NAME"

./scripts/start-tritonserver.sh -o $OUTPUT -m $MODEL_NAME