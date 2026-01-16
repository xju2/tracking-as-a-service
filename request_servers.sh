#!/bin/bash
#SBATCH -A m4439
#SBATCH --job-name=tritonserver
#SBATCH -C "gpu&hbm80g"
#SBATCH -N 1
#SBATCH --time=11:00:00
#SBATCH --output=tritonserver-%j.out
#SBATCH --error=tritonserver-%j.err
#SBATCH -n 1
#SBATCH --exclusive
#SBATCH -q preempt
##SBATCH --gres=gpu:2
#SBATCH --mail-type=ALL


OUTPUT=tracking_ready.json
MODELNAME="DoubleMetricLearning"
echo "Loading model: $MODELNAME"

./scripts/start-tritonserver.sh -o $OUTPUT -m $MODELNAME