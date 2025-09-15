#!/bin/bash
#SBATCH -A m3443
#SBATCH --job-name=tritonserver
#SBATCH -C "gpu&hbm80g"
#SBATCH -N 1
#SBATCH -G 4
#SBATCH --time=48:00:00
#SBATCH --output=tritonserver-%j.out
#SBATCH --error=tritonserver-%j.err
#SBATCH -n 1
#SBATCH --exclusive
#SBATCH -q regular
#SBATCH --mail-type=ALL


OUTPUT=tracking_ready.json
./scripts/start-tritonserver.sh -o $OUTPUT
