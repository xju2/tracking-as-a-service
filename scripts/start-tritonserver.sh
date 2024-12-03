#!/bin/bash

WORK_DIR="$( dirname "${BASH_SOURCE[0]}" )/../"
TRITON_MODELS="${WORK_DIR}/models"
TRITON_IMAGE="docker.io/docexoty/tritonserver:latest"

TRITON_JOBS_DIR="${WORK_DIR}/jobs"
TRITON_LOGS=$TRITON_JOBS_DIR/$SLURM_JOB_ID
mkdir -p $TRITON_LOGS

TRITON_LOG_VERBOSE=false


TRITON_LOG_VERBOSE_FLAGS=""
TRITON_SEVER_NAME="${SLURMD_NODENAME}"
echo "$SLURMD_NODENAME" > $TRITON_JOBS_DIR/node_id.txt

#Setup Triton flags
if [ "$TRITON_LOG_VERBOSE" = true ]
then
    TRITON_LOG_VERBOSE_FLAGS="--log-verbose=3 --log-info=1 --log-warning=1 --log-error=1"
fi

#Start Triton
echo "[slurm] starting $TRITON_SEVER_NAME"
podman-hpc run -it --rm --gpu --shm-size=20GB -p 8002:8002 -p 8001:8001 -p 8000:8000 \
    --volume="$TRITON_MODELS:/models" \
    $TRITON_IMAGE \
    tritonserver \
        --model-repository=/models \
        $TRITON_LOG_VERBOSE_FLAGS  2>&1 \
        | tee $TRITON_LOGS/$TRITON_SEVER_NAME.log
