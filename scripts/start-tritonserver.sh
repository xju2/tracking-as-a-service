#!/bin/bash

WORK_DIR="/pscratch/sd/x/xju/ITk/ForFinalPaper/tracking-as-a-service"
TRITON_MODELS="${WORK_DIR}/models"
TRITON_IMAGE="docexoty/tritonserver:latest"

TRITON_JOBS_DIR="${WORK_DIR}/jobs"
TRITON_LOGS=$TRITON_JOBS_DIR/$SLURM_JOB_ID

TRITON_LOG_VERBOSE=true


TRITON_LOG_VERBOSE_FLAGS=""
TRITON_SEVER_NAME="TritonServer_${SLURMD_NODENAME}"

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
