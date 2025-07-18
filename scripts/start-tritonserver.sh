#!/bin/bash

OUTPUTFILE="node_id.txt"

while getopts "o:" opt; do
  case $opt in
    o)
      OUTPUTFILE=$OPTARG
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
  esac
done

OUTPUTFILE="$(realpath $OUTPUTFILE)"


WORK_DIR="$(dirname "${BASH_SOURCE[0]}" )/../"
WORK_DIR=$(readlink -f $WORK_DIR)
TRITON_MODELS="${WORK_DIR}/models"
TRITON_IMAGE="docker.io/docexoty/tritonserver:latest"

TRITON_JOBS_DIR="${WORK_DIR}/jobs"
mkdir -p $TRITON_JOBS_DIR

TRITON_LOG_VERBOSE=false


TRITON_LOG_VERBOSE_FLAGS=""
TRITON_SEVER_NAME="${SLURMD_NODENAME}"

echo "{" > $OUTPUTFILE
echo "  \"url\": \"$SLURMD_NODENAME\"," > $OUTPUTFILE
echo "  \"port\": 8001," >> $OUTPUTFILE
echo "}" >> $OUTPUTFILE

#Setup Triton flags
if [ "$TRITON_LOG_VERBOSE" = true ]
then
    TRITON_LOG_VERBOSE_FLAGS="--log-verbose=3 --log-info=1 --log-warning=1 --log-error=1"
fi

#Start Triton
echo "[slurm] starting $TRITON_SEVER_NAME"
podman-hpc run -it --rm --gpu --shm-size=20GB -p 8002:8002 -p 8001:8001 -p 8000:8000 \
    --volume="$TRITON_MODELS:/models" -w $WORK_DIR \
    -v $WORK_DIR:$WORK_DIR \
    $TRITON_IMAGE \
    tritonserver \
        --model-repository=/models \
        $TRITON_LOG_VERBOSE_FLAGS  2>&1 \
        | tee $TRITON_JOBS_DIR/$TRITON_SEVER_NAME.log
