#!/bin/bash

OUTPUTFILE="node_id.txt"
MODEL_NAME=""

while getopts "o:m:" opt; do
  case $opt in
    o)
      OUTPUTFILE=$OPTARG
      ;;
    m)
      MODEL_NAME=$OPTARG
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
TAAS_UNIFIED_TRITON_IMAGE="${TAAS_UNIFIED_TRITON_IMAGE:-docker.io/jaychanhep/tritonserver-modulemap:latest}"
TAAS_DML_TRITON_IMAGE="${TAAS_DML_TRITON_IMAGE:-$TAAS_UNIFIED_TRITON_IMAGE}"
TAAS_MODULEMAP_TRITON_IMAGE="${TAAS_MODULEMAP_TRITON_IMAGE:-$TAAS_UNIFIED_TRITON_IMAGE}"

if [ -n "${TAAS_TRITON_IMAGE:-}" ]; then
    TRITON_IMAGE="$TAAS_TRITON_IMAGE"
    TRITON_IMAGE_REASON="TAAS_TRITON_IMAGE override"
else
    case "$MODEL_NAME" in
        ModuleMap)
            TRITON_IMAGE="$TAAS_MODULEMAP_TRITON_IMAGE"
            TRITON_IMAGE_REASON="ModuleMap default"
            ;;
        DoubleMetricLearning|MetricLearning|GNN4Pixel)
            TRITON_IMAGE="$TAAS_DML_TRITON_IMAGE"
            TRITON_IMAGE_REASON="${MODEL_NAME} default"
            ;;
        "")
            TRITON_IMAGE="${TAAS_ALL_TRITON_IMAGE:-$TAAS_MODULEMAP_TRITON_IMAGE}"
            TRITON_IMAGE_REASON="all-model default"
            ;;
        *)
            TRITON_IMAGE="${TAAS_DEFAULT_TRITON_IMAGE:-$TAAS_DML_TRITON_IMAGE}"
            TRITON_IMAGE_REASON="generic default for ${MODEL_NAME}"
            ;;
    esac
fi

echo "Using Triton image: $TRITON_IMAGE ($TRITON_IMAGE_REASON)"
TRITON_LD_LIBRARY_PATH="${TAAS_TRITON_LD_LIBRARY_PATH:-/usr/local/lib/python3.10/dist-packages/onnxruntime/capi:/usr/local/onnxruntime/lib:/usr/local/root/lib:/opt/hpcx/ucc/lib/:/opt/hpcx/ucx/lib/:/opt/tritonserver/backends/onnxruntime:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/lib:/usr/local/lib}"
TAAS_IMAGE_PULL_POLICY="${TAAS_IMAGE_PULL_POLICY:-missing}"
echo "Using image pull policy: $TAAS_IMAGE_PULL_POLICY"

TRITON_JOBS_DIR="${WORK_DIR}/jobs"
mkdir -p "$TRITON_JOBS_DIR"

TRITON_LOG_VERBOSE=false

TRITON_LOG_VERBOSE_FLAGS=""
TRITON_SERVER_NAME="${SLURMD_NODENAME}"
TRITON_MODEL_FLAGS="--model-repository=/models"
if [ -n "$MODEL_NAME" ]; then
    TRITON_MODEL_FLAGS="${TRITON_MODEL_FLAGS} --model-control-mode=explicit --load-model=${MODEL_NAME}"
    echo "Using explicit model load for: ${MODEL_NAME}"
else
    echo "No model provided; loading all models (implicit control mode)"
fi

echo "{" > "$OUTPUTFILE"
echo "  \"url\": \"$SLURMD_NODENAME\"," >> "$OUTPUTFILE"
echo "  \"port\": 8001" >> "$OUTPUTFILE"
echo "}" >> "$OUTPUTFILE"

#Setup Triton flags
if [ "$TRITON_LOG_VERBOSE" = true ]
then
    TRITON_LOG_VERBOSE_FLAGS="--log-verbose=3 --log-info=1 --log-warning=1 --log-error=1"
fi

#Start Triton
echo "[slurm] starting $TRITON_SERVER_NAME"
podman-hpc run -it --rm --gpu --shm-size=20GB -p 8002:8002 -p 8001:8001 -p 8000:8000 \
    --pull="$TAAS_IMAGE_PULL_POLICY" \
    --entrypoint=/opt/tritonserver/bin/tritonserver \
    --env LD_LIBRARY_PATH="$TRITON_LD_LIBRARY_PATH" \
    --volume="$TRITON_MODELS:/models" -w "$WORK_DIR" \
    -v "$WORK_DIR:$WORK_DIR" \
    $TRITON_IMAGE \
        $TRITON_MODEL_FLAGS \
        --allow-metrics=true \
        $TRITON_LOG_VERBOSE_FLAGS  2>&1 \
        | tee "$TRITON_JOBS_DIR/$TRITON_SERVER_NAME.log"
