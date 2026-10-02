# Tracking as a Service

This repository contains a set of Tracking models that can be used as a service.

## Instructions
1. Launch the server with an interactive job for 4 hours
```bash
srun --job-name=TritonTest -C "gpu&hbm80g" -N 1 -G 1 -c 10 -n 1 -t 4:00:00 -A m3443 \
  -q interactive /bin/bash -c "./scripts/start-tritonserver.sh -o triton_ready.txt"
```

Or, launch the server with a regular batch job for 48 hours
```bash
sbatch -G 1 request_servers.sh
```

2. Run the client:
```bash
podman-hpc run -it --rm --ipc=host --net=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v ${PWD}:/workspace/ -w /workspace \
  -v /global/cfs/cdirs/m3443/data/for_alina:/global/cfs/cdirs/m3443/data/for_alina \
  docker.io/docexoty/tritonserver:latest python models/MetricLearning/2/client.py -i /global/cfs/cdirs/m3443/data/for_alina/all_input_node_features.pt
```

## Models

Supported models are saved in the [model_repos](model_repos) directory.

## ExaTrkX Models

### Python backend


#### Docker container

```bash
podman-hpc build --format docker -f Dockerfile -t docexoty/tritonserver
```

The `frnn-eval` image adds [libFRNN](https://github.com/xju2/libFRNN) on top of it,
for the [FRNNEval](models/FRNNEval/README.md) validation model:
```bash
podman-hpc build --format docker -f Dockerfile.frnn-eval -t docexoty/frnn-eval .
```

FRNNEval needs `frnn_cuda`, which only this image provides. Make the image available
on compute nodes and launch the server with it (`-i`); the default image lacks libFRNN
and every FRNNEval request would fail:
```bash
podman-hpc migrate docexoty/frnn-eval
srun --job-name=FRNNEval -C "gpu&hbm80g" -N 1 -G 1 -c 10 -n 1 -t 4:00:00 -A m3443 \
  -q interactive /bin/bash -c "./scripts/start-tritonserver.sh -o triton_ready.txt -m FRNNEval -i localhost/docexoty/frnn-eval:latest"
```


### Install packages.
```
uv venv --python /global/common/software/nersc/pe/conda-envs/24.1.0/python-3.11/nersc-python/bin/python

uv sync
