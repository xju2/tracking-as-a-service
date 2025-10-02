# Tracking as a Service

This repository contains a set of Tracking models that can be used as a service.

## Instructions
1. Launch the server:
```bash
srun --job-name=TritonTest -C "gpu&hbm80g" -N 1 -G 1 -c 10 -n 1 -t 4:00:00 -A m3443 \
  -q interactive /bin/bash -c "./scripts/start-tritonserver.sh -o triton_ready.txt"
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


### Install packages.
```
poetry env use /global/common/software/nersc/pe/conda-envs/24.1.0/python-3.11/nersc-python/bin/python

poetry install

```
