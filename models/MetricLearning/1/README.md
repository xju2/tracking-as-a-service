## Metric Learning Pipeline For ITk

### Debugging data
Debugging data is created when `save_event` is enabled. The data is saved in `/global/cfs/cdirs/m3443/usr/xju/ITk/For2023Paper/metric_learning_testing/debug`
and copied in this directory.

### Local Testing.

To start the server
```bash
srun -C "gpu&hbm80g" -q interactive -N 1 -G 1 -c 32 -t 4:00:00 -A m3443 --pty /bin/bash -c "cd /pscratch/sd/x/xju/ITk/ForFinalPaper/tracking-as-a-service && ./scripts/start-tritonserver.sh"
```

To run the client
```bash
TRITON_IMAGE="docker.io/docexoty/tritonserver:latest"
podman-hpc run -it --rm --ipc=host --net=host --ulimit memlock=-1 --ulimit stack=67108864 -v /pscratch/sd/x/xju/ITk/ForFinalPaper/tracking-as-a-service:/workspace -w /workspace $TRITON_IMAGE bash


cd models/MetricLearning/1
python client.py -i event000006800_node_features.pt
```
