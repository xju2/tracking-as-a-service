# FRNNEval

Validation service ("frnn-eval") for [libFRNN](https://github.com/xju2/libFRNN).
It accepts the same `FEATURES` input as `MetricLearning` (`[N, 44]` FP32), runs the
MetricLearning embedding, builds the radius graph with both the original FRNN
(`frnn`) and libFRNN (`frnn_cuda`), and compares the two directed edge lists as sets.

Output `RESULT` (INT64, shape `[1]`):
- `0`: the edge lists agree.
- `>0`: number of edges found by only one of the two libraries.

On disagreement, the embedding and both edge lists are saved with `torch.save` to
`output_dir/frnn_diff_<time>_pid<pid>_<n>.pt`, as a dict with keys `embedding`,
`edges_frnn`, `edges_libfrnn`, `num_diff_edges`, `r_max` and `k_max`.

Parameters in `config.pbtxt`:
- `embedding_model_dir`: directory with `embedding.ckpt` and the MetricLearning modules,
  relative to this model directory (default `../MetricLearning/4`).
- `output_dir`: where disagreeing events are saved, relative to the server's working directory.
- `r_max`, `k_max`: radius and maximum number of neighbors, as in MetricLearning.
- `save_eval_metrics`: if `True`, append one row per request to
  `output_dir/eval_metrics_pid<pid>.csv` (one file per model instance) with the columns
  `request_id`, `num_space_points`, `num_frnn_edges`, `num_libfrnn_edges`,
  `num_diff_edges`, `frnn_latency_ms` and `libfrnn_latency_ms`. Latencies cover only
  the edge building, bracketed by `torch.cuda.synchronize()`. The first request
  includes one-time CUDA warm-up, so exclude it from timing studies.

The original FRNN keeps `k_max` neighbors including the point itself, whereas libFRNN
applies `max_neighbors` after excluding it. libFRNN is therefore called with
`max_neighbors = k_max - 1`.

This model requires the image built from `Dockerfile.frnn-eval`:
```bash
podman-hpc build --format docker -f Dockerfile.frnn-eval -t docexoty/frnn-eval .
```
and launched with it; `scripts/start-tritonserver.sh` defaults to an image without
libFRNN, so pass `-i`:
```bash
podman-hpc migrate docexoty/frnn-eval
./scripts/start-tritonserver.sh -o triton_ready.txt -m FRNNEval -i localhost/docexoty/frnn-eval:latest
```

Offline check without Triton (inside the same image):
```bash
cd models/FRNNEval/1
python frnn_eval.py -i /global/cfs/cdirs/m3443/data/for_alina/all_input_node_features.pt -v
# add -s to also write eval_metrics_pid<pid>.csv to the output directory (-o)
```
