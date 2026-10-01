# FRNNEval

Validation service ("frnn-eval") for [libFRNN](https://github.com/xju2/libFRNN).
It accepts the same `FEATURES` input as `MetricLearning` (`[N, 44]` FP32), runs the
MetricLearning embedding, builds the radius graph with both the original FRNN
(`frnn`) and libFRNN (`frnn_cuda`), and compares the two directed edge lists as sets.

Output `LABELS` (INT64, shape `[-1]`), the same as `MetricLearning` so that clients such
as the Athena job run unchanged. It is always the dummy track candidates `[0, 1, 2, 3, 4, 5]`.
The comparison results go to the server log (the number of differing edges is printed on
disagreement) and to `output_dir`. Errors are logged and do not fail the request.

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
  `num_diff_edges`, `frnn_latency_ms`, `libfrnn_latency_ms`, `frnn_peak_mem_mb` and
  `libfrnn_peak_mem_mb`. Latencies cover only the edge building, bracketed by
  `torch.cuda.synchronize()`. The first request includes one-time CUDA warm-up, so
  exclude it from timing studies. The memory columns are empty unless `measure_memory`
  is on.
- `measure_memory`: if `True`, run both libraries once more per request, untimed, and
  record the peak GPU memory each needs on top of what was already in use, including
  the output edge list. PyTorch's peak counter covers memory PyTorch allocates; memory
  allocated with raw `cudaMalloc` (libFRNN's per-call workspace) is sampled from a
  background thread as device memory in use minus PyTorch's reserved memory, so other
  processes on the same GPU add noise. Use a dedicated GPU for memory studies.
- `save_data`: if `True`, save every event (not only disagreeing ones) for the libFRNN
  benchmarks to `output_dir/data/<time>_pid<pid>_<n>/`:
  - `embedding.npy`: float32 `[N, D]`, C order, the exact tensor fed to both libraries.
  - `edges_frnn.npy`: int64 `[2, E]`, directed edges from the original FRNN (reference),
    self-loops removed.
  - `edges_libfrnn.npy`: int64 `[2, E']`, same from libFRNN.
  - `meta.json`: `request_id`, `num_nodes`, `dim`, `r_max`, `k_max`, `num_frnn_edges`,
    `num_libfrnn_edges` and `num_diff_edges`.

  `.npy` is a short text header followed by the raw little-endian array, so it is
  lossless, loads with `numpy.load(..., mmap_mode="r")`, and in C++ it takes a few lines
  to parse the header and `fread` the rest.

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
# add -d to also save the embedding and edge lists as .npy under <output-dir>/data/
# add -M to also record peak GPU memory (printed with -v, saved with -s)
```
