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

The original FRNN keeps `k_max` neighbors including the point itself, whereas libFRNN
applies `max_neighbors` after excluding it. libFRNN is therefore called with
`max_neighbors = k_max - 1`.

This model requires the image built from `Dockerfile.frnn-eval`:
```bash
podman-hpc build --format docker -f Dockerfile.frnn-eval -t docexoty/frnn-eval .
```

Offline check without Triton (inside the same image):
```bash
cd models/FRNNEval/1
python frnn_eval.py -i /global/cfs/cdirs/m3443/data/for_alina/all_input_node_features.pt -v
```
