Metric Learning pipeline for GNN tracking.


## Versions
- **1**: original FRNN for edge building; CC + walk-through from `fastwalkthrough.py` for
  graph segmentation.
- **2**: same models (checkpoints and model code are symlinks to `1/`), with
  [libFRNN](https://github.com/xju2/libFRNN) (`frnn_cuda`) for edge building and
  [graph_segment](https://github.com/xju2/graph_segment_for_gnn_tracking) for graph
  segmentation. Pick the algorithm with the `segment_backend` parameter in `config.pbtxt`
  (`cc`, `cc_walk`, `cc_bestpred_walk`, `cc_junction_removal`) and set its cuts with
  `segment_cuts` (JSON). The default, `cc_walk` with
  `{"cc_cut": 0.01, "th_min": 0.1, "th_add": 0.6}`, matches version 1's cuts.

Both versions take the `deterministic` parameter: sum the GNN messages in a fixed order, so
the same event always gives the same tracks (see `DoubleMetricLearning/README.md`).
`interaction_gnn.py` is a copy of `DoubleMetricLearning/1/interaction_gnn.py`; keep them in
sync. Latency per stage: `python scripts/benchmark_dml_stages.py -m MetricLearning -i <input> -D`.
