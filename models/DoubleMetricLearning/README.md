# Double Metric Learning pipeline

- GNN Filtering with 256 hidden dimension.


## Versions
- **1**: original FRNN for edge building; CC + walk-through from `fastwalkthrough.py`
  (or `dp_walkthrough.py` with `use_dpw`) for graph segmentation.
- **2**: same models (checkpoints and model code are symlinks to `1/`), with
  [libFRNN](https://github.com/xju2/libFRNN) (`frnn_cuda`) for edge building and
  [graph_segment](https://github.com/xju2/graph_segment_for_gnn_tracking) for graph
  segmentation. Pick the algorithm with the `segment_backend` parameter in `config.pbtxt`
  (`cc`, `cc_walk`, `cc_bestpred_walk`, `cc_junction_removal`) and set its cuts with
  `segment_cuts` (JSON). The default, `cc_walk` with
  `{"cc_cut": 0.01, "th_min": 0.1, "th_add": 0.6}`, matches version 1's cuts.
  A bad backend or cut name fails at model load.

Triton serves the latest version by default, so version 2 is served unless
`version_policy` says otherwise; it needs `frnn_cuda` and `graph_segment` in the image
(see `Dockerfile`). Offline:
```bash
cd models/DoubleMetricLearning/2
python inference.py -i ../all_input_node_features.pt -b cc_bestpred_walk \
  --segment-cuts '{"cc_cut": 0.01, "th_min": 0.3, "th_add": 0.8}'
```
