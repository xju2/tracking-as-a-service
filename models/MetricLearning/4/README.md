# Metric Learning Inference with Checkpoints

This document describes how to run the metric learning inference pipeline using pre-trained models from checkpoint files.

## How to Run

To run the inference script, use the following command from within the `models/MetricLearning/3` directory:

```bash
python inference.py -i <path_to_input_data> [options]
```

### Examples

Here are example commands to run the inference with verbose output on a sample input file:

- Running with default model path (current directory):
  ```bash
  time python inference.py -i ../all_input_node_features.pt -v
  ```

- Explicitly specifying the model path (e.g., current directory):
  ```bash
  time python inference.py -i ../all_input_node_features.pt -m . -v
  ```

---

Debug information:
Running on the event 6800 and printing the following information:

```text
(acorn_dev_forMR) cchan@login21:/global/cfs/projectdirs/m3443/usr/jay/tracking-as-a-service/models/MetricLearning/4> time python inference.py -i /global/cfs/cdirs/m3443/data/for_alina/all_input_node_features.pt -v
WARNING:root:FRNN is available
MetricLearningInferenceConfig(model_path=PosixPath('.'), device='cuda', auto_cast=False, compling=False, debug=True, save_debug_data=False, r_max=0.12, k_max=1000, filter_cut=0.05, filter_batches=10, cc_cut=0.01, walk_min=0.1, walk_max=0.6, embedding_node_features=['r', 'phi', 'z', 'cluster_x_1', 'cluster_y_1', 'cluster_z_1', 'cluster_x_2', 'cluster_y_2', 'cluster_z_2', 'count_1', 'charge_count_1', 'loc_eta_1', 'loc_phi_1', 'localDir0_1', 'localDir1_1', 'localDir2_1', 'lengthDir0_1', 'lengthDir1_1', 'lengthDir2_1', 'glob_eta_1', 'glob_phi_1', 'eta_angle_1', 'phi_angle_1', 'count_2', 'charge_count_2', 'loc_eta_2', 'loc_phi_2', 'localDir0_2', 'localDir1_2', 'localDir2_2', 'lengthDir0_2', 'lengthDir1_2', 'lengthDir2_2', 'glob_eta_2', 'glob_phi_2', 'eta_angle_2', 'phi_angle_2'], embedding_node_scale=[1000.0, 3.14, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1.0, 1.0, 3.14, 3.14, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.14, 3.14, 3.14, 3.14, 1.0, 1.0, 3.14, 3.14, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.14, 3.14, 3.14, 3.14], filter_node_features=['r', 'phi', 'z', 'cluster_x_1', 'cluster_y_1', 'cluster_z_1', 'cluster_x_2', 'cluster_y_2', 'cluster_z_2', 'count_1', 'charge_count_1', 'loc_eta_1', 'loc_phi_1', 'localDir0_1', 'localDir1_1', 'localDir2_1', 'lengthDir0_1', 'lengthDir1_1', 'lengthDir2_1', 'glob_eta_1', 'glob_phi_1', 'eta_angle_1', 'phi_angle_1', 'count_2', 'charge_count_2', 'loc_eta_2', 'loc_phi_2', 'localDir0_2', 'localDir1_2', 'localDir2_2', 'lengthDir0_2', 'lengthDir1_2', 'lengthDir2_2', 'glob_eta_2', 'glob_phi_2', 'eta_angle_2', 'phi_angle_2'], filter_node_scale=[1000.0, 3.14, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1.0, 1.0, 3.14, 3.14, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.14, 3.14, 3.14, 3.14, 1.0, 1.0, 3.14, 3.14, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.14, 3.14, 3.14, 3.14], gnn_node_features=['r', 'phi', 'z', 'eta', 'cluster_r_1', 'cluster_phi_1', 'cluster_z_1', 'cluster_eta_1', 'cluster_r_2', 'cluster_phi_2', 'cluster_z_2', 'cluster_eta_2'], gnn_node_scale=[1000.0, 3.14159265359, 1000.0, 1.0, 1000.0, 3.14159265359, 1000.0, 1.0, 1000.0, 3.14159265359, 1000.0, 1.0])
Loading checkpoint from embedding.ckpt
/global/cfs/cdirs/m3443/usr/jay/tracking-as-a-service/models/MetricLearning/4/inference.py:135: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  checkpoint = torch.load(embedding_path, map_location="cpu")
Loading checkpoint from filter.ckpt
/global/cfs/cdirs/m3443/usr/jay/tracking-as-a-service/models/MetricLearning/4/inference.py:141: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  checkpoint = torch.load(filtering_path, map_location="cpu")
Loading checkpoint from gnn.ckpt
/global/cfs/cdirs/m3443/usr/jay/tracking-as-a-service/models/MetricLearning/4/inference.py:148: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  checkpoint = torch.load(gnn_path, map_location="cpu")
Is a recurrent GNN?: False
Use ChainedInteractionGNN2
start a warm-up run.
/global/cfs/cdirs/m3443/usr/jay/tracking-as-a-service/models/MetricLearning/4/inference.py:578: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  node_features = torch.load(args.input)
after embedding, shape = torch.Size([271663, 12])
embedding data tensor([-0.0179, -0.1575, -0.2453,  0.1175,  0.1023, -0.1150,  0.0017, -0.5621,
        -0.0761, -0.0219,  0.7445, -0.0280], device='cuda:0')
embedding data type torch.float32
Number of edges after embedding: 9,279,672
after removing duplications: 4,639,836
edge_score tensor([9.1876e-01, 2.4591e-04, 8.8845e-02, 8.7648e-04, 1.8766e-03, 1.9007e-04,
        1.7902e-04, 6.0699e-03, 2.0114e-04, 2.7147e-04], device='cuda:0')
edge_index tensor([[14424, 14435, 85205,     3,     3,     5,     1, 14424, 14435, 14457],
        [    1,     1,     2,     5,     6,     6,     7,     7,     7,     7]],
       device='cuda:0')
Number of edges after filtering: 527,051
After GNN...
the graph information
Number of tracks found by CC: 3068
Number of tracks found by Walkthrough: 592
track_candidates [14424     1 14508    78 14581 14649 21910 21986 22055 22135    -1     2
    81   165    -1     6    80   161   231   308]
total tracks 3660

real    0m22.832s
user    0m35.496s
sys     0m5.315s
```
