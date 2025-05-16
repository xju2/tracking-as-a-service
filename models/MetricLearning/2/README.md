Debug information:
Running on the event 6800 and printing the following information:

```text
(.venv) [xju@login07] 2 >python inference.py -i ../all_input_node_features.pt -v
MetricLearningInferenceConfig(model_path=PosixPath('.'), device='cuda', auto_cast=False, compling=False, debug=True, save_debug_data=True, r_max=0.12, k_max=1000, filter_cut=0.05, filter_batches=10, cc_cut=0.01, walk_min=0.1, walk_max=0.6, embedding_node_features=['r', 'phi', 'z', 'cluster_x_1', 'cluster_y_1', 'cluster_z_1', 'cluster_x_2', 'cluster_y_2', 'cluster_z_2', 'count_1', 'charge_count_1', 'loc_eta_1', 'loc_phi_1', 'localDir0_1', 'localDir1_1', 'localDir2_1', 'lengthDir0_1', 'lengthDir1_1', 'lengthDir2_1', 'glob_eta_1', 'glob_phi_1', 'eta_angle_1', 'phi_angle_1', 'count_2', 'charge_count_2', 'loc_eta_2', 'loc_phi_2', 'localDir0_2', 'localDir1_2', 'localDir2_2', 'lengthDir0_2', 'lengthDir1_2', 'lengthDir2_2', 'glob_eta_2', 'glob_phi_2', 'eta_angle_2', 'phi_angle_2'], embedding_node_scale=[1000.0, 3.14, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1.0, 1.0, 3.14, 3.14, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.14, 3.14, 3.14, 3.14, 1.0, 1.0, 3.14, 3.14, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.14, 3.14, 3.14, 3.14], filter_node_features=['r', 'phi', 'z', 'cluster_x_1', 'cluster_y_1', 'cluster_z_1', 'cluster_x_2', 'cluster_y_2', 'cluster_z_2', 'count_1', 'charge_count_1', 'loc_eta_1', 'loc_phi_1', 'localDir0_1', 'localDir1_1', 'localDir2_1', 'lengthDir0_1', 'lengthDir1_1', 'lengthDir2_1', 'glob_eta_1', 'glob_phi_1', 'eta_angle_1', 'phi_angle_1', 'count_2', 'charge_count_2', 'loc_eta_2', 'loc_phi_2', 'localDir0_2', 'localDir1_2', 'localDir2_2', 'lengthDir0_2', 'lengthDir1_2', 'lengthDir2_2', 'glob_eta_2', 'glob_phi_2', 'eta_angle_2', 'phi_angle_2'], filter_node_scale=[1000.0, 3.14, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1.0, 1.0, 3.14, 3.14, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.14, 3.14, 3.14, 3.14, 1.0, 1.0, 3.14, 3.14, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.14, 3.14, 3.14, 3.14], gnn_node_features=['r', 'phi', 'z', 'eta', 'cluster_r_1', 'cluster_phi_1', 'cluster_z_1', 'cluster_eta_1', 'cluster_r_2', 'cluster_phi_2', 'cluster_z_2', 'cluster_eta_2'], gnn_node_scale=[1000.0, 3.14159265359, 1000.0, 1.0, 1000.0, 3.14159265359, 1000.0, 1.0, 1000.0, 3.14159265359, 1000.0, 1.0])
start a warm-up run.
/pscratch/sd/x/xju/ITk/ForFinalPaper/tracking-as-a-service/models/MetricLearning/2/inference.py:447: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  node_features = torch.load(args.input)
after embedding, shape = torch.Size([271663, 12])
embedding data tensor([ 0.1448, -0.1404, -0.2562,  0.0458,  0.0390, -0.8959, -0.2210, -0.1311,
         0.0817, -0.0497, -0.0924, -0.0613], device='cuda:0')
embedding data type torch.float32
Number of edges after embedding: 9,300,730
after removing duplications: 4,650,365
edge_score tensor([9.7556e-01, 2.0094e-04, 9.9307e-02, 7.6831e-04, 2.7306e-04, 1.0565e-04,
        7.6173e-05, 5.3496e-04, 1.1082e-04, 2.3861e-04], device='cuda:0')
edge_index tensor([[14424, 14435, 85205,     3,     3,     5,     1, 14424, 14435, 14457],
        [    1,     1,     2,     5,     6,     6,     7,     7,     7,     7]],
       device='cuda:0')
Number of edges after filtering: 652,337
After GNN...
the graph information
Number of tracks found by CC: 2834
Number of tracks found by Walkthrough: 722
track_candidates [    6    80   161   231   308 21969 23455 22034 23524 22111 22202 50155
 50212 67368 77663    -1 85205    11    86   166]
total tracks 3556
```
