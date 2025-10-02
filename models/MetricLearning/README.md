Metric Learning pipeline for GNN tracking.
A brief description of different versions of the pipeline:
1. The filtering stage is with the random flipping.
2. Re-trained the filtering and GNN stages without the random flipping. Using v5 dataset
3. Use v9 dataset for training, use pytorch checkpoints instead of torchscript for inference.
4. Add "look-back" in the `fastwalkthrough.py`.
