# Double Metric Learning Inference with DP-WALK with Checkpoints

This document describes how to run the metric learning inference pipeline using pre-trained models from checkpoint files.

## How to Run

To run the inference script, use the following command from within the `models/DoubleMetricLearningDPW/1` directory:

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

```
