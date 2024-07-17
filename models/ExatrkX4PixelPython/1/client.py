#!/usr/bin/env python
import sys
from pathlib import Path

import numpy as np
import torch
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype

rng_generator = np.random.default_rng(12345)


def labels_to_candidates(labels, num_nodes):
    num_tracks = np.max(labels) + 1
    track_candidates = [[] for _ in range(num_tracks)]
    for idx in range(num_nodes):
        label = labels[idx]
        if label == -1:
            continue
        track_candidates[label].append(idx)
    return track_candidates


def test_ExatrkX4PixelPython(host: str, port: int):
    if port not in {8000, 8001}:
        print(f"Invalid port: {port}")
        sys.exit(1)

    model_name = "ExatrkX4PixelPython"
    shape = [10, 15]
    input_fname = Path("node_features.pt")
    if input_fname.exists():
        input_data = torch.load(input_fname).float().cpu().numpy()
    else:
        input_data = rng_generator.random(shape).astype(np.float32)

    if port == 8000:
        client = httpclient.InferenceServerClient(f"{host}:{port}")
        inputs = [
            httpclient.InferInput(
                "FEATURES", input_data.shape, np_to_triton_dtype(input_data.dtype)
            )
        ]
        inputs[0].set_data_from_numpy(input_data)
        outputs = [
            httpclient.InferRequestedOutput("LABELS"),
        ]
    else:
        client = grpcclient.InferenceServerClient(f"{host}:{port}")
        inputs = [
            grpcclient.InferInput(
                "FEATURES", input_data.shape, np_to_triton_dtype(input_data.dtype)
            ),
        ]
        inputs[0].set_data_from_numpy(input_data)
        outputs = [
            grpcclient.InferRequestedOutput("LABELS"),
        ]

    response = client.infer(model_name, inputs, request_id=str(1), outputs=outputs)
    output0_data = response.as_numpy("LABELS")

    print(f"FEATURES: {input_data.shape}")
    print(f"LABELS: {output0_data.shape}")
    print(f"Finished: {model_name}")
    candidates = labels_to_candidates(output0_data, input_data.shape[0])
    print(f"Found {len(candidates)} tracks with more than 3 hits.")

    sys.exit(0)


if __name__ == "__main__":
    import argparse

    args = argparse.ArgumentParser()
    args.add_argument("-u", "--host", type=str, default="localhost")
    args.add_argument("-p", "--port", type=int, default=8001)
    args = args.parse_args()

    test_ExatrkX4PixelPython(args.host, args.port)
