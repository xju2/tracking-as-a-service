#!/usr/bin/env python
import sys
from pathlib import Path

import numpy as np
import torch
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype

rng_generator = np.random.default_rng(12345)


def labels_to_candidates(labels):
    track_candidates = []
    this_track = []
    for sp_idx in labels:
        if sp_idx == -1:
            track_candidates.append(this_track)
            this_track = []
            continue
        this_track.append(sp_idx)
    return track_candidates


def test_ExatrkX4PixelPython(host: str, port: int, input_fname="node_features.pt"):
    if port not in {8000, 8001}:
        print(f"Invalid port: {port}")
        sys.exit(1)

    model_name = "MetricLearning"
    shape = [10, 44]
    input_fname = Path(input_fname)
    if input_fname.exists():
        print(f"Loading input data from {input_fname}.")
        input_data = torch.load(input_fname, map_location="cpu").float().cpu().numpy()
    else:
        print("Generating random input data.")
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

    num_tracks = len(labels_to_candidates(output0_data))
    print(f"FEATURES: {input_data.shape}")
    print(f"Found: {num_tracks} tracks.")
    print(f"Finished: {model_name}")

    sys.exit(0)


if __name__ == "__main__":
    import argparse

    args = argparse.ArgumentParser()
    args.add_argument("-s", "--server", type=str, default="localhost")
    args.add_argument("-p", "--port", type=int, default=8001)
    args.add_argument("-i", "--input", type=str, default="node_features.pt")
    args = args.parse_args()

    test_ExatrkX4PixelPython(args.server, args.port, args.input)
