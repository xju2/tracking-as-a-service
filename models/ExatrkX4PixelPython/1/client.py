#!/usr/bin/env python
import sys
from pathlib import Path

import numpy as np
import torch
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype

rng_generator = np.random.default_rng(12345)


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
    sys.exit(0)


if __name__ == "__main__":
    host = "nid003420"
    port = 8001
    test_ExatrkX4PixelPython(host, port)
