#!/usr/bin/env python
import sys
import json
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


def test_ExaTrkX(host: str, port: int, input_fname="node_features.pt", allow_random=False):
    if port not in {8000, 8001}:
        print(f"Invalid port: {port}")
        sys.exit(1)

    model_name = "ModuleMap"
    shape = [10, 16]
    input_fname = Path(input_fname)
    if input_fname.exists():
        print(f"Loading input data from {input_fname}.")
        input_data = torch.load(input_fname, map_location="cpu").float().cpu().numpy()
    else:
        if not allow_random:
            raise FileNotFoundError(
                f"Input file not found: {input_fname}. "
                "Pass a valid -i path or use --allow-random for synthetic input."
            )
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

    def load_server_endpoint(repo_root: Path):
        endpoint_files = [
            repo_root / "triton_ready.txt",
            repo_root / "jobs" / "tracking_ready.json",
            repo_root / "jobs" / "node_id.txt",
        ]

        for endpoint_file in endpoint_files:
            if not endpoint_file.exists():
                continue

            text = endpoint_file.read_text().strip()
            if not text:
                continue

            if text.startswith("{"):
                data = json.loads(text)
                return data["url"], int(data.get("port", 8001))

            return text, 8001

        raise FileNotFoundError(
            "Could not find Triton endpoint file. Expected one of: "
            "triton_ready.txt, jobs/tracking_ready.json, jobs/node_id.txt"
        )

    args = argparse.ArgumentParser()
    args.add_argument("-i", "--input", type=str, default="node_features.pt")
    args.add_argument(
        "--allow-random",
        action="store_true",
        help="Generate random input when --input file does not exist.",
    )
    args = args.parse_args()

    # Find the server endpoint.
    this_file_location = Path(__file__).resolve()
    repo_root = this_file_location.parent.parent.parent.parent
    server, port = load_server_endpoint(repo_root)
    print(f"Using server {server} with port {port}.")

    test_ExaTrkX(server, port, args.input, allow_random=args.allow_random)
