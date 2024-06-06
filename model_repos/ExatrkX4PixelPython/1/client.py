import sys

import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype

model_name = "ExatrkX4PixelPython"
shape = [10, 15]

rng_generator = np.random.default_rng(12345)
with httpclient.InferenceServerClient("localhost:8000") as client:
    feature_data = rng_generator.random(shape).astype(np.float32)
    inputs = [
        httpclient.InferInput(
            "FEATURES", feature_data.shape, np_to_triton_dtype(feature_data.dtype)
        ),
    ]

    inputs[0].set_data_from_numpy(feature_data)

    outputs = [
        httpclient.InferRequestedOutput("LABELS"),
    ]

    response = client.infer(model_name, inputs, request_id=str(1), outputs=outputs)

    result = response.get_response()
    output0_data = response.as_numpy("LABELS")

    print(f"FEATURES: {feature_data.shape}")
    print(f"LABELS: {output0_data.shape}")

    print(f"Finished: {model_name}")
    sys.exit(0)
