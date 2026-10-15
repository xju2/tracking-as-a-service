"""ONNX Runtime InteractionGNN with GPU I/O binding."""

from __future__ import annotations

from typing import Sequence
import warnings

import numpy as np
import torch

from taas.edge_classification.base import EdgeClassifier


class InteractionGNNOnnx(EdgeClassifier):
    """Own one ONNX session and preserve tensors on CUDA when CUDA EP is active."""

    def __init__(
        self,
        model_path: str,
        device_id: int = 0,
        use_cuda: bool = True,
        apply_sigmoid: bool = True,
    ) -> None:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        providers: Sequence[object] = ["CPUExecutionProvider"]
        if use_cuda and torch.cuda.is_available():
            providers = [
                ("CUDAExecutionProvider", {"device_id": device_id}),
                "CPUExecutionProvider",
            ]
        self.session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=list(providers)
        )
        self.input_names = [item.name for item in self.session.get_inputs()]
        self.output_name = self.session.get_outputs()[0].name
        self.device_id = device_id
        self.apply_sigmoid = apply_sigmoid
        self.cuda_active = (
            use_cuda
            and torch.cuda.is_available()
            and "CUDAExecutionProvider" in self.session.get_providers()
        )
        if use_cuda and not self.cuda_active:
            warnings.warn(
                "ONNX CUDA is unavailable; falling back to ONNX CPU.",
                RuntimeWarning,
                stacklevel=2,
            )

    def score(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs = [node_features, edge_index]
        if edge_features is not None:
            inputs.append(edge_features)
        if len(inputs) != len(self.input_names):
            raise ValueError(
                f"ONNX InteractionGNN expects {len(self.input_names)} inputs, "
                f"received {len(inputs)}"
            )
        if self.cuda_active:
            contiguous = [tensor.contiguous() for tensor in inputs]
            output = torch.empty(
                (edge_index.shape[1],),
                dtype=torch.float32,
                device=node_features.device,
            )
            binding = self.session.io_binding()
            dtypes = {torch.float32: np.float32, torch.int64: np.int64}
            for name, tensor in zip(self.input_names, contiguous):
                binding.bind_input(
                    name=name,
                    device_type="cuda",
                    device_id=self.device_id,
                    element_type=dtypes[tensor.dtype],
                    shape=tuple(tensor.shape),
                    buffer_ptr=tensor.data_ptr(),
                )
            binding.bind_output(
                name=self.output_name,
                device_type="cuda",
                device_id=self.device_id,
                element_type=np.float32,
                shape=tuple(output.shape),
                buffer_ptr=output.data_ptr(),
            )
            self.session.run_with_iobinding(binding)
            scores = output.reshape(-1)
            return torch.sigmoid(scores) if self.apply_sigmoid else scores

        feed = {
            name: tensor.detach().cpu().numpy()
            for name, tensor in zip(self.input_names, inputs)
        }
        output = self.session.run([self.output_name], feed)[0]
        scores = torch.from_numpy(output).to(node_features.device).reshape(-1)
        return torch.sigmoid(scores) if self.apply_sigmoid else scores
