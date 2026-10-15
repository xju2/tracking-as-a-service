"""ONNX session factory."""

from taas.edge_classification.interaction_gnn_onnx import InteractionGNNOnnx


def create_onnx_session(model_path, device_id=0, use_cuda=True):
    return InteractionGNNOnnx(model_path, device_id=device_id, use_cuda=use_cuda)
