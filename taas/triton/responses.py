"""Triton response construction helpers."""


def inference_error(pb_utils, message):
    return pb_utils.InferenceResponse(error=pb_utils.TritonError(message))
