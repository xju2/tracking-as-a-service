from __future__ import annotations

import json
import os

import cudf
import cugraph
import frnn
import numpy as np
import pandas as pd
import torch
import triton_python_backend_utils as pb_utils
from torch.utils.dlpack import from_dlpack

os.environ["RAPIDS_NO_INITIALIZE"] = "1"


def build_edges(
    query: torch.Tensor,
    database: torch.Tensor,
    r_max: float = 0.1,
    k_max: int = 1000,
) -> torch.Tensor:
    # Compute edges
    _, idxs, _, _ = frnn.frnn_grid_points(
        points1=query.unsqueeze(0),
        points2=database.unsqueeze(0),
        lengths1=None,
        lengths2=None,
        K=k_max,
        r=r_max,
        grid=None,
        return_nn=False,
        return_sorted=True,
    )

    idxs: torch.Tensor = idxs.squeeze().int()
    ind = torch.arange(idxs.shape[0], device=query.device).repeat(idxs.shape[1], 1).T.int()
    positive_idxs = idxs >= 0
    edge_list = torch.stack([ind[positive_idxs], idxs[positive_idxs]]).long()

    # Remove self-loops
    edge_list = edge_list[:, edge_list[0] != edge_list[1]]
    return edge_list


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to intialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args["model_config"])
        self.model_instance_device_id = json.loads(args["model_instance_device_id"])

        self.embedding_model = torch.jit.load(model_config["embedding_fname"])
        self.embedding_model.eval()

        self.r_max = model_config["r_max"]
        self.k_max = model_config["k_max"]

        self.gnn_model = torch.jit.load(model_config["gnn_fname"])
        self.gnn_model.eval()

        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(model_config, "LABELS")

        # Convert Triton types to numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(output0_config["data_type"])

    def execute(self, requests):
        """`execute` MUST be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference request is made
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse.

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """
        output0_dtype = self.output0_dtype

        responses = []
        # print(torch.cuda.is_available())
        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:
            features = pb_utils.get_input_tensor_by_name(request, "FEATURES")
            features = from_dlpack(features.to_dlpack()).to(self.model_instance_device_id)

            # Embedding model
            embedding = self.embedding_model(features)

            edge_list = build_edges(embedding, embedding, r_max=self.r_max, k_max=self.k_max)

            # GNN model
            edge_list = edge_list.to(self.model_instance_device_id)
            edge_score = self.gnn_model(embedding, edge_list).squeeze().sigmoid()

            # connected components and track labeling
            num_nodes = embedding.shape[0]
            cut_edges = edge_list[:, edge_score > 0.75]

            if cut_edges.shape[1] > 0:
                cut_df = cudf.DataFrame(cut_edges.T)
                G = cugraph.Graph()
                G.from_cudf_edgelist(
                    cut_df, source=0, destination=1, edge_attr=None, renumber=False
                )
                labels = cugraph.components.connectivity.weakly_connected_components(G)
                labels = labels.to_pandas()

                # label those unlabeled nodes with the same number
                all_vertex = pd.DataFrame(np.arange(num_nodes), columns=["vertex"])
                max_label = np.max(labels.labels)
                labels = labels.merge(all_vertex, on="vertex", how="right").fillna(max_label + 1)

                out_0 = labels["labels"].to_numpy()
            else:
                out_0 = np.arange(num_nodes)

            # Create output tensors. You need pb_utils.Tensor
            # objects to create pb_utils.InferenceResponse.
            out_tensor_0 = pb_utils.Tensor("LABELS", out_0.astype(output0_dtype))
            # Create InferenceResponse. You can set an error here in case
            # there was a problem with handling this inference request.
            # Below is an example of how you can set errors in inference
            # response:
            #
            # pb_utils.InferenceResponse(
            #    output_tensors=..., TritonError("An error occured"))
            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor_0])
            responses.append(inference_response)

        # You should return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print("Cleaning up...")
