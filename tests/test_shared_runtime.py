import hashlib
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import networkx as nx
import torch

from taas.graph import TrackingGraph
from taas.graph_construction._frnn import build_edges
from taas.graph_construction.double_metric_learning import DoubleMetricLearningGraphConstructor
from taas.graph_construction.metric_learning import MetricLearningGraphConstructor
from taas.graph_segmentation.base import FunctionalGraphSegmenter
from taas.graph_segmentation.networkx_walkthrough import get_simple_path, get_tracks
from taas.graph_segmentation.fastwalkthrough import (
    FastWalkthroughProfile,
    max_add_cuts,
)
from taas.edge_classification.filter import FilterProfile
from torch_geometric.data import Data


REPO_ROOT = Path(__file__).resolve().parents[1]


class SharedRuntimeTest(unittest.TestCase):
    def test_tracking_graph_preserves_tensor_device_and_counts(self):
        nodes = torch.zeros((4, 3))
        edges = torch.tensor([[0, 1, 2], [1, 2, 3]])
        graph = TrackingGraph(nodes, edges)
        self.assertEqual(graph.num_nodes, 4)
        self.assertEqual(graph.num_edges, 3)
        self.assertEqual(graph.device, nodes.device)

    def test_frnn_edge_order_and_self_loop_removal(self):
        fake_frnn = types.SimpleNamespace()

        def frnn_grid_points(**kwargs):
            indexes = torch.tensor([[[0, 2, -1], [1, 0, -1]]])
            return None, indexes, None, None

        fake_frnn.frnn_grid_points = frnn_grid_points
        query = torch.zeros((2, 3))
        with mock.patch.dict(sys.modules, {"frnn": fake_frnn}):
            edges = build_edges(query, query, r_max=0.1, k_max=3)
            metric_edges = MetricLearningGraphConstructor.build_edges(
                query, query, r_max=0.1, k_max=3
            )
            double_metric_edges = DoubleMetricLearningGraphConstructor.build_edges(
                query, query, r_max=0.1, k_max=3
            )
        self.assertTrue(torch.equal(edges, torch.tensor([[0, 1], [2, 0]])))
        self.assertTrue(torch.equal(metric_edges, edges))
        self.assertTrue(torch.equal(double_metric_edges, edges))

    def test_networkx_walkthrough_keeps_historical_threshold_semantics(self):
        graph = nx.DiGraph()
        for node in range(4):
            graph.add_node(node, hit_id=100 + node)
        graph.add_edge(0, 1, score=0.8)
        graph.add_edge(1, 2, score=0.7)
        graph.add_edge(1, 3, score=0.2)
        tracks = get_tracks(graph, 0.1, 0.6, "score")
        self.assertEqual(tracks, [[100, 101, 102]])

        simple_graph = nx.DiGraph()
        for node in range(3):
            simple_graph.add_node(node, hit_id=10 + node)
        simple_graph.add_edges_from([(0, 1), (1, 2)])
        simple, remainder = get_simple_path(simple_graph)
        self.assertEqual(simple, [[10, 11, 12]])
        self.assertEqual(remainder.number_of_nodes(), 0)

    def test_functional_segmenter_binds_parameters_once(self):
        segmenter = FunctionalGraphSegmenter(lambda graph, cut: (graph, cut), 0.5)
        self.assertEqual(segmenter.segment("graph"), ("graph", 0.5))


class ConsolidationContractTest(unittest.TestCase):
    def test_tracking_models_only_deploy_version_one(self):
        for model_name in (
            "MetricLearning",
            "DoubleMetricLearning",
            "GNN4Pixel",
            "ModuleMap",
        ):
            model_dir = REPO_ROOT / "models" / model_name
            versions = sorted(
                path.name for path in model_dir.iterdir()
                if path.is_dir() and path.name.isdigit()
            )
            self.assertEqual(versions, ["1"], model_name)

    def test_fastwalk_profiles_preserve_current_incoming_edge_cut(self):
        graph = Data(
            edge_index=torch.tensor([[0, 1], [2, 2]]),
            hit_id=torch.arange(3),
            hit_r=torch.arange(3, dtype=torch.float32),
            hit_z=torch.zeros(3),
            num_nodes=3,
        )
        graph.scores = torch.tensor([0.8, 0.7])

        legacy = max_add_cuts(
            graph.clone(),
            "scores",
            0.1,
            0.9,
            FastWalkthroughProfile.LEGACY_LONGEST,
        )
        current = max_add_cuts(
            graph.clone(),
            "scores",
            0.1,
            0.9,
            FastWalkthroughProfile.CURRENT,
        )

        self.assertEqual(legacy.num_edges, 2)
        self.assertEqual(current.num_edges, 1)

    def test_profiles_replace_versioned_implementation_files(self):
        self.assertEqual(FilterProfile.METRIC_LEGACY.value, "metric_legacy")
        self.assertEqual(
            FastWalkthroughProfile.LEGACY_FIRST.value, "legacy_first"
        )
        removed = [
            "taas/edge_classification/filter_metric_v1.py",
            "taas/edge_classification/filter_metric_v2.py",
            "taas/edge_classification/filter_double_metric_legacy.py",
            "taas/graph_segmentation/fastwalkthrough_gnn4pixel.py",
            "taas/graph_segmentation/fastwalkthrough_metric_v2.py",
            "taas/graph_segmentation/networkx_walkthrough_topological.py",
            "taas/torch_models/double_metric_learning_legacy.py",
            "taas/torch_models/metric_learning_v3.py",
            "taas/torch_models/interaction_gnn_metric_v3.py",
        ]
        for relative_path in removed:
            self.assertFalse((REPO_ROOT / relative_path).exists(), relative_path)

    def test_modulemap_has_thin_triton_adapter(self):
        model_source = (REPO_ROOT / "models/ModuleMap/1/model.py").read_text()
        inference_source = (
            REPO_ROOT / "models/ModuleMap/1/inference.py"
        ).read_text()

        self.assertIn("class TritonPythonModel", model_source)
        self.assertIn("ModuleMapInference(config)", model_source)
        self.assertNotIn("def _run_edge_classifier", model_source)
        self.assertIn("class ModuleMapInference", inference_source)
        self.assertNotIn("triton_python_backend_utils", inference_source)



class TritonConfigContractTest(unittest.TestCase):
    EXPECTED_SHA256 = {
        "models/MetricLearning/config.pbtxt":
            "dc7693413e5632c8c998ef657d5b820d4dd8a21f15dad4899aa722eb4c371837",
        "models/DoubleMetricLearning/config.pbtxt":
            "f1afbd8ff356159be03dece75762db6fcf0a8ca15d250607a8fc1f2a4c3a75f2",
        "models/GNN4Pixel/config.pbtxt":
            "5e3e4d5687f7c9973f14b6d473da3598e15ccdcb56bdc0171f4098c0f3bc5125",
        "models/ModuleMap/config.pbtxt":
            "9f5cd3b9d011f1fae15239849e91f92b204150eadc321484872e777a457d8d71",
    }

    def test_model_configs_match_deployed_contracts(self):
        for relative_path, expected in self.EXPECTED_SHA256.items():
            digest = hashlib.sha256((REPO_ROOT / relative_path).read_bytes()).hexdigest()
            self.assertEqual(digest, expected, relative_path)


if __name__ == "__main__":
    unittest.main()
