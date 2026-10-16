#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <MMG/CUDA_edge_layer_connector>
#include <MMG/CUDA_graph>

#include <stdexcept>
#include <vector>

namespace {

void check_cuda_tensor(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

std::vector<torch::Tensor> edge_layer_connector_cuda(
    torch::Tensor hit_ids,
    torch::Tensor edge_src,
    torch::Tensor edge_dst,
    torch::Tensor edge_scores,
    double weights_cut,
    int64_t min_hits,
    int64_t block_size,
    int64_t max_hits_per_track) {
  check_cuda_tensor(hit_ids, "hit_ids");
  check_cuda_tensor(edge_src, "edge_src");
  check_cuda_tensor(edge_dst, "edge_dst");
  check_cuda_tensor(edge_scores, "edge_scores");
  TORCH_CHECK(hit_ids.scalar_type() == torch::kInt32, "hit_ids must be int32");
  TORCH_CHECK(edge_src.scalar_type() == torch::kInt32, "edge_src must be int32");
  TORCH_CHECK(edge_dst.scalar_type() == torch::kInt32, "edge_dst must be int32");
  TORCH_CHECK(
      edge_scores.scalar_type() == torch::kFloat32,
      "edge_scores must be float32");
  TORCH_CHECK(edge_src.numel() == edge_dst.numel(), "edge_src/dst size mismatch");
  TORCH_CHECK(
      edge_src.numel() == edge_scores.numel(),
      "edge_src/scores size mismatch");
  TORCH_CHECK(min_hits >= 1, "min_hits must be >= 1");
  TORCH_CHECK(block_size >= 1, "block_size must be >= 1");
  TORCH_CHECK(max_hits_per_track >= min_hits, "max_hits_per_track < min_hits");

  const c10::cuda::CUDAGuard device_guard(hit_ids.device());
  TORCH_CHECK(edge_src.device() == hit_ids.device(), "edge_src device mismatch");
  TORCH_CHECK(edge_dst.device() == hit_ids.device(), "edge_dst device mismatch");
  TORCH_CHECK(edge_scores.device() == hit_ids.device(), "edge_scores device mismatch");

  const auto num_hits = static_cast<int>(hit_ids.numel());
  const auto num_edges = static_cast<int>(edge_src.numel());
  if (num_hits == 0 || num_edges == 0) {
    auto empty_i32 = torch::empty({0}, hit_ids.options());
    return {empty_i32, empty_i32, torch::empty({0}, hit_ids.options())};
  }

  CUDA_graph<float> graph(
      hit_ids.data_ptr<int>(),
      num_hits,
      edge_src.data_ptr<int>(),
      edge_dst.data_ptr<int>(),
      edge_scores.data_ptr<float>(),
      num_edges);

  CUDA_edge_layer_connector<float> connector(
      &graph,
      static_cast<float>(weights_cut),
      static_cast<int>(min_hits),
      static_cast<int>(block_size),
      static_cast<int>(max_hits_per_track));

  connector.build_tracks();
  if (!connector.cuda_tracks() || connector.cuda_tracks()->nb_tracks() == 0) {
    auto empty_i32 = torch::empty({0}, hit_ids.options());
    return {empty_i32, empty_i32, torch::empty({0}, hit_ids.options())};
  }
  connector.filter();
  C10_CUDA_CHECK(cudaGetLastError());

  const int nb_tracks = connector.cuda_tracks()->nb_tracks();
  const int tracks_size = connector.cuda_tracks()->size();
  if (nb_tracks == 0 || tracks_size == 0) {
    auto empty_i32 = torch::empty({0}, hit_ids.options());
    return {empty_i32, empty_i32, torch::empty({0}, hit_ids.options())};
  }

  auto counts = torch::empty({nb_tracks}, hit_ids.options());
  auto padded_hits = torch::empty({tracks_size}, hit_ids.options());
  auto metadata = torch::empty({0}, hit_ids.options());
  auto stream = at::cuda::getCurrentCUDAStream();

  C10_CUDA_CHECK(cudaMemcpyAsync(
      counts.data_ptr<int>(),
      connector.cuda_tracks()->nb_hits(),
      nb_tracks * sizeof(int),
      cudaMemcpyDeviceToDevice,
      stream.stream()));
  C10_CUDA_CHECK(cudaMemcpyAsync(
      padded_hits.data_ptr<int>(),
      connector.cuda_tracks()->hits(),
      tracks_size * sizeof(int),
      cudaMemcpyDeviceToDevice,
      stream.stream()));
  C10_CUDA_CHECK(cudaStreamSynchronize(stream.stream()));
  C10_CUDA_CHECK(cudaGetLastError());
  return {padded_hits, counts, metadata};
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "edge_layer_connector_cuda",
      &edge_layer_connector_cuda,
      "ModuleMapGraph EdgeLayerConnector CUDA wrapper");
}
