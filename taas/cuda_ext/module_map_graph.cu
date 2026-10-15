#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_runtime.h>

#include <TFile.h>
#include <TTree.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#if __cplusplus < 202002L
namespace std {
namespace numbers {
template <class T>
inline constexpr T pi_v = static_cast<T>(3.141592653589793238462643383279502884L);
}  // namespace numbers
}  // namespace std
#endif

#include <MMG/CUDA_graph_creator>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr int kModuleMapInputFeatures = 16;
constexpr int kGnnNodeFeatures = 12;
constexpr int kEdgeFeatures = 6;

void check_2d_float_cpu(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(!tensor.is_cuda(), name, " must be a CPU tensor for this wrapper");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.dim() == 2, name, " must be 2D");
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat32, name, " must be float32");
}

void check_1d_int_cpu(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(!tensor.is_cuda(), name, " must be a CPU tensor for this wrapper");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.dim() == 1, name, " must be 1D");
  TORCH_CHECK(
      tensor.scalar_type() == torch::kInt64,
      name,
      " must be int64; values are cast to uint64_t internally");
}

void check_2d_float_cuda(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.dim() == 2, name, " must be 2D");
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat32, name, " must be float32");
}

void check_cuda_clean(const char* label) {
  auto err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, label, ": ", cudaGetErrorString(err));
  err = cudaDeviceSynchronize();
  TORCH_CHECK(err == cudaSuccess, label, " synchronize: ", cudaGetErrorString(err));
}

template <typename T>
class ScopedCudaPtr {
 public:
  ScopedCudaPtr() = default;
  ScopedCudaPtr(std::size_t size, cudaStream_t stream) : size_(size), stream_(stream) {
    if (size_ > 0) {
      C10_CUDA_CHECK(cudaMalloc(&ptr_, size_ * sizeof(T)));
    }
  }
  ScopedCudaPtr(const ScopedCudaPtr&) = delete;
  ScopedCudaPtr& operator=(const ScopedCudaPtr&) = delete;
  ScopedCudaPtr(ScopedCudaPtr&& other) noexcept
      : ptr_(other.ptr_), size_(other.size_), stream_(other.stream_) {
    other.ptr_ = nullptr;
    other.size_ = 0;
  }
  ScopedCudaPtr& operator=(ScopedCudaPtr&& other) noexcept {
    if (this != &other) {
      reset();
      ptr_ = other.ptr_;
      size_ = other.size_;
      stream_ = other.stream_;
      other.ptr_ = nullptr;
      other.size_ = 0;
    }
    return *this;
  }
  ~ScopedCudaPtr() { reset(); }
  T* data() const { return ptr_; }
  std::size_t size() const { return size_; }
  void reset() {
    if (ptr_ != nullptr) {
      cudaFree(ptr_);
      ptr_ = nullptr;
      size_ = 0;
    }
  }

 private:
  T* ptr_ = nullptr;
  std::size_t size_ = 0;
  cudaStream_t stream_ = cudaStreamLegacy;
};

struct RawCudaHits {
  int size = 0;
  float* r = nullptr;
  float* phi = nullptr;
  float* z = nullptr;
  float* x = nullptr;
  float* y = nullptr;
  float* eta = nullptr;
};

struct RawEdgeData {
  int n_edges = 0;
  int* edges = nullptr;
};

struct CastBoolToInt {
  __host__ __device__ int operator()(bool b) const { return b ? 1 : 0; }
};

__global__ void bool_to_int_kernel(const bool* src, int* dst, int64_t n) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    dst[i] = src[i] ? 1 : 0;
  }
}

__device__ float reset_angle(float angle) {
  if (angle > kPi) {
    return angle - 2.0f * kPi;
  }
  if (angle < -kPi) {
    return angle + 2.0f * kPi;
  }
  return angle;
}

__global__ void prepare_module_map_inputs(
    int64_t n_hits,
    const float* features,
    const int* sorted_indices,
    float* node_features,
    int* hit_ids,
    float* graph_r,
    float* graph_phi,
    float* graph_z,
    float* graph_x,
    float* graph_y,
    float* graph_eta) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n_hits) {
    return;
  }

  const int input_index = sorted_indices[i];
  const float* row = features + static_cast<int64_t>(input_index) * kModuleMapInputFeatures;
  const float z = row[2];
  const float r = row[5];
  const float phi = row[6];
  const float eta = row[7];

  hit_ids[i] = input_index;
  graph_r[i] = r;
  graph_phi[i] = phi;
  graph_z[i] = z;
  graph_x[i] = r * cosf(phi);
  graph_y[i] = r * sinf(phi);
  graph_eta[i] = eta;

  float* out = node_features + i * kGnnNodeFeatures;
  out[0] = r / 1000.0f;
  out[1] = phi / kPi;
  out[2] = z / 1000.0f;
  out[3] = eta;
  out[4] = row[8] / 1000.0f;
  out[5] = row[9] / kPi;
  out[6] = row[10] / 1000.0f;
  out[7] = row[11];
  out[8] = row[12] / 1000.0f;
  out[9] = row[13] / kPi;
  out[10] = row[14] / 1000.0f;
  out[11] = row[15];
}

__global__ void extract_module_ids_and_indices(
    int64_t n_hits,
    const float* features,
    uint64_t* module_ids,
    int* sorted_indices) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n_hits) {
    return;
  }

  const float module_id = features[i * kModuleMapInputFeatures + 3];
  module_ids[i] = static_cast<uint64_t>(module_id);
  sorted_indices[i] = static_cast<int>(i);
}

__global__ void map_module_ids_to_nb_hits(
    int* nb_hits_on_module,
    int64_t n_hits,
    const uint64_t* module_ids,
    std::size_t module_map_size,
    const uint64_t* module_map_key,
    const int* module_map_val) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n_hits) {
    return;
  }

  const auto module_id = module_ids[i];
  int left = 0;
  int right = static_cast<int>(module_map_size) - 1;
  while (left <= right) {
    int mid = left + (right - left) / 2;
    if (module_map_key[mid] == module_id) {
      atomicAdd(&nb_hits_on_module[module_map_val[mid]], 1);
      return;
    }
    if (module_map_key[mid] < module_id) {
      left = mid + 1;
    } else {
      right = mid - 1;
    }
  }
}

__global__ void make_edge_features_from_nodes(
    int64_t n_edges,
    const int* src_edges,
    const int* dst_edges,
    const float* node_features,
    float* edge_features) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n_edges) {
    return;
  }

  const int src = src_edges[i];
  const int dst = dst_edges[i];
  const float* src_features = node_features + src * kGnnNodeFeatures;
  const float* dst_features = node_features + dst * kGnnNodeFeatures;

  const float dr = dst_features[0] - src_features[0];
  const float dphi = reset_angle(kPi * (dst_features[1] - src_features[1])) / kPi;
  const float dz = dst_features[2] - src_features[2];
  const float deta = dst_features[3] - src_features[3];
  float phi_slope = 0.0f;
  float r_phi_slope = 0.0f;
  if (dr != 0.0f) {
    phi_slope = fminf(fmaxf(dphi / dr, -100.0f), 100.0f);
    const float avg_r = 0.5f * (dst_features[0] + src_features[0]);
    r_phi_slope = avg_r * phi_slope;
  }

  float* out = edge_features + i * kEdgeFeatures;
  out[0] = dr;
  out[1] = dphi;
  out[2] = dz;
  out[3] = deta;
  out[4] = phi_slope;
  out[5] = r_phi_slope;
}

__global__ void copy_i32_to_i64(const int* src, int64_t* dst, int64_t n) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    dst[i] = static_cast<int64_t>(src[i]);
  }
}

__global__ void copy_u64_to_i64(const uint64_t* src, int64_t* dst, int64_t n) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    dst[i] = static_cast<int64_t>(src[i]);
  }
}

__global__ void pack_edge_features(
    int64_t n,
    const float* dR,
    const float* dphi,
    const float* dz,
    const float* deta,
    const float* phi_slope,
    const float* r_phi_slope,
    float* out) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  float* row = out + i * 6;
  row[0] = dR[i];
  row[1] = dphi[i];
  row[2] = dz[i];
  row[3] = deta[i];
  row[4] = phi_slope[i];
  row[5] = r_phi_slope[i];
}

std::vector<uint64_t> int64_tensor_to_u64_vector(const torch::Tensor& tensor) {
  auto* data = tensor.data_ptr<int64_t>();
  std::vector<uint64_t> out(static_cast<std::size_t>(tensor.numel()));
  for (int64_t i = 0; i < tensor.numel(); ++i) {
    TORCH_CHECK(data[i] >= 0, "uint64 field contains negative value at index ", i);
    out[static_cast<std::size_t>(i)] = static_cast<uint64_t>(data[i]);
  }
  return out;
}

torch::Tensor u64_device_to_i64_cpu(const uint64_t* src, int64_t n) {
  auto out = torch::empty(
      {n}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  if (n == 0) {
    return out;
  }
  std::vector<uint64_t> host(static_cast<std::size_t>(n));
  C10_CUDA_CHECK(cudaMemcpy(
      host.data(), src, static_cast<std::size_t>(n) * sizeof(uint64_t), cudaMemcpyDeviceToHost));
  auto* out_data = out.data_ptr<int64_t>();
  for (int64_t i = 0; i < n; ++i) {
    const auto value = host[static_cast<std::size_t>(i)];
    TORCH_CHECK(
        value <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
        "hit ID exceeds int64 range: ",
        value);
    out_data[i] = static_cast<int64_t>(value);
  }
  return out;
}

torch::Tensor i32_device_to_i64_cpu(const int* src, int64_t n) {
  auto out = torch::empty(
      {n}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  if (n == 0) {
    return out;
  }
  std::vector<int> host(static_cast<std::size_t>(n));
  C10_CUDA_CHECK(cudaMemcpy(
      host.data(), src, static_cast<std::size_t>(n) * sizeof(int), cudaMemcpyDeviceToHost));
  auto* out_data = out.data_ptr<int64_t>();
  for (int64_t i = 0; i < n; ++i) {
    out_data[i] = static_cast<int64_t>(host[static_cast<std::size_t>(i)]);
  }
  return out;
}

torch::Tensor i32_device_to_i32_tensor(const int* src, int64_t n, c10::DeviceIndex device_index) {
  auto out = torch::empty(
      {n}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, device_index));
  if (n == 0) {
    return out;
  }
  C10_CUDA_CHECK(cudaMemcpy(
      out.data_ptr<int>(), src, static_cast<std::size_t>(n) * sizeof(int), cudaMemcpyDeviceToDevice));
  return out;
}

std::string strip_suffix(std::string path, const std::string& suffix) {
  if (path.size() >= suffix.size() &&
      path.compare(path.size() - suffix.size(), suffix.size(), suffix) == 0) {
    path.resize(path.size() - suffix.size());
  }
  return path;
}

std::string module_map_base_path(std::string module_map_path) {
  module_map_path = strip_suffix(std::move(module_map_path), ".triplets.root");
  module_map_path = strip_suffix(std::move(module_map_path), ".doublets.root");
  module_map_path = strip_suffix(std::move(module_map_path), ".root");
  return module_map_path;
}

void collect_module_ids_from_tree(
    const std::string& file_path,
    const char* tree_name,
    const std::vector<const char*>& branches,
    std::unordered_set<uint64_t>& ids) {
  TFile file(file_path.c_str(), "READ");
  TORCH_CHECK(!file.IsZombie(), "failed to open ModuleMap file: ", file_path);
  auto* tree = dynamic_cast<TTree*>(file.Get(tree_name));
  TORCH_CHECK(tree != nullptr, "missing tree ", tree_name, " in ", file_path);

  std::vector<ULong_t> values(branches.size(), 0);
  tree->SetBranchStatus("*", 0);
  for (std::size_t i = 0; i < branches.size(); ++i) {
    tree->SetBranchStatus(branches[i], 1);
    const auto status = tree->SetBranchAddress(branches[i], &values[i]);
    TORCH_CHECK(
        status >= 0,
        "failed to bind branch ",
        branches[i],
        " in ",
        file_path,
        "; SetBranchAddress returned ",
        status);
  }

  const Long64_t entries = tree->GetEntries();
  for (Long64_t entry = 0; entry < entries; ++entry) {
    tree->GetEntry(entry);
    for (const auto value : values) {
      ids.insert(static_cast<uint64_t>(value));
    }
  }
}

torch::Tensor load_exact_module_ids(const std::string& module_map_path) {
  TORCH_CHECK(!module_map_path.empty(), "module_map_path must not be empty");
  const auto base = module_map_base_path(module_map_path);
  std::unordered_set<uint64_t> ids;
  ids.reserve(100000);

  collect_module_ids_from_tree(
      base + ".doublets.root", "TreeModuleDoublet", {"Module1", "Module2"}, ids);
  collect_module_ids_from_tree(
      base + ".triplets.root",
      "TreeModuleTriplet",
      {"Module1", "Module2", "Module3"},
      ids);

  std::vector<uint64_t> sorted_ids(ids.begin(), ids.end());
  std::sort(sorted_ids.begin(), sorted_ids.end());
  auto out = torch::empty(
      {static_cast<int64_t>(sorted_ids.size())},
      torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  auto* out_data = out.data_ptr<int64_t>();
  for (std::size_t i = 0; i < sorted_ids.size(); ++i) {
    TORCH_CHECK(
        sorted_ids[i] <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
        "module ID exceeds int64 range: ",
        sorted_ids[i]);
    out_data[i] = static_cast<int64_t>(sorted_ids[i]);
  }
  return out;
}

class ModuleMapGraphCuda {
 public:
  ModuleMapGraphCuda(std::string module_map_path, int64_t gpu_blocks, int64_t gpu_device)
      : module_map_path_(std::move(module_map_path)),
        blocks_(static_cast<int>(gpu_blocks)),
        device_(static_cast<int>(gpu_device)) {
    TORCH_CHECK(!module_map_path_.empty(), "module_map_path must not be empty");
    TORCH_CHECK(blocks_ > 0, "gpu_blocks must be positive");
    TORCH_CHECK(device_ >= 0, "gpu_device must be non-negative");
    const c10::cuda::CUDAGuard device_guard(c10::Device(c10::kCUDA, device_));

    module_map_triplet<float> module_map_cpu;
    module_map_cpu.read_TTree(module_map_path_.c_str());
    TORCH_CHECK(
        !module_map_cpu == false,
        "Cannot retrieve ModuleMap from ",
        module_map_path_);

    cuda_module_map_doublet_ = std::make_unique<CUDA_module_map_doublet<float>>(module_map_cpu);
    cuda_module_map_doublet_->HostToDevice();
    cuda_module_map_triplet_ = std::make_unique<CUDA_module_map_triplet<float>>(module_map_cpu);
    cuda_module_map_triplet_->HostToDevice();

    std::map<uint64_t, int> unique_module_map;
    for (const auto& entry : cuda_module_map_doublet_->module_map()) {
      const auto inserted = unique_module_map.insert({entry.first, entry.second});
      TORCH_CHECK(inserted.second, "Duplicate key in module map: ", entry.first);
    }

    std::vector<uint64_t> keys;
    std::vector<int> vals;
    keys.reserve(unique_module_map.size());
    vals.reserve(unique_module_map.size());
    for (const auto& entry : unique_module_map) {
      keys.push_back(entry.first);
      vals.push_back(entry.second);
    }

    module_map_size_ = keys.size();
    C10_CUDA_CHECK(cudaMalloc(&cuda_module_map_keys_, module_map_size_ * sizeof(uint64_t)));
    C10_CUDA_CHECK(cudaMalloc(&cuda_module_map_vals_, module_map_size_ * sizeof(int)));
    C10_CUDA_CHECK(cudaMemcpy(
        cuda_module_map_keys_,
        keys.data(),
        module_map_size_ * sizeof(uint64_t),
        cudaMemcpyHostToDevice));
    C10_CUDA_CHECK(cudaMemcpy(
        cuda_module_map_vals_,
        vals.data(),
        module_map_size_ * sizeof(int),
        cudaMemcpyHostToDevice));
    C10_CUDA_CHECK(cudaDeviceSynchronize());
  }

  ~ModuleMapGraphCuda() {
    const c10::cuda::CUDAGuard device_guard(c10::Device(c10::kCUDA, device_));
    if (cuda_module_map_keys_ != nullptr) {
      cudaFree(cuda_module_map_keys_);
    }
    if (cuda_module_map_vals_ != nullptr) {
      cudaFree(cuda_module_map_vals_);
    }
    cudaGetLastError();
  }

  std::vector<torch::Tensor> build(torch::Tensor features) const {
    check_2d_float_cuda(features, "features");
    TORCH_CHECK(features.size(1) == kModuleMapInputFeatures, "features must have 16 columns");
    const auto device_index = static_cast<c10::DeviceIndex>(device_);
    TORCH_CHECK(
        features.get_device() == device_index,
        "features are on CUDA device ",
        features.get_device(),
        " but ModuleMapGraphCuda was initialized for device ",
        device_);
    const c10::cuda::CUDAGuard device_guard(c10::Device(c10::kCUDA, device_index));

    const int64_t n_hits_i64 = features.size(0);
    TORCH_CHECK(n_hits_i64 <= std::numeric_limits<int>::max(), "too many hits: ", n_hits_i64);
    const int n_hits = static_cast<int>(n_hits_i64);
    auto i64_opts = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA, device_index);
    auto i32_opts = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, device_index);
    auto f32_opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, device_index);

    if (n_hits == 0) {
      return {
          torch::empty({0, kGnnNodeFeatures}, f32_opts),
          torch::empty({2, 0}, i64_opts),
          torch::empty({0, kEdgeFeatures}, f32_opts),
          torch::empty({0}, i32_opts),
      };
    }

    auto node_features = torch::empty({n_hits_i64, kGnnNodeFeatures}, f32_opts);
    auto hit_ids = torch::empty({n_hits_i64}, i32_opts);
    ScopedCudaPtr<uint64_t> module_ids(n_hits, cudaStreamLegacy);
    ScopedCudaPtr<int> sorted_indices(n_hits, cudaStreamLegacy);
    ScopedCudaPtr<float> graph_features(static_cast<std::size_t>(6) * n_hits, cudaStreamLegacy);

    RawCudaHits hits;
    hits.size = n_hits;
    hits.r = graph_features.data() + 0 * n_hits;
    hits.phi = graph_features.data() + 1 * n_hits;
    hits.z = graph_features.data() + 2 * n_hits;
    hits.x = graph_features.data() + 3 * n_hits;
    hits.y = graph_features.data() + 4 * n_hits;
    hits.eta = graph_features.data() + 5 * n_hits;

    const int threads = 256;
    const int hit_blocks = static_cast<int>((n_hits_i64 + threads - 1) / threads);
    extract_module_ids_and_indices<<<hit_blocks, threads>>>(
        n_hits_i64,
        features.data_ptr<float>(),
        module_ids.data(),
        sorted_indices.data());
    check_cuda_clean("ModuleMapGraphCuda extract module IDs");

    thrust::sort_by_key(
        thrust::device,
        module_ids.data(),
        module_ids.data() + n_hits,
        sorted_indices.data());
    check_cuda_clean("ModuleMapGraphCuda sort hits by module ID");

    prepare_module_map_inputs<<<hit_blocks, threads>>>(
        n_hits_i64,
        features.data_ptr<float>(),
        sorted_indices.data(),
        node_features.data_ptr<float>(),
        hit_ids.data_ptr<int>(),
        hits.r,
        hits.phi,
        hits.z,
        hits.x,
        hits.y,
        hits.eta);
    check_cuda_clean("ModuleMapGraphCuda prepare inputs");

    RawEdgeData edge_data = make_edges(hits, module_ids.data());
    if (edge_data.n_edges == 0) {
      return {
          node_features,
          torch::empty({2, 0}, i64_opts),
          torch::empty({0, kEdgeFeatures}, f32_opts),
          hit_ids,
      };
    }

    const int64_t n_edges = static_cast<int64_t>(edge_data.n_edges);
    auto edge_index = torch::empty({2, n_edges}, i64_opts);
    auto edge_features = torch::empty({n_edges, kEdgeFeatures}, f32_opts);
    const int edge_blocks = static_cast<int>((n_edges + threads - 1) / threads);

    copy_i32_to_i64<<<edge_blocks, threads>>>(
        edge_data.edges, edge_index.data_ptr<int64_t>(), n_edges);
    copy_i32_to_i64<<<edge_blocks, threads>>>(
        edge_data.edges + n_edges, edge_index.data_ptr<int64_t>() + n_edges, n_edges);
    make_edge_features_from_nodes<<<edge_blocks, threads>>>(
        n_edges,
        edge_data.edges,
        edge_data.edges + n_edges,
        node_features.data_ptr<float>(),
        edge_features.data_ptr<float>());
    check_cuda_clean("ModuleMapGraphCuda pack outputs");
    C10_CUDA_CHECK(cudaFree(edge_data.edges));
    cudaGetLastError();

    return {node_features, edge_index, edge_features, hit_ids};
  }

 private:
  RawEdgeData make_edges(const RawCudaHits& hits, const uint64_t* module_ids) const {
    const int nb_doublets = cuda_module_map_doublet_->size();
    const dim3 block_dim(blocks_);
    dim3 grid_dim((nb_doublets + blocks_ - 1) / blocks_);

    ScopedCudaPtr<int> cuda_nb_hits(module_map_size_ + 1, cudaStreamLegacy);
    C10_CUDA_CHECK(cudaMemset(cuda_nb_hits.data(), 0, (module_map_size_ + 1) * sizeof(int)));
    const int hit_blocks = (hits.size + blocks_ - 1) / blocks_;
    map_module_ids_to_nb_hits<<<hit_blocks, block_dim>>>(
        cuda_nb_hits.data(),
        hits.size,
        module_ids,
        module_map_size_,
        cuda_module_map_keys_,
        cuda_module_map_vals_);
    check_cuda_clean("ModuleMapGraphCuda map module IDs");

    thrust::exclusive_scan(
        thrust::device,
        cuda_nb_hits.data(),
        cuda_nb_hits.data() + module_map_size_ + 1,
        cuda_nb_hits.data());
    check_cuda_clean("ModuleMapGraphCuda module hit scan");
    int* cuda_hit_indice = cuda_nb_hits.data();

    ScopedCudaPtr<int> cuda_nb_src_hits_per_doublet(nb_doublets + 1, cudaStreamLegacy);
    count_source_hits_per_doublet<<<grid_dim, block_dim>>>(
        cuda_nb_src_hits_per_doublet.data(),
        cuda_module_map_doublet_->cuda_module1(),
        cuda_hit_indice,
        nb_doublets);
    check_cuda_clean("ModuleMapGraphCuda count source hits");

    thrust::exclusive_scan(
        thrust::device,
        cuda_nb_src_hits_per_doublet.data(),
        cuda_nb_src_hits_per_doublet.data() + nb_doublets + 1,
        cuda_nb_src_hits_per_doublet.data());
    check_cuda_clean("ModuleMapGraphCuda source hit scan");

    int sum_nb_src_hits_per_doublet = 0;
    C10_CUDA_CHECK(cudaMemcpy(
        &sum_nb_src_hits_per_doublet,
        cuda_nb_src_hits_per_doublet.data() + nb_doublets,
        sizeof(int),
        cudaMemcpyDeviceToHost));
    if (sum_nb_src_hits_per_doublet == 0) {
      return {};
    }

    ScopedCudaPtr<int> cuda_edge_sum_per_src_hit(
        sum_nb_src_hits_per_doublet + 1, cudaStreamLegacy);
    dim3 grid_dim_shpd((sum_nb_src_hits_per_doublet + blocks_ - 1) / blocks_);
    count_doublet_edges<float><<<grid_dim_shpd, block_dim>>>(
        cuda_edge_sum_per_src_hit.data(),
        nb_doublets,
        cuda_nb_src_hits_per_doublet.data(),
        cuda_module_map_doublet_->cuda_module1(),
        cuda_module_map_doublet_->cuda_module2(),
        hits.r,
        hits.z,
        hits.eta,
        hits.phi,
        cuda_module_map_doublet_->cuda_z0_min(),
        cuda_module_map_doublet_->cuda_deta_min(),
        cuda_module_map_doublet_->cuda_phi_slope_min(),
        cuda_module_map_doublet_->cuda_dphi_min(),
        cuda_module_map_doublet_->cuda_z0_max(),
        cuda_module_map_doublet_->cuda_deta_max(),
        cuda_module_map_doublet_->cuda_phi_slope_max(),
        cuda_module_map_doublet_->cuda_dphi_max(),
        cuda_hit_indice,
        kPi,
        1e-8f,
        sum_nb_src_hits_per_doublet);
    check_cuda_clean("ModuleMapGraphCuda count doublet edges");

    thrust::exclusive_scan(
        thrust::device,
        cuda_edge_sum_per_src_hit.data(),
        cuda_edge_sum_per_src_hit.data() + sum_nb_src_hits_per_doublet + 1,
        cuda_edge_sum_per_src_hit.data());
    check_cuda_clean("ModuleMapGraphCuda doublet edge scan");

    int nb_doublet_edges = 0;
    C10_CUDA_CHECK(cudaMemcpy(
        &nb_doublet_edges,
        cuda_edge_sum_per_src_hit.data() + sum_nb_src_hits_per_doublet,
        sizeof(int),
        cudaMemcpyDeviceToHost));
    if (nb_doublet_edges == 0) {
      return {};
    }

    ScopedCudaPtr<int> cuda_reduced_m1_hits(nb_doublet_edges, cudaStreamLegacy);
    ScopedCudaPtr<int> cuda_reduced_m2_hits(nb_doublet_edges, cudaStreamLegacy);
    build_doublet_edges<float><<<grid_dim_shpd, block_dim>>>(
        cuda_reduced_m1_hits.data(),
        cuda_reduced_m2_hits.data(),
        nb_doublets,
        cuda_nb_src_hits_per_doublet.data(),
        cuda_hit_indice,
        cuda_edge_sum_per_src_hit.data(),
        cuda_module_map_doublet_->cuda_module1(),
        cuda_module_map_doublet_->cuda_module2(),
        hits.r,
        hits.z,
        hits.eta,
        hits.phi,
        cuda_module_map_doublet_->cuda_z0_min(),
        cuda_module_map_doublet_->cuda_deta_min(),
        cuda_module_map_doublet_->cuda_phi_slope_min(),
        cuda_module_map_doublet_->cuda_dphi_min(),
        cuda_module_map_doublet_->cuda_z0_max(),
        cuda_module_map_doublet_->cuda_deta_max(),
        cuda_module_map_doublet_->cuda_phi_slope_max(),
        cuda_module_map_doublet_->cuda_dphi_max(),
        kPi,
        1e-8f,
        sum_nb_src_hits_per_doublet);
    check_cuda_clean("ModuleMapGraphCuda build doublet edges");

    ScopedCudaPtr<int> cuda_edge_sum(nb_doublets + 1, cudaStreamLegacy);
    doublet_edge_sum<<<grid_dim, block_dim>>>(
        cuda_edge_sum.data(),
        cuda_nb_src_hits_per_doublet.data(),
        cuda_edge_sum_per_src_hit.data(),
        nb_doublets);
    check_cuda_clean("ModuleMapGraphCuda doublet edge sum");

    ScopedCudaPtr<float> cuda_z0(nb_doublet_edges, cudaStreamLegacy);
    ScopedCudaPtr<float> cuda_phi_slope(nb_doublet_edges, cudaStreamLegacy);
    ScopedCudaPtr<float> cuda_deta(nb_doublet_edges, cudaStreamLegacy);
    ScopedCudaPtr<float> cuda_dphi(nb_doublet_edges, cudaStreamLegacy);
    grid_dim = dim3((nb_doublet_edges + blocks_ - 1) / blocks_);
    hits_geometric_cuts<float><<<grid_dim, block_dim>>>(
        cuda_z0.data(),
        cuda_phi_slope.data(),
        cuda_deta.data(),
        cuda_dphi.data(),
        cuda_reduced_m1_hits.data(),
        cuda_reduced_m2_hits.data(),
        hits.r,
        hits.z,
        hits.eta,
        hits.phi,
        kPi,
        1e-8f,
        nb_doublet_edges);
    check_cuda_clean("ModuleMapGraphCuda doublet geometric cuts");

    ScopedCudaPtr<bool> cuda_mask(nb_doublet_edges + 1, cudaStreamLegacy);
    C10_CUDA_CHECK(cudaMemset(cuda_mask.data(), 0, (nb_doublet_edges + 1) * sizeof(bool)));
    const int nb_triplets = cuda_module_map_triplet_->size();
    grid_dim = dim3((nb_triplets + blocks_ - 1) / blocks_);

    ScopedCudaPtr<bool> cuda_vertices(hits.size, cudaStreamLegacy);
    C10_CUDA_CHECK(cudaMemset(cuda_vertices.data(), 0, hits.size * sizeof(bool)));
    ScopedCudaPtr<int> cuda_src_hits_per_triplet(nb_triplets + 1, cudaStreamLegacy);

    count_triplet_hits<<<grid_dim, block_dim>>>(
        cuda_src_hits_per_triplet.data(),
        cuda_module_map_triplet_->cuda_module12_map(),
        cuda_module_map_triplet_->cuda_module23_map(),
        cuda_edge_sum.data(),
        nb_triplets);
    check_cuda_clean("ModuleMapGraphCuda count triplet hits");

    thrust::exclusive_scan(
        thrust::device,
        cuda_src_hits_per_triplet.data(),
        cuda_src_hits_per_triplet.data() + nb_triplets + 1,
        cuda_src_hits_per_triplet.data());
    check_cuda_clean("ModuleMapGraphCuda triplet hit scan");

    int nb_src_hits_per_triplet_sum = 0;
    C10_CUDA_CHECK(cudaMemcpy(
        &nb_src_hits_per_triplet_sum,
        cuda_src_hits_per_triplet.data() + nb_triplets,
        sizeof(int),
        cudaMemcpyDeviceToHost));
    if (nb_src_hits_per_triplet_sum == 0) {
      return {};
    }

    dim3 grid_dim_shpt((nb_src_hits_per_triplet_sum + blocks_ - 1) / blocks_);
    triplet_cuts<float><<<grid_dim_shpt, block_dim>>>(
        cuda_vertices.data(),
        cuda_mask.data(),
        nb_triplets,
        cuda_src_hits_per_triplet.data(),
        cuda_module_map_triplet_->cuda_module12_map(),
        cuda_module_map_triplet_->cuda_module23_map(),
        hits.x,
        hits.y,
        hits.z,
        hits.r,
        cuda_z0.data(),
        cuda_phi_slope.data(),
        cuda_deta.data(),
        cuda_dphi.data(),
        cuda_module_map_triplet_->module12().cuda_z0_min(),
        cuda_module_map_triplet_->module12().cuda_phi_slope_min(),
        cuda_module_map_triplet_->module12().cuda_deta_min(),
        cuda_module_map_triplet_->module12().cuda_dphi_min(),
        cuda_module_map_triplet_->module12().cuda_z0_max(),
        cuda_module_map_triplet_->module12().cuda_phi_slope_max(),
        cuda_module_map_triplet_->module12().cuda_deta_max(),
        cuda_module_map_triplet_->module12().cuda_dphi_max(),
        cuda_module_map_triplet_->module23().cuda_z0_min(),
        cuda_module_map_triplet_->module23().cuda_phi_slope_min(),
        cuda_module_map_triplet_->module23().cuda_deta_min(),
        cuda_module_map_triplet_->module23().cuda_dphi_min(),
        cuda_module_map_triplet_->module23().cuda_z0_max(),
        cuda_module_map_triplet_->module23().cuda_phi_slope_max(),
        cuda_module_map_triplet_->module23().cuda_deta_max(),
        cuda_module_map_triplet_->module23().cuda_dphi_max(),
        cuda_module_map_triplet_->cuda_diff_dydx_min(),
        cuda_module_map_triplet_->cuda_diff_dydx_max(),
        cuda_module_map_triplet_->cuda_diff_dzdr_min(),
        cuda_module_map_triplet_->cuda_diff_dzdr_max(),
        cuda_reduced_m1_hits.data(),
        cuda_reduced_m2_hits.data(),
        cuda_edge_sum.data(),
        kPi,
        1e-8f,
        nb_src_hits_per_triplet_sum);
    check_cuda_clean("ModuleMapGraphCuda triplet cuts");

    ScopedCudaPtr<int> cuda_mask_sum(nb_doublet_edges + 1, cudaStreamLegacy);
    grid_dim = dim3((nb_doublet_edges + 1 + blocks_ - 1) / blocks_);
    bool_to_int_kernel<<<grid_dim, block_dim>>>(
        cuda_mask.data(), cuda_mask_sum.data(), nb_doublet_edges + 1);
    check_cuda_clean("ModuleMapGraphCuda mask bool-to-int");
    thrust::exclusive_scan(
        thrust::device,
        cuda_mask_sum.data(),
        cuda_mask_sum.data() + (nb_doublet_edges + 1),
        cuda_mask_sum.data());
    check_cuda_clean("ModuleMapGraphCuda mask scan");

    int nb_graph_edges = 0;
    C10_CUDA_CHECK(cudaMemcpy(
        &nb_graph_edges,
        cuda_mask_sum.data() + nb_doublet_edges,
        sizeof(int),
        cudaMemcpyDeviceToHost));
    if (nb_graph_edges == 0) {
      return {};
    }

    int* cuda_graph_edges = nullptr;
    C10_CUDA_CHECK(cudaMalloc(&cuda_graph_edges, 2 * nb_graph_edges * sizeof(int)));
    int* cuda_graph_m1_hits = cuda_graph_edges;
    int* cuda_graph_m2_hits = cuda_graph_edges + nb_graph_edges;
    grid_dim = dim3((nb_doublet_edges + blocks_ - 1) / blocks_);
    compact_stream<<<grid_dim, block_dim>>>(
        cuda_graph_m1_hits,
        cuda_reduced_m1_hits.data(),
        cuda_mask.data(),
        cuda_mask_sum.data(),
        nb_doublet_edges);
    compact_stream<<<grid_dim, block_dim>>>(
        cuda_graph_m2_hits,
        cuda_reduced_m2_hits.data(),
        cuda_mask.data(),
        cuda_mask_sum.data(),
        nb_doublet_edges);
    check_cuda_clean("ModuleMapGraphCuda compact graph edges");

    RawEdgeData out;
    out.n_edges = nb_graph_edges;
    out.edges = cuda_graph_edges;
    return out;
  }

  std::string module_map_path_;
  int blocks_ = 512;
  int device_ = 0;
  std::unique_ptr<CUDA_module_map_doublet<float>> cuda_module_map_doublet_;
  std::unique_ptr<CUDA_module_map_triplet<float>> cuda_module_map_triplet_;
  uint64_t* cuda_module_map_keys_ = nullptr;
  int* cuda_module_map_vals_ = nullptr;
  std::size_t module_map_size_ = 0;
};

std::vector<torch::Tensor> build_module_map_graph_cuda(
    torch::Tensor node_features,
    torch::Tensor module_ids,
    torch::Tensor hit_ids,
    const std::string& module_map_path,
    int64_t r_index,
    int64_t phi_index,
    int64_t z_index,
    int64_t gpu_blocks,
    int64_t gpu_device) {
  check_2d_float_cpu(node_features, "node_features");
  check_1d_int_cpu(module_ids, "module_ids");
  check_1d_int_cpu(hit_ids, "hit_ids");
  TORCH_CHECK(module_ids.numel() == node_features.size(0), "module_ids length mismatch");
  TORCH_CHECK(hit_ids.numel() == node_features.size(0), "hit_ids length mismatch");
  TORCH_CHECK(!module_map_path.empty(), "module_map_path must not be empty");
  TORCH_CHECK(gpu_blocks > 0, "gpu_blocks must be positive");
  TORCH_CHECK(gpu_device >= 0, "gpu_device must be non-negative");
  TORCH_CHECK(r_index >= 0 && r_index < node_features.size(1), "r_index out of range");
  TORCH_CHECK(phi_index >= 0 && phi_index < node_features.size(1), "phi_index out of range");
  TORCH_CHECK(z_index >= 0 && z_index < node_features.size(1), "z_index out of range");

  const auto device_index = static_cast<c10::DeviceIndex>(gpu_device);
  const c10::cuda::CUDAGuard device_guard(c10::Device(c10::kCUDA, device_index));
  const auto n_hits = static_cast<int>(node_features.size(0));
  if (n_hits == 0) {
    auto i64_opts = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA, device_index);
    auto f32_opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, device_index);
    auto i64_cpu_opts = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    return {
        torch::empty({2, 0}, i64_opts),
        torch::empty({0, 6}, f32_opts),
        torch::empty({0}, i64_cpu_opts),
        torch::empty({0}, i64_cpu_opts),
    };
  }

  const auto n_features = node_features.size(1);
  const float* features = node_features.data_ptr<float>();
  std::vector<float> x(static_cast<std::size_t>(n_hits));
  std::vector<float> y(static_cast<std::size_t>(n_hits));
  std::vector<float> z(static_cast<std::size_t>(n_hits));
  for (int i = 0; i < n_hits; ++i) {
    const float r = features[i * n_features + r_index];
    const float phi = features[i * n_features + phi_index];
    x[static_cast<std::size_t>(i)] = r * std::cos(phi);
    y[static_cast<std::size_t>(i)] = r * std::sin(phi);
    z[static_cast<std::size_t>(i)] = features[i * n_features + z_index];
  }

  auto hit_ids_u64 = int64_tensor_to_u64_vector(hit_ids);
  auto module_ids_u64 = int64_tensor_to_u64_vector(module_ids);

  auto graph_creator = std::make_unique<CUDA_graph_creator<float, float>>(
      static_cast<int>(gpu_blocks),
      static_cast<int>(gpu_device),
      module_map_path);
  CUDA_nodes<float> nodes;
  CUDA_edges<float> edges;
  graph_creator->build(
      nodes,
      edges,
      "taas",
      hit_ids_u64.data(),
      module_ids_u64.data(),
      x.data(),
      y.data(),
      z.data(),
      n_hits);
  check_cuda_clean("ModuleMapGraph after graph_creator.build");

  const int64_t n_nodes = static_cast<int64_t>(nodes.nb_nodes());
  const int64_t n_edges = static_cast<int64_t>(edges.nb_edges());
  auto i64_opts = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA, device_index);
  auto f32_opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, device_index);
  auto edge_index = torch::empty({2, n_edges}, i64_opts);
  auto edge_features = torch::empty({n_edges, 6}, f32_opts);

  auto stream = at::cuda::getCurrentCUDAStream();
  const int threads = 256;
  const int edge_blocks = static_cast<int>((n_edges + threads - 1) / threads);

  if (n_edges > 0) {
    copy_i32_to_i64<<<edge_blocks, threads, 0, stream.stream()>>>(
        edges.source(), edge_index.data_ptr<int64_t>(), n_edges);
    copy_i32_to_i64<<<edge_blocks, threads, 0, stream.stream()>>>(
        edges.target(), edge_index.data_ptr<int64_t>() + n_edges, n_edges);
    pack_edge_features<<<edge_blocks, threads, 0, stream.stream()>>>(
        n_edges,
        edges.dR(),
        edges.dphi(),
        edges.dz(),
        edges.deta(),
        edges.phi_slope(),
        edges.r_phi_slope(),
        edge_features.data_ptr<float>());
  }
  auto err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "ModuleMapGraph after edge tensor packing: ", cudaGetErrorString(err));
  err = cudaStreamSynchronize(stream.stream());
  TORCH_CHECK(
      err == cudaSuccess,
      "ModuleMapGraph after edge tensor packing synchronize: ",
      cudaGetErrorString(err));

  auto compact_hit_ids = u64_device_to_i64_cpu(nodes.cuda_hitID(), n_nodes);
  auto acorn_positions = i32_device_to_i64_cpu(nodes.cuda_ACORN_positions(), n_nodes);

  edges.kill();
  check_cuda_clean("ModuleMapGraph cleanup after edges.kill");
  nodes.kill();
  check_cuda_clean("ModuleMapGraph cleanup after nodes.kill");
  graph_creator.reset();
  // The ACTS/MMG graph creator owns only temporary buffers at this point. Some
  // CUDA runtimes report a stale cleanup status from its destructor after the
  // output tensors have already been materialized; clear it so it does not make
  // the completed Triton request fail or poison the next request.
  cudaGetLastError();
  return {edge_index, edge_features, compact_hit_ids, acorn_positions};
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "load_exact_module_ids",
      &load_exact_module_ids,
      "Load exact uint64 ModuleMap module IDs as an int64 CPU tensor");
  m.def(
      "build_module_map_graph_cuda",
      &build_module_map_graph_cuda,
      "Build a ModuleMapGraph CUDA edge graph from host node features and module IDs");
  pybind11::class_<ModuleMapGraphCuda>(m, "ModuleMapGraphCuda")
      .def(pybind11::init<std::string, int64_t, int64_t>())
      .def("build", &ModuleMapGraphCuda::build);
}
