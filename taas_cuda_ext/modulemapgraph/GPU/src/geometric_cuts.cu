/***********************************************************************
 * Module Map Graph library
 * Track reconstruction for HEP
 * copyright © 2024,2025 COLLARD Christophe
 * Laboratoire des 2 Infinis - Toulouse - France (L2IT)
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 *
 * contributors:
 * 2025 HUTH Benjamin
 **********************************************************************/


// If USE_CUDA_LAUNCH_BOUNDS is enabled, we try to force the compiler to optimize
// the allocation of registers, to fit a certain amount of blocks on a SM
// Note that this requires hardcoding the blockdim at compiletime
#ifdef USE_CUDA_LAUNCH_BOUNDS
#define MMG_LAUNCH_BOUNDS(MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP) __launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP)
#else
#define MMG_LAUNCH_BOUNDS(blockdim, blocks) /*nothing*/
#endif

// This file contains device functions and kernels to implement the geometetric cuts
// in with an additional loop unrolling

//=====GPU Device Kernels==========GPU Device Kernels==========GPU Device Kernels==========GPU Device Kernels==========

// This function uses a bisection, to find the index of the first occurance of an element in
// an array. For the algorithm to work, the input array must be sorted.
//---------------------------------------------------------------------------------------------------------------------
template<typename T>
__device__ void findFirstWithBisect (int left, int right, const T &query, int &result, const T *array){
//---------------------------------------------------------------------------------------------------------------------
  while (left <= right) {
    int mid = 0.5 * (right + left);

    // only terminate search if we found the first index of the hit
    // guard against array[mid-1] to be outside of the array
    if (array[mid] == query && (mid == left || array[mid - 1] != query)) {
      result = mid;
      break;
    }
    if (array[mid] < query)
      left = mid + 1;
    else
      right = mid - 1;
  }
}

/// Assume we have an query value i in the range (left, right),
/// and a prefix sum, also in this range.
/// This function finds the lower bound idx of the step in the prefix sum that
/// contains i
//---------------------------------------------------------------------------------------------------------------------
__device__ void locateInPrefixSumBisect (int left, int right, int i, int &result, const int *prefix_sum){
//---------------------------------------------------------------------------------------------------------------------
  while (left <= right) {
    int mid = left + 0.5 * (right - left);
    if (i >= prefix_sum[mid] && i < prefix_sum[mid + 1]) {
      result = mid;
      break;
    }
    if (i < prefix_sum[mid]) {
      right = mid - 1;
    } else {
      left = mid + 1;
    }
  }
}

//---------------------------------------------------------------------------------------------------------------------
template <class T>
__device__ void hits_geometric_cuts (T& z0, T& phi_slope, T&deta, T& dphi,
                                     const T& R1, const T& z1, const T& eta1, const T& phi1,
                                     const T& R2, const T& z2, const T& eta2, const T& phi2,
                                     const data_type2& pi, T epsilon){
//---------------------------------------------------------------------------------------------------------------------
  data_type2 dr = R2 - R1;
  deta = eta2 - eta1;
  dphi = phi2 - phi1;

  if (dphi > pi) dphi -= 2 * pi;
  else if (dphi < -pi) dphi += 2 * pi;

//  if (dr) {
  if (abs(dr) > epsilon) {
    phi_slope = dphi / dr;
    z0 = z1 - (data_type2) R1 * (z2-z1) / dr;
  }
  else {
    z0 = 0;
    phi_slope = 0;
  }
}

//---------------------------------------------------------------------------------------------------------------------
template <class T>
__device__ bool apply_geometric_cuts (int i, const T& z0, const T& phi_slope, const T& deta, const T& dphi,
                                      T *z0_min, T *phi_slope_min, T *deta_min, T *dphi_min,
                                      T *z0_max, T *phi_slope_max, T *deta_max, T *dphi_max){
//---------------------------------------------------------------------------------------------------------------------
  bool accept = (z0_min[i] <= z0) * (dphi_min[i] <= dphi) * (phi_slope_min[i] <= phi_slope) * (deta_min[i] <= deta);
  accept *= (z0 <= z0_max[i]) * (dphi <= dphi_max[i]) * (phi_slope <= phi_slope_max[i]) * (deta <= deta_max[i]);

  return accept;
}

//---------------------------------------------------------------------------------------------------------------------
template <class T>
__device__ T Diff_dydx (T *x, T *y, T *z, const int& it1, const int& it2, const int& it3, T epsilon){
//---------------------------------------------------------------------------------------------------------------------
  data_type2 dy_12 = y[it2] - y[it1];
  data_type2 dy_23 = y[it3] - y[it2];
  data_type2 dx_12 = x[it2] - x[it1];
  data_type2 dx_23 = x[it3] - x[it2];

  data_type2 diff_dydx = 0;
  if (abs(dx_12) < epsilon) dx_12 = 0;
  if (abs(dx_23) < epsilon) dx_23 = 0;

  if (dx_12 && dx_23)
    diff_dydx = (dy_12 / dx_12) - (dy_23 / dx_23);
  else if (dx_12)
    diff_dydx = dy_12 / dx_12; // force dy_23/dx_23 to 0
  else if (dx_23)
    diff_dydx = -dy_23 / dx_23; // force dy_12/dx_12 to 0

  return (T) diff_dydx;
}

//---------------------------------------------------------------------------------------------------------------------
template <class T>
__device__ T Diff_dzdr (T *R, T *z, const int& it1, const int& it2, const int& it3, T epsilon){
//---------------------------------------------------------------------------------------------------------------------
  data_type2 dz_12 = z[it2] - z[it1];
  data_type2 dz_23 = z[it3] - z[it2];
  data_type2 dr_12 = R[it2] - R[it1];
  data_type2 dr_23 = R[it3] - R[it2];

  T diff_dzdr = 0;
  if (abs(dr_12) < epsilon) dr_12 = 0;
  if (abs(dr_23) < epsilon) dr_23 = 0;

  if (dr_12 && dr_23)
    diff_dzdr = (dz_12 / dr_12) - (dz_23 / dr_23);
  else if (dr_12)
    diff_dzdr = dz_12 / dr_12; // force dz_23/dr_23 to 0
  else if (dr_23)
    diff_dzdr = -dz_23 / dr_23; // force dz_12/dr_12 to 0

  return diff_dzdr;
}


//=====GPU Kernels==========GPU Kernels==========GPU Kernels==========GPU Kernels==========GPU Kernels=================


//---------------------------------------------------------------------------------------------------------------------
template <class T>
__global__ void hits_geometric_cuts (T *z0, T *phi_slope, T *deta, T *dphi,
                                     int *SPi, int *SPo, T *R, T *z, T *eta, T *phi,
                                     data_type2 pi, T epsilon, int nb_doublets) {
//---------------------------------------------------------------------------------------------------------------------
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nb_doublets) return;

  int SP1 = SPi[i];
  int SP2 = SPo[i];
  data_type2 R1 = R[SP1];
  T R2 = R[SP2];
  T z1 = z[SP1];
  T z2 = z[SP2];

  data_type2 dr = R2 - R1;
  data_type2 dz = z2 - z1;
  deta[i] = eta[SP2] - eta[SP1];
  data_type2 dphi_loc = phi[SP2] - phi[SP1];
  T pipi = 2 * pi;

  if (dphi_loc > pi) dphi_loc -= pipi;
  else if (dphi_loc < -pi) dphi_loc += pipi;
  dphi[i] = dphi_loc;

//  if (dr) {
  if (abs(dr) > epsilon) {
    phi_slope[i] = dphi_loc / dr;
    z0[i] = z1 - R1 * dz / dr;
  }
  else {
    z0[i] = 0;
    phi_slope[i] = 0;
  }
}

//=====GPU Kernels==========GPU Kernels==========GPU Kernels==========GPU Kernels==========GPU Kernels=================

/*
//---------------------------------------------------------------------------------------------------------------------
template <class T>
__device__ bool apply_geometric_cuts (const T& z0, const T& phi_slope, const T& deta, const T& dphi,
                                      const T& z0_min, const T& phi_slope_min, const T& deta_min, const T& dphi_min,
                                      const T& z0_max, const T& phi_slope_max, const T& deta_max, const T& dphi_max){
//---------------------------------------------------------------------------------------------------------------------
  bool accept = (z0_min <= z0) * (dphi_min <= dphi) * (phi_slope_min <= phi_slope) * (deta_min <= deta);
  accept *= (z0 <= z0_max) * (dphi <= dphi_max) * (phi_slope <= phi_slope_max) * (deta <= deta_max);

  return accept;
}
*/

//---------------------------------------------------------------------------------------------------------------------
__global__ void count_source_hits_per_doublet (int *nb_src_hits_per_doublet,
                                               const int *modules1, const int *indices, int nb_doublets){
//---------------------------------------------------------------------------------------------------------------------
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nb_doublets) return;

  int module1 = modules1[i];
  nb_src_hits_per_doublet[i] = indices[module1 + 1] - indices[module1];
}

// Device function that includes the logic for the loop-unrolled doublet cut
// Can be supplied with a function that is executed for each selected doublet
// Used to implement the counting and the cut kernel without code duplication
//---------------------------------------------------------------------------------------------------------------------
template <typename T, typename F>
__device__ void doublet_cuts (F &&function, int i, int nb_doublets, const int *doublet_offsets,
                              const int *modules1, const int *modules2,
                              const T *R, const T *z, const T *eta, const T *phi,
                              T *z0_min, T *deta_min, T *phi_slope_min, T *dphi_min,
                              T *z0_max, T *deta_max, T *phi_slope_max, T *dphi_max,
                              const int *indices, T pi, T epsilon){
//---------------------------------------------------------------------------------------------------------------------
  // Since doublet_offsets should be the prefix sum of the
  // number of space points per doublet, we can construct the doublet index
  int doublet_idx = 0;
  locateInPrefixSumBisect(0, nb_doublets, i, doublet_idx, doublet_offsets);

  const int module1 = modules1[doublet_idx];
  const int module2 = modules2[doublet_idx];

  // we can reconstruct the SP1 index from start-index of SPs for module1,
  // and the difference between the current i and the start of the doublet
  // Note that several doublets can have the same module1
  const int k = indices[module1] + (i - doublet_offsets[doublet_idx]);

  T phi_SP1 = phi[k];
  T eta_SP1 = eta[k];
  T R_SP1 = R[k];
  T z_SP1 = z[k];

  for (int l = indices[module2]; l < indices[module2 + 1]; l++) {
    T z0, phi_slope, deta, dphi;
    hits_geometric_cuts<T> (z0, phi_slope, deta, dphi, 
                            R_SP1, z_SP1, eta_SP1, phi_SP1, R[l], z[l], eta[l], phi[l], pi, epsilon);

    if (apply_geometric_cuts (doublet_idx, z0, phi_slope, deta, dphi,
                              z0_min, phi_slope_min, deta_min, dphi_min, z0_max, phi_slope_max, deta_max, dphi_max))
      function(k, l);
  }
}

/// Kernel that implements the edge counting for loop-unrolled style
//---------------------------------------------------------------------------------------------------------------------
template <class T>
__global__ void count_doublet_edges (int *nb_edges_per_src_hit,
                                     int nb_doublets, const int *doublet_offsets,
                                     const int *modules1, const int *modules2,
                                     const T *R, const T *z, const T *eta, const T *phi,
                                     T *z0_min, T *deta_min, T *phi_slope_min, T *dphi_min,
                                     T *z0_max, T *deta_max, T *phi_slope_max, T *dphi_max,
                                     const int *indices, T pi, T epsilon, int sum_nb_src_hits_per_doublet){
//---------------------------------------------------------------------------------------------------------------------
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= sum_nb_src_hits_per_doublet)
    return;

  int edges = 0;
  doublet_cuts ([&] __device__(int, int) { edges++;},
                i, nb_doublets, doublet_offsets, modules1, modules2, R, z, eta, phi,
                z0_min, deta_min, phi_slope_min, dphi_min,
                z0_max, deta_max, phi_slope_max, dphi_max,
                indices, pi, epsilon);

  nb_edges_per_src_hit[i] = edges;
}

/// Kernel that implements the edge building for loop-unrolled style
//---------------------------------------------------------------------------------------------------------------------
template <class T>
//__global__ void MMG_LAUNCH_BOUNDS(64, 2) build_doublet_edges (int *reduced_M1_hits, int *reduced_M2_hits,
__global__ void build_doublet_edges (int *reduced_M1_hits, int *reduced_M2_hits,
                                     int nb_doublets, const int *doublet_offsets, const int *indices, int *edge_sum,
                                     const int *modules1, const int *modules2,
                                     const T *R, const T *z, const T *eta, const T *phi,
                                     T *z0_min, T *deta_min, T *phi_slope_min, T *dphi_min,
                                     T *z0_max, T *deta_max, T *phi_slope_max, T *dphi_max,
                                     T pi, T epsilon, int sum_nb_src_hits_per_doublet){
//---------------------------------------------------------------------------------------------------------------------
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= sum_nb_src_hits_per_doublet) {
    return;
  }

  int edges = 0;
  doublet_cuts ([&] __device__(int k, int l) {
                              reduced_M1_hits[edge_sum[i] + edges] = k;
                              reduced_M2_hits[edge_sum[i] + edges] = l;
                              edges++;},
                i, nb_doublets, doublet_offsets, modules1, modules2, R, z, eta, phi,
                z0_min, deta_min, phi_slope_min, dphi_min,
                z0_max, deta_max, phi_slope_max, dphi_max,
                indices, pi, epsilon);
}

//---------------------------------------------------------------------------------------------------------------------
//__global__ void MMG_LAUNCH_BOUNDS(64, 2) doublet_edge_sum (int *edge_sum,
__global__ void doublet_edge_sum (int *edge_sum,
                                  const int *doublet_offsets, const int *nb_edges_per_src_hit, int nb_doublets){
//---------------------------------------------------------------------------------------------------------------------
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  // Note that we want to get the maximum element as well
  if (i >= nb_doublets + 1) {
    return;
  }

  // Since nb_edges_per_src_hit is already a prefix sum, we can just copy the
  // elements on the boundary positions
  edge_sum[i] = nb_edges_per_src_hit[doublet_offsets[i]];
}


//=====GPU Kernels==========GPU Kernels==========GPU Kernels==========GPU Kernels==========GPU Kernels=================


// Kernel that implements the triplet hit counting, necessary for the triplet unrolling
//---------------------------------------------------------------------------------------------------------------------
__global__ void count_triplet_hits (int *src_hits_per_triplet, const int *modules12_map, const int *modules23_map,
                                    const int *edge_indices, int nb_triplets){
//---------------------------------------------------------------------------------------------------------------------
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nb_triplets) {
    return;
  }

  int module12 = modules12_map[i];
  int module23 = modules23_map[i];

  int nb_hits_M12 = edge_indices[module12 + 1] - edge_indices[module12];
  int nb_hits_M23 = edge_indices[module23 + 1] - edge_indices[module23];

  if (nb_hits_M12 == 0 || nb_hits_M23 == 0) {
    src_hits_per_triplet[i] = 0;
    return;
  }

  int shift12 = edge_indices[module12];
  int last12 = shift12 + nb_hits_M12 - 1;

  src_hits_per_triplet[i] = last12 - shift12 + 1;
}


// Device function that implements the inner loop body of the triplet cuts.
// Could be used to unify the implementation of the workflow with and without unrolling
//---------------------------------------------------------------------------------------------------------------------
template <typename T>
__device__ void triplet_cuts_inner_loop_body (bool *vertices, bool *edge_tag,
                                        int i, int k, int last12, int shift23, int ind23, T *x, T *y, T *z, T *R,
                                        T *z0, T *phi_slope, T *deta, T *dphi,
                                        T *MD12_z0_min, T *MD12_deta_min, T *MD12_phi_slope_min, T *MD12_dphi_min,
                                        T *MD12_z0_max, T *MD12_deta_max, T *MD12_phi_slope_max, T *MD12_dphi_max,
                                        T *MD23_z0_min, T *MD23_deta_min, T *MD23_phi_slope_min, T *MD23_dphi_min,
                                        T *MD23_z0_max, T *MD23_deta_max, T *MD23_phi_slope_max, T *MD23_dphi_max,
                                        T *diff_dydx_min, T *diff_dydx_max, T *diff_dzdr_min, T *diff_dzdr_max,
                                        int *M1_SP, int *M2_SP, int *edge_indices, T pi, T epsilon){
//---------------------------------------------------------------------------------------------------------------------
  int SP1 = M1_SP[k];
  int SP2 = M2_SP[k];
  bool next_ind = false;

  if (k < last12) {
    next_ind = (SP2 != (M2_SP[k + 1]));
  }

  if (!apply_geometric_cuts (i, z0[k], phi_slope[k], deta[k], dphi[k],
                            MD12_z0_min, MD12_phi_slope_min, MD12_deta_min, MD12_dphi_min,
                            MD12_z0_max, MD12_phi_slope_max, MD12_deta_max, MD12_dphi_max)) {
    return;
  }

  int l = shift23;
  findFirstWithBisect(shift23, ind23 - 1, SP2, l, M1_SP);

  bool new_elt = false;
  for (; l < ind23 && SP2 == M1_SP[l]; l++) {
    int SP3 = M2_SP[l];
    if (!apply_geometric_cuts (i, z0[l], phi_slope[l], deta[l], dphi[l],
                               MD23_z0_min, MD23_phi_slope_min, MD23_deta_min, MD23_dphi_min,
                               MD23_z0_max, MD23_phi_slope_max, MD23_deta_max, MD23_dphi_max)) {
      continue;
    }

    T diff_dydx = Diff_dydx(x, y, z, SP1, SP2, SP3, epsilon);
    if (!((diff_dydx >= diff_dydx_min[i]) * (diff_dydx <= diff_dydx_max[i]))) {
      continue;
    }

    T diff_dzdr = Diff_dzdr(R, z, SP1, SP2, SP3, epsilon);
    if (!((diff_dzdr >= diff_dzdr_min[i]) * (diff_dzdr <= diff_dzdr_max[i]))) {
      continue;
    }

    vertices[SP3] = edge_tag[l] = true;
    new_elt = true;
  }
  if (new_elt) {
    edge_tag[k] = vertices[SP1] = vertices[SP2] = true;
  }
  if (next_ind && new_elt) {
    shift23 = l;
  }
}

//---------------------------------------------------------------------------------------------------------------------
template <typename T>
__global__ void triplet_cuts (bool *vertices, bool *edge_tag,
                      int nb_triplets, const int *triplet_offsets,
                      const int *modules12_map, const int *modules23_map, T *x, T *y, T *z, T *R,
                      T *z0, T *phi_slope, T *deta, T *dphi,
                      T *MD12_z0_min, T *MD12_phi_slope_min, T *MD12_deta_min, T *MD12_dphi_min,
                      T *MD12_z0_max, T *MD12_phi_slope_max, T *MD12_deta_max, T *MD12_dphi_max,
                      T *MD23_z0_min, T *MD23_phi_slope_min, T *MD23_deta_min, T *MD23_dphi_min,
                      T *MD23_z0_max, T *MD23_phi_slope_max, T *MD23_deta_max, T *MD23_dphi_max,
                      T *diff_dydx_min, T *diff_dydx_max, T *diff_dzdr_min, T *diff_dzdr_max,
                      int *M1_SP, int *M2_SP, int *edge_indices, T pi, T epsilon, int nb_src_hits_per_triplet_sum){
//---------------------------------------------------------------------------------------------------------------------
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nb_src_hits_per_triplet_sum) {
    return;
  }

  // Find triplet index
  int triplet_index = 0;

  locateInPrefixSumBisect(0, nb_triplets, i, triplet_index, triplet_offsets);

  int module12 = modules12_map[triplet_index];
  int module23 = modules23_map[triplet_index];
  int nb_hits_M12 = edge_indices[module12 + 1] - edge_indices[module12];
  int nb_hits_M23 = edge_indices[module23 + 1] - edge_indices[module23];

  bool hits_on_modules = nb_hits_M12 * nb_hits_M23;
  if (!hits_on_modules) {
    return;
  }

  int shift12 = edge_indices[module12];
  int shift23 = edge_indices[module23];

  int last12 = shift12 + nb_hits_M12 - 1;
  int ind23 = shift23 + nb_hits_M23;
  const int k = shift12 + (i - triplet_offsets[triplet_index]);

  // From here on starts the original kernel, with the loop over k pulled out of the kernel
  triplet_cuts_inner_loop_body (vertices, edge_tag,
      triplet_index, k, last12, shift23, ind23, x, y, z, R, z0, phi_slope, deta, dphi,
      MD12_z0_min, MD12_deta_min, MD12_phi_slope_min, MD12_dphi_min,
      MD12_z0_max, MD12_deta_max, MD12_phi_slope_max, MD12_dphi_max,
      MD23_z0_min, MD23_deta_min, MD23_phi_slope_min, MD23_dphi_min,
      MD23_z0_max, MD23_deta_max, MD23_phi_slope_max, MD23_dphi_max,
      diff_dydx_min, diff_dydx_max, diff_dzdr_min, diff_dzdr_max,
      M1_SP, M2_SP, edge_indices, pi, epsilon);
}
