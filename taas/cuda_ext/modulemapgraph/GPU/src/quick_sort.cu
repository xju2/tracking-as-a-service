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

#include "MMG/quick_sort.cuh"

/*
template <class T>
struct tuple_compare
{
  __device__  bool operator() (const thrust::tuple<int, T>& t1, const thrust::tuple<int, T>& t2)
    {
      return thrust::get<1>(t1) < thrust::get<1>(t2);
    }
};
*/

//=====GPU Device Kernels==========GPU Device Kernels==========GPU Device Kernels==========GPU Device Kernels==========

//---------------------------------------------------------------------------------------------------------------------
__device__ void swap (int &a, int &b){
//---------------------------------------------------------------------------------------------------------------------
  if (a == b) return;
  int t = a;
  a = b;
  b = t;
}

//---------------------------------------------------------------------------------------------------------------------
template <class T>
__device__ void recursive_quick_sort (int *sorted_SP, T *SP, int start, int end){
//---------------------------------------------------------------------------------------------------------------------
  if (start > end) return;

  // select the lefttmost element as pivot
  T pivot = SP[sorted_SP[start]];
  int low = start;
  int high = start;

  // traverse each element of the array
  // compare them with the pivot
  for (int j=start+1; j<end; j++) {
    T SP2 = SP[sorted_SP[j]];
    // if element smaller than pivot is found swap it with the greater element pointed by i
    if (SP2 < pivot) {
      swap (sorted_SP[j], sorted_SP[++high]);
      swap (sorted_SP[high], sorted_SP[low++]);
    }
    else
      if (SP2 == pivot)
        swap (sorted_SP[++high], sorted_SP[j]);
  }

  if (low == start && high == end) return;
  recursive_quick_sort (sorted_SP, SP, start, low);
  recursive_quick_sort (sorted_SP, SP, high+1, end);
}

//---------------------------------------------------------------------------------------------------------------------
__device__ void quick_sort (int *sorted_SP, int *SP, int start, int end){
//---------------------------------------------------------------------------------------------------------------------
  if (start > end) return;

  // select the lefttmost element as pivot
  int pivot = SP[sorted_SP[start]];
  int low = start;
  int high = start;
  int stored_high [qs_max_steps];
  int stored_end [qs_max_steps];
  int step = 0;
  stored_high[0] = start;
  stored_end[0] = start;

  // traverse each element of the array and compare them with the pivot
  for (step=1; start < end || step;) {
    for (;start < end;) {
      low = start;
      high = start;
      pivot = SP[sorted_SP[start]];

      for (int j=start+1; j<end; j++) {
        int SP2 = SP[sorted_SP[j]];
        // if element smaller than pivot is found swap it with the greater element pointed by i
        if (SP2 < pivot) {
          swap (sorted_SP[j], sorted_SP[++high]);
          swap (sorted_SP[high], sorted_SP[low++]);
        }
        else
          if (SP2 == pivot)
            swap (sorted_SP[++high], sorted_SP[j]);
      }

      if (low == start && high == end) continue;
      assert (step < qs_max_steps); // if assertion fail change qs_max_steps value in CPU/headers/parameters
      stored_end[++step] = end;
      stored_high[step] = high;
      end = low;
    }
    start = stored_high[step]+1;
    end = stored_end[step--];
  }
}


//=====GPU Kernels==========GPU Kernels==========GPU Kernels==========GPU Kernels==========GPU Kernels=================


//---------------------------------------------------------------------------------------------------------------------
template <class T>
__global__ void sort (T* sorted_vector, T* vector, int* pos, int size){
//---------------------------------------------------------------------------------------------------------------------
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < size)
    sorted_vector[i] = vector[pos[i]];
}

//---------------------------------------------------------------------------------------------------------------------
__global__ void partial_recursive_quick_sort (int *sorted_vector, int *vector, int *mask_sum, int size){
//---------------------------------------------------------------------------------------------------------------------
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if ((i >= size) || (mask_sum[i+1] - mask_sum[i] < 2)) return;

  recursive_quick_sort (sorted_vector, vector, mask_sum[i], mask_sum[i+1]);
}

//---------------------------------------------------------------------------------------------------------------------
__global__ void partial_quick_sort (int *sorted_vector, int *vector, int *mask_sum, int size){
//---------------------------------------------------------------------------------------------------------------------
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if ((i >= size) || (mask_sum[i+1] - mask_sum[i] < 2)) return;

  quick_sort (sorted_vector, vector, mask_sum[i], mask_sum[i+1]);
}

/*
//-------------------------------------------------------------------------------------------
template <class T>
__global__ void thrust_partial_quick_sort (int *sorted_indices, T *vector, int *mask_sum, int size)
//-------------------------------------------------------------------------------------------
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if ((i >= size) || (mask_sum[i+1] - mask_sum[i] < 2)) return;

//  thrust::zip_iterator<thrust::tuple<thrust::device_ptr<int>,thrust::device_ptr<T>>> iter thrust::tuple<thrust::device_ptr<int>,thrust::device_ptr<T>>;
  thrust::zip_iterator<thrust::tuple<int*,T*>> iter (thrust::make_tuple(sorted_indices, vector));
  thrust::sort (thrust::make_zip_iterator (
                thrust::make_tuple (thrust::device_ptr<int>(sorted_indices + mask_sum[i]),
                                    thrust::device_ptr<T>(vector + mask_sum[i]))),
              thrust::make_zip_iterator (
                thrust::make_tuple (thrust::device_ptr<int>(sorted_indices + mask_sum[i+1]-1),
                                    thrust::device_ptr<T>(vector + mask_sum[i+1]-1))),
                   tuple_compare<int>());
}
*/

//---------------------------------------------------------------------------------------------------------------------
__global__ void quick_sort (int *sorted_vector, int *vector, int size){
//---------------------------------------------------------------------------------------------------------------------
  quick_sort (sorted_vector, vector, 0, size-1);
}

//---------------------------------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(512, 4)
    block_odd_even_sort (int *M2_idxs, const int *M2_hits, const int *cuda_edge_sum, int size){
//---------------------------------------------------------------------------------------------------------------------
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= size) return;

  bool sorted{};
  auto comparison = thrust::less<int>{};

  const int begin = cuda_edge_sum[blockIdx.x];
  const int end = cuda_edge_sum[blockIdx.x + 1];
  if (end - begin == 0) {
    return;
  }

  do {
    sorted = true;

    for (std::uint32_t j =
             begin + 2 * static_cast<std::uint32_t>(threadIdx.x) + 1;
         j < end - 1; j += 2 * blockDim.x) {
      if (comparison(M2_hits[M2_idxs[j + 1]], M2_hits[M2_idxs[j]])) {
        swap(M2_idxs[j + 1], M2_idxs[j]);
        sorted = false;
      }
    }

    __syncthreads();

    for (std::uint32_t j = begin + 2 * static_cast<std::uint32_t>(threadIdx.x);
         j < end - 1; j += 2 * blockDim.x) {
      if (comparison(M2_hits[M2_idxs[j + 1]], M2_hits[M2_idxs[j]])) {
        swap(M2_idxs[j + 1], M2_idxs[j]);
        sorted = false;
      }
    }
  } while (__syncthreads_or(!sorted));
}


//=====CPU Functions==========CPU Functions==========CPU Functions==========CPU Functions==========CPU Functions=======


//---------------------------------------------------------------------------------------------------------------------
void order (int* sorted_vector, int *vector, int size, int blocks){
//---------------------------------------------------------------------------------------------------------------------
  int *sorted_positions;
  cudaMalloc (&sorted_positions, size * sizeof(int));
  dim3 grid_dim = ((size + blocks - 1) / blocks);
  init_vector<<<grid_dim,blocks>>> (sorted_vector, size);
  quick_sort<<<1,1>>> (sorted_positions, vector, size);
  sort<<<grid_dim,blocks>>> (sorted_vector, vector, sorted_positions, size);
  cudaFree (sorted_positions);
}

//---------------------------------------------------------------------------------------------------------------------
template <class T>
void sort (T **vector, int *positions, int size, int blocks){
//---------------------------------------------------------------------------------------------------------------------
  dim3 grid_dim = (size + blocks - 1) / blocks;

  T *sorted_vector;
  cudaMalloc (&sorted_vector, size*sizeof(T));
  sort<<<grid_dim,blocks>>> (sorted_vector, *vector, positions, size);

  // swap pointers
  T *old_vector;
  old_vector = *vector;
  *vector = sorted_vector;
  sorted_vector = old_vector;
  cudaFree (sorted_vector);
}

//---------------------------------------------------------------------------------------------------------------------
template <class T>
void thrust_sort (thrust::device_vector<int>& sorted_vector, T* key, int size, int blocks){
//---------------------------------------------------------------------------------------------------------------------
  dim3 grid_dim = (size + blocks - 1) / blocks;
  init_vector<<<grid_dim, blocks>>> (thrust::raw_pointer_cast(&sorted_vector[0]), size);

  thrust::device_ptr<T> thrust_ptr2 = thrust::device_pointer_cast (key);
  thrust::device_vector<T> thrust_key (thrust_ptr2, thrust_ptr2 + size);

  thrust::sort_by_key (thrust::device, thrust_key.begin(), thrust_key.end(), sorted_vector.begin());
}

//---------------------------------------------------------------------------------------------------------------------
template <class T, class V>
void thrust_sort (V **sorted_vector, T* key, int size){
//---------------------------------------------------------------------------------------------------------------------
  thrust::device_ptr<V> thrust_ptr = thrust::device_pointer_cast (*sorted_vector);
  thrust::device_vector<V> thrust_sorted_vector (thrust_ptr, thrust_ptr + size);

  thrust::device_ptr<T> thrust_ptr2 = thrust::device_pointer_cast (key);
  thrust::device_vector<T> thrust_key (thrust_ptr2, thrust_ptr2 + size);

  thrust::sort_by_key (thrust::device, thrust_key.begin(), thrust_key.end(), thrust_sorted_vector.begin());
  *sorted_vector = thrust::raw_pointer_cast (&thrust_sorted_vector[0]);
}

//---------------------------------------------------------------------------------------------------------------------
template <class T>
void thrust_sort (int* sorted_vector, T* key, int size, int blocks){
//---------------------------------------------------------------------------------------------------------------------
  dim3 grid_dim = (size + blocks - 1) / blocks;
  init_vector<<<grid_dim, blocks>>> (sorted_vector, size);

  // key and values are reordered
  thrust::sort_by_key (thrust::device, key, key+size, sorted_vector);
}

//---------------------------------------------------------------------------------------------------------------------
template <class T>
void thrust_sort_nokeysort (int* sorted_vector, T* key, int size, int blocks){
//---------------------------------------------------------------------------------------------------------------------
  dim3 grid_dim = (size + blocks - 1) / blocks;
  init_vector<<<grid_dim, blocks>>> (sorted_vector, size);
  T *key_copy;
  cudaMalloc (&key_copy, size * sizeof(T));
  cudaMemcpyAsync (key_copy, key, size * sizeof(T), cudaMemcpyDeviceToDevice);

  // key_copy and values are reordered
  thrust::sort_by_key (thrust::device, key_copy, key_copy+size, sorted_vector);
  cudaFree (key_copy);
}

/*
//----------------------------------------------------------
template <class T>
void thrust_sort (T *vector, int *key, int size, int blocks)
//----------------------------------------------------------
{
  dim3 grid_dim = (size + blocks - 1) / blocks;
//  init_vector<<<grid_dim, blocks>>> (thrust::raw_pointer_cast(&sorted_vector[0]), size);
  init_vector<<<grid_dim, blocks>>> (vector, size);

  thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast (vector);
  thrust::device_vector<int> thrust_vector (thrust_ptr, thrust_ptr + size);

  thrust::device_ptr<int> thrust_ptr2 = thrust::device_pointer_cast (key);
  thrust::device_vector<int> thrust_key (thrust_ptr2, thrust_ptr2 + size);

  thrust::sort_by_key (thrust::device, thrust_key.begin(), thrust_key.end(), sorted_vector.begin());
  vector = thrust::raw_pointer_cast(&sorted_vector[0]);
}
*/

