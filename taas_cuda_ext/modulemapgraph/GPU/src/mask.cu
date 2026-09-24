/***********************************************************************
 * Module Map Graph library
 * Track reconstruction for HEP
 * copyright © 2024,2025 COLLARD Christophe
 * Laboratoire des 2 Infinis - Toulouse - France (L2IT)
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 **********************************************************************/

#include "MMG/mask.cuh"

__global__ void convert (int* output, bool* input, int size){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= size) return;

  output[i] = input[i];
}


//=====GPU functions for mask======================================================================

//-------------------------------------------------------------------------------------------------
// do not call this function directly for mask objects - use the "reverse" method
template <class T>
__global__ void reverse_mask (T* mask, int size){
//-------------------------------------------------------------------------------------------------
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= size) return;

  mask[i] = !mask[i];
}

//-------------------------------------------------------------------------------------------------
template <class T>
__global__ void reverse_mask (T* rmask,
                              const T* mask, int size){
//-------------------------------------------------------------------------------------------------
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < size)
    rmask[i] = !mask[i];
}

//-------------------------------------------------------------------------------------------------     // mask          1 1 0 0 1 0 1 0 1 1 0 0 0 0 1 0 1
__global__ void mask_filter (bool* mask,                                                                // filter        1 0 0 1 0 0 0 1 1 0 0 0 0 0 1 0 0
                             const bool* filter, int size){                                             // filtered mask 0 1 0 0 1 0 1 0 0 1 0 0 0 0 0 0 1
//-------------------------------------------------------------------------------------------------
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= size) return;

  if (filter[i])
    mask[i] = false;
}

//-------------------------------------------------------------------------------------------------     // mask          0 1 0 0 1 0 1 0 0 1 0 0 0 0 0 0 1 (already filtered)
__global__ void mask_reverse_filter (bool* mask,                                                        // filter        1 0 0 1 0 0 0 1 1 0 0 0 0 0 1 0 0
                                    const bool* filter, int size){                                      // rev filt mask 1 1 0 1 1 0 1 1 1 1 0 0 0 0 1 0 1
//-------------------------------------------------------------------------------------------------
  int i = blockIdx.x * blockDim.x + threadIdx.x;                                                        // mask          1 1 0 0 1 0 1 0 1 1 0 0 0 0 1 0 1 (origin)
  if (i >= size) return;                                                                                 // filter        1 0 0 1 0 0 0 1 1 0 0 0 0 0 1 0 0
                                                                                                        // rev filt mask 1 1 0 1 1 0 1 1 1 1 0 0 0 0 1 0 1
  if (filter[i])
    mask[i] = true;
}

//-------------------------------------------------------------------------------------------------     // mask          0 1 0 0 1 0 1 0 0 1 0 0 0 0 0 0 1 (already filtered)
__global__ void reverse_mask_reverse_filter (bool* mask,                                                        // filter        1 0 0 1 0 0 0 1 1 0 0 0 0 0 1 0 0
                                             const bool* filter, int size){                                      // rev filt mask 1 1 0 1 1 0 1 1 1 1 0 0 0 0 1 0 1
//-------------------------------------------------------------------------------------------------
  int i = blockIdx.x * blockDim.x + threadIdx.x;                                                        // mask          1 1 0 0 1 0 1 0 1 1 0 0 0 0 1 0 1 (origin)
  if (i >= size) return;                                                                                 // filter        1 0 0 1 0 0 0 1 1 0 0 0 0 0 1 0 0
                                                                                                        // rev filt mask 1 1 0 1 1 0 1 1 1 1 0 0 0 0 1 0 1
  if (!filter[i])
    mask[i] = true;
}

//-------------------------------------------------------------------------------------------------
template <class T>
__global__ void tag_mask (T* mask,
                          const int* vector, int size){
//-------------------------------------------------------------------------------------------------
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < size)
    mask[vector[i]] = true;  
}


//=====GPU functions for mask======================================================================

//-------------------------------------------------------------------------------------------------
CUDA_mask::CUDA_mask (int size){
//-------------------------------------------------------------------------------------------------
  _size = size;
  _sum = 0;
  cudaMalloc (&_cuda_mask, _size * sizeof(int));
  cudaMalloc (&_cuda_mask_sum, (_size+1) * sizeof(int));
  cudaMemsetAsync (_cuda_mask, 0, _size * sizeof(int));
}

//-------------------------------------------------------------------------------------------------
CUDA_mask::CUDA_mask (int size, bool value, int blocks){
//-------------------------------------------------------------------------------------------------
  _size = size;
  _sum = 0;
  cudaMalloc (&_cuda_mask, _size * sizeof(int));
  cudaMalloc (&_cuda_mask_sum, (_size+1) * sizeof(int));

  dim3 grid_dim = (_size + blocks - 1) / blocks;
  init_vector<<<grid_dim,blocks>>> (_cuda_mask, value, _size);
}

//-------------------------------------------------------------------------------------------------
CUDA_mask::~CUDA_mask (){
//-------------------------------------------------------------------------------------------------
  if (_size) {
    cudaFree (_cuda_mask_sum);
    cudaFree (_cuda_mask);
  }
}

//-------------------------------------------------------------------------------------------------
void CUDA_mask::init (int size)
//-------------------------------------------------------------------------------------------------
{
  assert (!_size);
  _size = size;
  _sum = 0;
  cudaMalloc (&_cuda_mask, _size * sizeof(int));
  cudaMalloc (&_cuda_mask_sum, (_size+1) * sizeof(int));
  cudaMemsetAsync (_cuda_mask, 0, _size * sizeof(int));
}

//-------------------------------------------------------------------------------------------------
void CUDA_mask::init (int size, bool value, int blocks)
//-------------------------------------------------------------------------------------------------
{
  assert (!_size);
  _size = size;
  _sum = 0;
  cudaMalloc (&_cuda_mask, _size * sizeof(int));
  cudaMalloc (&_cuda_mask_sum, (_size+1) * sizeof(int));

  dim3 grid_dim = (_size + blocks - 1) / blocks;
  init_vector<<<grid_dim,blocks>>> (_cuda_mask, value, _size);
}

//-------------------------------------------------------------------------------------------------
int CUDA_mask::scan (int blocks)
//-------------------------------------------------------------------------------------------------
{
  if (!_size) {
    _sum = 0;
    return _sum;
  }
  ::scan (_cuda_mask_sum, _cuda_mask, blocks, _size);
  cudaMemcpyAsync (&_sum, &_cuda_mask_sum[_size], sizeof(int), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  return _sum;
}

//-------------------------------------------------------------------------------------------------
void CUDA_mask::reverse (int blocks)
//-------------------------------------------------------------------------------------------------
{
  if (!_size) return;
  dim3 grid_dim = (_size + blocks - 1) / blocks;
  reverse_mask<<<grid_dim,blocks>>> (_cuda_mask, _size);
  _sum = 0;
}
