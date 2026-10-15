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

#include <MMG/CUDA_mask>

//=====GPU functions for stream compaction=============================================================================

//---------------------------------------------------------------------------------------------------------------------
template <class T, class Tintbool>
__global__ void compact_stream (T *output,
                                const T *input, const Tintbool *mask, const int *mask_sum, int size){
//---------------------------------------------------------------------------------------------------------------------
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= size) return;

  if (mask[i])
    output[mask_sum[i]] = input[i];
}

//---------------------------------------------------------------------------------------------------------------------
template <class T>
__global__ void compact_stream (T *output,
                                const T *input, int max_hits_per_track, const int *initial_positions, int size){
//---------------------------------------------------------------------------------------------------------------------
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= size) return;

  int old_pos = initial_positions[i]*max_hits_per_track;
  int new_pos = i*max_hits_per_track;
  for (int p=0; p<max_hits_per_track; p++)
    output[new_pos+p] = input[old_pos+p];
}

//---------------------------------------------------------------------------------------------------------------------
template <class T>
__global__ void compact_stream (T *output,
                                const T *input, const int *mask_sum, int step, int size){
//---------------------------------------------------------------------------------------------------------------------
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= size) return;

  int shift = i * step;

  for (int k=mask_sum[i]; k<mask_sum[i+1]; k++, shift++)
    output[k] = input[shift];
}

//---------------------------------------------------------------------------------------------------------------------
template <class T, class Tintbool>
__global__ void compact_stream_reverse_mask (T *output,
                                             const T *input, const Tintbool *mask, const int *mask_sum, int size){
//---------------------------------------------------------------------------------------------------------------------
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= size) return;

  if (mask[i]) return;

  int p = i - mask_sum[i];
  output[p] = input[i];
}

//=====CPU functions for stream compaction=============================================================================

//---------------------------------------------------------------------------------------------------------------------
template <class T>
void compact_stream (T **vector,
                     CUDA_mask& mask, int new_size, int blocks){
//---------------------------------------------------------------------------------------------------------------------
  int size = mask.size();
  dim3 grid_dim ((size + blocks - 1) / blocks);

  T *reduced_vector, *old_vector;
  cudaMalloc (&reduced_vector, new_size * sizeof(T));
  compact_stream<<<grid_dim,blocks>>> (reduced_vector, *vector, mask.cuda_mask(), mask.cuda_mask_sum(), size);

  // swap pointers
  old_vector = *vector;
  *vector = reduced_vector;
  reduced_vector = old_vector;
  cudaFree (reduced_vector);
}

//---------------------------------------------------------------------------------------------------------------------
template <class T>
void compact_stream_reverse_mask (T **vector,
                                  CUDA_mask& mask, int new_size, int blocks){
//---------------------------------------------------------------------------------------------------------------------
  int size = mask.size();
  dim3 grid_dim ((size + blocks - 1) / blocks);

  T *reduced_vector, *old_vector;
  cudaMalloc (&reduced_vector, new_size * sizeof(T));
  compact_stream_reverse_mask<<<grid_dim,blocks>>> (reduced_vector, *vector, mask.cuda_mask(), mask.cuda_mask_sum(), size);

  // swap pointers
  old_vector = *vector;
  *vector = reduced_vector;
  reduced_vector = old_vector;
  cudaFree (reduced_vector);
}
