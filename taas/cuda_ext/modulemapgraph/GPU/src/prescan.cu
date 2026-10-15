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
 * prescan algorithm based on CUDA Parallel Prefix Sum (SCAN) - see Mark Harris doc
 **********************************************************************/

#include <stdio.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <MMG/CUDA_vector_tools>
#include <MMG/CUDA_error_check>

#define NUM_BANKS 16
#define LOG_NUM_BANKS 4
//int THREADS_PER_BLOCK = 512;
//int ELEMENTS_PER_BLOCK = THREADS_PER_BLOCK * 2;

#ifdef ZERO_BANK_CONFLICTS
  #define CONFLICT_FREE_OFFSET(n) ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))
#else
  #define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)
#endif

//---------------------------------------------------------------------------------------------------------------------
template <class I>
__global__ void prescan_large (int *output, const I *input, int size, int *sums, int blocks)
//---------------------------------------------------------------------------------------------------------------------
{
	extern __shared__ int temp[];

	int blockID = blockIdx.x;
	int threadID = threadIdx.x;
	int blockOffset = blockID * blocks;
	int partial_sum = 0;
	if (blockOffset + threadID > size) return;

	int ai = threadID;
	int bi = threadID + (blocks / 2);
	int bankOffsetA = CONFLICT_FREE_OFFSET (ai);
	int bankOffsetB = CONFLICT_FREE_OFFSET (bi);

	temp[ai + bankOffsetA] = input[blockOffset + ai];
	if (blockOffset + bi < size)
		temp[bi + bankOffsetB] = input[blockOffset + bi];
	else
		temp[bi + bankOffsetB] = 0;

	int offset = 1;
	for (int d = blocks >> 1; d > 0; d >>= 1) { // build sum in place up the tree
	  __syncthreads();
	  if (threadID < d) {
		int ai = offset * (2 * threadID + 1) - 1;
		int bi = offset * (2 * threadID + 2) - 1;
		ai += CONFLICT_FREE_OFFSET (ai);
		bi += CONFLICT_FREE_OFFSET (bi);
		temp[bi] += temp[ai];
	  }
    offset *= 2;
	}

	if (threadID == 0) {
		partial_sum = temp[blocks - 1 + CONFLICT_FREE_OFFSET(blocks - 1)];
		temp[blocks - 1 + CONFLICT_FREE_OFFSET(blocks - 1)] = 0;
	}

	for (int d = 1; d < blocks; d *= 2) { // traverse down tree & build scan
		offset >>= 1;
		__syncthreads();
		if (threadID < d) {
			int ai = offset * (2 * threadID + 1) - 1;
			int bi = offset * (2 * threadID + 2) - 1;
			ai += CONFLICT_FREE_OFFSET (ai);
			bi += CONFLICT_FREE_OFFSET (bi);
			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();

// write results to device memory
	if (blockOffset + ai > size) return;

	output[blockOffset + ai] = temp[ai + bankOffsetA];
//	output[blockOffset + bi] = temp[bi + bankOffsetB];

	if (!threadID)
		sums[blockID] = partial_sum;
}

//---------------------------------------------------------------------------------------------------------------------
template <class I>
__global__ void scan_sum (int *output,
													const I *input, int size)
//---------------------------------------------------------------------------------------------------------------------
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i) return;
	int value = 0;

	output[0] = 0;
	for (int k=1; k<=size; k++) {
		value += input[k-1];
		output[k] = value;
	}
}

//---------------------------------------------------------------------------------------------------------------------
template <class I>
__global__ void scan_sum (I *vector, int size)
//---------------------------------------------------------------------------------------------------------------------
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i) return;

	for (int k=0; k<size-1; k++) {
		vector[k+1] += vector[k];
	}
}

//---------------------------------------------------------------------------------------------------------------------
__global__ void add (int *output,
										 int length, const int *n, int size)
//---------------------------------------------------------------------------------------------------------------------
{
	int blockID = blockIdx.x;
	int threadID = threadIdx.x;
	int blockOffset = blockID * length;

	if (blockOffset + threadID > size) return;
	output[blockOffset + threadID] += n[blockID];
}

//---------------------------------------------------------------------------------------------------------------------
template <class I>
void recursive_scan (int *output, const I *input, int blocks, int n, bool last_call=false)
//---------------------------------------------------------------------------------------------------------------------
{
  int *sums;
  int nb_blocks_input = (n + blocks - 1) / blocks; // = nb of elements for 'sums' (nb of elements to store 'input' elements)
  int nb_blocks_output = (n + blocks) / blocks; // = nb of blocks to store 'output' elements
	int nb_blocks_subsums = (nb_blocks_input + blocks - 1) / blocks; // nb of blocks to store 'sums'
	if (nb_blocks_subsums == 1 && !last_call) {
		last_call = true;
		nb_blocks_subsums++;
	}

  cudaMalloc (&sums, nb_blocks_subsums * blocks * sizeof(int));
	cudaMemset (sums, 0, nb_blocks_subsums * blocks * sizeof(int));

//  prescan_large<<<nb_blocks_output, THREADS_PER_BLOCK, 2 * ELEMENTS_PER_BLOCK * sizeof(int)>>> (output, input, n, sums, THREADS_PER_BLOCK);
  prescan_large<<<nb_blocks_output, blocks, 4 * blocks * sizeof(int)>>> (output, input, n, sums, blocks);

	if (nb_blocks_output > 1) {
		int *cumulative_sums;
		cudaMalloc (&cumulative_sums, ((nb_blocks_subsums+1) * blocks) * sizeof(int));
		cudaMemset (cumulative_sums, 0, ((nb_blocks_subsums+1) * blocks) * sizeof(int));
		if (nb_blocks_subsums > 1)
			recursive_scan (cumulative_sums, sums, blocks, nb_blocks_input, last_call);
		add<<<nb_blocks_output+1,blocks>>> (output, blocks, cumulative_sums, n);
		cudaFree (cumulative_sums);
	}
	cudaFree (sums);
}

//---------------------------------------------------------------------------------------------------------------------
template <class I>
int scan (int *output, I *input, int blocks, int n)
//---------------------------------------------------------------------------------------------------------------------
{
	int sum;

	recursive_scan (output, input, blocks, n);
	cudaMemcpyAsync (&sum, &output[n], sizeof (int), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	return sum;
}

//---------------------------------------------------------------------------------------------------------------------
template <class T>
int thrust_scan (T *vector, int n)
//---------------------------------------------------------------------------------------------------------------------
{
	int sum = 0;
  thrust::exclusive_scan (thrust::device, vector, vector+n+1, vector);
  cudaMemcpyAsync (&sum, &vector[n], sizeof(T), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	return sum;
}
