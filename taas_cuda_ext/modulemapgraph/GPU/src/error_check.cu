#include <MMG/colors>

/***********************************************************************
 * Module Map Graph library
 * Track reconstruction for HEP
 * copyright © 2025 HUTH Benjamin
 * Laboratoire des 2 Infinis - Toulouse - France (L2IT)
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 **********************************************************************/

 #if 0
#define CUDA_PRINTV(ptr, size) \
	std::cout << #ptr << ": "; \
  cudaDeviceSynchronize(); \
	print<<<1,1>>>(ptr, size); \
  cudaDeviceSynchronize(); \
	std::cout << std::endl; 
#else
#define CUDA_PRINTV(ptr, size) /*nothing*/
#endif

inline void gpuAssert(cudaError_t code, const char *file, int line)
{
   if (code != cudaSuccess) 
   {
	 		std::stringstream ss;
			ss << red << "CUDA error: " << cudaGetErrorString(code) << ", " << file << ":" << line;
   		throw std::runtime_error(ss.str());
	 }
   cudaError_t sync_code = cudaDeviceSynchronize();
   if (sync_code != cudaSuccess)
   {
	 		std::stringstream ss;
			ss << red << "CUDA synchronize error: " << cudaGetErrorString(sync_code) << ", " << file << ":" << line;
   		throw std::runtime_error(ss.str());
   }
}

// use it with GPU_ERROR_CHECK (cudaGetLastError());
#if 1
#define GPU_ERROR_CHECK(ans) { cudaDeviceSynchronize(); gpuAssert ((ans), __FILE__, __LINE__); }
#else 
#define GPU_ERROR_CHECK(ans) /*nothing*/
#endif
